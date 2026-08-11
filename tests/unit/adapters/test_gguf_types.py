from __future__ import annotations

import re
from dataclasses import replace

import pytest

from vramfit.adapters.outbound.gguf.types import (
    BASE_FTYPE_BY_BITS,
    GGML_TYPE_BY_BITS,
    PackError,
    base_type,
    check_runtime,
    ggml_type_for,
    gguf_tensor_name,
    imatrix_exclusion_names,
    output_tensor_type,
    protection_overrides,
    tensor_overrides,
    token_embedding_type,
)
from vramfit.domain.errors import VramfitError
from vramfit.domain.model import Assignment, PlanMeta, ProtectedTensor, Recipe
from vramfit.domain.pack import TypeOverride
from vramfit.domain.runtime import EFFECTIVE_BITS, LLAMA_CPP, RUNTIME_CAPABILITIES

pytestmark = pytest.mark.unit

# Block-layout costs of the K-quant types this backend drives, in
# bits per weight. Independent of the domain table on purpose: the
# pairing test below fails if either side drifts.
BITS_PER_GGML_TYPE = {
    "q8_0": 8.5,
    "q6_k": 6.5625,
    "q5_k": 5.5,
    "q4_k": 4.5,
    "q3_k": 3.4375,
    "q2_k": 2.625,
}


def make_recipe(*assignments: tuple[str, int]) -> Recipe:
    return Recipe(
        model_id="test/model",
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=3_000,
            predicted_total_bytes=2_000,
            predicted_damage=0.01,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=tuple(
            Assignment(group=group, bits=bits, bytes=100, damage=0.001)
            for group, bits in assignments
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


def test_type_table_covers_the_llama_cpp_capability_set() -> None:
    assert set(GGML_TYPE_BY_BITS) == {8, 6, 5, 4, 3, 2}
    assert set(BASE_FTYPE_BY_BITS) == {8, 6, 5, 4, 3, 2}
    assert set(GGML_TYPE_BY_BITS) == RUNTIME_CAPABILITIES["llama.cpp"]


def test_gguf_types_spend_the_domains_effective_bits() -> None:
    # The domain's EFFECTIVE_BITS values are only correct while this
    # backend maps each nominal precision to the type they price
    # (ADR-0014). Swapping q5_k for q5_1 must fail here, loudly.
    for bits, ggml_type in GGML_TYPE_BY_BITS.items():
        assert EFFECTIVE_BITS[LLAMA_CPP][bits] == BITS_PER_GGML_TYPE[ggml_type]


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        (8, "q8_0"),
        (6, "q6_k"),
        (5, "q5_k"),
        (4, "q4_k"),
        (3, "q3_k"),
        (2, "q2_k"),
    ],
    ids=["8-bit", "6-bit", "5-bit", "4-bit", "3-bit", "2-bit"],
)
def test_ggml_type_for_maps_the_fixed_table(bits: int, expected: str) -> None:
    assert ggml_type_for(bits) == expected


def test_ggml_type_for_unknown_bits_raises_pack_error() -> None:
    with pytest.raises(PackError, match="no GGUF type maps 16-bit"):
        ggml_type_for(16)


def test_check_runtime_accepts_llama_cpp_and_none() -> None:
    recipe = make_recipe(("model.layers.0", 4))

    check_runtime(recipe)
    check_runtime(replace(recipe, runtime="llama.cpp"))


def test_check_runtime_rejects_a_foreign_runtime() -> None:
    recipe = replace(make_recipe(("model.layers.0", 4)), runtime="vllm")

    with pytest.raises(PackError, match=r"packs for llama\.cpp"):
        check_runtime(recipe)


def test_pack_error_inherits_the_vramfit_root() -> None:
    assert issubclass(PackError, VramfitError)
    assert issubclass(PackError, RuntimeError)


@pytest.mark.parametrize(
    ("floor_bits", "expected"),
    [(6, "Q6_K"), (5, "Q5_K_S"), (3, "Q3_K_S")],
    ids=["6-bit", "5-bit", "3-bit"],
)
def test_base_type_is_the_recipe_floor(floor_bits: int, expected: str) -> None:
    recipe = make_recipe(("model.layers.0", 8), ("model.layers.1", floor_bits))

    assert base_type(recipe) == expected


def test_base_type_with_unmapped_floor_raises_pack_error() -> None:
    recipe = make_recipe(("model.layers.0", 8), ("model.layers.1", 7))

    with pytest.raises(PackError, match="no GGUF base type maps 7-bit"):
        base_type(recipe)


def test_token_embedding_type_maps_the_embedding_group() -> None:
    recipe = make_recipe(("model.embed_tokens", 8), ("model.layers.0", 4))

    assert token_embedding_type(recipe) == "q8_0"


def test_token_embedding_type_without_embedding_group_is_none() -> None:
    recipe = make_recipe(("model.layers.0", 4))

    assert token_embedding_type(recipe) is None


def test_output_tensor_type_with_lm_head_group_uses_its_own_assignment() -> None:
    recipe = make_recipe(
        ("model.embed_tokens", 8),
        ("lm_head", 4),
        ("model.layers.0", 2),
    )

    assert output_tensor_type(recipe) == "q4_k"


def test_output_tensor_type_without_lm_head_group_pins_to_the_embedding() -> None:
    recipe = make_recipe(("model.embed_tokens", 8), ("model.layers.0", 4))

    assert output_tensor_type(recipe) == "q8_0"


def test_output_tensor_type_without_either_group_is_none() -> None:
    recipe = make_recipe(("model.layers.0", 4))

    assert output_tensor_type(recipe) is None


def test_tensor_overrides_escape_dots_so_layer_1_never_matches_layer_11() -> None:
    recipe = make_recipe(("model.layers.1", 8), ("model.layers.11", 4))

    patterns = [override.pattern for override in tensor_overrides(recipe)]

    assert patterns == [r"blk\.1\.", r"blk\.11\."]
    assert re.search(patterns[0], "blk.1.attn_q.weight")
    assert not re.search(patterns[0], "blk.11.attn_q.weight")


def test_tensor_overrides_keep_recipe_order_and_skip_the_flag_groups() -> None:
    recipe = make_recipe(
        ("model.embed_tokens", 8),
        ("lm_head", 4),
        ("model.layers.0", 4),
        ("model.layers.1", 2),
    )

    overrides = tensor_overrides(recipe)

    assert [(o.pattern, o.quant_type) for o in overrides] == [
        (r"blk\.0\.", "q4_k"),
        (r"blk\.1\.", "q2_k"),
    ]


def test_tensor_overrides_reject_tensor_level_groups() -> None:
    recipe = make_recipe(("model.layers.0.self_attn.q_proj.weight", 4))

    with pytest.raises(PackError, match="no GGUF tensor mapping"):
        tensor_overrides(recipe)


def make_protected_recipe(
    pairs: tuple[tuple[str, int], ...],
    *assignments: tuple[str, int],
    excluded: tuple[str, ...] = (),
) -> Recipe:
    base = make_recipe(*assignments)
    return replace(
        base,
        plan=replace(
            base.plan,
            protections={"user-pattern": min(b for _, b in pairs)},
            imatrix_exclusions=("user-exclusion",) if excluded else (),
        ),
        protected_tensors=tuple(
            ProtectedTensor(
                tensor=tensor, bits=bits, exclude_imatrix=tensor in excluded
            )
            for tensor, bits in pairs
        ),
    )


class TestGgufTensorName:
    @pytest.mark.parametrize(
        ("tensor", "expected"),
        [
            ("model.layers.4.self_attn.v_proj.weight", "blk.4.attn_v.weight"),
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.11.self_attn.k_proj.weight", "blk.11.attn_k.weight"),
            ("model.layers.7.self_attn.o_proj.weight", "blk.7.attn_output.weight"),
            ("model.layers.3.mlp.gate_proj.weight", "blk.3.ffn_gate.weight"),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            ("model.layers.79.mlp.down_proj.weight", "blk.79.ffn_down.weight"),
        ],
        ids=["v", "q", "k", "o", "gate", "up", "down"],
    )
    def test_class_table_maps_the_seven_projections(
        self, tensor: str, expected: str
    ) -> None:
        assert gguf_tensor_name(tensor) == expected

    @pytest.mark.parametrize(
        "tensor",
        [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.4.input_layernorm.weight",
            "model.layers.4.self_attn.v_proj.bias",
            "transformer.h.4.attn.c_attn.weight",
        ],
        ids=["embedding", "head", "norm", "bias", "foreign-arch"],
    )
    def test_unmappable_tensor_raises_pack_error(self, tensor: str) -> None:
        with pytest.raises(PackError, match="no GGUF mapping"):
            gguf_tensor_name(tensor)


class TestProtectionOverrides:
    def test_resolved_pair_becomes_escaped_override(self) -> None:
        recipe = make_protected_recipe(
            (("model.layers.4.self_attn.v_proj.weight", 5),),
            ("model.layers.4", 3),
        )

        assert protection_overrides(recipe) == (
            TypeOverride(pattern=r"blk\.4\.attn_v\.", quant_type="q5_k"),
        )

    def test_escaping_keeps_layer_four_from_matching_layer_forty(self) -> None:
        recipe = make_protected_recipe(
            (("model.layers.4.self_attn.v_proj.weight", 5),),
            ("model.layers.4", 3),
        )

        pattern = protection_overrides(recipe)[0].pattern
        assert re.search(pattern, "blk.4.attn_v.weight")
        assert not re.search(pattern, "blk.40.attn_v.weight")
        assert not re.search(pattern, "blk.14.attn_v.weight")

    def test_unprotected_recipe_yields_no_overrides(self) -> None:
        assert protection_overrides(make_recipe(("model.layers.0", 4))) == ()

    def test_unmapped_precision_raises_pack_error(self) -> None:
        recipe = make_protected_recipe(
            (("model.layers.4.self_attn.v_proj.weight", 7),),
            ("model.layers.4", 3),
        )

        with pytest.raises(PackError, match="no GGUF type maps 7-bit"):
            protection_overrides(recipe)


class TestImatrixExclusionNames:
    def test_marked_pair_yields_the_full_gguf_name(self) -> None:
        recipe = make_protected_recipe(
            (
                ("model.layers.1.self_attn.v_proj.weight", 5),
                ("model.layers.4.self_attn.v_proj.weight", 5),
            ),
            ("model.layers.1", 3),
            ("model.layers.4", 3),
            excluded=("model.layers.1.self_attn.v_proj.weight",),
        )

        assert imatrix_exclusion_names(recipe) == ("blk.1.attn_v.weight",)

    def test_recipe_without_marks_yields_nothing(self) -> None:
        recipe = make_protected_recipe(
            (("model.layers.4.self_attn.v_proj.weight", 5),),
            ("model.layers.4", 3),
        )

        assert imatrix_exclusion_names(recipe) == ()

    def test_unmappable_excluded_tensor_raises_pack_error(self) -> None:
        recipe = make_protected_recipe(
            (("model.layers.4.input_layernorm.weight", 5),),
            ("model.layers.4", 3),
            excluded=("model.layers.4.input_layernorm.weight",),
        )

        with pytest.raises(PackError, match="no GGUF mapping"):
            imatrix_exclusion_names(recipe)
