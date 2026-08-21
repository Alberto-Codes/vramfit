from __future__ import annotations

import re
from dataclasses import replace

import pytest

from vramfit.adapters.outbound.gguf.types import (
    _LAYER_TENSOR,
    BASE_FTYPE_BY_BITS,
    GGML_TYPE_BY_BITS,
    PackError,
    all_overrides,
    base_type,
    check_runtime,
    expert_stack_type_for,
    ggml_type_for,
    gguf_stack_prefix,
    gguf_tensor_name,
    imatrix_exclusion_names,
    output_group_type,
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
    "f16": 16.0,
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
    # 16 is the ADR-0029 F16 passthrough, which the capability table
    # carries so a recipe can hold an unmeasured group at reference.
    assert set(GGML_TYPE_BY_BITS) == {16, 8, 6, 5, 4, 3, 2}
    assert set(BASE_FTYPE_BY_BITS) == {16, 8, 6, 5, 4, 3, 2}
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
        (16, "f16"),
        (8, "q8_0"),
        (6, "q6_k"),
        (5, "q5_k"),
        (4, "q4_k"),
        (3, "q3_k"),
        (2, "q2_k"),
    ],
    ids=["16-bit", "8-bit", "6-bit", "5-bit", "4-bit", "3-bit", "2-bit"],
)
def test_ggml_type_for_maps_the_fixed_table(bits: int, expected: str) -> None:
    assert ggml_type_for(bits) == expected


def test_ggml_type_for_unknown_bits_raises_pack_error() -> None:
    with pytest.raises(PackError, match="no GGUF type maps 7-bit"):
        ggml_type_for(7)


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


def test_output_group_type_with_lm_head_group_reads_its_assignment() -> None:
    recipe = make_recipe(("model.embed_tokens", 8), ("lm_head", 4))

    assert output_group_type(recipe) == "q4_k"


def test_output_group_type_without_lm_head_group_is_none() -> None:
    # The distinction the pack step reads (#306). `output_tensor_type`
    # returns the embedding's `q8_0` here, and that flag is a ruled
    # no-op on a tied model — so it must not be held against the file.
    recipe = make_recipe(("model.embed_tokens", 8), ("model.layers.0", 4))

    assert output_tensor_type(recipe) == "q8_0"
    assert output_group_type(recipe) is None


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


def test_tensor_overrides_name_the_group_they_cannot_map() -> None:
    # The refusal is the backend's only guard against mispacking a
    # group it does not understand, so the message must name the
    # group and what it does map (#180). The dense-MLP `mixer.up_proj`
    # has no class-table row — no scanned target carries it.
    recipe = make_recipe(("backbone.layers.0.mixer.up_proj", 4))

    with pytest.raises(PackError) as caught:
        tensor_overrides(recipe)

    message = str(caught.value)
    assert '"backbone.layers.0.mixer.up_proj"' in message
    assert "layer groups, routed-expert stacks" in message


@pytest.mark.parametrize(
    ("group", "pattern"),
    [
        ("model.layers.0", r"blk\.0\."),
        ("backbone.layers.7", r"blk\.7\."),
        ("transformer.h.3", r"blk\.3\."),
        ("gpt_neox.blocks.11", r"blk\.11\."),
    ],
    ids=["llama", "nemotron", "gpt2", "blocks"],
)
def test_tensor_overrides_derive_the_layer_index_from_any_naming_family(
    group: str, pattern: str
) -> None:
    # GGUF numbers every decoder layer `blk.<n>.` whatever the
    # checkpoint calls it, so the index is derived, not matched
    # against one fixed prefix (#160, #180).
    recipe = make_recipe((group, 4))

    assert tensor_overrides(recipe) == (TypeOverride(pattern, "q4_k"),)


@pytest.mark.parametrize(
    ("group", "tensor"),
    [
        ("model.layers.0.mlp.experts.up_proj", "blk.0.ffn_up_exps.weight"),
        ("backbone.layers.9.mixer.experts.down_proj", "blk.9.ffn_down_exps.weight"),
        (
            "model.layers.2.block_sparse_moe.experts.gate_proj",
            "blk.2.ffn_gate_exps.weight",
        ),
    ],
    ids=["up", "down", "gate"],
)
def test_tensor_overrides_map_a_routed_expert_stack_to_its_fused_tensor(
    group: str, tensor: str
) -> None:
    # llama.cpp fuses one layer's experts into a single tensor with
    # one type, so the stack is what a pack addresses (#159, #161).
    recipe = make_recipe((group, 4))

    overrides = tensor_overrides(recipe)

    assert len(overrides) == 1
    assert re.search(overrides[0].pattern, tensor)
    assert overrides[0].quant_type == "q4_0"


def test_tensor_overrides_escape_the_stack_pattern_so_layer_1_never_matches_11() -> (
    None
):
    recipe = make_recipe(("model.layers.1.mlp.experts.up_proj", 8))

    pattern = tensor_overrides(recipe)[0].pattern

    assert re.search(pattern, "blk.1.ffn_up_exps.weight")
    assert not re.search(pattern, "blk.11.ffn_up_exps.weight")
    assert not re.search(pattern, "blk.1.ffn_up_exps_scale.weight")


def test_tensor_overrides_put_stacks_before_layers_so_the_stack_wins() -> None:
    # The quantizer applies the first matching pattern, and the
    # layer pattern `blk\.1\.` also matches `blk.1.ffn_up_exps.
    # weight`. A stack override placed second would never apply.
    recipe = make_recipe(
        ("model.layers.1", 8),
        ("model.layers.1.mlp.experts.up_proj", 2),
    )

    overrides = tensor_overrides(recipe)

    assert [o.quant_type for o in overrides] == ["q2_0", "q8_0"]
    assert re.search(overrides[0].pattern, "blk.1.ffn_up_exps.weight")


def test_tensor_overrides_keep_recipe_order_within_the_stack_bucket() -> None:
    recipe = make_recipe(
        ("backbone.layers.5.mixer.experts.down_proj", 2),
        ("backbone.layers.1.mixer.experts.up_proj", 8),
    )

    assert [o.quant_type for o in tensor_overrides(recipe)] == ["q2_0", "q8_0"]


@pytest.mark.parametrize(
    ("bits", "quant_type"), [(8, "q8_0"), (4, "q4_0"), (2, "q2_0")]
)
def test_expert_stack_type_for_maps_the_adr_0028_table(
    bits: int, quant_type: str
) -> None:
    # Every k-quant packs 256-element super-blocks and the stack rows
    # (2688, 1856) do not divide by 256, so the stack table carries
    # only types whose block size divides both (ADR-0028).
    assert expert_stack_type_for(bits, "model.layers.0.mlp.experts.up_proj") == (
        quant_type
    )


def test_tensor_overrides_refuse_nominal_3_on_an_expert_stack_naming_the_gap() -> None:
    # No GGUF type lands between 2.25 and 4.25 bits per weight on the
    # stack rows (ADR-0028 decision 2). The refusal names the group,
    # the gap, and both neighboring table entries.
    recipe = make_recipe(("backbone.layers.3.mixer.experts.up_proj", 3))

    with pytest.raises(PackError) as caught:
        tensor_overrides(recipe)

    message = str(caught.value)
    assert '"backbone.layers.3.mixer.experts.up_proj"' in message
    assert "2.25" in message
    assert "4.25" in message
    assert "q2_0" in message
    assert "q4_0" in message


def test_expert_stack_type_for_refuses_a_precision_outside_the_table() -> None:
    # The dense table maps 6-bit to Q6_K, but the stack table has no
    # 6-bit row — silently keeping the k-quant would let the
    # quantizer substitute (ADR-0028 decision 1).
    with pytest.raises(PackError) as caught:
        expert_stack_type_for(6, "model.layers.0.mlp.experts.up_proj")

    message = str(caught.value)
    assert '"model.layers.0.mlp.experts.up_proj"' in message
    assert "[2, 4, 8, 16]" in message


def test_tensor_overrides_reject_an_expert_projection_outside_the_stack_table() -> None:
    # Mixtral spells its projections w1/w2/w3. Guessing which fused
    # tensor those become would mispack silently (#159).
    recipe = make_recipe(("model.layers.0.block_sparse_moe.experts.w1", 4))

    with pytest.raises(PackError) as caught:
        tensor_overrides(recipe)

    message = str(caught.value)
    assert '"model.layers.0.block_sparse_moe.experts.w1"' in message
    assert "down_proj" in message


@pytest.mark.parametrize(
    "group",
    ["model.embed_tokens", "backbone.embeddings", "model.embeddings"],
    ids=["llama", "nemotron", "nemotron-reconciled"],
)
def test_token_embedding_type_maps_every_embedding_naming_family(group: str) -> None:
    # `--token-embedding-type` binds one tensor whatever the
    # checkpoint calls the group. Missing a name refuses the whole
    # recipe, because the group then reaches the pattern branch.
    recipe = make_recipe((group, 8), ("backbone.layers.0", 4))

    assert token_embedding_type(recipe) == "q8_0"
    assert [o.pattern for o in tensor_overrides(recipe)] == [r"blk\.0\."]


def test_tensor_overrides_reject_two_layer_stacks_naming_both() -> None:
    # GGUF numbers one layer stack `blk.<n>.`. The target carries
    # `mtp.layers.<n>` beside `backbone.layers.<n>`, and a
    # multimodal checkpoint carries a vision tower. Mapping both
    # onto `blk.0.` would drop one silently (#183).
    recipe = make_recipe(("backbone.layers.0", 8), ("mtp.layers.0", 4))

    with pytest.raises(PackError) as caught:
        tensor_overrides(recipe)

    message = str(caught.value)
    assert "two layer stacks" in message
    assert '"backbone.layers.0"' in message
    assert '"mtp.layers.0"' in message


def test_tensor_overrides_reject_a_vision_tower_beside_the_language_model() -> None:
    # GGUF names vision blocks `v.blk.<n>.`, not `blk.<n>.`.
    recipe = make_recipe(
        ("model.layers.0", 4),
        ("vision_tower.transformer.layers.0", 4),
    )

    with pytest.raises(PackError, match="two layer stacks"):
        tensor_overrides(recipe)


def test_tensor_overrides_accept_one_root_across_layers_and_stacks() -> None:
    recipe = make_recipe(
        ("backbone.layers.0", 8),
        ("backbone.layers.1.mixer.experts.up_proj", 4),
        ("backbone.layers.2", 2),
    )

    assert len(tensor_overrides(recipe)) == 3


def test_tensor_overrides_map_a_shared_expert_as_its_own_class() -> None:
    # A shared expert is a separate GGUF tensor, `ffn_up_shexp` —
    # never part of the fused routed stack. The class table maps it
    # under a free prefix at three suffix segments, through the
    # ADR-0028 type table (the 2026-08-20 amendment).
    recipe = make_recipe(("backbone.layers.1.mixer.shared_experts.up_proj", 4))

    assert tensor_overrides(recipe) == (
        TypeOverride(r"blk\.1\.ffn_up_shexp\.", "q4_0"),
    )


@pytest.mark.parametrize(
    ("group", "bits", "pattern", "quant_type"),
    [
        ("model.layers.3.mixer.in_proj", 16, r"blk\.3\.ssm_in\.", "f16"),
        ("backbone.layers.3.mixer.in_proj", 4, r"blk\.3\.ssm_in\.", "q4_0"),
        ("model.layers.7.mixer.out_proj", 2, r"blk\.7\.ssm_out\.", "q2_0"),
        ("model.layers.2.mixer.q_proj", 8, r"blk\.2\.attn_q\.", "q8_0"),
        (
            "backbone.layers.1.mixer.shared_experts.down_proj",
            4,
            r"blk\.1\.ffn_down_shexp\.",
            "q4_0",
        ),
    ],
    ids=["ssm-in-f16", "ssm-in-free-prefix", "ssm-out", "attn-q", "shexp-down"],
)
def test_tensor_overrides_map_a_layer_class_through_the_adr_0028_table(
    group: str, bits: int, pattern: str, quant_type: str
) -> None:
    # The nine Nemotron-H rows map under a free prefix, and their
    # rows refuse the 256 super-block, so the ADR-0028 table supplies
    # the type (the 2026-08-20 amendment).
    recipe = make_recipe((group, bits))

    assert tensor_overrides(recipe) == (TypeOverride(pattern, quant_type),)


def test_tensor_overrides_route_a_llama_class_through_the_kquant_table() -> None:
    # A llama-family class row keeps decision 1's table: its rows
    # divide the 256 super-block, so k-quants reach them.
    recipe = make_recipe(("model.layers.2.self_attn.q_proj", 4))

    assert tensor_overrides(recipe) == (TypeOverride(r"blk\.2\.attn_q\.", "q4_k"),)


def test_tensor_overrides_refuse_nominal_3_on_a_layer_class_naming_the_gap() -> None:
    recipe = make_recipe(("model.layers.3.mixer.in_proj", 3))

    with pytest.raises(PackError) as caught:
        tensor_overrides(recipe)

    message = str(caught.value)
    assert 'layer-class group "model.layers.3.mixer.in_proj"' in message
    assert "between 2.25 and 4.25" in message


@pytest.mark.parametrize("bits", [6, 5])
def test_tensor_overrides_refuse_5_and_6_bit_on_an_adr_0028_routed_class(
    bits: int,
) -> None:
    # The ADR-0028 table carries no 5- or 6-bit row (#232). The class
    # rows at 2688 would take Q5_0's block of 32, so the gap is the
    # table's, not the rows' — the refusal names the table.
    recipe = make_recipe(("model.layers.3.mixer.in_proj", bits))

    with pytest.raises(PackError, match="ADR-0028 table covers"):
        tensor_overrides(recipe)


def test_tensor_overrides_escape_the_class_pattern_so_layer_1_never_matches_11() -> (
    None
):
    recipe = make_recipe(("model.layers.1.mixer.in_proj", 16))

    (override,) = tensor_overrides(recipe)

    assert re.search(override.pattern, "blk.1.ssm_in.weight")
    assert not re.search(override.pattern, "blk.11.ssm_in.weight")


@pytest.mark.parametrize(
    "group",
    ["model.layers.4.mixer.gate", "model.layers.4.mixer.conv1d"],
    ids=["router", "conv1d"],
)
def test_tensor_overrides_hold_an_unquantizable_class_without_an_override(
    group: str,
) -> None:
    # llama-quantize refuses these tensors and holds them at the
    # convert dtype, so the F16 passthrough needs no override (the
    # 2026-08-20 amendment).
    recipe = make_recipe((group, 16), ("model.layers.0", 4))

    assert tensor_overrides(recipe) == (TypeOverride(r"blk\.0\.", "q4_k"),)


@pytest.mark.parametrize(
    ("group", "filter_name"),
    [
        ("model.layers.4.mixer.gate", "ffn_gate_inp.weight"),
        ("backbone.layers.9.mixer.conv1d", "ssm_conv1d"),
    ],
    ids=["router", "conv1d"],
)
def test_tensor_overrides_refuse_a_width_below_the_passthrough_naming_the_filter(
    group: str, filter_name: str
) -> None:
    # The quantizer drops an override on a refused tensor and exits
    # 0, so a lower width would record a type the artifact cannot
    # carry (the 2026-08-20 amendment).
    recipe = make_recipe((group, 8))

    with pytest.raises(PackError) as caught:
        tensor_overrides(recipe)

    message = str(caught.value)
    assert f'"{group}"' in message
    assert filter_name in message
    assert "F16 passthrough" in message


def test_tensor_overrides_put_classes_between_stacks_and_layers() -> None:
    # The quantizer applies the first matching pattern. `blk\.0\.`
    # also matches every class tensor of layer 0, so the class
    # pattern must come first.
    recipe = make_recipe(
        ("model.layers.0", 4),
        ("model.layers.0.mixer.in_proj", 16),
        ("model.layers.0.mixer.experts.up_proj", 2),
    )

    assert tensor_overrides(recipe) == (
        TypeOverride(r"blk\.0\.ffn_up_exps\.", "q2_0"),
        TypeOverride(r"blk\.0\.ssm_in\.", "f16"),
        TypeOverride(r"blk\.0\.", "q4_k"),
    )


def test_tensor_overrides_reject_a_class_group_under_a_second_root() -> None:
    # A layer-class group claims its root like any other mapped
    # group, so the two-root refusal stands (the 2026-08-20
    # amendment).
    recipe = make_recipe(
        ("model.layers.0.mixer.in_proj", 16),
        ("mtp.layers.0", 4),
    )

    with pytest.raises(PackError, match="two layer stacks"):
        tensor_overrides(recipe)


def test_gguf_stack_prefix_maps_experts_attached_straight_to_the_layer() -> None:
    assert gguf_stack_prefix("model.layers.4.experts.up_proj") == "blk.4.ffn_up_exps."


def test_gguf_stack_prefix_returns_none_for_a_plain_layer_group() -> None:
    assert gguf_stack_prefix("model.layers.4") is None
    assert gguf_stack_prefix("model.embed_tokens") is None


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

    @pytest.mark.parametrize(
        ("tensor", "expected"),
        [
            ("model.layers.4.mixer.q_proj.weight", "blk.4.attn_q.weight"),
            ("backbone.layers.4.mixer.v_proj.weight", "blk.4.attn_v.weight"),
        ],
        ids=["model-root", "backbone-root"],
    )
    def test_mixer_classes_map_under_either_root(
        self, tensor: str, expected: str
    ) -> None:
        # The 2026-08-20 amendment's rows reach this path under
        # either scan root (#365).
        assert gguf_tensor_name(tensor) == expected

    def test_a_three_segment_row_maps_through_the_class_table(self) -> None:
        # The widened suffix capture reaches the three-segment rows
        # (#365), and the class table holds two of them (the
        # 2026-08-20 amendment).
        tensor = "model.layers.4.mixer.shared_experts.up_proj.weight"

        assert gguf_tensor_name(tensor) == "blk.4.ffn_up_shexp.weight"

    @pytest.mark.parametrize(
        ("tensor", "filter_name"),
        [
            ("model.layers.4.mixer.gate.weight", "ffn_gate_inp.weight"),
            ("model.layers.9.mixer.conv1d.weight", "ssm_conv1d"),
        ],
        ids=["router", "conv1d"],
    )
    def test_an_unquantizable_class_refuses_naming_the_filter(
        self, tensor: str, filter_name: str
    ) -> None:
        # The quantizer drops an override on a refused tensor and
        # exits 0, so a protection pair here would record a type the
        # artifact does not carry (the 2026-08-20 amendment). The
        # filter check runs before the class-table match, so
        # `conv1d` — which has no class-table row — still names its
        # filter rather than a missing mapping.
        with pytest.raises(PackError) as caught:
            gguf_tensor_name(tensor)

        message = str(caught.value)
        assert f'"{tensor}"' in message
        assert filter_name in message

    def test_a_protection_pair_on_an_unquantizable_class_refuses(self) -> None:
        recipe = make_protected_recipe(
            (("model.layers.4.mixer.gate.weight", 8),),
            ("model.layers.4", 4),
        )

        with pytest.raises(PackError, match="refuses to quantize"):
            protection_overrides(recipe)

    @pytest.mark.parametrize(
        "root",
        ["model", "backbone"],
        ids=["model-root", "backbone-root"],
    )
    def test_class_table_maps_either_scan_root(self, root: str) -> None:
        name = f"{root}.layers.4.self_attn.v_proj.weight"
        assert gguf_tensor_name(name) == "blk.4.attn_v.weight"

    def test_suffix_capture_holds_three_segments(self) -> None:
        match = _LAYER_TENSOR.match(
            "backbone.layers.1.mixer.shared_experts.down_proj.weight"
        )
        assert match is not None
        assert match.group(2) == "mixer.shared_experts.down_proj"

    @pytest.mark.parametrize(
        "tensor",
        [
            "mtp.layers.0.self_attn.v_proj.weight",
            "transformer.h.4.self_attn.v_proj.weight",
        ],
        ids=["mtp-root", "foreign-family"],
    )
    def test_root_outside_the_two_scan_roots_raises_pack_error(
        self, tensor: str
    ) -> None:
        with pytest.raises(PackError, match="no GGUF mapping"):
            gguf_tensor_name(tensor)

    @pytest.mark.parametrize(
        "tensor",
        [
            "backbone.layers.1.mixer.experts.0.up_proj.weight",
            "model.layers.4.self_attn.v_proj.weight.weight",
        ],
        ids=["expert-stack", "double-suffix"],
    )
    def test_suffix_outside_the_class_table_raises_pack_error(
        self, tensor: str
    ) -> None:
        # The class table holds the mixer and shared-expert rows now
        # (the 2026-08-20 amendment), so only a suffix outside the
        # table refuses here.
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


class TestAllOverrides:
    def test_protection_overrides_lead_the_group_overrides(self) -> None:
        # The quantizer applies the first matching pattern, so a
        # per-tensor pattern must precede its group's (ADR-0022).
        recipe = make_protected_recipe(
            (("model.layers.4.self_attn.v_proj.weight", 5),),
            ("model.layers.4", 3),
        )

        assert all_overrides(recipe) == (
            TypeOverride(pattern=r"blk\.4\.attn_v\.", quant_type="q5_k"),
            TypeOverride(pattern=r"blk\.4\.", quant_type="q3_k"),
        )

    def test_composition_equals_its_two_halves(self) -> None:
        recipe = make_protected_recipe(
            (("model.layers.4.self_attn.v_proj.weight", 5),),
            ("model.layers.4", 3),
        )

        assert all_overrides(recipe) == protection_overrides(recipe) + tensor_overrides(
            recipe
        )

    def test_unprotected_recipe_yields_the_group_overrides_alone(self) -> None:
        recipe = make_recipe(("model.layers.0", 4))

        assert all_overrides(recipe) == tensor_overrides(recipe)

    @pytest.mark.parametrize("root", ["model", "backbone"], ids=["model", "backbone"])
    def test_protection_under_the_groups_root_passes(self, root: str) -> None:
        recipe = make_protected_recipe(
            ((f"{root}.layers.4.self_attn.v_proj.weight", 5),),
            (f"{root}.layers.4", 4),
        )

        assert all_overrides(recipe) == (
            TypeOverride(pattern=r"blk\.4\.attn_v\.", quant_type="q5_k"),
            TypeOverride(pattern=r"blk\.4\.", quant_type="q4_k"),
        )

    @pytest.mark.parametrize(
        ("protection_root", "group_root"),
        [("model", "backbone"), ("backbone", "model")],
        ids=["model-protection", "backbone-protection"],
    )
    def test_protection_under_a_second_root_refuses_naming_both_roots(
        self, protection_root: str, group_root: str
    ) -> None:
        # A protection is not a group, so `_claim_root` never saw it.
        # Both roots emitted overrides onto one `blk.<n>.` namespace,
        # and the protection held the other root's tensor (#367).
        recipe = make_protected_recipe(
            ((f"{protection_root}.layers.4.self_attn.v_proj.weight", 5),),
            (f"{group_root}.layers.4", 4),
        )

        with pytest.raises(PackError) as caught:
            all_overrides(recipe)

        message = str(caught.value)
        assert "two layer stacks" in message
        assert f'root "{protection_root}"' in message
        assert f'root "{group_root}"' in message
        assert f'"{protection_root}.layers.4.self_attn.v_proj.weight"' in message
        assert f'"{group_root}.layers.4"' in message


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
