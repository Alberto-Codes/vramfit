from __future__ import annotations

import re
from dataclasses import replace

import pytest

from quantfit.adapters.outbound.gguf.types import (
    BASE_FTYPE_BY_BITS,
    GGML_TYPE_BY_BITS,
    PackError,
    base_type,
    check_runtime,
    ggml_type_for,
    tensor_overrides,
    token_embedding_type,
)
from quantfit.domain.errors import QuantfitError
from quantfit.domain.model import Assignment, PlanMeta, Recipe
from quantfit.domain.runtime import RUNTIME_CAPABILITIES

pytestmark = pytest.mark.unit


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
            format_overhead=0.05,
            trace=(),
        ),
        assignments=tuple(
            Assignment(group=group, bits=bits, bytes=100, damage=0.001)
            for group, bits in assignments
        ),
        runtime=None,
    )


def test_type_table_covers_the_llama_cpp_capability_set() -> None:
    assert set(GGML_TYPE_BY_BITS) == {8, 6, 5, 4, 3, 2}
    assert set(BASE_FTYPE_BY_BITS) == {8, 6, 5, 4, 3, 2}
    assert set(GGML_TYPE_BY_BITS) == RUNTIME_CAPABILITIES["llama.cpp"]


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


def test_pack_error_inherits_the_quantfit_root() -> None:
    assert issubclass(PackError, QuantfitError)
    assert issubclass(PackError, RuntimeError)


def test_base_type_is_the_recipe_floor() -> None:
    recipe = make_recipe(("model.layers.0", 8), ("model.layers.1", 3))

    assert base_type(recipe) == "Q3_K_S"


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


def test_tensor_overrides_escape_dots_so_layer_1_never_matches_layer_11() -> None:
    recipe = make_recipe(("model.layers.1", 8), ("model.layers.11", 4))

    patterns = [override.pattern for override in tensor_overrides(recipe)]

    assert patterns == [r"blk\.1\.", r"blk\.11\."]
    assert re.search(patterns[0], "blk.1.attn_q.weight")
    assert not re.search(patterns[0], "blk.11.attn_q.weight")


def test_tensor_overrides_keep_recipe_order_and_skip_the_embedding() -> None:
    recipe = make_recipe(
        ("model.embed_tokens", 8),
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
