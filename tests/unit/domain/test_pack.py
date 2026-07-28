from __future__ import annotations

import pytest

from quantfit.domain.model import Assignment, PlanMeta, Recipe
from quantfit.domain.pack import PackResult, TypeOverride, weight_budget_margin

pytestmark = pytest.mark.unit


def make_recipe(weight_budget_bytes: int) -> Recipe:
    return Recipe(
        model_id="test/model",
        plan=PlanMeta(
            vram_budget_bytes=weight_budget_bytes + 1_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=weight_budget_bytes,
            predicted_total_bytes=weight_budget_bytes - 500,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
        ),
        runtime=None,
    )


class TestPackResult:
    def test_valid_result_constructs(self) -> None:
        result = PackResult(
            packed_bytes=100,
            base_type="Q4_K_S",
            token_embedding_type=None,
            overrides=(TypeOverride(pattern=r"blk\.0\.", quant_type="q4_k"),),
        )

        assert result.packed_bytes == 100

    def test_empty_token_embedding_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="token_embedding_type"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type="",
                overrides=(),
            )

    def test_duplicate_override_patterns_raise_value_error(self) -> None:
        duplicate = TypeOverride(pattern=r"blk\.0\.", quant_type="q4_k")
        shadowed = TypeOverride(pattern=r"blk\.0\.", quant_type="q8_0")

        with pytest.raises(ValueError, match="unique"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                overrides=(duplicate, shadowed),
            )

    @pytest.mark.parametrize("bad_bytes", [0, -1], ids=["zero", "negative"])
    def test_non_positive_packed_bytes_raises_value_error(self, bad_bytes) -> None:
        with pytest.raises(ValueError, match="packed_bytes"):
            PackResult(
                packed_bytes=bad_bytes,
                base_type="Q4_K_S",
                token_embedding_type=None,
                overrides=(),
            )

    def test_empty_base_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="base_type"):
            PackResult(
                packed_bytes=1, base_type="", token_embedding_type=None, overrides=()
            )

    @pytest.mark.parametrize(
        ("pattern", "quant_type"),
        [("", "q4_k"), (r"blk\.0\.", "")],
        ids=["empty-pattern", "empty-type"],
    )
    def test_empty_override_half_raises_value_error(self, pattern, quant_type) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            TypeOverride(pattern=pattern, quant_type=quant_type)


class TestWeightBudgetMargin:
    def test_under_budget_margin_is_positive(self) -> None:
        assert weight_budget_margin(make_recipe(3_000), 2_900) == 100

    def test_over_budget_margin_is_negative(self) -> None:
        assert weight_budget_margin(make_recipe(3_000), 3_001) == -1

    def test_exact_fit_margin_is_zero(self) -> None:
        assert weight_budget_margin(make_recipe(3_000), 3_000) == 0

    def test_non_positive_packed_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="packed_bytes"):
            weight_budget_margin(make_recipe(3_000), 0)
