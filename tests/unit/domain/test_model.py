from __future__ import annotations

import pytest

from quantfit.domain.model import Assignment, PlanMeta, Recipe

pytestmark = pytest.mark.unit


def make_plan() -> PlanMeta:
    return PlanMeta(
        vram_budget_bytes=4_000,
        kv_headroom_bytes=1_000,
        weight_budget_bytes=3_000,
        predicted_total_bytes=2_500,
        predicted_damage=0.05,
        solver="greedy-damage-per-byte",
        pins={},
        format_overhead=0.05,
        trace=(),
    )


class TestRecipeInvariants:
    def test_unique_groups_construct_succeeds(self) -> None:
        recipe = Recipe(
            model_id="test/model",
            plan=make_plan(),
            assignments=(
                Assignment(group="model.layers.0", bits=8, bytes=1_500, damage=0.01),
                Assignment(group="model.layers.1", bits=4, bytes=1_000, damage=0.02),
            ),
        )

        assert len(recipe.assignments) == 2

    def test_duplicate_assignment_groups_raise_value_error(self) -> None:
        duplicated = Assignment(
            group="model.layers.0", bits=4, bytes=1_000, damage=0.02
        )

        with pytest.raises(ValueError, match="unique"):
            Recipe(
                model_id="test/model",
                plan=make_plan(),
                assignments=(
                    Assignment(
                        group="model.layers.0", bits=8, bytes=1_500, damage=0.01
                    ),
                    duplicated,
                ),
            )

    def test_empty_assignments_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Recipe(model_id="test/model", plan=make_plan(), assignments=())
