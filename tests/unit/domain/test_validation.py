from __future__ import annotations

import math

import pytest

from quantfit.domain.model import Assignment, PlanMeta, Recipe
from quantfit.domain.validation import (
    ValidationResult,
    summed_marginal_damage,
    validation_result,
)

pytestmark = pytest.mark.unit


def make_recipe(damages: tuple[float, ...]) -> Recipe:
    return Recipe(
        model_id="test/model",
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=3_000,
            predicted_total_bytes=2_500,
            predicted_damage=sum(damages),
            solver="greedy-damage-per-byte",
            pins={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=tuple(
            Assignment(group=f"model.layers.{i}", bits=4, bytes=500, damage=damage)
            for i, damage in enumerate(damages)
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
    )


class TestValidationResult:
    def test_gap_is_measured_minus_predicted(self) -> None:
        result = ValidationResult(predicted_damage=0.10, measured_damage=0.15)

        assert result.gap == pytest.approx(0.05)

    def test_ratio_is_measured_over_predicted(self) -> None:
        result = ValidationResult(predicted_damage=0.10, measured_damage=0.15)

        assert result.ratio == pytest.approx(1.5)

    def test_ratio_with_zero_prediction_is_none(self) -> None:
        result = ValidationResult(predicted_damage=0.0, measured_damage=0.01)

        assert result.ratio is None

    @pytest.mark.parametrize(
        "predicted",
        [-0.1, math.nan, math.inf],
        ids=["negative", "nan", "inf"],
    )
    def test_non_finite_or_negative_predicted_raises_value_error(
        self, predicted: float
    ) -> None:
        with pytest.raises(ValueError, match="predicted_damage"):
            ValidationResult(predicted_damage=predicted, measured_damage=0.1)

    @pytest.mark.parametrize(
        "measured",
        [-0.1, math.nan, math.inf],
        ids=["negative", "nan", "inf"],
    )
    def test_non_finite_or_negative_measured_raises_value_error(
        self, measured: float
    ) -> None:
        with pytest.raises(ValueError, match="measured_damage"):
            ValidationResult(predicted_damage=0.1, measured_damage=measured)


class TestValidationOfRecipe:
    def test_summed_marginal_damage_sums_the_assignments(self) -> None:
        recipe = make_recipe((0.01, 0.02, 0.005))

        assert summed_marginal_damage(recipe) == pytest.approx(0.035)

    def test_validation_result_predicts_from_the_assignments(self) -> None:
        recipe = make_recipe((0.01, 0.02))

        result = validation_result(recipe, measured_damage=0.045)

        assert result.predicted_damage == pytest.approx(0.03)
        assert result.measured_damage == pytest.approx(0.045)
        assert result.gap == pytest.approx(0.015)

    def test_validation_result_with_bad_measurement_raises_value_error(self) -> None:
        recipe = make_recipe((0.01,))

        with pytest.raises(ValueError, match="measured_damage"):
            validation_result(recipe, measured_damage=math.nan)
