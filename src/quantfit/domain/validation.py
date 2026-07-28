"""Validation-pass logic: measured whole-recipe damage vs the prediction.

Marginal scanning assumes per-group damage is additive (ADR-0006). The
validation pass tests that assumption: replay a whole recipe through
the scan's own quantization in one pass, then compare the measured
damage against the recipe's summed marginal damages. This module holds
the pure comparison — the measurement itself runs through the
`DamageMeter` port, and the command lives in
[quantfit.adapters.inbound.cli_validate][].

Examples:
    Compare a measurement against a recipe's prediction:

    ```python
    from quantfit.domain.validation import validation_result

    result = validation_result(recipe, measured_damage=0.07)
    print(result.gap, result.ratio)
    ```

See Also:
    - [quantfit.domain.model][]: `Recipe`, the prediction's source.
    - [quantfit.ports.outbound][]: `DamageMeter.measure_recipe`, the
      measurement side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfit.domain.model import Recipe


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """One validation pass: the prediction next to the measurement.

    Construction validates both damages, so a result that exists is
    safe to report. The derived numbers are properties — they cannot
    drift from the two facts they derive from.

    Attributes:
        predicted_damage (float): The recipe's summed marginal damages.
        measured_damage (float): Whole-recipe damage measured in one
            pass through the scan's own quantization.

    Examples:
        A measurement 8 % above the prediction:

        ```python
        from quantfit.domain.validation import ValidationResult

        result = ValidationResult(predicted_damage=0.10, measured_damage=0.108)
        assert result.ratio is not None and round(result.ratio, 2) == 1.08
        ```
    """

    predicted_damage: float
    measured_damage: float

    def __post_init__(self) -> None:
        """Enforce the damage invariants.

        Raises:
            ValueError: If either damage is negative or not finite.
        """
        for name, value in (
            ("predicted_damage", self.predicted_damage),
            ("measured_damage", self.measured_damage),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

    @property
    def gap(self) -> float:
        """Measured minus predicted damage.

        Returns:
            The signed gap. Positive means the additivity assumption
            under-predicted the damage.
        """
        return self.measured_damage - self.predicted_damage

    @property
    def ratio(self) -> float | None:
        """Measured damage as a multiple of the prediction.

        Returns:
            ``measured / predicted``, or None when the prediction is
            zero — a ratio against zero carries no information.
        """
        if self.predicted_damage == 0.0:
            return None
        return self.measured_damage / self.predicted_damage


def summed_marginal_damage(recipe: Recipe) -> float:
    """Sum the recipe's per-assignment marginal damages.

    This is the additive prediction the validation pass tests. The
    solver records the same sum as ``plan.predicted_damage`` — summing
    the assignments here keeps the prediction tied to the cells it
    came from.

    Args:
        recipe: The recipe under validation.

    Returns:
        The summed marginal damage.

    Examples:
        The sum over a two-assignment recipe:

        ```python
        from quantfit.domain.validation import summed_marginal_damage

        assert summed_marginal_damage(recipe) == sum(
            a.damage for a in recipe.assignments
        )
        ```
    """
    return sum(assignment.damage for assignment in recipe.assignments)


def validation_result(recipe: Recipe, measured_damage: float) -> ValidationResult:
    """Build the validation result for one recipe and one measurement.

    Args:
        recipe: The recipe under validation.
        measured_damage: Whole-recipe damage from the
            `quantfit.ports.outbound.DamageMeter` port's
            ``measure_recipe``.

    Returns:
        The comparison record.

    Raises:
        ValueError: If ``measured_damage`` is negative or not finite.

    Examples:
        Report the gap for one pass:

        ```python
        from quantfit.domain.validation import validation_result

        result = validation_result(recipe, measured_damage=0.07)
        ```
    """
    return ValidationResult(
        predicted_damage=summed_marginal_damage(recipe),
        measured_damage=measured_damage,
    )
