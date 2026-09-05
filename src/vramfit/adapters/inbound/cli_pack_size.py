"""Size-check stage for the ``vramfit pack`` command.

The stage stats the packed file and reports it twice (ADR-0012
decision 4, amended 2026-09-04). The budget line compares the bytes
against ``plan.weight_budget_bytes`` and gates the pack. The
prediction line compares the same bytes against
``plan.predicted_total_bytes``, with the signed delta and its
fraction, and warns past the predicted-bytes tolerance. It never
refuses. Publication #2 fit its budget on two cancelling pricing
errors and reported only its margin (#409), which is the gap the
second line closes. The prediction comes from the recipe the pack
already read, never from a fresh fetch, so the run stays replayable.

Examples:
    The pack command drives the stage like this:

    ```python
    _size_check_stage(run_log, recipe, result.packed_bytes, out)
    ```

See Also:
    - [vramfit.adapters.inbound.cli_pack][]: The command that
      drives this stage.
    - [vramfit.domain.pack][]: The margin and delta arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer

from vramfit.adapters.inbound.cli_pack_smoke import _halt
from vramfit.adapters.inbound.run_log import SafeRunLog
from vramfit.domain.budget import format_size
from vramfit.domain.model import Recipe
from vramfit.domain.pack import (
    PREDICTED_BYTES_TOLERANCE,
    predicted_bytes_delta,
    predicted_bytes_within_tolerance,
    weight_budget_margin,
)


@dataclass(frozen=True, slots=True)
class PredictedReport:
    """The prediction line and its run-log fields.

    Attributes:
        line (str): The console line.
        warns (bool): True when the line is a warning, routed to
            stderr.
        event (dict[str, object]): Fields the ``size_checked`` event
            carries. Every field is null when the prediction is
            absent.

    Examples:
        Route the line by its warning flag:

        ```python
        report = _predicted_report(10_000, 10_050)
        typer.echo(report.line, err=report.warns)
        ```
    """

    line: str
    warns: bool
    event: dict[str, object]


def _predicted_report(predicted_total_bytes: int, packed_bytes: int) -> PredictedReport:
    """Compare the packed bytes against the recipe's prediction.

    A recipe that records no positive prediction gets a line that
    says so, never a crash. The loader requires the field, so the
    absent case reaches this stage only from a zero prediction.

    Args:
        predicted_total_bytes: ``plan.predicted_total_bytes``.
        packed_bytes: Real size of the packed model file.

    Returns:
        The report: line, warning flag, and event fields.
    """
    if predicted_total_bytes <= 0:
        return PredictedReport(
            line=(
                "predicted bytes absent: the recipe records no positive "
                "plan.predicted_total_bytes, so the packed bytes have no "
                "prediction to compare against"
            ),
            warns=False,
            event={
                "predicted_total_bytes": None,
                "predicted_delta_bytes": None,
                "predicted_delta_fraction": None,
                "predicted_within_tolerance": None,
            },
        )
    delta = predicted_bytes_delta(predicted_total_bytes, packed_bytes)
    fraction = delta / predicted_total_bytes
    within = predicted_bytes_within_tolerance(predicted_total_bytes, packed_bytes)
    sign = "+" if delta >= 0 else "-"
    verdict = (
        f"predicted {format_size(predicted_total_bytes)}, delta "
        f"{sign}{format_size(abs(delta))} ({fraction:+.2%}), tolerance "
        f"±{PREDICTED_BYTES_TOLERANCE:.1%}"
    )
    if within:
        line = f"{verdict} — within (ADR-0012)"
    else:
        line = (
            f"warning: {verdict} — OUTSIDE. The recipe's "
            "plan.predicted_total_bytes does not describe this file: the "
            "size model mispriced a class, or the base GGUF carries "
            "floored layers the prediction never counted (#307). The "
            "budget line alone decides the pack (ADR-0012 decision 4)"
        )
    return PredictedReport(
        line=line,
        warns=not within,
        event={
            "predicted_total_bytes": predicted_total_bytes,
            "predicted_delta_bytes": delta,
            "predicted_delta_fraction": fraction,
            "predicted_within_tolerance": within,
        },
    )


def _size_check_stage(
    run_log: SafeRunLog, recipe: Recipe, packed_bytes: int, out: Path
) -> None:
    """Re-check the packed bytes against the budget and the prediction.

    Nominal-bit predictions undershoot GGUF's effective bits, so the
    recipe's promise is re-proven on the artifact (ADR-0014). The
    budget line gates the pack. The prediction line follows it, lands
    in the same ``size_checked`` event, and only warns.

    Args:
        run_log: The pack run's event log.
        recipe: The recipe the pack applied.
        packed_bytes: Real size of the packed model file.
        out: The packed model path, for the report.

    Raises:
        typer.Exit: With code 1 when the packed bytes exceed the
            weight budget (via ``_halt``); the file is kept.
    """
    margin = weight_budget_margin(recipe, packed_bytes)
    fits = margin >= 0
    prediction = _predicted_report(recipe.plan.predicted_total_bytes, packed_bytes)
    run_log.emit(
        "size_checked",
        {
            "packed_bytes": packed_bytes,
            "weight_budget_bytes": recipe.plan.weight_budget_bytes,
            "margin_bytes": margin,
            "fits": fits,
            **prediction.event,
        },
    )
    typer.echo(
        f"packed {len(recipe.assignments)} groups -> {out} "
        f"({format_size(packed_bytes)}), weight budget "
        f"{format_size(recipe.plan.weight_budget_bytes)}, margin "
        f"{format_size(abs(margin))} {'under' if fits else 'OVER'}"
    )
    typer.echo(prediction.line, err=prediction.warns)
    if not fits:
        error = RuntimeError(
            f"packed model exceeds the weight budget by {format_size(-margin)} "
            f"— the file is kept at {out}"
        )
        raise _halt(run_log, "size_check", error)
