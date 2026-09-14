"""The refinement pass's measure loop: pack an arm, measure it, compare.

The orchestration ADR-0031 decision 1 requires. Every arm is packed
and measured in the runtime frame. Nothing is ranked by the
sensitivity map, and no arm is skipped because the map predicted
poorly for it.

The control runs first. A candidate's number is unreadable until the
control reproduces the frame it claims to be measured in, so a caller
that cannot check its control has no result to read.

When the neighbourhood is larger than the caller's arm budget, the
loop takes an evenly spaced stride through the enumeration. The
stride is deterministic and independent of the map, which is the
property that matters — a map-ranked subset is the one selection
ADR-0031 decision 1 forbids.

The sidecar records the whole neighbourhood's size beside the arms
the stride took, so a reader never mistakes a sample for a search.
The count comes from the enumeration, never from the arms.

Examples:
    Run a pass over the first few neighbours:

    ```python
    sidecar = run_pass(
        recipe,
        map_,
        packer_for,
        meter,
        frame,
        bar=7.8,
        limit=15,
        out_dir=out_dir,
        row_widths=row_widths,
    )
    ```

See Also:
    - [vramfit.domain.refinement][]: Enumerates the arms.
    - [vramfit.domain.paired][]: Compares and selects between them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from vramfit.domain.model import Recipe, SensitivityMap
from vramfit.domain.paired import PairedResult, compare, select
from vramfit.domain.refinement import Candidate, decline_reason, neighbours
from vramfit.domain.refinement_record import (
    CONTROL_ARM,
    ArmRecord,
    MeasurementFrame,
    RefinementSidecar,
)
from vramfit.ports.outbound import RecipePacker, RuntimeDivergenceMeter

# What the loop tells its caller as it goes. The caller decides
# whether that reaches a terminal, a run log, or nothing.
Reporter = Callable[[str, Mapping[str, object]], None]


def _silent(event: str, fields: Mapping[str, object]) -> None:
    """Discard a progress report.

    Args:
        event: Event name.
        fields: Event payload.
    """


def stride(candidates: Sequence[Candidate], limit: int) -> tuple[Candidate, ...]:
    """Take an evenly spaced, map-independent subset of a neighbourhood.

    Args:
        candidates: Every neighbour, in enumeration order.
        limit: Most arms the caller will pay to measure.

    Returns:
        At most ``limit`` candidates, evenly spaced across the
        enumeration. Every candidate when ``limit`` covers them.

    Raises:
        ValueError: If ``limit`` is not positive.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(candidates) <= limit:
        return tuple(candidates)
    step = len(candidates) / limit
    return tuple(candidates[int(i * step)] for i in range(limit))


def _arm_recipe(recipe: Recipe, candidate: Candidate) -> Recipe:
    """Build the packable recipe for one arm.

    The arm's plan keeps the budget it competes inside and names the
    pass in ``solver``. It carries no trace, because the solver's
    downgrade steps no longer explain these assignments (ADR-0031
    decision 4).

    Args:
        recipe: The recipe under refinement.
        candidate: The arm to build.

    Returns:
        The arm's recipe.
    """
    plan = replace(
        recipe.plan,
        predicted_total_bytes=candidate.total_bytes(),
        predicted_damage=recipe.plan.predicted_damage + candidate.predicted_delta,
        solver=f"{recipe.plan.solver}+refine",
        trace=(),
    )
    return replace(recipe, plan=plan, assignments=candidate.assignments)


def _measure(
    name: str,
    arm_recipe: Recipe,
    packer_for: Callable[[str], RecipePacker],
    meter: RuntimeDivergenceMeter,
    out_dir: Path,
    keep_packs: bool,
    report: Reporter,
) -> tuple[tuple[float, ...], int]:
    """Pack one arm, measure it, and drop the file unless asked to keep it.

    Args:
        name: The arm's name, which names its packed file.
        arm_recipe: The recipe to pack.
        packer_for: Builds a packer writing to the given path.
        meter: The runtime-frame divergence meter.
        out_dir: Directory the packed files go in.
        keep_packs: Keep each packed file instead of deleting it.
        report: Progress reporter.

    Returns:
        The arm's per-chunk divergences and its real packed bytes.
    """
    packed = out_dir / f"{name}.gguf"
    report("arm_packing", {"arm": name, "out": str(packed)})
    result = packer_for(str(packed)).pack(arm_recipe)
    report("arm_packed", {"arm": name, "packed_bytes": result.packed_bytes})
    divergences = meter.measure(str(packed))
    report("arm_measured", {"arm": name, "chunks": len(divergences)})
    if not keep_packs:
        packed.unlink(missing_ok=True)
    return divergences, result.packed_bytes


def _record(
    name: str,
    candidate: Candidate | None,
    paired: PairedResult,
    packed_bytes: int,
) -> ArmRecord:
    """Turn one arm's measurement into its record.

    Args:
        name: The arm's name.
        candidate: The arm's move, or None for the control.
        paired: The arm's standing against the control.
        packed_bytes: Real size of the arm's packed file.

    Returns:
        The arm record.
    """
    move = None if candidate is None else candidate.move
    return ArmRecord(
        arm=name,
        promoted=None if move is None else move.promoted,
        demoted=None if move is None else move.demoted,
        from_bits=None if move is None else move.from_bits,
        to_bits=None if move is None else move.to_bits,
        mean=paired.mean,
        delta=paired.delta,
        sigma=paired.sigma,
        better_chunks=paired.better_chunks,
        chunks=paired.chunks,
        predicted_delta=None if candidate is None else candidate.predicted_delta,
        packed_bytes=packed_bytes,
    )


def run_pass(  # noqa: PLR0913 - the pass surface: two ports, a frame, and its budget
    recipe: Recipe,
    map_: SensitivityMap,
    packer_for: Callable[[str], RecipePacker],
    meter: RuntimeDivergenceMeter,
    frame: MeasurementFrame,
    *,
    bar: float,
    limit: int,
    out_dir: Path,
    row_widths: Mapping[str, int],
    keep_packs: bool = False,
    report: Reporter = _silent,
) -> RefinementSidecar:
    """Measure a recipe's neighbourhood and keep the best arm that wins.

    Args:
        recipe: The solved recipe to refine.
        map_: The map that priced it.
        packer_for: Builds a packer writing to the given path.
        meter: The runtime-frame divergence meter.
        frame: Where the measurements come from.
        bar: Evidence bar in sigma, stated positive.
        limit: Most arms to measure.
        out_dir: Directory the packed files go in.
        row_widths: Elements per row per group, which bind each
            group's effective-bits table for candidate pricing.
        keep_packs: Keep each packed file instead of deleting it.
        report: Progress reporter.

    Returns:
        The pass's complete search record, carrying the arms measured
        and the whole neighbourhood they were drawn from. A recipe
        with no legal swap returns a declined record, having measured
        nothing and reached no card.
    """
    declined = decline_reason(recipe, map_, row_widths)
    if declined is not None:
        report("refine_declined", {"reason": declined})
        return RefinementSidecar(
            model_id=recipe.model_id,
            frame=frame,
            bar=bar,
            control=None,
            arms=(),
            winner=None,
            declined=declined,
            neighbourhood_moves=0,
        )
    candidates = neighbours(recipe, map_, row_widths)
    arms = stride(candidates, limit)
    report(
        "refine_started",
        {"arms": len(arms), "neighbourhood_moves": len(candidates), "bar": bar},
    )
    # Every arm quantizes the same full-precision base, so the convert
    # stage runs once for the whole pass. It reuses an existing base
    # GGUF, and `pack` refuses without one.
    base_bytes = packer_for(str(out_dir / f"{CONTROL_ARM}.gguf")).convert()
    report("base_converted", {"base_bytes": base_bytes})
    control_chunks, control_bytes = _measure(
        CONTROL_ARM, recipe, packer_for, meter, out_dir, keep_packs, report
    )
    control = _record(
        CONTROL_ARM, None, compare(control_chunks, control_chunks), control_bytes
    )
    records: list[ArmRecord] = []
    results: dict[str, PairedResult] = {}
    for index, candidate in enumerate(arms, start=1):
        name = f"arm{index:02d}"
        chunks, packed_bytes = _measure(
            name,
            _arm_recipe(recipe, candidate),
            packer_for,
            meter,
            out_dir,
            keep_packs,
            report,
        )
        paired = compare(chunks, control_chunks)
        results[name] = paired
        records.append(_record(name, candidate, paired, packed_bytes))
    winner, refusal = select(results, bar)
    report("refine_finished", {"winner": winner, "refusal": refusal})
    return RefinementSidecar(
        model_id=recipe.model_id,
        frame=frame,
        bar=bar,
        control=control,
        arms=tuple(records),
        winner=winner,
        declined=None,
        neighbourhood_moves=len(candidates),
    )
