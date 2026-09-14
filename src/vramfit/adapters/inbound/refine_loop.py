"""The refinement pass's measure loop: pack an arm, measure it, compare.

The orchestration ADR-0031 decision 1 requires. Every arm is packed
and measured in the runtime frame. Nothing is ranked by the
sensitivity map, and no arm is skipped because the map predicted
poorly for it.

The control runs first. A candidate's number is unreadable until the
control reproduces the frame it claims to be measured in, so a caller
that cannot check its control has no result to read. A control that
packs over the weight budget stops the pass for the same reason.

Every packed arm is judged against that budget with
`vramfit.domain.pack.weight_budget_margin`, the rule `pack` gates on.
Byte-neutrality equalizes predicted bytes, so a swap can still pack
over. An arm that does stays in the record with its measurement
intact and leaves the selection: it cost card time, so it is
excluded rather than dropped.

`_pack` and `_measure` are two calls for that reason. Packing yields
the size, and the size yields the verdict, so a caller holds the
verdict before it spends the measurement — the pass's dominant cost.
The control's refusal lands between them because that is where the
seam is, not because a rule says to check early.

`select` judges the arms it is given, and an excluded arm is not
among them, so `_refusal_with_exclusions` names the exclusions the
selection never saw. Without it a budget exclusion would read as an
arm that failed the bar.

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
from dataclasses import dataclass, replace
from pathlib import Path

from vramfit.domain.model import Recipe, SensitivityMap
from vramfit.domain.pack import weight_budget_margin
from vramfit.domain.paired import PairedResult, compare, select
from vramfit.domain.refinement import Candidate, decline_reason, neighbours
from vramfit.domain.refinement_record import (
    CONTROL_ARM,
    ArmRecord,
    MeasurementFrame,
    RefinementRecordError,
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


@dataclass(frozen=True, slots=True)
class PackedArm:
    """One arm packed and judged, before anything is spent measuring it.

    The seam this stage turns on. Packing produces a real file and a
    real size, and the weight-budget verdict follows from that size
    alone — so the caller holds the verdict before it decides whether
    to pay for the measurement. `run_pass` refuses an over-budget
    control here because this is where the knowledge is, not because
    anyone remembered to check early.

    Attributes:
        name (str): The arm's name, which names its packed file.
        path (Path): The packed file, still on disk.
        packed_bytes (int): The file's real size.
        budget_margin (int): ``weight_budget_bytes - packed_bytes``,
            from `vramfit.domain.pack.weight_budget_margin`.
            Non-negative means the arm fits.

    Examples:
        Judge before spending:

        ```python
        from pathlib import Path

        from vramfit.adapters.inbound.refine_loop import PackedArm

        packed = PackedArm(
            name="arm01",
            path=Path("arm01.gguf"),
            packed_bytes=500,
            budget_margin=-1,
        )
        assert not packed.fits_budget()
        ```
    """

    name: str
    path: Path
    packed_bytes: int
    budget_margin: int

    def fits_budget(self) -> bool:
        """Judge whether this pack fits the budget it was solved for.

        Returns:
            True when the margin is non-negative, which is what
            `vramfit.domain.pack.weight_budget_margin` documents as
            fitting.
        """
        return self.budget_margin >= 0


def _pack(
    name: str,
    arm_recipe: Recipe,
    packer_for: Callable[[str], RecipePacker],
    out_dir: Path,
    report: Reporter,
) -> PackedArm:
    """Pack one arm and judge its size, spending nothing further.

    Args:
        name: The arm's name, which names its packed file.
        arm_recipe: The recipe to pack.
        packer_for: Builds a packer writing to the given path.
        out_dir: Directory the packed files go in.
        report: Progress reporter.

    Returns:
        The packed arm and its weight-budget verdict. The file is
        still on disk — the caller drops it, measured or not.
    """
    packed = out_dir / f"{name}.gguf"
    report("arm_packing", {"arm": name, "out": str(packed)})
    result = packer_for(str(packed)).pack(arm_recipe)
    margin = weight_budget_margin(arm_recipe, result.packed_bytes)
    report(
        "arm_packed",
        {
            "arm": name,
            "packed_bytes": result.packed_bytes,
            "budget_margin": margin,
        },
    )
    return PackedArm(
        name=name,
        path=packed,
        packed_bytes=result.packed_bytes,
        budget_margin=margin,
    )


def _measure(
    packed: PackedArm,
    meter: RuntimeDivergenceMeter,
    report: Reporter,
) -> tuple[float, ...]:
    """Measure a packed arm and drop its file.

    A pass packs one file per arm and the 30B target's are about
    21 GiB each, so a sixteen-arm pass on a rented pod keeps none of
    them past its own measurement.

    Args:
        packed: The packed arm, already judged.
        meter: The runtime-frame divergence meter.
        report: Progress reporter.

    Returns:
        The arm's per-chunk divergences.
    """
    divergences = meter.measure(str(packed.path))
    report("arm_measured", {"arm": packed.name, "chunks": len(divergences)})
    packed.path.unlink(missing_ok=True)
    return divergences


def _refusal_with_exclusions(
    refusal: str | None, excluded: Sequence[str], measured: int
) -> str | None:
    """Name the budget exclusions the selection never saw.

    `select` judges the arms it is given. An arm the weight budget
    excluded is not among them, so an unqualified refusal would
    report it as one that failed the bar — and when every arm was
    excluded, `select` says none was measured at all, after the pass
    paid to measure them.

    Args:
        refusal: The refusal `select` returned, or None for a winner.
        excluded: Names of the arms the budget excluded.
        measured: How many arms the pass measured in total.

    Returns:
        The refusal, naming the exclusions when there were any.
        None when an arm won.
    """
    if refusal is None or not excluded:
        return refusal
    over = f"{len(excluded)} of {measured} measured arms packed over the weight budget"
    if len(excluded) == measured:
        return f"{over}, so no arm was judged on merit"
    return f"{refusal}; {over} and were not judged"


def _record(
    name: str,
    candidate: Candidate | None,
    paired: PairedResult,
    packed_bytes: int,
    budget_margin: int,
) -> ArmRecord:
    """Turn one arm's measurement into its record.

    Args:
        name: The arm's name.
        candidate: The arm's move, or None for the control.
        paired: The arm's standing against the control.
        packed_bytes: Real size of the arm's packed file.
        budget_margin: The arm's weight-budget margin, negative when
            the packed file exceeds the budget.

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
        budget_margin=budget_margin,
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
        report: Progress reporter.

    The neighbourhood is enumerated once. The decline reads that
    same enumeration, so no caller relies on a decline and an
    enumeration agreeing across two runs, and a declined record
    carries the moves that enumeration found rather than zero.

    Every packed arm is judged against the weight budget between its
    pack and its measurement. An arm that exceeds it is still
    measured, stays in the record, and leaves the selection, so a
    pass never keeps a file `pack` would refuse. The control is
    refused at that same seam, before its measurement is spent.

    Returns:
        The pass's complete search record, carrying the arms measured
        — including any the budget excluded — and the whole
        neighbourhood they were drawn from. A recipe with no legal
        swap returns a declined record, having measured nothing and
        reached no card.

    Raises:
        RefinementRecordError: If the control packs over the weight
            budget. Every other arm is read against it, so nothing
            downstream is readable.
    """
    candidates = neighbours(recipe, map_, row_widths)
    declined = decline_reason(recipe, map_, candidates)
    if declined is not None:
        report(
            "refine_declined",
            {"reason": declined, "neighbourhood_moves": len(candidates)},
        )
        return RefinementSidecar(
            model_id=recipe.model_id,
            frame=frame,
            bar=bar,
            control=None,
            arms=(),
            winner=None,
            declined=declined,
            neighbourhood_moves=len(candidates),
        )
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
    control_pack = _pack(CONTROL_ARM, recipe, packer_for, out_dir, report)
    if not control_pack.fits_budget():
        control_pack.path.unlink(missing_ok=True)
        raise RefinementRecordError(
            f"the control packed {-control_pack.budget_margin} bytes over the "
            "weight budget, so no arm is readable against it"
        )
    control_chunks = _measure(control_pack, meter, report)
    control = _record(
        CONTROL_ARM,
        None,
        compare(control_chunks, control_chunks),
        control_pack.packed_bytes,
        control_pack.budget_margin,
    )
    records: list[ArmRecord] = []
    results: dict[str, PairedResult] = {}
    excluded: list[str] = []
    for index, candidate in enumerate(arms, start=1):
        name = f"arm{index:02d}"
        packed = _pack(
            name, _arm_recipe(recipe, candidate), packer_for, out_dir, report
        )
        chunks = _measure(packed, meter, report)
        paired = compare(chunks, control_chunks)
        records.append(
            _record(name, candidate, paired, packed.packed_bytes, packed.budget_margin)
        )
        if packed.fits_budget():
            results[name] = paired
        else:
            excluded.append(name)
            report(
                "arm_over_budget",
                {"arm": name, "budget_margin": packed.budget_margin},
            )
    winner, refusal = select(results, bar)
    refusal = _refusal_with_exclusions(refusal, excluded, len(records))
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
