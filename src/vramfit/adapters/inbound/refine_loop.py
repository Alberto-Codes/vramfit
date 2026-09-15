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
among them, so it reports a winner and never a reason for none.
`RefinementSidecar.outcome` classifies the finished pass instead —
judged arms, excluded arms, and the arm kept — and the run log
renders that structure. The terminal renders the same one, so a
budget exclusion cannot read as an arm that failed the bar on one
surface and as an exclusion on the other.

When the neighbourhood is larger than the caller's arm budget, the
loop takes an evenly spaced stride through the enumeration. The
stride is deterministic and independent of the map, which is the
property that matters — a map-ranked subset is the one selection
ADR-0031 decision 1 forbids.

The sidecar records the whole neighbourhood's size beside the arms
the stride took, so a reader never mistakes a sample for a search.
The count comes from the enumeration, never from the arms.

The pass banks that record itself, through `_PassRecorder`: once when
the control is measured, again after every arm, and last with the
winner. Holding the arms in memory and writing once lost every
measurement a pass did not finish, and each arm costs about 0.48 USD
on a rented card. The sink is a parameter of `run_pass` rather than a
step its caller takes afterwards, so the losing shape cannot be
written.

The control's bank is the same change's other half. It lands about
four minutes into a 59-minute pass on the 30B target, before the
first arm packs, so an operator can read the gate and stop a pass
whose control did not reproduce its frame.

Examples:
    Run a pass over the first few neighbours:

    ```python
    sidecar = run_pass(
        recipe,
        map_,
        packer_for,
        meter,
        frame,
        sink,
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
from vramfit.domain.pack import fits_weight_budget, weight_budget_margin
from vramfit.domain.paired import PairedResult, compare, select
from vramfit.domain.refinement import Candidate, decline_reason, neighbours
from vramfit.domain.refinement_record import (
    CONTROL_ARM,
    ArmRecord,
    MeasurementFrame,
    RefinementRecordError,
    RefinementSidecar,
)
from vramfit.ports.outbound import (
    RecipePacker,
    RefinementSidecarSink,
    RuntimeDivergenceMeter,
)

# What the loop tells its caller as it goes. The caller decides
# whether that reaches a terminal, a run log, or nothing.
Reporter = Callable[[str, Mapping[str, object]], None]

# The event carrying the control's measurement, reported before the
# first arm packs. A caller that shows the operator one event shows
# this one: every arm below is read against it.
CONTROL_MEASURED = "control_measured"

# The event carrying one candidate arm's measurement.
ARM_MEASURED = "arm_measured"

# Keys into a measurement payload. Named here rather than spelled at
# each surface, so a caller reading a figure back cannot miss a
# rename that `_measurement` makes.
MEAN_FIELD = "mean"
CHUNKS_FIELD = "chunks"


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
            `vramfit.domain.pack.fits_weight_budget` of the margin,
            the one definition of the rule `vramfit pack` gates on.
        """
        return fits_weight_budget(self.budget_margin)


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


def _measure(packed: PackedArm, meter: RuntimeDivergenceMeter) -> tuple[float, ...]:
    """Measure a packed arm and drop its file.

    A pass packs one file per arm and the 30B target's are about
    21 GiB each, so a sixteen-arm pass on a rented pod keeps none of
    them past its own measurement.

    Nothing is reported here. Divergences alone are not a result —
    the mean, the delta, and the sigma appear once the arm is paired
    against the control — and a pass that reported an arm from here
    reported a name and a chunk count, which is what made a lost pass
    unreconstructable.

    Args:
        packed: The packed arm, already judged.
        meter: The runtime-frame divergence meter.

    Returns:
        The arm's per-chunk divergences.
    """
    divergences = meter.measure(str(packed.path))
    packed.path.unlink(missing_ok=True)
    return divergences


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


def _measurement(arm: ArmRecord) -> dict[str, object]:
    """Report one arm's result, control or candidate, one way.

    Both events carry the same fields because both describe the same
    thing: an arm the pass paid for. A reported name and chunk count
    is not a result, and a pass reporting one could not be read back.

    Args:
        arm: The arm's finished record.

    Returns:
        The event payload.
    """
    return {
        "arm": arm.arm,
        MEAN_FIELD: arm.mean,
        "delta": arm.delta,
        "sigma": arm.sigma,
        "better_chunks": arm.better_chunks,
        CHUNKS_FIELD: arm.chunks,
    }


@dataclass(frozen=True, slots=True)
class _PassRecorder:
    """Builds the pass's record and banks it, in one step.

    The loop reaches `RefinementSidecar` only through `bank`, so
    there is no way to hold a measurement without writing it. An arm
    costs about 0.48 USD of card time and a pass on a rented pod runs
    against a deletion deadline, so a measurement that lives only in
    memory is a measurement the pass can lose.

    That is a shape, not a rule. ADR-0031 already records that a rule
    someone must remember is the weakest guarantee available, and the
    write-once version of this loop is the case it was written about.

    Attributes:
        sink (RefinementSidecarSink): Where each record lands. The
            JSON adapter replaces the file in one step, so a pass
            interrupted mid-write keeps the record before it.
        model_id (str): The refined recipe's model identifier.
        frame (MeasurementFrame): Where the numbers came from.
        bar (float): The evidence bar the caller stated.
        neighbourhood_moves (int): How large the neighbourhood was.

    Examples:
        Bank the control before the first arm packs:

        ```python
        recorder.bank(control, ())
        ```
    """

    sink: RefinementSidecarSink
    model_id: str
    frame: MeasurementFrame
    bar: float
    neighbourhood_moves: int

    def bank(
        self,
        control: ArmRecord | None,
        arms: tuple[ArmRecord, ...],
        *,
        winner: str | None = None,
        declined: str | None = None,
        finished: bool = False,
    ) -> RefinementSidecar:
        """Write what the pass holds so far and hand it back.

        Args:
            control: The control's measurement, or None before it
                runs and on a declined pass.
            arms: Every arm measured so far, in measurement order.
            winner: The arm the pass kept, once selection has run.
            declined: Why the recipe had no neighbourhood to search.
            finished: Whether the pass measured every arm it selected
                and ran selection.

        Returns:
            The record it wrote.
        """
        sidecar = RefinementSidecar(
            model_id=self.model_id,
            frame=self.frame,
            bar=self.bar,
            control=control,
            arms=arms,
            winner=winner,
            declined=declined,
            neighbourhood_moves=self.neighbourhood_moves,
            finished=finished,
        )
        self.sink.save(sidecar)
        return sidecar


def run_pass(  # noqa: PLR0913 - the pass surface: two ports, a frame, and its budget
    recipe: Recipe,
    map_: SensitivityMap,
    packer_for: Callable[[str], RecipePacker],
    meter: RuntimeDivergenceMeter,
    frame: MeasurementFrame,
    sink: RefinementSidecarSink,
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
        sink: Where the record is banked, after the control and after
            every arm. It is the pass's own parameter rather than the
            caller's closing step, so no caller can hold a pass's
            measurements and then lose them.
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

    The finished record classifies itself. `refine_finished` reports
    `RefinementSidecar.outcome` — the judged count, the excluded arm
    names, and the refusal when no arm was kept — so the run log
    carries the exclusions whether or not an arm won.

    The record is banked as the pass runs: once when the control is
    measured, again after every arm, and last with the winner. A pass
    that stops at arm 12 of 16 therefore leaves twelve measured arms
    and a reproduced control behind, each with its mean, delta,
    sigma, and better-chunk count. Those records carry
    ``finished=False``, which is what separates them from a pass that
    judged every arm and kept none.

    The control's bank is also what makes the gate readable early. It
    lands about four minutes into a 59-minute pass on the 30B target,
    before the first arm packs, so an operator whose control did not
    reproduce its frame can stop the pass rather than pay for it.

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
    recorder = _PassRecorder(
        sink=sink,
        model_id=recipe.model_id,
        frame=frame,
        bar=bar,
        neighbourhood_moves=len(candidates),
    )
    declined = decline_reason(recipe, map_, candidates)
    if declined is not None:
        report(
            "refine_declined",
            {"reason": declined, "neighbourhood_moves": len(candidates)},
        )
        return recorder.bank(None, (), declined=declined, finished=True)
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
    control_chunks = _measure(control_pack, meter)
    control = _record(
        CONTROL_ARM,
        None,
        compare(control_chunks, control_chunks),
        control_pack.packed_bytes,
        control_pack.budget_margin,
    )
    # The gate, banked and reported before the first arm packs. Every
    # candidate below is read against this number, so an operator who
    # cannot recognize it has nothing to gain from the next 55 minutes.
    recorder.bank(control, ())
    report(CONTROL_MEASURED, _measurement(control))
    records: list[ArmRecord] = []
    results: dict[str, PairedResult] = {}
    for index, candidate in enumerate(arms, start=1):
        name = f"arm{index:02d}"
        packed = _pack(
            name, _arm_recipe(recipe, candidate), packer_for, out_dir, report
        )
        chunks = _measure(packed, meter)
        paired = compare(chunks, control_chunks)
        records.append(
            _record(name, candidate, paired, packed.packed_bytes, packed.budget_margin)
        )
        recorder.bank(control, tuple(records))
        report(ARM_MEASURED, _measurement(records[-1]))
        if packed.fits_budget():
            results[name] = paired
        else:
            report(
                "arm_over_budget",
                {"arm": name, "budget_margin": packed.budget_margin},
            )
    sidecar = recorder.bank(
        control, tuple(records), winner=select(results, bar), finished=True
    )
    outcome = sidecar.outcome()
    report(
        "refine_finished",
        {
            "winner": sidecar.winner,
            "judged": len(outcome.judged),
            "excluded": [arm.arm for arm in outcome.excluded],
            "refusal": outcome.refusal(),
        },
    )
    return sidecar
