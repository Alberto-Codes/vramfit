"""The refinement sidecar: what a refinement pass evaluated and kept.

A refined recipe's assignments no longer follow from the greedy solve,
so the recipe alone cannot explain itself. ADR-0031 decision 3 puts
the explanation in a sidecar beside the recipe rather than in the
recipe, the line ADR-0025 already drew for evaluation evidence. The
recipe schema does not change, and no recipe that was never refined
gains a field.

The record carries every arm the pass measured, not only the winner.
An arm that lost is evidence about the neighbourhood, and the losing
arms are what show the map did not order it.

It also carries how large the neighbourhood was. A caller's arm
budget is usually smaller than the neighbourhood, so the arms are a
sample of it. `neighbourhood_moves` beside ``len(arms)`` is what
tells a reader 15 arms of 15 from 15 arms of 385, and no outcome
reads correctly without both.

One rule governs the predicted delta. It records as provenance and
nothing may order, filter, or select on it (decision 6). Spearman rho
between prediction and measured outcome was +0.146 over the fifteen
arms of the 2026-09-11 sweep, which is why the pass measures at all.
`vramfit.domain.paired.select` cannot see the field, because it reads
`PairedResult` and that type carries no prediction.

Examples:
    Read the winning arm's measured mean:

    ```python
    winner = sidecar.winning_arm()
    print(winner.mean if winner is not None else sidecar.declined)
    ```

See Also:
    - [vramfit.domain.refinement][]: Generates the arms.
    - [vramfit.domain.paired][]: Measures and selects between them.
      `cleared_bar` states the win rule this module validates
      against.
"""

from __future__ import annotations

from dataclasses import dataclass

from vramfit.domain.errors import VramfitError
from vramfit.domain.evals import CorpusReference
from vramfit.domain.paired import cleared_bar

# The arm name reserved for the unmodified recipe. The pass measures
# it first, because a candidate is only readable against a control
# that reproduced the published frame.
CONTROL_ARM = "control"


class RefinementRecordError(VramfitError, ValueError):
    """A refinement sidecar does not describe a readable pass.

    Examples:
        A winner that never cleared the stated bar:

        ```python
        raise RefinementRecordError("winner arm03 did not clear 7.8 sigma")
        ```
    """


@dataclass(frozen=True, slots=True)
class MeasurementFrame:
    """Where a refinement pass's numbers came from.

    ADR-0027 binds damage numbers to one instrument, so the record
    names the instrument rather than assuming it. Two passes measured
    in different frames do not compare.

    Attributes:
        runtime_build (str): The runtime binary's build identity,
            e.g. ``b10362``.
        hardware (str): The card the pass ran on, e.g. ``H100 SXM``.
        corpus (CorpusReference): The evaluation text, named by
            content where the pass recorded it.
        reference (str): What the divergences were measured against,
            e.g. ``f16 base logits``.

    Examples:
        The 2026-09-11 frame:

        ```python
        from vramfit.domain.evals import CorpusReference
        from vramfit.domain.refinement_record import MeasurementFrame

        frame = MeasurementFrame(
            runtime_build="b10362",
            hardware="H100 SXM",
            corpus=CorpusReference(file="wiki.test.raw"),
            reference="f16 base logits",
        )
        ```
    """

    runtime_build: str
    hardware: str
    corpus: CorpusReference
    reference: str

    def __post_init__(self) -> None:
        """Enforce that the frame names its instrument.

        Raises:
            RefinementRecordError: If any field is empty. A frame
                that names nothing cannot bound a comparison.
        """
        for field_name in ("runtime_build", "hardware", "reference"):
            if not getattr(self, field_name):
                raise RefinementRecordError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class ArmRecord:
    """One arm the refinement pass packed and measured.

    Attributes:
        arm (str): The arm's name. `CONTROL_ARM` for the unmodified
            recipe.
        promoted (str | None): Group whose precision rose, or None on
            the control.
        demoted (str | None): Group whose precision fell, or None on
            the control.
        from_bits (int | None): Precision the promoted group left, or
            None on the control.
        to_bits (int | None): Precision the promoted group reached, or
            None on the control.
        mean (float): The arm's own mean divergence.
        delta (float): Mean paired difference against the control.
            Zero on the control itself.
        sigma (float): ``delta`` in units of its standard error.
        better_chunks (int): Chunks measuring below the control.
        chunks (int): Chunks measured.
        predicted_delta (float | None): The map's predicted damage
            change for the swap. Provenance only — nothing orders,
            filters, or selects on it (ADR-0031 decision 6).
        packed_bytes (int | None): Real size of the arm's packed
            file, or None when the pass did not record it.

    Examples:
        The winning arm of the 2026-09-11 sweep:

        ```python
        from vramfit.domain.refinement_record import ArmRecord

        arm = ArmRecord(
            arm="arm11",
            promoted="g1",
            demoted="g2",
            from_bits=2,
            to_bits=4,
            mean=0.191855,
            delta=-0.012368,
            sigma=-14.4,
            better_chunks=436,
            chunks=594,
        )
        ```
    """

    arm: str
    promoted: str | None
    demoted: str | None
    from_bits: int | None
    to_bits: int | None
    mean: float
    delta: float
    sigma: float
    better_chunks: int
    chunks: int
    predicted_delta: float | None = None
    packed_bytes: int | None = None

    def __post_init__(self) -> None:
        """Enforce that an arm describes a whole move, or none.

        Raises:
            RefinementRecordError: If the arm is unnamed, reports no
                chunks, counts more better chunks than it measured,
                or half-describes its swap. The control names no
                swap and every other arm names all four parts.
        """
        if not self.arm:
            raise RefinementRecordError("arm must be named")
        if self.chunks <= 0:
            raise RefinementRecordError(f"arm {self.arm} measured no chunks")
        if not 0 <= self.better_chunks <= self.chunks:
            raise RefinementRecordError(
                f"arm {self.arm} counts {self.better_chunks} better chunks "
                f"of {self.chunks} measured"
            )
        parts = (self.promoted, self.demoted, self.from_bits, self.to_bits)
        named = [p for p in parts if p is not None]
        if self.arm == CONTROL_ARM and named:
            raise RefinementRecordError("the control arm names no swap")
        if self.arm != CONTROL_ARM and len(named) != len(parts):
            raise RefinementRecordError(f"arm {self.arm} half-describes its swap")
        if self.packed_bytes is not None and self.packed_bytes <= 0:
            raise RefinementRecordError(f"arm {self.arm} has no packed bytes")

    def improved(self, bar: float) -> bool:
        """Judge whether this arm beat the control past a bar.

        Reads `vramfit.domain.paired.cleared_bar`, the one
        definition of the rule, so a recorded winner is a winner
        `vramfit.domain.paired.select` would have chosen.

        Args:
            bar: Evidence bar in sigma, stated positive.

        Returns:
            True when the arm measured lower and cleared the bar.

        Raises:
            ValueError: If ``bar`` is negative.
        """
        return cleared_bar(self.delta, self.sigma, bar)


@dataclass(frozen=True, slots=True)
class RefinementSidecar:
    """One refinement pass's complete search record.

    Attributes:
        model_id (str): The refined recipe's model identifier.
        frame (MeasurementFrame): Where the numbers came from.
        bar (float): Evidence bar in sigma the pass was run against.
        control (ArmRecord | None): The unmodified recipe's
            measurement, or None when the pass declined. A declined
            pass reaches no card, so it measures nothing at all.
        arms (tuple[ArmRecord, ...]): Every candidate measured, in
            measurement order. Empty when the pass declined.
        winner (str | None): Name of the arm the pass kept, or None
            when no arm cleared the bar.
        declined (str | None): Why the recipe had no neighbourhood
            worth searching, or None when the pass ran arms.
        neighbourhood_moves (int): How many byte-neutral moves the
            neighbourhood held, before any stride took a sample. Read
            it beside ``len(arms)``: the two differ whenever the
            caller's arm budget was smaller than the neighbourhood,
            and a reader cannot judge an outcome without both. Zero
            for a declined pass, which enumerated no move worth
            measuring.

    Examples:
        A declined pass measured nothing:

        ```python
        assert sidecar.arms == () and sidecar.control is None
        ```

        A sampled pass names the fraction it read:

        ```python
        print(f"{len(sidecar.arms)} of {sidecar.neighbourhood_moves}")
        ```
    """

    model_id: str
    frame: MeasurementFrame
    bar: float
    control: ArmRecord | None
    arms: tuple[ArmRecord, ...]
    winner: str | None
    declined: str | None
    neighbourhood_moves: int

    def __post_init__(self) -> None:
        """Enforce that the record's claims match its measurements.

        Raises:
            RefinementRecordError: If the bar is negative, the
                neighbourhood count is negative or smaller than the
                arms measured, a pass that ran carries no control,
                arm names repeat, an arm measured a different chunk
                count from the control, the winner names no measured
                arm or one that never cleared the bar, or a declined
                pass still carries a measurement.
        """
        if not self.model_id:
            raise RefinementRecordError("model_id must not be empty")
        if self.bar < 0:
            raise RefinementRecordError("bar must not be negative")
        if self.neighbourhood_moves < 0:
            raise RefinementRecordError("neighbourhood_moves must not be negative")
        if self.declined is not None:
            self._check_declined()
            return
        if len(self.arms) > self.neighbourhood_moves:
            raise RefinementRecordError(
                f"the pass measured {len(self.arms)} arms of a neighbourhood "
                f"of {self.neighbourhood_moves}, so the arms are no sample of it"
            )
        self._check_measured()
        self._check_winner()

    def _check_declined(self) -> None:
        """Enforce that a declined pass reached no card.

        Raises:
            RefinementRecordError: If the record carries any
                measurement, or counts a neighbourhood move.
                Declining happens before the first pack, which is
                what keeps an unreachable target free.
        """
        if self.arms or self.winner is not None or self.control is not None:
            raise RefinementRecordError(
                "a declined pass measures nothing and keeps nothing"
            )
        if self.neighbourhood_moves:
            raise RefinementRecordError(
                "a declined pass enumerated no move worth measuring, so its "
                "neighbourhood count is zero"
            )

    def _check_measured(self) -> None:
        """Enforce that the arms pair against one control.

        Raises:
            RefinementRecordError: If the control is missing or
                misnamed, arm names repeat, or an arm measured a
                different chunk count from the control.
        """
        if self.control is None:
            raise RefinementRecordError("a pass that ran needs its control measurement")
        if self.control.arm != CONTROL_ARM:
            raise RefinementRecordError(
                f'the control arm must be named "{CONTROL_ARM}"'
            )
        names = [a.arm for a in self.arms]
        if len(set(names)) != len(names):
            raise RefinementRecordError("arm names must be unique")
        for arm in self.arms:
            if arm.chunks != self.control.chunks:
                raise RefinementRecordError(
                    f"arm {arm.arm} measured {arm.chunks} chunks against the "
                    f"control's {self.control.chunks}, so the pairing is broken"
                )

    def _check_winner(self) -> None:
        """Enforce that a kept arm actually won.

        Raises:
            RefinementRecordError: If the winner names no measured
                arm, or names one that never cleared the stated bar.
        """
        if self.winner is None:
            return
        kept = self.winning_arm()
        if kept is None:
            raise RefinementRecordError(f"winner {self.winner} names no measured arm")
        if not kept.improved(self.bar):
            raise RefinementRecordError(
                f"winner {self.winner} reached {kept.sigma:.1f} sigma, "
                f"which does not clear the stated {self.bar}"
            )

    def winning_arm(self) -> ArmRecord | None:
        """Find the kept arm's record.

        Returns:
            The winning arm, or None when the pass kept none.
        """
        return next((a for a in self.arms if a.arm == self.winner), None)
