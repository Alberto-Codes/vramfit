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

The frame carries every input whose substitution would change the
number (ADR-0031 decision 5): the runtime build, the hardware, and
the content identity of the evaluation corpus, the reference logits,
and the importance matrix. An input the pass did not use records as
None, never as an absent field.

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

`outcome` classifies a pass once. It splits the arms into the ones
selection judged and the ones the weight budget excluded, and it
names the arm kept. Every surface that reports a pass renders that
structure rather than re-deriving the split, so the terminal and the
run log cannot describe one pass two ways.

The record is written as the pass runs, not once at the end. An arm
costs about 0.48 USD of card time, so a pass that stops at arm 12 of
16 must leave those twelve behind. `finished` is what keeps the two
empty-winner states apart: a pass that judged every arm and kept
none, and a pass that stopped before selection ran.

Examples:
    Read the winning arm's measured mean:

    ```python
    winner = sidecar.winning_arm()
    print(winner.mean if winner is not None else sidecar.declined)
    ```

See Also:
    - [vramfit.domain.refinement][]: Generates the arms.
    - [vramfit.domain.pack][]: `fits_weight_budget` states the
      weight-budget rule this module validates against.
    - [vramfit.domain.paired][]: Measures and selects between them.
      `cleared_bar` states the win rule this module validates
      against.
"""

from __future__ import annotations

from dataclasses import dataclass

from vramfit.domain.errors import VramfitError
from vramfit.domain.evals import CorpusReference
from vramfit.domain.pack import fits_weight_budget
from vramfit.domain.paired import cleared_bar

# The arm name reserved for the unmodified recipe. The pass measures
# it first, because a candidate is only readable against a control
# that reproduced the published frame.
CONTROL_ARM = "control"

# A SHA-256 digest as the frame records it: 64 lowercase hex digits.
_SHA256_HEX_LEN = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class RefinementRecordError(VramfitError, ValueError):
    """A refinement sidecar does not describe a readable pass.

    Examples:
        A winner that never cleared the stated bar:

        ```python
        raise RefinementRecordError("winner arm03 did not clear 7.8 sigma")
        ```
    """


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """One file a pass consumed, named by its bytes.

    The frame names the bytes, not the path alone: a path names no
    bytes once the file behind it changes. One type serves every such
    input — the importance matrix an assisted pack consumes
    (ADR-0016, ADR-0020) and the stored reference logits every
    divergence is measured against — because they are one concept and
    a second definition would drift from the first.

    Attributes:
        file (str): The path the pass was given.
        sha256 (str): SHA-256 of those bytes, 64 lowercase hex
            digits.
        size_bytes (int): Size of those bytes. Pairs with `sha256`
            as the glossary's content identity.

    Examples:
        The 30B pack's matrix:

        ```python
        from vramfit.domain.refinement_record import FileIdentity

        matrix = FileIdentity(file="30b.imatrix", sha256="ab" * 32, size_bytes=1024)
        ```
    """

    file: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Enforce that the reference names bytes.

        Raises:
            RefinementRecordError: If the path is empty, the digest
                is not 64 lowercase hex digits, or the byte count is
                not positive. Half a content identity names nothing.
        """
        if not self.file:
            raise RefinementRecordError("file must not be empty")
        if len(self.sha256) != _SHA256_HEX_LEN or not set(self.sha256) <= _HEX_DIGITS:
            raise RefinementRecordError("sha256 must be 64 lowercase hex digits")
        if self.size_bytes <= 0:
            raise RefinementRecordError("size_bytes must be positive")


@dataclass(frozen=True, slots=True)
class MeasurementFrame:
    """Where a refinement pass's numbers came from.

    ADR-0027 binds damage numbers to one instrument, so the record
    names the instrument rather than assuming it. Two passes measured
    in different frames do not compare.

    The frame carries every input whose substitution would change the
    number (ADR-0031 decision 5). That rule is what puts the matrix
    and the reference logits beside the corpus: an assisted pass and
    an unassisted one produce different numbers, and two passes
    against different reference logits are not comparable at all, so
    a record that omitted either would serialize them identically.

    Attributes:
        runtime_build (str): The runtime binary's build identity,
            e.g. ``b10362``.
        hardware (str): The card the pass ran on, e.g. ``H100 SXM``.
        corpus (CorpusReference): The evaluation text, named by
            content where the pass recorded it.
        reference (FileIdentity): The stored reference logits every
            divergence was measured against, named by content. It is
            the most load-bearing input of all: every figure in the
            record is computed relative to those bytes.
        imatrix (FileIdentity | None): The importance matrix every
            arm packed with, or None when the pass ran unassisted.
            None is a recorded state, never an absent field.

    Examples:
        The 2026-09-11 frame:

        ```python
        from vramfit.domain.evals import CorpusReference
        from vramfit.domain.refinement_record import (
            FileIdentity,
            MeasurementFrame,
        )

        frame = MeasurementFrame(
            runtime_build="b10362",
            hardware="H100 SXM",
            corpus=CorpusReference(file="wiki.test.raw"),
            reference=FileIdentity(
                file="base.logits", sha256="cd" * 32, size_bytes=1024
            ),
            imatrix=None,
        )
        ```
    """

    runtime_build: str
    hardware: str
    corpus: CorpusReference
    reference: FileIdentity
    imatrix: FileIdentity | None

    def __post_init__(self) -> None:
        """Enforce that the frame names its instrument.

        The reference and the matrix validate themselves, so this
        checks only the two fields the operator types by hand.

        Raises:
            RefinementRecordError: If ``runtime_build`` or
                ``hardware`` is empty. A frame that names nothing
                cannot bound a comparison.
        """
        for field_name in ("runtime_build", "hardware"):
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
        packed_bytes (int): Real size of the arm's packed file.
            Every arm is packed before it is measured, so every
            record has one.
        budget_margin (int): ``weight_budget_bytes - packed_bytes``,
            the same figure `pack` gates on
            (`vramfit.domain.pack.weight_budget_margin`). Negative
            means the arm packed over the budget its recipe was
            solved for, and `fits_budget` reads it. Byte-neutrality
            equalizes *predicted* bytes, so a swap can still pack
            over: the real GGUF size is a different number, which is
            why this is measured rather than assumed.

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
            packed_bytes=21_860_214_272,
            budget_margin=1_293_252_608,
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
    packed_bytes: int
    budget_margin: int
    predicted_delta: float | None = None

    def __post_init__(self) -> None:
        """Enforce that an arm describes a whole move, or none.

        Raises:
            RefinementRecordError: If the arm is unnamed, reports no
                chunks, counts more better chunks than it measured,
                half-describes its swap, or reports no packed bytes.
                The control names no swap and every other arm names
                all four parts.
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
        if self.packed_bytes <= 0:
            raise RefinementRecordError(f"arm {self.arm} has no packed bytes")

    def fits_budget(self) -> bool:
        """Judge whether this arm packed inside its weight budget.

        An arm that did not is measured and recorded, never dropped
        from the record — it cost card time and its number is real.
        It is excluded from selection instead, because handing that
        file to `pack` would exit 1 on the same rule.

        Reads `vramfit.domain.pack.fits_weight_budget`, the one
        definition of the rule, so a recorded arm is judged the way
        `vramfit pack` gates.

        Returns:
            True when the packed file fits the recipe's weight
            budget.
        """
        return fits_weight_budget(self.budget_margin)

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
class PassOutcome:
    """What a finished pass judged, what it excluded, what it kept.

    One classification, read by every surface that reports a pass.
    The terminal summary and the run log render this structure, so
    neither re-derives the split and neither can describe one pass
    differently. Two surfaces re-deriving it drifted inside one
    change: one said "1 of 2 measured arms packed over the weight
    budget", the other said "1 packed over the weight budget", and
    the second dropped the neighbourhood it was supposed to name.

    It carries records, never sentences. A string would let each
    reader re-derive meaning from text, which is what drifted.

    Attributes:
        judged (tuple[ArmRecord, ...]): The arms selection could
            choose between — every arm that packed inside its weight
            budget.
        excluded (tuple[ArmRecord, ...]): The arms the weight budget
            kept out of selection, each carrying its negative margin.
            Measured and recorded, never judged on merit.
        winner (ArmRecord | None): The arm the pass kept, or None
            when it kept none.
        bar (float): The evidence bar in sigma the pass ran against.
        neighbourhood_moves (int): How many byte-neutral moves the
            neighbourhood held, before any stride sampled it.

    Examples:
        Report a finished pass:

        ```python
        outcome = sidecar.outcome()
        print(outcome.summary())
        ```
    """

    judged: tuple[ArmRecord, ...]
    excluded: tuple[ArmRecord, ...]
    winner: ArmRecord | None
    bar: float
    neighbourhood_moves: int

    def measured(self) -> int:
        """Count the arms the pass packed and measured.

        Returns:
            The judged arms and the excluded arms together. Every one
            of them cost card time.
        """
        return len(self.judged) + len(self.excluded)

    def strongest_judged(self) -> ArmRecord | None:
        """Find the judged arm that came closest to winning.

        Returns:
            The judged arm with the lowest sigma, or None when the
            budget excluded every arm the pass measured.
        """
        if not self.judged:
            return None
        return min(self.judged, key=lambda arm: arm.sigma)

    def sample_phrase(self) -> str:
        """Word which arms the outcome speaks for.

        Returns:
            The arms measured and the neighbourhood they came from,
            naming the judged count separately whenever the budget
            excluded any. No sample supports a conclusion about the
            arms it never measured, and an excluded arm is not one
            the selection weighed.
        """
        phrase = f"{self.measured()} evaluated"
        if self.measured() != self.neighbourhood_moves:
            phrase = f"{phrase} of a neighbourhood of {self.neighbourhood_moves}"
        if self.excluded and self.judged:
            return f"{len(self.judged)} judged of {phrase}"
        return phrase

    def summary(self) -> str:
        """State the outcome in one sentence.

        Returns:
            What the pass kept or why it kept nothing, against the
            arms it judged and the neighbourhood they came from. An
            excluded arm is named as excluded and never as one that
            failed the bar.
        """
        over = f"{len(self.excluded)} packed over the weight budget"
        if self.winner is not None:
            kept = (
                f"winner: {self.winner.arm} at {self.winner.mean:.6f}, "
                f"{self.winner.sigma:+.1f} sigma against the control, "
                f"among the {self.sample_phrase()}"
            )
            return kept if not self.excluded else f"{kept}, and {over}"
        strongest = self.strongest_judged()
        if strongest is None:
            return (
                f"no arm of the {self.sample_phrase()} was judged on merit: all {over}"
            )
        lost = (
            f"no arm among the {self.sample_phrase()} cleared "
            f"{self.bar} sigma, and the strongest reached "
            f"{strongest.sigma:+.1f}"
        )
        return lost if not self.excluded else f"{lost}; {over}"

    def refusal(self) -> str | None:
        """State why the pass kept no arm.

        Returns:
            The summary when no arm was kept. None when one won: a
            winning pass refuses nothing, and the run log records its
            exclusions in their own field.
        """
        return None if self.winner is not None else self.summary()


@dataclass(frozen=True, slots=True)
class RefinementSidecar:
    """One refinement pass's complete search record.

    `outcome` reads these fields and nothing else. It is the one
    classifier of a finished pass, so every surface that reports one
    renders the same split.

    Attributes:
        model_id (str): The refined recipe's model identifier.
        frame (MeasurementFrame): Where the numbers came from.
        bar (float): Evidence bar in sigma the pass was run against.
        control (ArmRecord | None): The unmodified recipe's
            measurement, or None when the pass declined. A declined
            pass packs nothing and measures nothing, so it reaches no
            card at all.
        arms (tuple[ArmRecord, ...]): Every candidate measured, in
            measurement order, including any the weight budget
            excluded from selection — a measured arm is never dropped
            from the record. Empty when the pass declined, which is
            the only state meaning "never evaluated".
        winner (str | None): Name of the arm the pass kept, or None
            when no arm cleared the bar.
        declined (str | None): Why the recipe had no neighbourhood
            worth searching, or None when the pass ran arms.
        neighbourhood_moves (int): How many byte-neutral moves the
            neighbourhood held, before any stride took a sample. Read
            it beside ``len(arms)``: the two differ whenever the
            caller's arm budget was smaller than the neighbourhood,
            and a reader cannot judge an outcome without both.

            It is a count of what was enumerated, never a summary of
            the outcome. A declined pass records the moves it
            enumerated, which is zero only when the neighbourhood was
            genuinely empty. A pass that declines for another reason
            — a pin this map cannot resolve — records the moves it
            found, because "no neighbourhood" and "a neighbourhood
            the pass could not prove safe" are different facts about
            the recipe.
        finished (bool): Whether the pass measured every arm it
            selected and ran selection. The pass banks this record
            after the control and after every arm, so a pass that
            stops partway leaves the arms it paid for. False marks
            such a record.

            Read it before ``winner``. Both a finished pass that kept
            nothing and a pass that stopped before selecting carry an
            empty winner, and only this field separates them. A
            declined pass is finished: declining is an outcome
            (ADR-0031 decision 8), not an interruption.

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
    finished: bool

    def __post_init__(self) -> None:
        """Enforce that the record's claims match its measurements.

        Raises:
            RefinementRecordError: If the bar is negative, the
                neighbourhood count is negative or smaller than the
                arms measured, a pass that ran carries no control,
                arm names repeat, an arm measured a different chunk
                count from the control, the winner names no measured
                arm or one that never cleared the bar, a pass that
                stopped before selecting names a winner, or a
                declined pass still carries a measurement.
        """
        if not self.model_id:
            raise RefinementRecordError("model_id must not be empty")
        if self.bar < 0:
            raise RefinementRecordError("bar must not be negative")
        if self.neighbourhood_moves < 0:
            raise RefinementRecordError("neighbourhood_moves must not be negative")
        if not self.finished and self.winner is not None:
            raise RefinementRecordError(
                f"the pass stopped before selecting, so it cannot name "
                f"{self.winner} as its winner"
            )
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

        The neighbourhood count is not checked here. A decline that
        enumerated moves records them, because a recipe with 385
        moves the pass could not prove safe is not a recipe with no
        neighbourhood.

        Raises:
            RefinementRecordError: If the record carries any
                measurement, or reports that the pass stopped before
                finishing. Declining happens before the first pack,
                which is what keeps an unreachable target off the
                card, and it is an outcome rather than an
                interruption (ADR-0031 decision 8).
        """
        if self.arms or self.winner is not None or self.control is not None:
            raise RefinementRecordError(
                "a declined pass measures nothing and keeps nothing"
            )
        if not self.finished:
            raise RefinementRecordError("a declined pass is an outcome, so it finished")

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
                arm, names one that never cleared the stated bar, or
                names one that packed over the weight budget.
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
        if not kept.fits_budget():
            raise RefinementRecordError(
                f"winner {self.winner} packed {-kept.budget_margin} bytes over "
                "the weight budget, so no pass may keep it"
            )

    def winning_arm(self) -> ArmRecord | None:
        """Find the kept arm's record.

        Returns:
            The winning arm, or None when the pass kept none.
        """
        return next((a for a in self.arms if a.arm == self.winner), None)

    def outcome(self) -> PassOutcome:
        """Classify this pass, once, for every surface.

        Returns:
            The arms selection judged, the arms the weight budget
            excluded, and the arm the pass kept. A declined pass
            measured nothing, so every part is empty.
        """
        return PassOutcome(
            judged=tuple(a for a in self.arms if a.fits_budget()),
            excluded=tuple(a for a in self.arms if not a.fits_budget()),
            winner=self.winning_arm(),
            bar=self.bar,
            neighbourhood_moves=self.neighbourhood_moves,
        )
