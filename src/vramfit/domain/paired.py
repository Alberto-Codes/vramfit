"""Paired comparison of a refinement candidate against its control.

The refinement pass measures every candidate in the runtime frame and
compares it to the control on the same chunks. Pairing is what makes
the comparison sharp: chunk-to-chunk variance in the evaluation text
is large beside the effect, and it cancels when both arms read the
same chunk. The C4 sweep resolved a 6.06 percent mean-KLD improvement
at 14.4 sigma paired on 594 chunks.

Selection reads measurements only. The map that priced the recipe does
not order its neighbourhood, so nothing here consults a prediction.
`select` takes its evidence bar from the caller and has no default,
because the bar a result must clear is the caller's to state.

`cleared_bar` states what clearing the bar means, once. `select`
reads it to pick a winner and
`vramfit.domain.refinement_record.ArmRecord` reads it to validate a
recorded one, so no record can claim a winner selection would refuse.

`select` reports a winner and never a reason for none. The arms it is
handed are the judged arms alone, so it cannot see the ones the
weight budget excluded, and a refusal it worded would describe a pass
it only half read.

Examples:
    Compare one candidate against the control:

    ```python
    from vramfit.domain.paired import compare

    result = compare(candidate_chunks, control_chunks)
    print(result.sigma)
    ```

See Also:
    - [vramfit.domain.refinement][]: Generates the candidates.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from vramfit.domain.errors import VramfitError

# The smallest paired sample that has a spread at all. One chunk
# yields no standard deviation, so no sigma exists to report.
MIN_CHUNKS = 2


def cleared_bar(delta: float, sigma: float, bar: float) -> bool:
    """Judge one measured standing against an evidence bar.

    The one definition of what "cleared the bar" means. `select`
    reads it to pick a winner and `RefinementSidecar` reads it to
    validate the winner a record claims, so the two cannot disagree
    about which arm becomes a published claim.

    Args:
        delta: Mean paired difference, arm less control. Negative
            means the arm measured better.
        sigma: ``delta`` in units of its own standard error.
        bar: Evidence bar in sigma, stated positive.

    Returns:
        True when the arm measured lower and cleared the bar.

    Raises:
        ValueError: If ``bar`` is negative. A negative bar would
            accept an arm that measured worse.

    Examples:
        The winning arm of the 2026-09-11 sweep:

        ```python
        from vramfit.domain.paired import cleared_bar

        assert cleared_bar(-0.012368, -14.4, 7.8)
        ```
    """
    if bar < 0:
        raise ValueError("bar must not be negative")
    return delta < 0 and sigma <= -bar


class PairedError(VramfitError, ValueError):
    """Two arms cannot be compared as paired measurements.

    Examples:
        Arms measured over different chunk counts:

        ```python
        raise PairedError("arms measured 594 and 512 chunks")
        ```
    """


@dataclass(frozen=True, slots=True)
class PairedResult:
    """One candidate's measured standing against the control.

    Attributes:
        mean (float): The candidate's own mean divergence.
        control_mean (float): The control's mean divergence.
        delta (float): Mean paired difference, candidate less
            control. Negative means the candidate measured better.
        sigma (float): ``delta`` in units of its own standard error.
            Negative for an improvement, so a winner's sigma is
            below the negated bar.
        better_chunks (int): Chunks where the candidate measured
            lower divergence than the control.
        chunks (int): Chunks both arms measured.

    Examples:
        Read the improvement:

        ```python
        print(result.delta, result.sigma)
        ```
    """

    mean: float
    control_mean: float
    delta: float
    sigma: float
    better_chunks: int
    chunks: int

    def improved(self, bar: float) -> bool:
        """Judge whether this candidate beat the control past a bar.

        Reads `cleared_bar`, the one definition of the rule.

        Args:
            bar: Evidence bar in sigma, stated positive.

        Returns:
            True when the candidate measured lower and cleared the
            bar.

        Raises:
            ValueError: If ``bar`` is negative. A negative bar would
                accept a candidate that measured worse.
        """
        return cleared_bar(self.delta, self.sigma, bar)


def per_chunk(cumulative: Sequence[float]) -> tuple[float, ...]:
    """Recover per-chunk values from a running mean.

    The runtime's evaluation tool reports a running mean after each
    chunk. Chunk ``n``'s own value is ``n`` times the mean at ``n``
    less ``n - 1`` times the mean before it.

    Args:
        cumulative: Running mean after each chunk, in chunk order.

    Returns:
        One value per chunk.

    Raises:
        PairedError: If ``cumulative`` is empty.
    """
    if not cumulative:
        raise PairedError("no chunks to recover")
    out: list[float] = []
    previous = 0.0
    for index, mean in enumerate(cumulative, start=1):
        out.append(index * mean - (index - 1) * previous)
        previous = mean
    return tuple(out)


def _sigma(delta: float, error: float) -> float:
    """Express a paired difference in units of its standard error.

    A sample with no spread carries no finite sigma. A nonzero
    difference that every chunk agrees on is unbounded evidence, so it
    reports infinity with the difference's sign. Arms that measured
    identically report zero.

    Args:
        delta: Mean paired difference.
        error: Standard error of that difference.

    Returns:
        The signed sigma.
    """
    if error:
        return delta / error
    return 0.0 if delta == 0 else math.copysign(math.inf, delta)


def compare(candidate: Sequence[float], control: Sequence[float]) -> PairedResult:
    """Compare one candidate against the control, chunk by chunk.

    Args:
        candidate: The candidate's per-chunk divergences.
        control: The control's per-chunk divergences, same chunks in
            the same order.

    Returns:
        The paired standing.

    Raises:
        PairedError: If the two arms measured different chunk counts,
            or fewer than two chunks. Either leaves no paired sample
            with a spread.
    """
    if len(candidate) != len(control):
        raise PairedError(f"arms measured {len(candidate)} and {len(control)} chunks")
    if len(candidate) < MIN_CHUNKS:
        raise PairedError(
            f"a paired comparison needs at least {MIN_CHUNKS} chunks, "
            f"got {len(candidate)}"
        )
    differences = [c - b for c, b in zip(candidate, control, strict=True)]
    delta = statistics.fmean(differences)
    spread = statistics.stdev(differences)
    error = spread / math.sqrt(len(differences))
    return PairedResult(
        mean=statistics.fmean(candidate),
        control_mean=statistics.fmean(control),
        delta=delta,
        sigma=_sigma(delta, error),
        better_chunks=sum(1 for d in differences if d < 0),
        chunks=len(differences),
    )


def select(results: dict[str, PairedResult], bar: float) -> str | None:
    """Choose the winning arm, if one cleared the bar.

    The winner is the arm with the lowest measured mean among those
    that cleared the bar. Measurement decides alone.

    It names no loser and states no refusal. Why a pass kept nothing
    is a fact about the whole pass, including the arms the weight
    budget kept out of `results`, and
    `vramfit.domain.refinement_record.PassOutcome` states it once for
    every surface.

    Args:
        results: One paired standing per arm name.
        bar: Evidence bar in sigma, stated positive.

    Returns:
        The winning arm's name, or None when no arm cleared the bar.

    Raises:
        ValueError: If ``bar`` is negative.
    """
    if bar < 0:
        raise ValueError("bar must not be negative")
    winners = {name: r for name, r in results.items() if r.improved(bar)}
    if not winners:
        return None
    return min(winners, key=lambda name: winners[name].mean)
