"""Equal-byte neighbour generation for an assembled recipe.

The refinement pass searches the neighbourhood of a solved recipe. A
neighbour swaps the precisions of two assignments and spends the same
bytes, so it competes inside the same weight budget. The C4 sweep
measured fifteen neighbours of the published 30B recipe on 2026-09-11.
Nine measured lower full-window KLD than the recipe they came from.

The generator does not rank. Spearman rho between the map's predicted
penalty and the measured delta was +0.146 over those fifteen arms, so
the map does not order the neighbourhood it prices. Every candidate
this module emits carries `predicted_delta` as recorded provenance,
never as a selection key. `vramfit.domain.paired` selects from
measurements instead.

Pricing rides on the recipe, not on a rebuilt table. A swap is
byte-neutral exactly when the two groups price identically at every
precision, which holds when they carry the same reference size. The
candidate then trades one recorded byte figure for the other and the
total cannot move. `refuse_unpriced_move` states that condition.

Examples:
    Enumerate a recipe's neighbourhood:

    ```python
    from vramfit.domain.refinement import neighbours

    candidates = neighbours(recipe, map_)
    ```

See Also:
    - [vramfit.domain.paired][]: Selects a winner from measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vramfit.domain.errors import VramfitError
from vramfit.domain.model import Assignment, Recipe, SensitivityMap

# The smallest number of distinct precisions a neighbourhood needs. A
# recipe holding one precision has no pair to swap between.
MIN_LEVELS = 2


class RefinementError(VramfitError, ValueError):
    """A neighbourhood cannot be generated as asked.

    Examples:
        A recipe and a map that describe different models:

        ```python
        raise RefinementError("recipe names a group the map omits: g9")
        ```
    """


@dataclass(frozen=True, slots=True)
class Move:
    """One byte-neutral swap between two assignments.

    Attributes:
        promoted (str): Group whose precision rises.
        demoted (str): Group whose precision falls.
        from_bits (int): The promoted group's precision before the
            swap, which the demoted group takes after it.
        to_bits (int): The promoted group's precision after the swap,
            which the demoted group held before it.

    Examples:
        Trade a 2-bit stack up against a 4-bit stack:

        ```python
        from vramfit.domain.refinement import Move

        move = Move(promoted="g1", demoted="g2", from_bits=2, to_bits=4)
        ```
    """

    promoted: str
    demoted: str
    from_bits: int
    to_bits: int

    def __post_init__(self) -> None:
        """Enforce that the swap moves precision in both directions.

        Raises:
            ValueError: If the two groups are the same, or
                ``to_bits`` does not exceed ``from_bits``. A swap
                that raises nothing frees nothing.
        """
        if self.promoted == self.demoted:
            raise ValueError("a swap needs two distinct groups")
        if self.to_bits <= self.from_bits:
            raise ValueError(
                f"to_bits {self.to_bits} must exceed from_bits {self.from_bits}"
            )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One neighbour of the recipe under refinement.

    Attributes:
        move (Move): The swap that produced this neighbour.
        assignments (tuple[Assignment, ...]): The neighbour's full
            assignment list, in the recipe's own order.
        predicted_delta (float): Damage change the map predicts for
            the swap. Recorded provenance, never a selection key —
            the map does not order the neighbourhood (#486).

    Examples:
        Read a candidate's move:

        ```python
        print(candidate.move.promoted, candidate.move.demoted)
        ```
    """

    move: Move
    assignments: tuple[Assignment, ...]
    predicted_delta: float

    def total_bytes(self) -> int:
        """Sum the candidate's predicted assignment sizes.

        Returns:
            Predicted total bytes across every assignment.
        """
        return sum(a.bytes for a in self.assignments)


def _group_sensitivity(map_: SensitivityMap) -> Mapping[str, Mapping[int, float]]:
    """Index the map's per-group damage curves by group name.

    Args:
        map_: The map that priced the recipe.

    Returns:
        Damage per precision, per group name.
    """
    return {g.name: g.sensitivity for g in map_.groups}


def _reference_bytes(map_: SensitivityMap) -> Mapping[str, int]:
    """Index the map's per-group reference sizes by group name.

    Args:
        map_: The map that priced the recipe.

    Returns:
        Reference-precision bytes per group name.
    """
    return {g.name: g.bytes_fp16 for g in map_.groups}


def refuse_unpriced_move(
    move: Move,
    reference_bytes: Mapping[str, int],
    sensitivity: Mapping[str, Mapping[int, float]],
) -> str | None:
    """Report why a swap cannot be priced from the recipe alone.

    The swap trades the two groups' recorded byte figures. That trade
    is exact only when both groups price identically at every
    precision, which this function tests by reference size. It also
    refuses a precision the map never measured, because the
    neighbour's recorded damage would then name no measurement.

    Args:
        move: The swap to check.
        reference_bytes: Reference-precision bytes per group name.
        sensitivity: Damage per precision, per group name.

    Returns:
        The refusal, or None when the swap prices exactly.
    """
    for name in (move.promoted, move.demoted):
        if name not in reference_bytes:
            return f"the map omits group {name}"
    if reference_bytes[move.promoted] != reference_bytes[move.demoted]:
        return (
            f"groups {move.promoted} and {move.demoted} carry different "
            "reference sizes, so swapping their bytes is not byte-neutral"
        )
    for name in (move.promoted, move.demoted):
        for bits in (move.from_bits, move.to_bits):
            if bits not in sensitivity[name]:
                return f"the map never measured {name} at {bits} bits"
    return None


def apply_move(
    move: Move,
    assignments: tuple[Assignment, ...],
    sensitivity: Mapping[str, Mapping[int, float]],
) -> tuple[Assignment, ...]:
    """Rewrite the two swapped assignments, leaving the rest in place.

    Each rewritten assignment takes the other group's recorded byte
    figure, which keeps the predicted total fixed. Its damage comes
    from the map at the new precision.

    Args:
        move: The swap to apply.
        assignments: The recipe's assignments, in recipe order.
        sensitivity: Damage per precision, per group name.

    Returns:
        The neighbour's assignments, in the same order.

    Raises:
        RefinementError: If an assignment named by the move is
            missing, or either group does not sit at the precision
            the move expects.
    """
    current = {a.group: a for a in assignments}
    for name in (move.promoted, move.demoted):
        if name not in current:
            raise RefinementError(f"the recipe omits group {name}")
    if current[move.promoted].bits != move.from_bits:
        raise RefinementError(
            f"{move.promoted} sits at {current[move.promoted].bits} bits, "
            f"not the {move.from_bits} the swap promotes from"
        )
    if current[move.demoted].bits != move.to_bits:
        raise RefinementError(
            f"{move.demoted} sits at {current[move.demoted].bits} bits, "
            f"not the {move.to_bits} the swap demotes from"
        )
    swapped = {
        move.promoted: Assignment(
            group=move.promoted,
            bits=move.to_bits,
            bytes=current[move.demoted].bytes,
            damage=sensitivity[move.promoted][move.to_bits],
        ),
        move.demoted: Assignment(
            group=move.demoted,
            bits=move.from_bits,
            bytes=current[move.promoted].bytes,
            damage=sensitivity[move.demoted][move.from_bits],
        ),
    }
    return tuple(swapped.get(a.group, a) for a in assignments)


def predicted_delta(
    move: Move, sensitivity: Mapping[str, Mapping[int, float]]
) -> float:
    """Predict the swap's damage change from the map.

    Recorded provenance, never a selection key. The map did not order
    the measured neighbourhood (#486).

    Args:
        move: The swap to price.
        sensitivity: Damage per precision, per group name.

    Returns:
        Damage the demotion adds, less damage the promotion removes.
    """
    demoted = sensitivity[move.demoted]
    promoted = sensitivity[move.promoted]
    added = demoted[move.from_bits] - demoted[move.to_bits]
    removed = promoted[move.from_bits] - promoted[move.to_bits]
    return added - removed


def neighbours(recipe: Recipe, map_: SensitivityMap) -> tuple[Candidate, ...]:
    """Enumerate every byte-neutral neighbour of a recipe.

    One neighbour per ordered pair of assignments whose precisions
    differ and whose swap prices exactly. The result carries no
    ordering the caller should read as a ranking — it follows the
    recipe's own assignment order.

    Args:
        recipe: The solved recipe to search around.
        map_: The map that priced it.

    Returns:
        Every legal neighbour, empty when the recipe has none.
    """
    sensitivity = _group_sensitivity(map_)
    reference = _reference_bytes(map_)
    found: list[Candidate] = []
    for low in recipe.assignments:
        for high in recipe.assignments:
            if low.bits >= high.bits:
                continue
            move = Move(
                promoted=low.group,
                demoted=high.group,
                from_bits=low.bits,
                to_bits=high.bits,
            )
            if refuse_unpriced_move(move, reference, sensitivity) is not None:
                continue
            found.append(
                Candidate(
                    move=move,
                    assignments=apply_move(move, recipe.assignments, sensitivity),
                    predicted_delta=predicted_delta(move, sensitivity),
                )
            )
    return tuple(found)


def decline_reason(recipe: Recipe, map_: SensitivityMap) -> str | None:
    """Report why a recipe has no neighbourhood worth searching.

    A recipe whose groups all sit at one precision has no swap at
    all. The published 49B recipe places 81 of its 82 groups at the
    3-bit floor, and its one 8-bit group prices differently from
    every other, so the protocol's move does not exist there.

    Args:
        recipe: The solved recipe to search around.
        map_: The map that priced it.

    Returns:
        The refusal, or None when at least one neighbour exists.
    """
    levels = {a.bits for a in recipe.assignments}
    if len(levels) < MIN_LEVELS:
        only = next(iter(levels))
        return f"every group sits at {only} bits, so no swap moves precision"
    if not neighbours(recipe, map_):
        return (
            "no pair of groups both prices exactly and differs in precision, "
            "so the recipe has no byte-neutral neighbour"
        )
    return None
