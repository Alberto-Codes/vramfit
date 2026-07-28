"""Greedy damage-per-byte solver that turns a sensitivity map into a recipe.

Implements ADR-0007: start every group at the highest candidate precision
(or its pin), then repeatedly apply the downgrade with the best
damage-per-byte-freed ratio until the total fits the weight budget. The
ordered downgrade log is recorded in the recipe as its explanation
trace, and the final downgrade is refined when a milder step also fits.
Inputs are validated at the API boundary: a negative ``format_overhead``
raises ``ValueError`` before any solving starts.

Attributes:
    SOLVER_NAME (str): Identifier recorded in ``plan.solver`` for
        reproducibility.
    DEFAULT_FORMAT_OVERHEAD (float): Default quantization-format overhead
        fraction (scales, zero-points) applied to size predictions.

Examples:
    Solve a map against a byte budget:

    ```python
    from quantfit.domain.solver import solve

    recipe = solve(
        map_,  # a quantfit.domain.model.SensitivityMap
        weight_budget_bytes=20 * 2**30,
        vram_budget_bytes=24 * 2**30,
        kv_headroom_bytes=4 * 2**30,
    )
    ```

See Also:
    - [quantfit.domain.model][]: The `SensitivityMap` input and `Recipe`
      output types.
    - [quantfit.domain.budget][]: Computes the weight budget this solver
      packs to.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from typing import Final

from quantfit.domain.model import (
    Assignment,
    PlanMeta,
    Recipe,
    SensitivityMap,
    TraceStep,
)

SOLVER_NAME: Final[str] = "greedy-damage-per-byte"
DEFAULT_FORMAT_OVERHEAD: Final[float] = 0.05


class PinError(ValueError):
    """A ``--pin`` pattern is unusable.

    Raised when a pin names a precision the scan did not measure, or when
    its pattern matches no group (usually a typo).

    Examples:
        A pin against an unscanned precision:

        ```python
        from quantfit.domain.solver import PinError

        try:
            solve_with_pins(pins={"g*": 6})
        except PinError as exc:
            print(exc)
        ```
    """


class InfeasibleBudgetError(Exception):
    """No recipe fits the weight budget, even at minimum precision.

    Attributes:
        gap_bytes (int): How far the best possible total overshoots the
            budget.
        minimum_bytes (int): The smallest achievable total (pins
            respected).
        weight_budget_bytes (int): The budget that could not be met.

    Examples:
        Report the gap to the user:

        ```python
        from quantfit.domain.solver import InfeasibleBudgetError

        try:
            solve_with_tiny_budget()
        except InfeasibleBudgetError as exc:
            print(f"over budget by {exc.gap_bytes} bytes")
        ```
    """

    def __init__(
        self, gap_bytes: int, minimum_bytes: int, weight_budget_bytes: int
    ) -> None:
        """Build the error from the budget arithmetic.

        Args:
            gap_bytes: Overshoot of the smallest achievable total.
            minimum_bytes: The smallest achievable total in bytes.
            weight_budget_bytes: The budget that could not be met.
        """
        super().__init__(
            f"no recipe fits: minimum achievable size is {minimum_bytes} bytes, "
            f"{gap_bytes} bytes over the {weight_budget_bytes}-byte weight budget"
        )
        self.gap_bytes = gap_bytes
        self.minimum_bytes = minimum_bytes
        self.weight_budget_bytes = weight_budget_bytes


def group_bytes(bytes_fp16: int, bits: int, format_overhead: float) -> int:
    """Predict a group's size at a target precision.

    Args:
        bytes_fp16: The group's size at 16-bit reference precision.
        bits: Target precision.
        format_overhead: Overhead fraction for quantization metadata.

    Returns:
        Predicted bytes, rounded up.

    Examples:
        4-bit with 5% overhead is ~26% of the fp16 size:

        ```python
        from quantfit.domain.solver import group_bytes

        assert group_bytes(1600, 4, 0.05) == 420
        ```
    """
    return math.ceil(bytes_fp16 * bits / 16 * (1 + format_overhead))


def _expand_pins(
    pins: Mapping[str, int],
    map_: SensitivityMap,
) -> dict[str, int]:
    """Resolve pin patterns to concrete per-group precisions.

    Args:
        pins: Ordered mapping of glob pattern to forced precision; later
            patterns override earlier ones for overlapping groups.
        map_: The sensitivity map whose groups are matched.

    Returns:
        Mapping of group name to pinned precision.

    Raises:
        PinError: If a pin uses an unscanned precision or matches no
            group.
    """
    candidates = set(map_.scan.precisions)
    pinned: dict[str, int] = {}
    for pattern, bits in pins.items():
        if bits not in candidates:
            raise PinError(
                f'pin "{pattern}={bits}": precision {bits} is not in the scanned '
                f"set {sorted(candidates, reverse=True)}"
            )
        matched = [g.name for g in map_.groups if fnmatchcase(g.name, pattern)]
        if not matched:
            raise PinError(f'pin "{pattern}={bits}" matches no group')
        for name in matched:
            pinned[name] = bits
    return pinned


def _best_move(
    sensitivity_map: SensitivityMap,
    pinned: dict[str, int],
    state: dict[str, int],
    candidates: tuple[int, ...],
    size: Callable[[int, int], int],
) -> tuple[str, int, int, float] | None:
    """Pick the downgrade with the minimum greedy selection key.

    Considers every lower candidate precision of every unpinned group.
    Moves that free no bytes (ceil rounding on tiny groups) are skipped —
    they never help and would divide by zero in the ratio.

    Args:
        sensitivity_map: Damage curves for every group.
        pinned: Group names whose precision is user-forced.
        state: Current precision per group name.
        candidates: The scan's candidate precisions, descending.
        size: Group-size predictor for the solver's overhead setting.

    Returns:
        ``(group, target_bits, bytes_freed, damage_delta)`` for the best
        move, or None when no downgrade can free bytes.
    """
    best_key: tuple[float, str, int] | None = None
    best: tuple[str, int, int, float] | None = None
    for group in sensitivity_map.groups:
        if group.name in pinned:
            continue
        current = state[group.name]
        current_bytes = size(group.bytes_fp16, current)
        for target in candidates:
            if target >= current:
                continue
            bytes_freed = current_bytes - size(group.bytes_fp16, target)
            if bytes_freed <= 0:
                continue
            damage_delta = group.sensitivity[target] - group.sensitivity[current]
            key = (damage_delta / bytes_freed, group.name, -target)
            if best_key is None or key < best_key:
                best_key = key
                best = (group.name, target, bytes_freed, damage_delta)
    return best


def _refine_last_step(
    sensitivity_map: SensitivityMap,
    trace: list[TraceStep],
    state: dict[str, int],
    total: int,
    weight_budget_bytes: int,
    candidates: tuple[int, ...],
    size: Callable[[int, int], int],
) -> int:
    """Shrink the final downgrade if a smaller step also fits the budget.

    The greedy loop ranks moves by ratio, so the step that crosses under
    the budget can overshoot: a milder downgrade of the same group might
    fit with less damage. This post-pass replaces the last trace step
    with the least-damaging sufficient alternative, keeping the recipe
    deterministic and the trace all-downgrades.

    Args:
        sensitivity_map: Damage curves for every group.
        trace: The downgrade log; its last step may be replaced in place.
        state: Current precision per group name; updated on refinement.
        total: Current predicted weight total.
        weight_budget_bytes: Hard ceiling for the predicted total.
        candidates: The scan's candidate precisions, descending.
        size: Group-size predictor for the solver's overhead setting.

    Returns:
        The (possibly reduced-overshoot) predicted total in bytes.
    """
    if not trace:
        return total
    last = trace[-1]
    group = next(g for g in sensitivity_map.groups if g.name == last.group)
    best: tuple[float, int, int, int] | None = None
    for target in candidates:
        if not last.to_bits < target < last.from_bits:
            continue
        bytes_freed = size(group.bytes_fp16, last.from_bits) - size(
            group.bytes_fp16, target
        )
        if bytes_freed <= 0:
            continue
        new_total = (
            total
            - size(group.bytes_fp16, last.to_bits)
            + size(group.bytes_fp16, target)
        )
        if new_total > weight_budget_bytes:
            continue
        damage = group.sensitivity[target]
        key = (damage, -target)
        if best is None or key < (best[0], best[1]):
            best = (damage, -target, bytes_freed, new_total)
    if best is None or best[0] >= group.sensitivity[last.to_bits]:
        return total
    damage, neg_target, bytes_freed, new_total = best
    target = -neg_target
    damage_delta = damage - group.sensitivity[last.from_bits]
    trace[-1] = TraceStep(
        step=last.step,
        group=last.group,
        from_bits=last.from_bits,
        to_bits=target,
        damage_delta=damage_delta,
        bytes_freed=bytes_freed,
        ratio=damage_delta / bytes_freed,
    )
    state[last.group] = target
    return new_total


def solve(
    sensitivity_map: SensitivityMap,
    weight_budget_bytes: int,
    *,
    vram_budget_bytes: int,
    kv_headroom_bytes: int,
    pins: Mapping[str, int] | None = None,
    format_overhead: float = DEFAULT_FORMAT_OVERHEAD,
) -> Recipe:
    """Assign a precision to every group so the total fits the budget.

    Every group starts at the highest scanned precision (or its pin).
    While the total exceeds the budget, the solver applies the downgrade
    with the minimum ``(damage_delta / bytes_freed, group name, smallest
    step)`` key, considering all lower candidate precisions of every
    unpinned group. Moves that free no bytes (possible on tiny groups
    where sizes round to the same value) are never considered. The
    selection is a total order, so the result is deterministic and
    independent of group input order. After the loop, the final downgrade
    is refined: if a milder step of the same group also fits with less
    damage, it replaces the overshooting one.

    Args:
        sensitivity_map: Damage curves for every group.
        weight_budget_bytes: Hard ceiling for the predicted weight total.
        vram_budget_bytes: Total VRAM ceiling, recorded for provenance.
        kv_headroom_bytes: Reserved KV/runtime bytes, recorded for
            provenance.
        pins: Ordered glob-pattern pins forcing precisions; later patterns
            override earlier ones.
        format_overhead: Overhead fraction for quantization metadata.

    Returns:
        The recipe, with assignments in sensitivity-map group order and
        the downgrade trace in ``plan.trace``.

    Raises:
        ValueError: If ``format_overhead`` is negative.
        PinError: If a pin is malformed with respect to the map.
        InfeasibleBudgetError: If even minimum precision (pins respected)
            exceeds the budget.

    Examples:
        Pin the first layer high and solve:

        ```python
        from quantfit.domain.solver import solve

        recipe = solve(
            map_,
            weight_budget_bytes=20 * 2**30,
            vram_budget_bytes=24 * 2**30,
            kv_headroom_bytes=4 * 2**30,
            pins={"model.layers.0.*": 8},
        )
        ```
    """
    if format_overhead < 0:
        raise ValueError(f"format_overhead must be non-negative, got {format_overhead}")
    pins = dict(pins or {})
    candidates = sensitivity_map.scan.precisions
    pinned = _expand_pins(pins, sensitivity_map)

    def size(bytes_fp16: int, bits: int) -> int:
        """Shorthand for `group_bytes` with the solver's overhead.

        Args:
            bytes_fp16: Group size at reference precision.
            bits: Target precision.

        Returns:
            Predicted bytes at the target precision.
        """
        return group_bytes(bytes_fp16, bits, format_overhead)

    state: dict[str, int] = {}
    for group in sensitivity_map.groups:
        state[group.name] = pinned.get(group.name, candidates[0])

    total = sum(size(g.bytes_fp16, state[g.name]) for g in sensitivity_map.groups)
    floor_total = sum(
        size(g.bytes_fp16, pinned.get(g.name, candidates[-1]))
        for g in sensitivity_map.groups
    )
    if floor_total > weight_budget_bytes:
        raise InfeasibleBudgetError(
            gap_bytes=floor_total - weight_budget_bytes,
            minimum_bytes=floor_total,
            weight_budget_bytes=weight_budget_bytes,
        )

    trace: list[TraceStep] = []
    while total > weight_budget_bytes:
        best_move = _best_move(sensitivity_map, pinned, state, candidates, size)
        if best_move is None:  # pragma: no cover - guarded by the precheck
            raise RuntimeError(
                "solver invariant broken: over budget but no freeing move "
                "exists despite the feasibility precheck"
            )
        name, target, bytes_freed, damage_delta = best_move
        trace.append(
            TraceStep(
                step=len(trace) + 1,
                group=name,
                from_bits=state[name],
                to_bits=target,
                damage_delta=damage_delta,
                bytes_freed=bytes_freed,
                ratio=damage_delta / bytes_freed,
            )
        )
        state[name] = target
        total -= bytes_freed

    total = _refine_last_step(
        sensitivity_map, trace, state, total, weight_budget_bytes, candidates, size
    )
    assignments = tuple(
        Assignment(
            group=g.name,
            bits=state[g.name],
            bytes=size(g.bytes_fp16, state[g.name]),
            damage=g.sensitivity[state[g.name]],
        )
        for g in sensitivity_map.groups
    )
    return Recipe(
        model_id=sensitivity_map.model_id,
        plan=PlanMeta(
            vram_budget_bytes=vram_budget_bytes,
            kv_headroom_bytes=kv_headroom_bytes,
            weight_budget_bytes=weight_budget_bytes,
            predicted_total_bytes=total,
            predicted_damage=sum(a.damage for a in assignments),
            solver=SOLVER_NAME,
            pins=pins,
            format_overhead=format_overhead,
            trace=tuple(trace),
        ),
        assignments=assignments,
    )
