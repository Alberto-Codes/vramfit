"""Greedy damage-per-byte solver that turns a sensitivity map into a recipe.

Implements ADR-0007. Errors sit under the `VramfitError` root
(ADR-0011) and carry user-facing messages the CLI prints verbatim.
A target runtime narrows the candidate set through the ADR-0013
capability table before any solving starts, and an infeasible
budget names the precisions that narrowing removed.
The algorithm: start every group at the highest candidate precision
(or its pin), then repeatedly apply the downgrade with the best
damage-per-byte-freed ratio until the total fits the weight budget. The
ordered downgrade log is recorded in the recipe as its explanation
trace, and the final downgrade is refined when a milder step also fits.
Size predictions follow ADR-0014: a runtime with an effective-bits
table prices each precision at its real per-weight cost, and the
overhead fraction shrinks to a residual for what the table cannot
see (unquantized tensors, file metadata). Without a table the
nominal-bits prediction and the 0.05 scalar remain. Protections
(ADR-0022) enter through the size model only: a protected tensor
prices at the higher of the candidate precision and its floor
([vramfit.domain.protection][]), so downgrading a protected group
frees fewer bytes and the ranking shifts. Imatrix exclusions
(ADR-0023) change no size and no damage — the solver only expands
their patterns against the protected set and records the marked
pairs in the recipe. A pair resolves only where its floor exceeds
the final assignment, and the solver refuses an exclusion left with
no surviving pair (issue #59). Inputs are
validated at the API boundary: a negative ``format_overhead`` raises
``ValueError`` before any solving starts.

Attributes:
    SOLVER_NAME (str): Identifier recorded in ``plan.solver`` for
        reproducibility.
    DEFAULT_FORMAT_OVERHEAD (float): Default overhead fraction when no
        effective-bits table applies — one scalar has to cover scales,
        zero-points, and everything else.
    DEFAULT_RESIDUAL_OVERHEAD (float): Default overhead fraction when
        an effective-bits table prices the quantization metadata —
        covers only unquantized tensors and file metadata (ADR-0014).

Examples:
    Solve a map against a byte budget:

    ```python
    from vramfit.domain.solver import solve

    recipe = solve(
        map_,  # a vramfit.domain.model.SensitivityMap
        weight_budget_bytes=20 * 2**30,
        vram_budget_bytes=24 * 2**30,
        kv_headroom_bytes=4 * 2**30,
    )
    ```

See Also:
    - [vramfit.domain.model][]: The `SensitivityMap` input and `Recipe`
      output types.
    - [vramfit.domain.budget][]: Computes the weight budget this solver
      packs to.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from typing import Final

from vramfit.domain.budget import format_size
from vramfit.domain.errors import VramfitError
from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    SensitivityMap,
    TraceStep,
)
from vramfit.domain.protection import (
    expand_exclusions,
    expand_protections,
    protected_group_bytes,
    refuse_dead_exclusions,
    resolve_protected,
)
from vramfit.domain.runtime import effective_bits, servable_precisions

SOLVER_NAME: Final[str] = "greedy-damage-per-byte"
DEFAULT_FORMAT_OVERHEAD: Final[float] = 0.05
DEFAULT_RESIDUAL_OVERHEAD: Final[float] = 0.005


class PinError(VramfitError, ValueError):
    """A ``--pin`` pattern is unusable. Under the `VramfitError` root.

    Raised when a pin names a precision the scan did not measure, or when
    its pattern matches no group (usually a typo).

    Examples:
        A pin against an unscanned precision:

        ```python
        from vramfit.domain.solver import PinError

        try:
            solve_with_pins(pins={"g*": 6})
        except PinError as exc:
            print(exc)
        ```
    """


class InfeasibleBudgetError(VramfitError):
    """No recipe fits the weight budget. Under the `VramfitError` root.

    Attributes:
        gap_bytes (int): How far the best possible total overshoots the
            budget.
        minimum_bytes (int): The smallest achievable total (pins
            respected).
        weight_budget_bytes (int): The budget that could not be met.
        runtime (str | None): Target runtime the solve was constrained
            to, when one was given.
        dropped_precisions (tuple[int, ...]): Scanned precisions the
            runtime cannot serve — the floor the message reports
            excludes them.

    Examples:
        Report the gap to the user:

        ```python
        from vramfit.domain.solver import InfeasibleBudgetError

        try:
            solve_with_tiny_budget()
        except InfeasibleBudgetError as exc:
            print(f"over budget by {exc.gap_bytes} bytes")
        ```
    """

    def __init__(
        self,
        gap_bytes: int,
        minimum_bytes: int,
        weight_budget_bytes: int,
        *,
        runtime: str | None = None,
        dropped_precisions: tuple[int, ...] = (),
        protected_count: int = 0,
    ) -> None:
        """Build the error from the budget arithmetic.

        The message renders every size with `format_size`, so the CLI
        can print it verbatim as its ``error:`` line. When a runtime
        filter removed scanned precisions, the message names them —
        the reported floor is higher than the scan alone allows, and
        the user must see why. Protections raise the floor the same
        way (ADR-0022), so the message counts them too.

        Args:
            gap_bytes: Overshoot of the smallest achievable total.
            minimum_bytes: The smallest achievable total in bytes.
            weight_budget_bytes: The budget that could not be met.
            runtime: Target runtime that constrained the solve, when
                one was given.
            dropped_precisions: Scanned precisions the runtime cannot
                serve.
            protected_count: Tensors held at a protection floor.
        """
        message = (
            f"no recipe fits the {format_size(weight_budget_bytes)} weight "
            f"budget — minimum achievable is {format_size(minimum_bytes)} "
            f"({format_size(gap_bytes)} over)"
        )
        if runtime is not None and dropped_precisions:
            message += (
                f'. The target runtime "{runtime}" cannot serve the scanned '
                f"precisions {list(dropped_precisions)}, so the floor sits "
                f"higher than the scan alone allows"
            )
        if protected_count:
            message += (
                f". Protections hold {protected_count} tensors at their "
                "floors, raising the minimum (ADR-0022)"
            )
        super().__init__(message)
        self.gap_bytes = gap_bytes
        self.minimum_bytes = minimum_bytes
        self.weight_budget_bytes = weight_budget_bytes
        self.runtime = runtime
        self.dropped_precisions = dropped_precisions
        self.protected_count = protected_count


def group_bytes(bytes_fp16: int, bits: float, format_overhead: float) -> int:
    """Predict a group's size at a per-weight bit cost.

    Args:
        bytes_fp16: The group's size at 16-bit reference precision.
        bits: Bits spent per weight — a nominal precision, or a
            fractional effective-bits value from a runtime table
            (ADR-0014).
        format_overhead: Overhead fraction for whatever ``bits`` does
            not price in.

    Returns:
        Predicted bytes, rounded up.

    Examples:
        A 4-bit group priced at Q4_K's 4.5 effective bits:

        ```python
        from vramfit.domain.solver import group_bytes

        assert group_bytes(1600, 4.5, 0.0) == 450
        ```
    """
    return math.ceil(bytes_fp16 * bits / 16 * (1 + format_overhead))


def _expand_pins(
    pins: Mapping[str, int],
    map_: SensitivityMap,
    candidates: tuple[int, ...],
) -> dict[str, int]:
    """Resolve pin patterns to concrete per-group precisions.

    Args:
        pins: Ordered mapping of glob pattern to forced precision; later
            patterns override earlier ones for overlapping groups.
        map_: The sensitivity map whose groups are matched.
        candidates: The solver's candidate precisions — the scanned
            set, runtime-filtered when a target runtime is given.

    Returns:
        Mapping of group name to pinned precision.

    Raises:
        PinError: If a pin uses a precision outside the candidate set
            or matches no group.
    """
    allowed = set(candidates)
    pinned: dict[str, int] = {}
    for pattern, bits in pins.items():
        if bits not in allowed:
            raise PinError(
                f'pin "{pattern}={bits}": precision {bits} is not in the candidate '
                f"set {sorted(allowed, reverse=True)}"
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
    size: Callable[[LayerGroup, int], int],
) -> tuple[str, int, int, float] | None:
    """Pick the downgrade with the minimum greedy selection key.

    Considers every lower candidate precision of every unpinned group.
    Moves that free no bytes (ceil rounding on tiny groups) are skipped —
    they never help and would divide by zero in the ratio. Downgrading
    a group with protected tensors frees fewer bytes (ADR-0022), so
    its ratio worsens and the ranking shifts accordingly.

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
        current_bytes = size(group, current)
        for target in candidates:
            if target >= current:
                continue
            bytes_freed = current_bytes - size(group, target)
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
    size: Callable[[LayerGroup, int], int],
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
        size: Group-size predictor ``(group, bits) -> bytes``, carrying
            the solver's overhead setting and protection floors
            (ADR-0022).

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
        bytes_freed = size(group, last.from_bits) - size(group, target)
        if bytes_freed <= 0:
            continue
        new_total = total - size(group, last.to_bits) + size(group, target)
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


def solve(  # noqa: PLR0913 - the plan surface: budget triple + pins, protections, overhead, runtime
    sensitivity_map: SensitivityMap,
    weight_budget_bytes: int,
    *,
    vram_budget_bytes: int,
    kv_headroom_bytes: int,
    pins: Mapping[str, int] | None = None,
    protections: Mapping[str, int] | None = None,
    imatrix_exclusions: tuple[str, ...] = (),
    format_overhead: float | None = None,
    runtime: str | None = None,
) -> Recipe:
    """Assign a precision to every group so the total fits the budget.

    When a target runtime is given, the candidate set first filters
    through the ADR-0013 capability table — a precision the runtime
    cannot serve is never assigned, and the recipe records the
    runtime. An infeasible budget then names the precisions the
    runtime removed, so the reported floor is explainable. A runtime
    with an effective-bits table prices every candidate at its real
    per-weight cost (ADR-0014) — Q4_K spends 4.5 bits, not 4. Every
    group starts at the highest candidate precision (or its pin).
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
        protections: Ordered fnmatch protection rules, pattern to
            floor (ADR-0022); later patterns override earlier ones
            for overlapping tensors. A protected tensor prices at
            the higher of the candidate precision and its floor —
            by size only, never by damage.
        imatrix_exclusions: Fnmatch patterns over protected tensor
            names (ADR-0023). Each matched tensor keeps its
            promotion and quantizes without its imatrix row. Size
            and damage predictions do not change — the exclusion
            swaps the fit, not the type.
        format_overhead: Overhead fraction on top of the per-weight
            bit cost. None means the default for the size model:
            `DEFAULT_RESIDUAL_OVERHEAD` when the runtime has an
            effective-bits table, `DEFAULT_FORMAT_OVERHEAD` otherwise.
            The recipe records the resolved value.
        runtime: Target runtime name, or None for no capability
            constraint.

    Returns:
        The recipe, with assignments in sensitivity-map group order,
        the downgrade trace in ``plan.trace``, and the map's
        within-group method token and imatrix path in
        ``within_group`` and ``imatrix`` — the validation pass
        matches its frame against them (ADR-0019, ADR-0020).

    Raises:
        ValueError: If ``format_overhead`` is negative, NaN, or
            infinite.
        RuntimeCapabilityError: If the runtime is unknown or serves
            none of the scanned precisions.
        PinError: If a pin is malformed with respect to the candidate
            set.
        ProtectionError: If a protection floor is unservable, a
            pattern matches no tensor or a single-tensor group, the
            map lacks per-tensor sizes (ADR-0022), an imatrix
            exclusion misses the protected set (ADR-0023), or every
            pair an exclusion matches drops as a per-tensor no-op
            (issue #59).
        InfeasibleBudgetError: If even minimum precision (pins and
            protections respected) exceeds the budget — the message
            counts the protected tensors that raised the floor.

    Examples:
        Pin the first layer high and solve:

        ```python
        from vramfit.domain.solver import solve

        recipe = solve(
            map_,
            weight_budget_bytes=20 * 2**30,
            vram_budget_bytes=24 * 2**30,
            kv_headroom_bytes=4 * 2**30,
            pins={"model.layers.0.*": 8},
        )
        ```
    """
    table = effective_bits(runtime)
    if format_overhead is None:
        format_overhead = (
            DEFAULT_RESIDUAL_OVERHEAD if table is not None else DEFAULT_FORMAT_OVERHEAD
        )
    if not (math.isfinite(format_overhead) and format_overhead >= 0):
        raise ValueError(
            f"format_overhead must be finite and non-negative, got {format_overhead}"
        )
    pins = dict(pins or {})
    protections = dict(protections or {})
    candidates = sensitivity_map.scan.precisions
    if runtime is not None:
        candidates = servable_precisions(candidates, runtime)
    dropped = tuple(p for p in sensitivity_map.scan.precisions if p not in candidates)
    pinned = _expand_pins(pins, sensitivity_map, candidates)
    floors = expand_protections(protections, sensitivity_map, runtime)
    excluded = expand_exclusions(imatrix_exclusions, floors, sensitivity_map)

    def price(bytes_fp16: int, bits: int) -> int:
        """Shorthand for `group_bytes` with the solver's size model.

        Args:
            bytes_fp16: Size at reference precision.
            bits: Target nominal precision.

        Returns:
            Predicted bytes at the target precision — priced at the
            runtime's effective bits when a table exists.
        """
        spent = table[bits] if table is not None else bits
        return group_bytes(bytes_fp16, spent, format_overhead)

    def size(group: LayerGroup, bits: int) -> int:
        """Price one group, holding its protected tensors at floor.

        Args:
            group: The group to price.
            bits: Candidate precision for the group.

        Returns:
            Predicted bytes, protections included (ADR-0022).
        """
        return protected_group_bytes(group, bits, floors, price)

    state: dict[str, int] = {}
    for group in sensitivity_map.groups:
        state[group.name] = pinned.get(group.name, candidates[0])

    total = sum(size(g, state[g.name]) for g in sensitivity_map.groups)
    floor_total = sum(
        size(g, pinned.get(g.name, candidates[-1])) for g in sensitivity_map.groups
    )
    if floor_total > weight_budget_bytes:
        # Count only the floors that raise the minimum — a floor the
        # minimum state already meets is a per-tensor no-op and holds
        # nothing (issue #59).
        group_of = {t: g.name for g in sensitivity_map.groups for t in g.tensors}
        raised = sum(
            1
            for name, floor in floors.items()
            if floor > pinned.get(group_of[name], candidates[-1])
        )
        raise InfeasibleBudgetError(
            gap_bytes=floor_total - weight_budget_bytes,
            minimum_bytes=floor_total,
            weight_budget_bytes=weight_budget_bytes,
            runtime=runtime,
            dropped_precisions=dropped,
            protected_count=raised,
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
    # Resolved pairs, not raw floors — a pair exists only where the
    # floor exceeds the final assignment (ADR-0022, issue #59). An
    # exclusion whose every pair dropped refuses here: the state it
    # rides on is only known after solving.
    protected_pairs = resolve_protected(sensitivity_map, state, floors, excluded)
    refuse_dead_exclusions(imatrix_exclusions, protected_pairs)
    assignments = tuple(
        Assignment(
            group=g.name,
            bits=state[g.name],
            bytes=size(g, state[g.name]),
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
            protections=protections,
            format_overhead=format_overhead,
            trace=tuple(trace),
            imatrix_exclusions=imatrix_exclusions,
        ),
        assignments=assignments,
        runtime=runtime,
        # The map's method token and imatrix path ride into the
        # recipe (ADR-0019, ADR-0020): the validation pass refuses a
        # frame that does not match them.
        within_group=sensitivity_map.scan.within_group,
        imatrix=sensitivity_map.scan.imatrix,
        protected_tensors=protected_pairs,
    )
