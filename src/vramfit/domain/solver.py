"""Greedy damage-per-byte solver that turns a sensitivity map into a recipe.

Implements ADR-0007, as amended by ADR-0029: the solver prices
every discovered group and not only the groups its input map carries.
A group the checkpoint holds and the map omits is uncovered. It
prices at reference precision, and the recipe assigns it there —
`pack` runs the quantizer at the recipe's precision floor, so a
group the recipe leaves unnamed would reach the artifact at that
floor rather than at the reference bytes the plan reserved. An
uncovered group carries no damage curve, so the solver never ranks a
downgrade for it. A target runtime that cannot serve reference
precision refuses the solve, because the recipe's own reader
rejects an assignment the runtime cannot serve. Errors
sit under the `VramfitError` root
(ADR-0011) and carry user-facing messages the CLI prints verbatim.
A target runtime narrows the candidate set through the ADR-0013
capability table before any solving starts, and an infeasible
budget names the precisions that narrowing removed.
Pins resolve through [vramfit.domain.pins][] (the 2026-08-22
ADR-0007 amendment): a pin may name any runtime-servable width and
land on any checkpoint-discovered group, and an unmeasured width
records 0.0 damage. The infeasible message counts only the groups
held at reference, because a scan recovers nothing from a pinned
group. A group of an unquantizable class gets its own clause there,
because a scan skips it (#204) and it reserves the convert dtype's
bytes (#409).
The algorithm: start every group at the highest candidate precision
(or its pin), then repeatedly apply the downgrade with the best
damage-per-byte-freed ratio until the total fits the weight budget. The
ordered downgrade log is recorded in the recipe as its explanation
trace, and the final downgrade is refined when a milder step also fits.
On expert-stack groups, a downgrade to the cheapest in-budget width
also passes the spread placement rule with its projection tie-break
([vramfit.domain.placement][], the 2026-08-21 ADR-0007 amendment) —
the rule narrows the candidates and the selection key stays unchanged.
Size predictions follow ADR-0014: a runtime with an effective-bits
table prices each precision at its real per-weight cost, and the
overhead fraction shrinks to a residual for what the table cannot
see (unquantized tensors, file metadata). Without a table the
nominal-bits prediction and the 0.05 scalar remain. A routed-expert-
stack group prices through the expert-stack table instead (ADR-0028)
— 2.25 bits at nominal 2, not Q2_K's 2.625 — and so does a
layer-class group whose rows refuse the 256 super-block. A group of
a class the runtime's quantizer refuses holds at the F16 passthrough
whatever the map measured (the 2026-08-20 ADR-0012 amendment). A
pin pattern that resolves to that one group refuses, and a pattern
that resolves to more than one group skips it (ADR-0007, 2026-09-04
amendment, #371). Such a group prices at the dtype the
converter wrote it at, 32 bits per weight on llama.cpp, because the
packed file holds it there and not at `F16`'s 16 (#409). Protections
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
    - [vramfit.domain.solver_errors][]: `PinError` and
      `InfeasibleBudgetError`, both re-exported here.
    - [vramfit.domain.model][]: The `SensitivityMap` input and `Recipe`
      output types.
    - [vramfit.domain.budget][]: Computes the weight budget this solver
      packs to.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Final

from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    SensitivityMap,
    TraceStep,
)
from vramfit.domain.pins import assignment_damage, resolve_pins
from vramfit.domain.placement import refused_cheapest_stack_moves
from vramfit.domain.protection import (
    expand_exclusions,
    expand_protections,
    protected_group_bytes,
    refuse_dead_exclusions,
    resolve_protected,
)
from vramfit.domain.runtime import (
    convert_dtype_bits,
    effective_bits,
    expert_stack_effective_bits,
    rows_refuse_super_block,
    servable_precisions,
    unquantizable_class,
)
from vramfit.domain.scan import is_expert_stack
from vramfit.domain.sizes import REFERENCE_BITS, held_assignments
from vramfit.domain.solver_errors import (
    InfeasibleBudgetError as InfeasibleBudgetError,  # noqa: PLC0414 - re-export: the solver's errors read from this module
)
from vramfit.domain.solver_errors import (
    PinError as PinError,  # noqa: PLC0414 - re-export: the solver's errors read from this module
)

SOLVER_NAME: Final[str] = "greedy-damage-per-byte"
DEFAULT_FORMAT_OVERHEAD: Final[float] = 0.05
DEFAULT_RESIDUAL_OVERHEAD: Final[float] = 0.005


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
    its ratio worsens and the ranking shifts accordingly. A move that
    takes an expert-stack group to the cheapest in-budget width also
    passes the spread placement rule
    ([vramfit.domain.placement][], the 2026-08-21 ADR-0007
    amendment) — the rule narrows the candidates and the key stays
    unchanged.

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
    cheapest = candidates[-1]
    refused = refused_cheapest_stack_moves(
        sensitivity_map.groups, pinned, state, cheapest, size
    )
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
            if target == cheapest and group.name in refused:
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


def _predictor(
    runtime: str | None, format_overhead: float
) -> Callable[[str], Callable[[int, int], int]]:
    """Build the per-group size predictor for one solve.

    An expert-stack group prices through the ADR-0028 table where it
    has a row (8, 6, 5, 4, 2), and so does a layer-class group whose rows
    refuse the 256 super-block (the 2026-08-20 amendment). Nominal 3
    keeps the dense entry: the plan-time refusal is an ADR-0028 open
    question, so pack refuses first and the plan still prices the
    assignment. At the passthrough a group prices at its table's 16
    row, `F16`'s 16.0 bits (ADR-0029 decision 4). A group of a class
    the quantizer refuses prices at the convert dtype instead, which
    is what the packed file holds (#409).

    Args:
        runtime: Target runtime name, or None to price at nominal
            bits.
        format_overhead: Overhead fraction on top of the per-weight
            bit cost.

    Returns:
        A builder from group name to a ``(bytes_fp16, bits) -> bytes``
        predictor.
    """
    table = effective_bits(runtime)
    stack_table = expert_stack_effective_bits(runtime)
    merged = (
        {**table, **stack_table}
        if table is not None and stack_table is not None
        else None
    )

    def price_for(name: str) -> Callable[[int, int], int]:
        """Bind one group's tables into its predictor.

        Args:
            name: The group's name, which selects its tables.

        Returns:
            The group's predictor, carrying the overhead setting.
        """
        stacked = is_expert_stack(name) or rows_refuse_super_block(name)
        spent_table = merged if stacked and merged is not None else table

        def price(bytes_fp16: int, bits: int) -> int:
            """Predict bytes at one precision under the bound table.

            Args:
                bytes_fp16: Size at reference precision.
                bits: Target nominal precision.

            Returns:
                Predicted bytes, rounded up.
            """
            converted = (
                convert_dtype_bits(name, runtime) if bits == REFERENCE_BITS else None
            )
            if converted is not None:
                spent: float = converted
            elif spent_table is not None:
                spent = spent_table[bits]
            else:
                spent = bits
            return group_bytes(bytes_fp16, spent, format_overhead)

        return price

    return price_for


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
    discovered_bytes: Mapping[str, int] | None = None,
) -> Recipe:
    """Assign a precision to every group so the total fits the budget.

    When a target runtime is given, the candidate set first filters
    through the ADR-0013 capability table — a precision the runtime
    cannot serve is never assigned, and the recipe records the
    runtime. An infeasible budget then names the precisions the
    runtime removed, so the reported floor is explainable. A runtime
    with an effective-bits table prices every candidate at its real
    per-weight cost (ADR-0014) — Q4_K spends 4.5 bits, not 4. A
    routed-expert-stack group prices through the expert-stack table
    instead (ADR-0028): 2.25 bits at nominal 2. A layer-class group
    whose rows refuse the 256 super-block prices through the same
    table, and a group of a class the runtime's quantizer refuses
    holds at the F16 passthrough whatever the map measured — a pin
    pattern that resolves to that one group refuses, a multi-group
    pattern skips it (ADR-0007, 2026-09-04 amendment, #371), and
    the hold records 0.0 damage unless the map scanned reference
    precision (the 2026-08-20 ADR-0012 amendment).
    The hold prices at the convert dtype, 32 bits per weight, because
    the packed file stores the class there (#409). Every
    group starts at the highest candidate precision (or its pin).
    While the total exceeds the budget, the solver applies the downgrade
    with the minimum ``(damage_delta / bytes_freed, group name, smallest
    step)`` key, considering all lower candidate precisions of every
    unpinned group. Moves that free no bytes (possible on tiny groups
    where sizes round to the same value) are never considered. A move
    that takes an expert-stack group to the cheapest in-budget width
    also passes the spread placement rule with its projection
    tie-break ([vramfit.domain.placement][], the 2026-08-21 ADR-0007
    amendment) — dense groups keep the plain damage-per-byte order.
    The selection is a total order and the rule is a deterministic
    function of the allocation state, so the result is deterministic
    and independent of group input order. After the loop, the final downgrade
    is refined: if a milder step of the same group also fits with less
    damage, it replaces the overshooting one.

    Args:
        sensitivity_map: Damage curves for every group.
        weight_budget_bytes: Hard ceiling for the predicted weight total.
        vram_budget_bytes: Total VRAM ceiling, recorded for provenance.
        kv_headroom_bytes: Reserved KV/runtime bytes, recorded for
            provenance.
        pins: Ordered glob-pattern pins forcing precisions. Later
            patterns override earlier ones. A pin may name any width the
            target runtime serves, and it may land on any
            checkpoint-discovered group (the 2026-08-22 ADR-0007
            amendment). At a width the map never measured the
            assignment records 0.0 damage.
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
        discovered_bytes: Bytes at reference precision per group the
            checkpoint holds (ADR-0029), from
            `vramfit.domain.sizes.discovered_group_bytes`. A group
            here that the map does not carry is uncovered: it prices
            at reference precision and the recipe assigns it there.
            None means the map defines the model, which is the
            behavior ADR-0029 replaced. An uncovered expert-stack
            group prices through the ADR-0028 table, like a measured
            one.

    Returns:
        The recipe, with assignments in sensitivity-map group order
        followed by the uncovered groups in name order, the
        downgrade trace in ``plan.trace``, and the map's
        within-group method token and imatrix path in
        ``within_group`` and ``imatrix`` — the validation pass
        matches its frame against them (ADR-0019, ADR-0020).

    Raises:
        ValueError: If ``format_overhead`` is negative, NaN, or
            infinite.
        RuntimeCapabilityError: If the runtime is unknown, serves
            none of the scanned precisions, or cannot serve reference
            precision while ``discovered_bytes`` leaves a group
            uncovered.
        PinError: If a pin uses a precision neither scanned nor
            runtime-servable, matches no group, or lands on an
            unquantizable-class group.
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
    pinned, uncovered_pins, user_pinned = resolve_pins(
        pins, sensitivity_map, candidates, runtime, discovered_bytes
    )
    floors = expand_protections(protections, sensitivity_map, runtime)
    excluded = expand_exclusions(imatrix_exclusions, floors, sensitivity_map)

    price_for = _predictor(runtime, format_overhead)

    def size(group: LayerGroup, bits: int) -> int:
        """Price one group, holding its protected tensors at floor.

        Args:
            group: The group to price.
            bits: Candidate precision for the group.

        Returns:
            Predicted bytes, protections included (ADR-0022).
        """
        return protected_group_bytes(group, bits, floors, price_for(group.name))

    # Every group the checkpoint holds and the map does not (ADR-0029
    # decision 3). Each holds at reference precision: no measurement
    # ranks a downgrade for it, so it is a constant in the budget and
    # never a move. The recipe still assigns it, because `pack` runs
    # `--pure` at the recipe's floor and would otherwise quantize the
    # group the plan just reserved reference bytes for.
    held = held_assignments(
        discovered_bytes, sensitivity_map, runtime, price_for, pins=uncovered_pins
    )
    held_total = sum(a.bytes for a in held)

    state: dict[str, int] = {}
    for group in sensitivity_map.groups:
        state[group.name] = pinned.get(group.name, candidates[0])

    total = held_total + sum(size(g, state[g.name]) for g in sensitivity_map.groups)
    floor_total = held_total + sum(
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
        # The message's held clause reads "at reference precision.
        # Scan them to spend it", so it counts only the unpinned
        # holds of a class a scan can measure — a pinned uncovered
        # group sits at its pin, and a scan skips an unquantizable
        # class (#204). The second set gets its own clause.
        unquantizable = [a for a in held if unquantizable_class(a.group) is not None]
        refused = {a.group for a in unquantizable}
        scannable = [
            a for a in held if a.bits == REFERENCE_BITS and a.group not in refused
        ]
        raise InfeasibleBudgetError(
            gap_bytes=floor_total - weight_budget_bytes,
            minimum_bytes=floor_total,
            weight_budget_bytes=weight_budget_bytes,
            runtime=runtime,
            dropped_precisions=dropped,
            protected_count=raised,
            held_count=len(scannable),
            held_bytes=sum(a.bytes for a in scannable),
            unquantizable_count=len(unquantizable),
            unquantizable_bytes=sum(a.bytes for a in unquantizable),
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
    assignments = (
        tuple(
            Assignment(
                group=g.name,
                bits=state[g.name],
                bytes=size(g, state[g.name]),
                damage=assignment_damage(g, state[g.name], user_pinned),
            )
            for g in sensitivity_map.groups
        )
        + held
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
