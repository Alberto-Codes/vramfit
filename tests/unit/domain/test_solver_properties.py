from __future__ import annotations

import random
from typing import Any

import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from tests.strategies import raw_protected_maps, raw_sensitivity_maps
from tests.unit.conftest import make_map
from vramfit.adapters.outbound.sensitivity_map_json import map_from_dict
from vramfit.domain.runtime import (
    EFFECTIVE_BITS,
    EXPERT_STACK_EFFECTIVE_BITS,
    RUNTIME_CAPABILITIES,
    RuntimeCapabilityError,
)
from vramfit.domain.solver import InfeasibleBudgetError, group_bytes, solve

overheads = st.floats(
    min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False
)


def bounds(raw: dict[str, Any], overhead: float) -> tuple[int, int]:
    map_ = map_from_dict(raw)
    lowest = map_.scan.precisions[-1]
    highest = map_.scan.precisions[0]
    floor = sum(group_bytes(g.bytes_fp16, lowest, overhead) for g in map_.groups)
    ceiling = sum(group_bytes(g.bytes_fp16, highest, overhead) for g in map_.groups)
    return floor, ceiling


def solve_simple(map_, budget: int, overhead: float, **kwargs: Any):
    return solve(
        map_,
        weight_budget_bytes=budget,
        vram_budget_bytes=budget + 1000,
        kv_headroom_bytes=1000,
        format_overhead=overhead,
        **kwargs,
    )


@pytest.mark.unit
class TestSolverProperties:
    @given(raw=raw_sensitivity_maps(), overhead=overheads, data=st.data())
    def test_feasible_budget_always_respected(self, raw, overhead, data) -> None:
        floor, ceiling = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling + 1000))

        recipe = solve_simple(map_from_dict(raw), budget, overhead)

        assert recipe.plan.predicted_total_bytes <= budget
        assert recipe.plan.predicted_total_bytes == sum(
            a.bytes for a in recipe.assignments
        )
        assert len(recipe.assignments) == len(raw["groups"])
        assert all(
            a.bits in map_from_dict(raw).scan.precisions for a in recipe.assignments
        )
        assert recipe.plan.format_overhead == overhead

    @given(raw=raw_sensitivity_maps(), overhead=overheads, data=st.data())
    def test_infeasible_budget_reports_exact_gap(self, raw, overhead, data) -> None:
        floor, _ = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=0, max_value=floor - 1))

        with pytest.raises(InfeasibleBudgetError) as excinfo:
            solve_simple(map_from_dict(raw), budget, overhead)

        assert excinfo.value.minimum_bytes == floor
        assert excinfo.value.gap_bytes == floor - budget

    @given(
        raw=raw_sensitivity_maps(),
        overhead=st.floats(max_value=-0.0001, min_value=-100, allow_nan=False),
    )
    def test_negative_overhead_always_rejected(self, raw, overhead) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            solve_simple(map_from_dict(raw), 10_000, overhead)

    @given(
        raw=raw_sensitivity_maps(),
        runtime=st.sampled_from(sorted(RUNTIME_CAPABILITIES)),
        data=st.data(),
    )
    def test_infeasible_gap_under_a_runtime_is_priced_by_its_table(
        self, raw, runtime, data
    ) -> None:
        map_ = map_from_dict(raw)
        capability = RUNTIME_CAPABILITIES[runtime]
        servable = [p for p in map_.scan.precisions if p in capability]
        if not servable:
            event("no servable precision")
            return
        # The reported floor must be the table-priced floor — a
        # nominal-bits precheck would let budgets between the two
        # floors reach the downgrade loop and break its invariant.
        table = EFFECTIVE_BITS.get(runtime)

        def spent(bits: int) -> float:
            return table[bits] if table is not None else bits

        floor = sum(
            group_bytes(g.bytes_fp16, spent(servable[-1]), 0.0) for g in map_.groups
        )
        budget = data.draw(st.integers(min_value=0, max_value=floor - 1))

        with pytest.raises(InfeasibleBudgetError) as excinfo:
            solve_simple(map_, budget, 0.0, runtime=runtime)

        assert excinfo.value.minimum_bytes == floor
        assert excinfo.value.gap_bytes == floor - budget

    @given(
        raw=raw_sensitivity_maps(),
        runtime=st.sampled_from(sorted(RUNTIME_CAPABILITIES)),
        data=st.data(),
    )
    def test_runtime_never_assigns_an_unservable_precision(
        self, raw, runtime, data
    ) -> None:
        map_ = map_from_dict(raw)
        capability = RUNTIME_CAPABILITIES[runtime]
        servable = [p for p in map_.scan.precisions if p in capability]

        if not servable:
            event("no servable precision")
            with pytest.raises(RuntimeCapabilityError):
                solve_simple(map_, 10**12, 0.0, runtime=runtime)
            return
        # Budget between the *filtered* floor and ceiling, so the
        # downgrade loop actually runs — an unfiltered candidate could
        # only leak under budget pressure. A runtime with an
        # effective-bits table prices each precision at that table
        # (ADR-0014), so the bounds must too.
        table = EFFECTIVE_BITS.get(runtime)

        def spent(bits: int) -> float:
            return table[bits] if table is not None else bits

        floor = sum(
            group_bytes(g.bytes_fp16, spent(servable[-1]), 0.0) for g in map_.groups
        )
        ceiling = sum(
            group_bytes(g.bytes_fp16, spent(servable[0]), 0.0) for g in map_.groups
        )
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        event("pressured" if budget < ceiling else "unpressured")

        recipe = solve_simple(map_, budget, 0.0, runtime=runtime)

        assert all(a.bits in capability for a in recipe.assignments)
        assert recipe.plan.predicted_total_bytes <= budget
        assert recipe.runtime == runtime
        by_name = {g.name: g for g in map_.groups}
        assert all(
            a.bytes == group_bytes(by_name[a.group].bytes_fp16, spent(a.bits), 0.0)
            for a in recipe.assignments
        )

    @given(raw=raw_sensitivity_maps(), overhead=overheads, data=st.data())
    def test_group_order_never_changes_the_recipe(self, raw, overhead, data) -> None:
        floor, ceiling = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        seed = data.draw(st.integers(min_value=0, max_value=2**16))
        shuffled = dict(raw)
        shuffled["groups"] = list(raw["groups"])
        random.Random(seed).shuffle(shuffled["groups"])

        a = solve_simple(map_from_dict(raw), budget, overhead)
        b = solve_simple(map_from_dict(shuffled), budget, overhead)

        assert {x.group: x.bits for x in a.assignments} == {
            x.group: x.bits for x in b.assignments
        }
        assert a.plan.predicted_total_bytes == b.plan.predicted_total_bytes

    @given(raw=raw_sensitivity_maps(), overhead=overheads, data=st.data())
    def test_trace_replay_reproduces_assignments(self, raw, overhead, data) -> None:
        floor, ceiling = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        map_ = map_from_dict(raw)

        recipe = solve_simple(map_, budget, overhead)

        by_name = {g.name: g for g in map_.groups}
        state = {g.name: map_.scan.precisions[0] for g in map_.groups}
        for i, step in enumerate(recipe.plan.trace, start=1):
            g = by_name[step.group]
            assert step.step == i
            assert state[step.group] == step.from_bits
            assert step.bytes_freed == group_bytes(
                g.bytes_fp16, step.from_bits, overhead
            ) - group_bytes(g.bytes_fp16, step.to_bits, overhead)
            assert step.damage_delta == (
                g.sensitivity[step.to_bits] - g.sensitivity[step.from_bits]
            )
            assert step.ratio == step.damage_delta / step.bytes_freed
            state[step.group] = step.to_bits
        assert state == {a.group: a.bits for a in recipe.assignments}

    @given(raw=raw_sensitivity_maps(), overhead=overheads, data=st.data())
    def test_pins_always_honored_under_budget_pressure(
        self, raw, overhead, data
    ) -> None:
        map_ = map_from_dict(raw)
        lowest = map_.scan.precisions[-1]
        pinned_group = data.draw(st.sampled_from([g.name for g in map_.groups]))
        pinned_bits = data.draw(st.sampled_from(list(map_.scan.precisions)))
        # Budget between the pin-respecting floor and the pinned starting
        # total, so downgrades are forced *around* the pinned group.
        floor = sum(
            group_bytes(
                g.bytes_fp16,
                pinned_bits if g.name == pinned_group else lowest,
                overhead,
            )
            for g in map_.groups
        )
        start = sum(
            group_bytes(
                g.bytes_fp16,
                pinned_bits if g.name == pinned_group else map_.scan.precisions[0],
                overhead,
            )
            for g in map_.groups
        )
        budget = data.draw(st.integers(min_value=floor, max_value=max(floor, start)))
        event("pressured" if start > budget else "unpressured")

        recipe = solve_simple(map_, budget, overhead, pins={pinned_group: pinned_bits})

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group[pinned_group] == pinned_bits
        assert all(step.group != pinned_group for step in recipe.plan.trace)
        assert recipe.plan.predicted_total_bytes <= budget


@pytest.mark.unit
class TestProtectionProperties:
    @given(drawn=raw_protected_maps(), overhead=overheads, data=st.data())
    def test_protection_never_shrinks_the_predicted_total(
        self, drawn, overhead, data
    ) -> None:
        raw, floor = drawn
        map_ = map_from_dict(raw)
        _, ceiling = bounds(raw, overhead)
        # Split pricing rounds each piece up, so a protected group can
        # cost one byte more than its one-piece ceiling.
        slack = ceiling + len(map_.groups)
        budget = data.draw(st.integers(min_value=slack, max_value=slack + 1000))
        protections = {"model.layers.*.t0": floor}

        bare = solve_simple(map_, budget, overhead)
        protected = solve_simple(map_, budget, overhead, protections=protections)

        event(f"floor={floor}")
        assert protected.plan.predicted_total_bytes >= bare.plan.predicted_total_bytes

    @given(drawn=raw_protected_maps(), overhead=overheads, data=st.data())
    def test_resolved_pairs_exist_exactly_where_floor_exceeds_assignment(
        self, drawn, overhead, data
    ) -> None:
        # A pair at the group's own assignment would pack identically
        # to the unprotected reference and falsely fail the
        # reconstruction gate's strict inequality (issue #59).
        raw, floor = drawn
        map_ = map_from_dict(raw)
        _, ceiling = bounds(raw, overhead)
        slack = ceiling + len(map_.groups)
        budget = data.draw(st.integers(min_value=slack, max_value=slack + 1000))
        protections = {"model.layers.*.t0": floor}

        recipe = solve_simple(map_, budget, overhead, protections=protections)

        resolved = {p.tensor: p.bits for p in recipe.protected_tensors}
        for assignment in recipe.assignments:
            tensor = f"{assignment.group}.t0"
            if floor > assignment.bits:
                assert resolved[tensor] == floor
            else:
                assert tensor not in resolved


@pytest.mark.unit
class TestExpertStackPricingProperties:
    @given(
        bytes_fp16=st.integers(min_value=1, max_value=10**9),
        bits=st.sampled_from([8, 4, 3, 2]),
        overhead=overheads,
    )
    def test_expert_stack_assignment_always_prices_at_the_stack_table(
        self, bytes_fp16: int, bits: int, overhead: float
    ) -> None:
        # An expert-stack group prices at the ADR-0028 row where one
        # exists, and at the dense ADR-0014 entry otherwise (nominal
        # 3 — the plan-time refusal is an open question there). A
        # dense group beside it never moves off the dense table.
        stack = "model.layers.0.mlp.experts.up_proj"
        dense = "model.layers.1"
        curve = {8: 0.001, 4: 0.01, 3: 0.1, 2: 1.0}
        raw = make_map([(stack, bytes_fp16, curve), (dense, bytes_fp16, curve)])

        recipe = solve_simple(
            map_from_dict(raw),
            budget=10**12,
            overhead=overhead,
            runtime="llama.cpp",
            pins={stack: bits, dense: bits},
        )

        table = {
            **EFFECTIVE_BITS["llama.cpp"],
            **EXPERT_STACK_EFFECTIVE_BITS["llama.cpp"],
        }
        by_group = {a.group: a.bytes for a in recipe.assignments}
        event(f"stack row: {bits in EXPERT_STACK_EFFECTIVE_BITS['llama.cpp']}")
        assert by_group[stack] == group_bytes(bytes_fp16, table[bits], overhead)
        assert by_group[dense] == group_bytes(
            bytes_fp16, EFFECTIVE_BITS["llama.cpp"][bits], overhead
        )
