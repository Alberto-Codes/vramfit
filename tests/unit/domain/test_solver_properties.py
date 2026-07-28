from __future__ import annotations

import random
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantfit.adapters.outbound.sensitivity_map_json import map_from_dict
from quantfit.domain.solver import InfeasibleBudgetError, group_bytes, solve
from tests.strategies import raw_sensitivity_maps

OVERHEAD = 0.05


def bounds(raw: dict[str, Any]) -> tuple[int, int]:
    map_ = map_from_dict(raw)
    lowest = map_.scan.precisions[-1]
    highest = map_.scan.precisions[0]
    floor = sum(group_bytes(g.bytes_fp16, lowest, OVERHEAD) for g in map_.groups)
    ceiling = sum(group_bytes(g.bytes_fp16, highest, OVERHEAD) for g in map_.groups)
    return floor, ceiling


def solve_simple(map_, budget: int, **kwargs: Any):
    return solve(
        map_,
        weight_budget_bytes=budget,
        vram_budget_bytes=budget + 1000,
        kv_headroom_bytes=1000,
        format_overhead=OVERHEAD,
        **kwargs,
    )


@pytest.mark.unit
class TestSolverProperties:
    @given(raw=raw_sensitivity_maps(), data=st.data())
    def test_feasible_budget_always_respected(self, raw, data) -> None:
        floor, ceiling = bounds(raw)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling + 1000))

        recipe = solve_simple(map_from_dict(raw), budget)

        assert recipe.plan.predicted_total_bytes <= budget
        assert len(recipe.assignments) == len(raw["groups"])

    @given(raw=raw_sensitivity_maps(), data=st.data())
    def test_infeasible_budget_reports_exact_gap(self, raw, data) -> None:
        floor, _ = bounds(raw)
        budget = data.draw(st.integers(min_value=0, max_value=floor - 1))

        with pytest.raises(InfeasibleBudgetError) as excinfo:
            solve_simple(map_from_dict(raw), budget)

        assert excinfo.value.minimum_bytes == floor
        assert excinfo.value.gap_bytes == floor - budget

    @given(raw=raw_sensitivity_maps(), data=st.data())
    def test_group_order_never_changes_the_recipe(self, raw, data) -> None:
        floor, ceiling = bounds(raw)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        seed = data.draw(st.integers(min_value=0, max_value=2**16))
        shuffled = dict(raw)
        shuffled["groups"] = list(raw["groups"])
        random.Random(seed).shuffle(shuffled["groups"])

        a = solve_simple(map_from_dict(raw), budget)
        b = solve_simple(map_from_dict(shuffled), budget)

        assert {x.group: x.bits for x in a.assignments} == {
            x.group: x.bits for x in b.assignments
        }
        assert a.plan.predicted_total_bytes == b.plan.predicted_total_bytes

    @given(raw=raw_sensitivity_maps(), data=st.data())
    def test_trace_replay_reproduces_assignments(self, raw, data) -> None:
        floor, ceiling = bounds(raw)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        map_ = map_from_dict(raw)

        recipe = solve_simple(map_, budget)

        state = {g.name: map_.scan.precisions[0] for g in map_.groups}
        for i, step in enumerate(recipe.plan.trace, start=1):
            assert step.step == i
            assert state[step.group] == step.from_bits
            state[step.group] = step.to_bits
        assert state == {a.group: a.bits for a in recipe.assignments}

    @given(raw=raw_sensitivity_maps(), data=st.data())
    def test_pins_always_honored_when_feasible(self, raw, data) -> None:
        map_ = map_from_dict(raw)
        _, ceiling = bounds(raw)
        pinned_group = data.draw(st.sampled_from([g.name for g in map_.groups]))
        pinned_bits = data.draw(st.sampled_from(list(map_.scan.precisions)))

        recipe = solve_simple(map_, ceiling, pins={pinned_group: pinned_bits})

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group[pinned_group] == pinned_bits
        assert all(step.group != pinned_group for step in recipe.plan.trace)
