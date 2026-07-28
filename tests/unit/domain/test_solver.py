from __future__ import annotations

import random
from typing import Any

import pytest

from quantfit.adapters.outbound.sensitivity_map_json import map_from_dict
from quantfit.domain.budget import format_size
from quantfit.domain.model import SensitivityMap
from quantfit.domain.solver import (
    SOLVER_NAME,
    InfeasibleBudgetError,
    PinError,
    group_bytes,
    solve,
)
from tests.unit.conftest import make_map

CONVEX_CURVE = {8: 0.001, 4: 0.010, 3: 0.100, 2: 1.000}


def load(raw: dict[str, Any]) -> SensitivityMap:
    return map_from_dict(raw)


def solve_simple(map_: SensitivityMap, budget: int, **kwargs: Any):
    return solve(
        map_,
        weight_budget_bytes=budget,
        vram_budget_bytes=budget + 1000,
        kv_headroom_bytes=1000,
        **kwargs,
    )


@pytest.mark.unit
class TestGroupBytes:
    def test_scales_with_bits(self) -> None:
        assert group_bytes(1600, 8, 0.0) == 800
        assert group_bytes(1600, 4, 0.0) == 400

    def test_format_overhead_inflates_size(self) -> None:
        assert group_bytes(1600, 4, 0.05) == 420

    def test_rounds_up(self) -> None:
        assert group_bytes(3, 4, 0.0) == 1


@pytest.mark.unit
class TestSolve:
    def test_budget_fits_at_highest_precision_keeps_max_and_empty_trace(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000)

        assert recipe.assignments[0].bits == 8
        assert recipe.plan.trace == ()
        assert recipe.plan.solver == SOLVER_NAME

    def test_downgrades_best_ratio_group_first(self) -> None:
        map_ = load(
            make_map(
                [
                    # tolerant: tiny damage per byte freed
                    ("tolerant", 8000, {8: 0.0, 4: 0.001, 3: 0.002, 2: 0.004}),
                    # fragile: large damage per byte freed
                    ("fragile", 8000, {8: 0.0, 4: 1.0, 3: 2.0, 2: 4.0}),
                ]
            )
        )
        # Budget forces exactly one group from 8 -> 4 (with 0 overhead:
        # both at 8 = 8000; one downgrade to 4 frees 2000).
        recipe = solve_simple(map_, budget=6100, format_overhead=0.0)

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group == {"tolerant": 4, "fragile": 8}
        assert recipe.plan.trace[0].group == "tolerant"

    def test_total_bytes_never_exceeds_budget(self, nemotron_like_map) -> None:
        map_ = load(nemotron_like_map)
        floor = sum(group_bytes(g.bytes_fp16, 2, 0.05) for g in map_.groups)
        ceiling = sum(group_bytes(g.bytes_fp16, 8, 0.05) for g in map_.groups)

        for budget in range(floor, ceiling + 1, 97):
            recipe = solve_simple(map_, budget=budget)
            assert recipe.plan.predicted_total_bytes <= budget

    def test_predicted_damage_sums_measured_values_at_chosen_bits(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000)

        # Even an all-8-bit recipe carries the measured 8-bit damage.
        assert recipe.plan.predicted_damage == pytest.approx(0.001)

    def test_pin_exact_name_forces_precision(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE), ("g1", 1000, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000, pins={"g1": 4})

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group == {"g0": 8, "g1": 4}

    def test_pin_glob_matches_multiple_groups(self) -> None:
        map_ = load(
            make_map(
                [
                    ("model.layers.0.attn", 1000, CONVEX_CURVE),
                    ("model.layers.0.mlp", 1000, CONVEX_CURVE),
                    ("model.embed", 1000, CONVEX_CURVE),
                ]
            )
        )

        recipe = solve_simple(map_, budget=10_000, pins={"model.layers.0.*": 4})

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group["model.layers.0.attn"] == 4
        assert by_group["model.layers.0.mlp"] == 4
        assert by_group["model.embed"] == 8

    def test_pin_no_match_raises_pin_error(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        with pytest.raises(PinError, match="matches no group"):
            solve_simple(map_, budget=10_000, pins={"nope*": 4})

    def test_pin_unscanned_precision_raises_pin_error(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        with pytest.raises(PinError, match="not in the candidate set"):
            solve_simple(map_, budget=10_000, pins={"g0": 6})

    def test_runtime_filters_candidates_and_is_recorded(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        # A budget that forces below 8-bit: with vLLM's {8, 4} the
        # solver must land on 4, never on the scanned 3 or 2.
        recipe = solve_simple(map_, budget=300, runtime="vllm")

        assert recipe.assignments[0].bits == 4
        assert recipe.runtime == "vllm"

    def test_no_runtime_leaves_candidates_unfiltered(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=150)

        assert recipe.assignments[0].bits == 2
        assert recipe.runtime is None

    def test_runtime_excluding_the_floor_can_make_a_budget_infeasible(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        # 150 bytes needs the 2-bit floor, which vLLM cannot serve.
        with pytest.raises(InfeasibleBudgetError):
            solve_simple(map_, budget=150, runtime="vllm")

    def test_pin_outside_runtime_set_raises_pin_error(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        with pytest.raises(PinError, match="not in the candidate set"):
            solve_simple(map_, budget=10_000, runtime="vllm", pins={"g0": 3})

    def test_later_pin_overrides_earlier_for_same_group(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000, pins={"g*": 4, "g0": 8})

        assert recipe.assignments[0].bits == 8
        assert recipe.plan.pins == {"g*": 4, "g0": 8}

    def test_unreachable_budget_raises_with_gap_bytes(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))
        floor = group_bytes(1600, 2, 0.05)

        with pytest.raises(InfeasibleBudgetError) as excinfo:
            solve_simple(map_, budget=floor - 10)

        assert excinfo.value.minimum_bytes == floor
        assert excinfo.value.gap_bytes == 10
        # The CLI prints str(exc) verbatim — the message must carry
        # human-readable sizes, not raw byte counts.
        assert "no recipe fits" in str(excinfo.value)
        assert format_size(floor) in str(excinfo.value)

    @pytest.mark.parametrize(
        "overhead", [float("nan"), float("inf")], ids=["nan", "inf"]
    )
    def test_non_finite_format_overhead_raises_value_error(self, overhead) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        with pytest.raises(ValueError, match="finite"):
            solve_simple(map_, budget=10_000, format_overhead=overhead)

    def test_pins_alone_over_budget_raises_infeasible(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE), ("g1", 1600, CONVEX_CURVE)]))
        # g0 pinned at 8 already exceeds the budget even with g1 at 2.
        budget = group_bytes(1600, 8, 0.05)

        with pytest.raises(InfeasibleBudgetError):
            solve_simple(map_, budget=budget, pins={"g0": 8})

    def test_repeated_runs_produce_identical_recipe(self, nemotron_like_map) -> None:
        map_ = load(nemotron_like_map)

        first = solve_simple(map_, budget=4000)
        second = solve_simple(map_, budget=4000)

        assert first == second

    def test_shuffled_group_order_produces_identical_assignments(
        self, nemotron_like_map
    ) -> None:
        map_ = load(nemotron_like_map)
        shuffled_raw = dict(nemotron_like_map)
        shuffled_raw["groups"] = list(nemotron_like_map["groups"])
        random.Random(7).shuffle(shuffled_raw["groups"])
        shuffled = load(shuffled_raw)

        a = solve_simple(map_, budget=4000)
        b = solve_simple(shuffled, budget=4000)

        assert {x.group: x.bits for x in a.assignments} == {
            x.group: x.bits for x in b.assignments
        }
        assert a.plan.trace == b.plan.trace

    def test_equal_ratios_tie_break_by_group_name(self) -> None:
        curve = {8: 0.0, 4: 0.100, 3: 0.200, 2: 0.400}
        map_ = load(make_map([("b", 1000, curve), ("a", 1000, curve)]))
        # Identical curves and sizes: alphabetical group name breaks the tie.
        recipe = solve_simple(map_, budget=940, format_overhead=0.0)

        assert recipe.plan.trace[0].group == "a"

    def test_nonconvex_curve_takes_multi_step_downgrade(self) -> None:
        # 3-bit is worse than 2-bit here: a direct 8 -> 2 jump has a better
        # ratio than 8 -> 3, so the solver must consider all lower targets.
        curve = {8: 0.0, 4: 0.5, 3: 1.0, 2: 0.6}
        map_ = load(make_map([("g0", 1600, curve)]))

        recipe = solve_simple(map_, budget=250, format_overhead=0.0)

        assert recipe.assignments[0].bits == 2
        assert len(recipe.plan.trace) == 1
        assert recipe.plan.trace[0].from_bits == 8
        assert recipe.plan.trace[0].to_bits == 2

    def test_negative_damage_delta_downgraded_first(self) -> None:
        map_ = load(
            make_map(
                [
                    # "improver": 4-bit measured better than 8-bit.
                    ("improver", 1000, {8: 0.5, 4: 0.4, 3: 0.9, 2: 1.5}),
                    ("normal", 1000, {8: 0.0, 4: 0.2, 3: 0.5, 2: 1.0}),
                ]
            )
        )

        recipe = solve_simple(map_, budget=940, format_overhead=0.0)

        assert recipe.plan.trace[0].group == "improver"
        assert recipe.plan.trace[0].damage_delta < 0

    def test_trace_steps_numbered_and_replayable(self, nemotron_like_map) -> None:
        map_ = load(nemotron_like_map)

        recipe = solve_simple(map_, budget=4000)

        state = {g.name: 8 for g in map_.groups}
        for i, step in enumerate(recipe.plan.trace, start=1):
            assert step.step == i
            assert state[step.group] == step.from_bits
            state[step.group] = step.to_bits
        assert state == {a.group: a.bits for a in recipe.assignments}

    def test_tiny_group_zero_byte_downgrade_skipped(self) -> None:
        # A 1-byte group rounds to 1 byte at every precision, so its
        # downgrades free nothing and must be skipped, not divided by.
        map_ = load(make_map([("tiny", 1, CONVEX_CURVE), ("big", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=300)

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group["tiny"] == 8
        assert all(step.group == "big" for step in recipe.plan.trace)
        assert recipe.plan.predicted_total_bytes <= 300

    def test_pinned_group_never_moves_under_pressure(self) -> None:
        map_ = load(
            make_map(
                [("fragile", 1600, CONVEX_CURVE), ("tolerant", 1600, CONVEX_CURVE)]
            )
        )
        # At 8 bits both groups are 800 (overhead 0); budget forces the
        # unpinned group all the way down while the pin holds.
        recipe = solve_simple(
            map_, budget=1100, format_overhead=0.0, pins={"fragile": 8}
        )

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group["fragile"] == 8
        assert recipe.plan.trace != ()
        assert all(step.group == "tolerant" for step in recipe.plan.trace)

    def test_ratio_prefers_bigger_group_at_equal_damage(self) -> None:
        # Identical curves, very different sizes: the big group frees far
        # more bytes for the same damage, so damage-per-BYTE must pick it.
        # The small group sorts first alphabetically, so a wrong
        # denominator (for example per-bit) would tie and pick "a".
        map_ = load(make_map([("a", 100, CONVEX_CURVE), ("z", 10_000, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=4000, format_overhead=0.0)

        assert recipe.plan.trace[0].group == "z"

    def test_overshooting_final_step_is_refined_to_milder_downgrade(self) -> None:
        # Best ratio at budget-crossing time is the 8->2 jump, but 4-bit
        # also fits with less damage; the refinement pass must take it.
        curve = {8: 0.0, 4: 0.5, 3: 1.0, 2: 0.6}
        map_ = load(make_map([("g0", 1600, curve)]))

        recipe = solve_simple(map_, budget=450, format_overhead=0.0)

        assert recipe.assignments[0].bits == 4
        assert recipe.assignments[0].damage == 0.5
        assert recipe.plan.trace[-1].to_bits == 4

    def test_recipe_carries_provenance(self) -> None:
        map_ = load(make_map([("g0", 1000, CONVEX_CURVE)]))

        recipe = solve(
            map_,
            weight_budget_bytes=5000,
            vram_budget_bytes=6000,
            kv_headroom_bytes=1000,
            format_overhead=0.07,
        )

        assert recipe.model_id == "test/model"
        assert recipe.plan.vram_budget_bytes == 6000
        assert recipe.plan.kv_headroom_bytes == 1000
        assert recipe.plan.weight_budget_bytes == 5000
        assert recipe.plan.format_overhead == 0.07
