from __future__ import annotations

import random
from typing import Any

import pytest

from tests.unit.conftest import make_map
from vramfit.adapters.outbound.sensitivity_map_json import map_from_dict
from vramfit.domain.budget import format_size
from vramfit.domain.model import SensitivityMap
from vramfit.domain.runtime import (
    RuntimeCapabilityError,
    expert_stack_effective_bits,
)
from vramfit.domain.solver import (
    DEFAULT_FORMAT_OVERHEAD,
    DEFAULT_RESIDUAL_OVERHEAD,
    SOLVER_NAME,
    InfeasibleBudgetError,
    PinError,
    group_bytes,
    solve,
)

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

    def test_recipe_records_the_maps_within_group_method(self) -> None:
        # The pass that validates this recipe must match the map's
        # frame — the token rides along (ADR-0019).
        raw = make_map([("g0", 1000, CONVEX_CURVE)])
        raw["scan"]["within_group"] = "kquant-imx"
        raw["scan"]["imatrix"] = "/runs/model.imatrix.gguf"
        map_ = load(raw)

        recipe = solve_simple(map_, budget=10_000)

        assert recipe.within_group == "kquant-imx"
        assert recipe.imatrix == "/runs/model.imatrix.gguf"

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
        with pytest.raises(InfeasibleBudgetError) as excinfo:
            solve_simple(map_, budget=150, runtime="vllm")

        # The message must name the removed precisions — the reported
        # floor is higher than the scan alone allows.
        assert excinfo.value.runtime == "vllm"
        assert excinfo.value.dropped_precisions == (3, 2)
        assert 'runtime "vllm" cannot serve' in str(excinfo.value)
        assert "[3, 2]" in str(excinfo.value)

    def test_runtime_dropping_nothing_keeps_the_plain_message(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))
        # The llama.cpp floor: Q2_K's 2.625 effective bits plus the
        # auto-resolved 0.005 residual.
        floor = group_bytes(1600, 2.625, 0.005)

        # llama.cpp serves the whole scanned set — the message must
        # not blame the runtime for the floor.
        with pytest.raises(InfeasibleBudgetError) as excinfo:
            solve_simple(map_, budget=floor - 10, runtime="llama.cpp")

        assert "cannot serve" not in str(excinfo.value)
        assert excinfo.value.minimum_bytes == floor

    def test_infeasible_floor_is_priced_at_effective_bits(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        # 250 fits the nominal 2-bit floor (200) but not Q2_K's real
        # 263 — the precheck must price the floor at the table, or
        # the loop would run and break the solver invariant.
        with pytest.raises(InfeasibleBudgetError) as excinfo:
            solve_simple(map_, budget=250, runtime="llama.cpp", format_overhead=0.0)

        assert excinfo.value.minimum_bytes == 263
        assert excinfo.value.gap_bytes == 13

    def test_pinned_group_is_priced_at_effective_bits(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE), ("g1", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_,
            budget=2000,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={"g0": 8},
        )

        by_group = {a.group: a.bytes for a in recipe.assignments}
        assert by_group["g0"] == 850

    def test_pinned_floor_is_priced_at_effective_bits(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE), ("g1", 1600, CONVEX_CURVE)]))

        # 1100 fits the nominal pinned floor (800 + 200) but not the
        # effective one (850 + 263).
        with pytest.raises(InfeasibleBudgetError):
            solve_simple(
                map_,
                budget=1100,
                runtime="llama.cpp",
                format_overhead=0.0,
                pins={"g0": 8},
            )

        # The same budget is feasible without the runtime — the
        # regression signal that the pinned floor uses the table.
        unconstrained = solve_simple(
            map_, budget=1100, format_overhead=0.0, pins={"g0": 8}
        )
        assert unconstrained.plan.predicted_total_bytes <= 1100

    def test_refined_final_step_is_priced_at_effective_bits(self) -> None:
        # Best ratio at budget-crossing time is the 8->2 jump, but
        # Q4_K's 450 bytes also fit; the refinement must price the
        # milder step at the table.
        curve = {8: 0.0, 4: 0.5, 3: 1.0, 2: 0.6}
        map_ = load(make_map([("g0", 1600, curve)]))

        recipe = solve_simple(
            map_, budget=460, runtime="llama.cpp", format_overhead=0.0
        )

        assert recipe.assignments[0].bits == 4
        assert recipe.assignments[0].bytes == 450
        assert recipe.plan.trace[-1].to_bits == 4
        assert recipe.plan.trace[-1].bytes_freed == 850 - 450

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

    def test_llama_cpp_runtime_prices_groups_at_effective_bits(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_, budget=10_000, runtime="llama.cpp", format_overhead=0.0
        )

        # Q8_0 spends 8.5 bits per weight, not the nominal 8.
        assert recipe.assignments[0].bytes == 850

    def test_expert_stack_group_prices_at_the_stack_table(self) -> None:
        # An expert-stack group maps through the ADR-0028 type table,
        # so nominal 2 spends Q2_0's 2.25 bits, not Q2_K's 2.625.
        stack = "model.layers.0.mlp.experts.up_proj"
        map_ = load(make_map([(stack, 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_,
            budget=10_000,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={stack: 2},
        )

        assert recipe.assignments[0].bytes == 225

    def test_expert_stack_group_at_nominal_3_keeps_the_dense_entry(self) -> None:
        # The ADR-0028 stack table has no 3-bit row and the plan-time
        # refusal is an open question there — pack refuses first, so
        # the plan still prices the assignment at Q3_K's 3.4375.
        stack = "model.layers.0.mlp.experts.up_proj"
        map_ = load(make_map([(stack, 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_,
            budget=10_000,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={stack: 3},
        )

        assert recipe.assignments[0].bytes == 344

    def test_dense_group_at_nominal_2_keeps_the_q2_k_pricing(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_,
            budget=10_000,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={"model.layers.0": 2},
        )

        assert recipe.assignments[0].bytes == 263

    def test_no_runtime_keeps_nominal_bit_pricing(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000, format_overhead=0.0)

        assert recipe.assignments[0].bytes == 800

    def test_runtime_without_a_table_keeps_nominal_bit_pricing(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000, runtime="vllm", format_overhead=0.0)

        assert recipe.assignments[0].bytes == 800

    def test_effective_bits_force_a_downgrade_nominal_bits_would_skip(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        # 820 fits the nominal 8-bit size (800) but not Q8_0's real
        # 850 — the effective-bits solver must downgrade, the
        # unconstrained one must not.
        constrained = solve_simple(
            map_, budget=820, runtime="llama.cpp", format_overhead=0.0
        )
        unconstrained = solve_simple(map_, budget=820, format_overhead=0.0)

        assert constrained.assignments[0].bits == 4
        assert unconstrained.assignments[0].bits == 8

    def test_auto_overhead_resolves_to_residual_with_a_table(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000, runtime="llama.cpp")

        assert recipe.plan.format_overhead == DEFAULT_RESIDUAL_OVERHEAD

    @pytest.mark.parametrize("runtime", [None, "vllm"], ids=["none", "vllm"])
    def test_auto_overhead_resolves_to_scalar_without_a_table(self, runtime) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(map_, budget=10_000, runtime=runtime)

        assert recipe.plan.format_overhead == DEFAULT_FORMAT_OVERHEAD

    def test_explicit_overhead_overrides_the_auto_default(self) -> None:
        map_ = load(make_map([("g0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_, budget=10_000, runtime="llama.cpp", format_overhead=0.02
        )

        assert recipe.plan.format_overhead == 0.02
        # ceil(1600 * 8.5 / 16 * 1.02): the residual rides on top of
        # the effective bits.
        assert recipe.assignments[0].bytes == 867

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


def make_protected_map(
    layers: int = 2,
    curve: dict[int, float] | None = None,
    v_bytes: int = 200,
    rest_bytes: int = 800,
) -> SensitivityMap:
    from vramfit.domain.model import LayerGroup, ScanMeta

    curve = curve or CONVEX_CURVE

    def layer(index: int) -> LayerGroup:
        v_proj = f"model.layers.{index}.self_attn.v_proj.weight"
        down = f"model.layers.{index}.mlp.down_proj.weight"
        return LayerGroup(
            name=f"model.layers.{index}",
            tensors=(v_proj, down),
            bytes_fp16=v_bytes + rest_bytes,
            sensitivity=curve,
            tensor_bytes={v_proj: v_bytes, down: rest_bytes},
        )

    return SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="wikitext",
            calibration_tokens=1024,
            precisions=tuple(sorted(curve, reverse=True)),
            group_by="layer",
            started_at="2026-08-08T00:00:00Z",
        ),
        groups=tuple(layer(i) for i in range(layers)),
    )


@pytest.mark.unit
class TestSolveWithProtections:
    def test_recipe_records_patterns_and_resolved_pairs(self) -> None:
        map_ = make_protected_map()

        # 600 forces both groups to 4-bit, below the floor of 5.
        recipe = solve_simple(
            map_,
            budget=600,
            protections={"*.self_attn.v_proj.weight": 5},
            format_overhead=0.0,
        )

        assert dict(recipe.plan.protections) == {"*.self_attn.v_proj.weight": 5}
        assert [p.tensor for p in recipe.protected_tensors] == [
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
        ]
        assert all(p.bits == 5 for p in recipe.protected_tensors)

    def test_noop_floor_resolves_to_no_pairs(self) -> None:
        # Budget fits at 8-bit, above the floor of 5 — every pair
        # would pack identically to the unprotected reference and
        # falsely fail the reconstruction gate (issue #59). The
        # verbatim rule stays as provenance.
        map_ = make_protected_map()

        recipe = solve_simple(
            map_,
            budget=10_000,
            protections={"*.self_attn.v_proj.weight": 5},
            format_overhead=0.0,
        )

        assert dict(recipe.plan.protections) == {"*.self_attn.v_proj.weight": 5}
        assert recipe.protected_tensors == ()

    def test_imatrix_exclusions_reach_the_recipe(self) -> None:
        map_ = make_protected_map()

        # 600 forces both groups to 4-bit, so the floor of 5 holds.
        recipe = solve_simple(
            map_,
            budget=600,
            protections={"*.self_attn.v_proj.weight": 5},
            imatrix_exclusions=("model.layers.0.*",),
            format_overhead=0.0,
        )

        assert recipe.plan.imatrix_exclusions == ("model.layers.0.*",)
        marks = {p.tensor: p.exclude_imatrix for p in recipe.protected_tensors}
        assert marks["model.layers.0.self_attn.v_proj.weight"] is True
        assert marks["model.layers.1.self_attn.v_proj.weight"] is False

    def test_exclusion_riding_only_dropped_pairs_rejected(self) -> None:
        # Budget fits at 8-bit, so every floor-5 pair drops — nothing
        # survives for the exclusion to ride (issue #59).
        from vramfit.domain.protection import ProtectionError

        map_ = make_protected_map()

        with pytest.raises(ProtectionError, match="per-tensor no-op"):
            solve_simple(
                map_,
                budget=10_000,
                protections={"*.self_attn.v_proj.weight": 5},
                imatrix_exclusions=("model.layers.0.*",),
                format_overhead=0.0,
            )

    def test_exclusion_does_not_change_size_or_damage(self) -> None:
        map_ = make_protected_map(layers=1)
        kwargs = {
            "budget": 300,
            "protections": {"*.self_attn.v_proj.weight": 5},
            "format_overhead": 0.0,
        }

        plain = solve_simple(map_, **kwargs)
        excluded = solve_simple(
            map_, imatrix_exclusions=("model.layers.0.*",), **kwargs
        )

        assert excluded.assignments == plain.assignments
        assert excluded.plan.predicted_total_bytes == plain.plan.predicted_total_bytes
        assert excluded.plan.predicted_damage == plain.plan.predicted_damage

    def test_protected_tensor_holds_floor_through_a_downgrade(self) -> None:
        map_ = make_protected_map(layers=1)

        # 1000 bytes at 8-bit = 500; the budget forces a downgrade.
        recipe = solve_simple(
            map_,
            budget=300,
            protections={"*.self_attn.v_proj.weight": 5},
            format_overhead=0.0,
        )

        assignment = recipe.assignments[0]
        assert assignment.bits == 4
        assert recipe.protected_tensors[0].bits == 5
        # 800 bytes at 4-bit (200) + 200 bytes at 5-bit floor (63).
        assert assignment.bytes == 200 + 63

    def test_protection_makes_group_size_larger_than_unprotected(self) -> None:
        map_ = make_protected_map(layers=1)

        unprotected = solve_simple(map_, budget=300, format_overhead=0.0)
        protected = solve_simple(
            map_,
            budget=300,
            protections={"*.self_attn.v_proj.weight": 5},
            format_overhead=0.0,
        )

        assert (
            protected.plan.predicted_total_bytes
            > unprotected.plan.predicted_total_bytes
        )

    def test_protected_group_frees_fewer_bytes_and_ranks_second(self) -> None:
        # Two identical groups; the protection makes layer 0's
        # downgrade free fewer bytes, so layer 1 goes first.
        map_ = make_protected_map(layers=2)

        recipe = solve_simple(
            map_,
            budget=700,
            protections={"model.layers.0.self_attn.v_proj.weight": 8},
            format_overhead=0.0,
        )

        assert recipe.plan.trace[0].group == "model.layers.1"

    def test_floor_counts_against_feasibility(self) -> None:
        map_ = make_protected_map(layers=1)

        # Unprotected floor: 1000 at 2-bit = 125 bytes -> feasible.
        # With the v_proj held at 8-bit: 100 + 100 = 200 -> infeasible.
        with pytest.raises(InfeasibleBudgetError):
            solve_simple(
                map_,
                budget=150,
                protections={"*.self_attn.v_proj.weight": 8},
                format_overhead=0.0,
            )

    def test_effective_bits_price_the_floor_under_a_runtime_table(self) -> None:
        map_ = make_protected_map(layers=1)

        recipe = solve_simple(
            map_,
            budget=10_000,
            protections={"*.self_attn.v_proj.weight": 5},
            format_overhead=0.0,
            runtime="llama.cpp",
        )

        # 8-bit assignment prices at 8.5 effective bits for both
        # pieces: ceil(800 * 8.5/16) + ceil(200 * 8.5/16).
        assert recipe.assignments[0].bytes == 425 + 107

    def test_unservable_floor_rejected(self) -> None:
        from vramfit.domain.protection import ProtectionError

        with pytest.raises(ProtectionError, match="cannot serve"):
            solve_simple(
                make_protected_map(),
                budget=10_000,
                protections={"*.self_attn.v_proj.weight": 7},
                runtime="llama.cpp",
            )

    def test_no_protections_solves_identically_to_before(self) -> None:
        map_ = make_protected_map()

        bare = solve_simple(map_, budget=1_200, format_overhead=0.0)
        explicit = solve_simple(map_, budget=1_200, protections={}, format_overhead=0.0)

        assert bare == explicit
        assert bare.protected_tensors == ()

    def test_infeasible_budget_names_the_protection_count(self) -> None:
        map_ = make_protected_map(layers=1)

        with pytest.raises(InfeasibleBudgetError, match="hold 1 tensors"):
            solve_simple(
                map_,
                budget=150,
                protections={"*.self_attn.v_proj.weight": 8},
                format_overhead=0.0,
            )


@pytest.mark.unit
class TestUncoveredGroups:
    """ADR-0029 decision 3: price every discovered group, assign it too."""

    def test_a_group_outside_the_map_reaches_the_recipe(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_, 100_000, discovered_bytes={"model.layers.0": 1600, "lm_head": 800}
        )

        assert [a.group for a in recipe.assignments] == ["model.layers.0", "lm_head"]

    def test_an_uncovered_group_holds_at_reference_precision(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_, 100_000, discovered_bytes={"model.layers.0": 1600, "lm_head": 800}
        )

        held = next(a for a in recipe.assignments if a.group == "lm_head")
        assert held.bits == 16

    def test_an_uncovered_group_carries_no_damage(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_, 100_000, discovered_bytes={"model.layers.0": 1600, "lm_head": 800}
        )

        held = next(a for a in recipe.assignments if a.group == "lm_head")
        assert held.damage == 0.0

    def test_uncovered_bytes_enter_the_predicted_total(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        without = solve_simple(map_, 100_000)
        with_source = solve_simple(
            map_, 100_000, discovered_bytes={"model.layers.0": 1600, "lm_head": 1600}
        )

        held = next(a for a in with_source.assignments if a.group == "lm_head")
        assert (
            with_source.plan.predicted_total_bytes
            == without.plan.predicted_total_bytes + held.bytes
        )

    def test_uncovered_bytes_force_downgrades_of_the_measured_groups(self) -> None:
        map_ = load(
            make_map(
                [
                    ("model.layers.0", 16_000, CONVEX_CURVE),
                    ("model.layers.1", 16_000, CONVEX_CURVE),
                ]
            )
        )

        without = solve_simple(map_, 18_000)
        with_source = solve_simple(
            map_,
            18_000,
            discovered_bytes={
                "model.layers.0": 16_000,
                "model.layers.1": 16_000,
                "model.layers.2": 8_000,
            },
        )

        assert len(with_source.plan.trace) > len(without.plan.trace)

    def test_uncovered_bytes_can_make_the_budget_infeasible(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        with pytest.raises(InfeasibleBudgetError):
            solve_simple(
                map_, 2000, discovered_bytes={"model.layers.0": 1600, "big": 100_000}
            )

    def test_an_uncovered_group_is_never_downgraded(self) -> None:
        map_ = load(make_map([("model.layers.0", 16_000, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_,
            12_000,
            discovered_bytes={"model.layers.0": 16_000, "held": 4_000},
        )

        assert [step.group for step in recipe.plan.trace] == ["model.layers.0"]

    def test_a_fully_covered_source_changes_nothing(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        without = solve_simple(map_, 100_000)
        with_source = solve_simple(
            map_, 100_000, discovered_bytes={"model.layers.0": 1}
        )

        assert with_source == without

    def test_no_source_leaves_the_map_defining_the_model(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(map_, 100_000)

        assert [a.group for a in recipe.assignments] == ["model.layers.0"]

    def test_uncovered_groups_land_in_name_order(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_,
            100_000,
            discovered_bytes={"model.layers.0": 1600, "zeta": 8, "alpha": 8},
        )

        assert [a.group for a in recipe.assignments[1:]] == ["alpha", "zeta"]

    def test_an_uncovered_expert_stack_prices_at_the_passthrough(self) -> None:
        # `F16` has no super-block, so a stack row costs the same 16.0
        # bits per weight as a dense one (ADR-0029 decision 4).
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))
        stack = "model.layers.1.mixer.experts.up_proj"

        recipe = solve_simple(
            map_,
            100_000,
            discovered_bytes={"model.layers.0": 1600, stack: 1600},
            runtime="llama.cpp",
        )

        held = next(a for a in recipe.assignments if a.group == stack)
        assert held.bytes == group_bytes(1600, 16.0, DEFAULT_RESIDUAL_OVERHEAD)

    def test_a_runtime_without_reference_precision_refuses(self) -> None:
        # vLLM serves 8 and 4. A recipe carrying 16 fails its own
        # reader, so the solver refuses at solve time rather than
        # writing a file every downstream command rejects.
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        with pytest.raises(RuntimeCapabilityError, match="reference precision"):
            solve_simple(
                map_,
                100_000,
                discovered_bytes={"model.layers.0": 1600, "lm_head": 800},
                runtime="vllm",
            )

    def test_a_runtime_without_reference_precision_still_plans_a_full_map(self) -> None:
        # The refusal is about uncovered groups, not about the runtime.
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(
            map_, 100_000, discovered_bytes={"model.layers.0": 1600}, runtime="vllm"
        )

        assert [a.bits for a in recipe.assignments] == [8]

    def test_an_uncovered_expert_stack_prices_through_the_stack_table(self) -> None:
        # The stack table and the dense table both read 16.0 today, so
        # this pins the routing rather than a difference. A stack group
        # priced through the dense table would drift the day either
        # table moves.
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))
        stack = "model.layers.1.mixer.experts.up_proj"

        recipe = solve_simple(
            map_,
            100_000,
            discovered_bytes={"model.layers.0": 1600, stack: 1600},
            runtime="llama.cpp",
        )

        held = next(a for a in recipe.assignments if a.group == stack)
        table = expert_stack_effective_bits("llama.cpp")
        assert table is not None
        assert held.bytes == group_bytes(1600, table[16], DEFAULT_RESIDUAL_OVERHEAD)

    def test_the_infeasible_message_names_the_held_groups(self) -> None:
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))

        with pytest.raises(InfeasibleBudgetError) as caught:
            solve_simple(
                map_, 2000, discovered_bytes={"model.layers.0": 1600, "big": 100_000}
            )

        message = str(caught.value)
        assert "1 groups the map does not measure" in message
        assert caught.value.held_count == 1


@pytest.mark.unit
class TestUnquantizableClasses:
    """The F16 hold for classes llama-quantize refuses (2026-08-20)."""

    def test_a_measured_unquantizable_group_holds_at_the_passthrough(self) -> None:
        # The hold wins over the measurement: llama-quantize refuses
        # the tensor, so any lower width would record a type the
        # artifact cannot carry (ADR-0012, 2026-08-20 amendment).
        map_ = load(
            make_map(
                [
                    ("model.layers.0.mixer.gate", 1600, CONVEX_CURVE),
                    ("model.layers.0", 8000, CONVEX_CURVE),
                ]
            )
        )

        # The budget forces downgrades. Only the plain layer group may
        # supply them. The hold prices at the convert dtype, 32 bits
        # per weight, because the packed file holds the router there
        # (#409).
        recipe = solve_simple(map_, 6000, runtime="llama.cpp")

        gate = next(
            a for a in recipe.assignments if a.group == "model.layers.0.mixer.gate"
        )
        assert gate.bits == 16
        assert gate.bytes == group_bytes(1600, 32.0, DEFAULT_RESIDUAL_OVERHEAD)
        assert gate.damage == 0.0
        assert all(step.group != gate.group for step in recipe.plan.trace)

    def test_the_hold_needs_a_runtime_with_a_filter_table(self) -> None:
        # The filter list is llama.cpp's. An unconstrained plan keeps
        # the measured candidates.
        map_ = load(make_map([("model.layers.0.mixer.gate", 1600, CONVEX_CURVE)]))

        recipe = solve_simple(map_, 100_000)

        assert recipe.assignments[0].bits == 8

    def test_a_glob_pin_sweeping_a_held_group_refuses_too(self) -> None:
        # Current policy: any pin overlap refuses, loudly. Whether a
        # wildcard that merely sweeps the group should skip it instead
        # is an open question raised on #368's build — this pins the
        # loud behavior until a record answers.
        map_ = load(
            make_map(
                [
                    ("model.layers.0.mixer.gate", 1600, CONVEX_CURVE),
                    ("model.layers.0.mixer.in_proj", 1600, CONVEX_CURVE),
                ]
            )
        )

        with pytest.raises(PinError, match="F16 passthrough"):
            solve_simple(
                map_, 100_000, pins={"model.layers.0.*": 8}, runtime="llama.cpp"
            )

    def test_a_pin_on_an_unquantizable_group_refuses_naming_the_filter(self) -> None:
        map_ = load(make_map([("model.layers.0.mixer.gate", 1600, CONVEX_CURVE)]))

        with pytest.raises(PinError) as caught:
            solve_simple(
                map_,
                100_000,
                pins={"model.layers.0.mixer.gate": 8},
                runtime="llama.cpp",
            )

        message = str(caught.value)
        assert '"model.layers.0.mixer.gate"' in message
        assert "ffn_gate_inp.weight" in message

    def test_a_measured_super_block_refused_class_prices_through_the_stack_table(
        self,
    ) -> None:
        # A Nemotron-H dense class routes through the ADR-0028 table
        # (the 2026-08-20 amendment): nominal 2 spends q2_0's 2.25
        # bits per weight, not q2_k's 2.625.
        map_ = load(
            make_map(
                [("model.layers.0.mixer.in_proj", 160_000, {8: 0.001, 2: 1.0})],
                precisions=(8, 2),
            )
        )

        recipe = solve_simple(map_, 30_000, runtime="llama.cpp")

        assignment = recipe.assignments[0]
        assert assignment.bits == 2
        assert assignment.bytes == group_bytes(160_000, 2.25, DEFAULT_RESIDUAL_OVERHEAD)


@pytest.mark.unit
class TestPassthroughPricing:
    """The F16 passthrough prices at what the packed file holds (#409)."""

    # Publication #2's 46 passthrough groups, from the #409 tables:
    # 23 `mixer.conv1d` at 24,576 weights and 23 `mixer.gate` at
    # 344,064 weights, all packed at F32.
    CONV1D_FP16 = 24_576 * 2
    GATE_FP16 = 344_064 * 2
    PACKED_BYTES = 33_914_880
    RECIPE_BYTES_AT_16 = 16_991_388
    OVERHEAD = 0.002

    @staticmethod
    def _discovered() -> dict[str, int]:
        layers = [n for n in range(52) if n % 2 == 0][:23]
        conv = {f"model.layers.{n}.mixer.conv1d": 49_152 for n in layers}
        gate = {f"model.layers.{n + 1}.mixer.gate": 688_128 for n in layers}
        return {"model.layers.0.mixer.in_proj": 1600, **conv, **gate}

    def _held(self, format_overhead: float) -> int:
        map_ = load(make_map([("model.layers.0.mixer.in_proj", 1600, CONVEX_CURVE)]))
        recipe = solve_simple(
            map_,
            10**9,
            runtime="llama.cpp",
            discovered_bytes=self._discovered(),
            format_overhead=format_overhead,
        )
        passthrough = [a for a in recipe.assignments if a.bits == 16]
        assert len(passthrough) == 46
        return sum(a.bytes for a in passthrough)

    def test_the_46_groups_price_at_their_packed_bytes(self) -> None:
        # At zero overhead the prediction equals the packed file's
        # F32 bytes exactly. The 16.0 row read half of that.
        assert self._held(0.0) == self.PACKED_BYTES

    def test_the_publication_2_shortfall_no_longer_reproduces(self) -> None:
        # The published recipe priced these groups at 16,991,388 B
        # under its 0.002 overhead, 16,923,492 B short of the packed
        # bytes and past the 16,874,535 B margin. The corrected
        # prediction covers the packed bytes.
        shortfall = self.PACKED_BYTES - self.RECIPE_BYTES_AT_16
        assert shortfall == 16_923_492
        assert self._held(self.OVERHEAD) >= self.PACKED_BYTES

    def test_a_quantizable_class_still_spends_16_bits(self) -> None:
        # The `f16` override holds a refused-nothing class at two
        # bytes per weight, so only the unquantizable classes move.
        map_ = load(make_map([("model.layers.0", 1600, CONVEX_CURVE)]))
        recipe = solve_simple(
            map_,
            10**6,
            runtime="llama.cpp",
            discovered_bytes={
                "model.layers.0": 1600,
                "model.layers.1.mixer.in_proj": 1600,
            },
        )
        held = next(a for a in recipe.assignments if a.group.endswith("in_proj"))
        assert held.bytes == group_bytes(1600, 16.0, DEFAULT_RESIDUAL_OVERHEAD)
