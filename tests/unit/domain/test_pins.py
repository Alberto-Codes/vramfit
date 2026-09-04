"""Tests for the widened pin surface (the 2026-08-22 ADR-0007 amendment)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.conftest import make_map
from vramfit.adapters.outbound.sensitivity_map_json import map_from_dict
from vramfit.domain.model import SensitivityMap
from vramfit.domain.runtime import RuntimeCapabilityError
from vramfit.domain.solver import PinError, solve


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
class TestWidenedPinSurface:
    """The 2026-08-22 ADR-0007 amendment (#301).

    The pin surface widens to any runtime-servable width and any
    discovered group.
    """

    def test_pin_runtime_servable_width_outside_map_candidates_is_accepted(
        self,
    ) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        recipe = solve_simple(map_, budget=10_000, runtime="llama.cpp", pins={"g0": 8})

        assert recipe.assignments[0].bits == 8

    def test_pin_at_unmeasured_width_records_zero_damage(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        recipe = solve_simple(map_, budget=10_000, runtime="llama.cpp", pins={"g0": 8})

        assert recipe.assignments[0].damage == 0.0
        assert recipe.plan.predicted_damage == 0.0

    def test_pin_at_measured_width_keeps_the_measured_damage(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        recipe = solve_simple(map_, budget=10_000, runtime="llama.cpp", pins={"g0": 4})

        assert recipe.assignments[0].damage == pytest.approx(0.01)

    def test_pin_width_no_runtime_serves_names_the_runtime(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        with pytest.raises(PinError, match="does not serve it"):
            solve_simple(map_, budget=10_000, runtime="llama.cpp", pins={"g0": 7})

    def test_pin_without_runtime_keeps_the_candidate_bound(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        with pytest.raises(PinError, match="not in the candidate set"):
            solve_simple(map_, budget=10_000, pins={"g0": 8})

    def test_pin_lands_on_uncovered_group_and_prices_at_pinned_width(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        recipe = solve_simple(
            map_,
            budget=10_000,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={"dense": 8},
            discovered_bytes={"g0": 1000, "dense": 1600},
        )

        by_group = {a.group: a for a in recipe.assignments}
        assert by_group["dense"].bits == 8
        # Q8_0 spends 8.5 effective bits (ADR-0014): 1600 * 8.5 / 16.
        assert by_group["dense"].bytes == 850
        assert by_group["dense"].damage == 0.0

    def test_pinned_uncovered_group_counts_toward_the_budget(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        recipe = solve_simple(
            map_,
            budget=10_000,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={"dense": 8},
            discovered_bytes={"g0": 1000, "dense": 1600},
        )

        assert recipe.plan.predicted_total_bytes == sum(
            a.bytes for a in recipe.assignments
        )

    def test_pinned_uncovered_group_never_enters_the_downgrade_loop(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        # The budget forces the measured group down while the pinned
        # uncovered group holds its width.
        recipe = solve_simple(
            map_,
            budget=1100,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={"dense": 8},
            discovered_bytes={"g0": 1000, "dense": 1600},
        )

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group["dense"] == 8
        assert by_group["g0"] == 2
        assert all(step.group != "dense" for step in recipe.plan.trace)

    def test_pin_on_unquantizable_uncovered_group_refuses(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        with pytest.raises(PinError, match="F16 passthrough"):
            solve_simple(
                map_,
                budget=10_000,
                runtime="llama.cpp",
                pins={"model.layers.0.mixer.conv1d": 8},
                discovered_bytes={"g0": 1000, "model.layers.0.mixer.conv1d": 64},
            )

    def test_redundant_pin_on_unquantizable_group_says_so(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        # A pin at the passthrough asks for what the hold already
        # gives — the refusal must not claim the pin moves anything.
        with pytest.raises(PinError, match="the pin is redundant"):
            solve_simple(
                map_,
                budget=10_000,
                runtime="llama.cpp",
                pins={"model.layers.0.mixer.conv1d": 16},
                discovered_bytes={"g0": 1000, "model.layers.0.mixer.conv1d": 64},
            )

    def test_all_uncovered_pinned_skips_the_reference_capability_check(self) -> None:
        # vLLM serves no reference precision, and every uncovered
        # group carries a pin at a width it does serve.
        map_ = load(make_map([("g0", 1000, {8: 0.001, 4: 0.01})], precisions=(8, 4)))

        recipe = solve_simple(
            map_,
            budget=10_000,
            runtime="vllm",
            pins={"dense": 4},
            discovered_bytes={"g0": 1000, "dense": 1600},
        )

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group["dense"] == 4

    def test_uncovered_group_without_a_pin_still_refuses_an_unservable_runtime(
        self,
    ) -> None:
        map_ = load(make_map([("g0", 1000, {8: 0.001, 4: 0.01})], precisions=(8, 4)))

        with pytest.raises(RuntimeCapabilityError, match="reference precision"):
            solve_simple(
                map_,
                budget=10_000,
                runtime="vllm",
                pins={"dense": 4},
                discovered_bytes={"g0": 1000, "dense": 1600, "unpinned": 800},
            )

    def test_later_pin_overrides_earlier_across_the_two_universes(self) -> None:
        map_ = load(make_map([("g0", 1000, {4: 0.01, 2: 1.0})], precisions=(4, 2)))

        recipe = solve_simple(
            map_,
            budget=10_000,
            runtime="llama.cpp",
            pins={"*": 8, "g0": 4},
            discovered_bytes={"g0": 1000, "dense": 1600},
        )

        by_group = {a.group: a for a in recipe.assignments}
        assert by_group["g0"].bits == 4
        assert by_group["g0"].damage == pytest.approx(0.01)
        assert by_group["dense"].bits == 8
        assert by_group["dense"].damage == 0.0

    def test_pin_below_the_cheapest_candidate_counts_toward_the_spread_rule(
        self,
    ) -> None:
        # A stack pinned below the cheapest candidate is a
        # cheapest-or-below stack in its layer, so the placement
        # rule sends the forced move to the layer with none.
        stacks = [
            ("model.layers.0.mixer.experts.up_proj", 160_000, {8: 0.001, 4: 0.01}),
            ("model.layers.0.mixer.experts.down_proj", 160_000, {8: 0.001, 4: 0.01}),
            ("model.layers.1.mixer.experts.up_proj", 160_000, {8: 0.001, 4: 0.01}),
            ("model.layers.1.mixer.experts.down_proj", 160_000, {8: 0.001, 4: 0.01}),
        ]
        map_ = load(make_map(stacks, precisions=(8, 4)))

        recipe = solve_simple(
            map_,
            budget=240_000,
            runtime="llama.cpp",
            format_overhead=0.0,
            pins={"model.layers.0.mixer.experts.up_proj": 2},
            discovered_bytes={name: 160_000 for name, _, _ in stacks},
        )

        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group["model.layers.0.mixer.experts.up_proj"] == 2
        assert by_group["model.layers.0.mixer.experts.down_proj"] == 8
        layer1 = [bits for group, bits in by_group.items() if ".layers.1." in group]
        assert sorted(layer1) == [4, 8]
