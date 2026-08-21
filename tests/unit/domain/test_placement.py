from __future__ import annotations

import random
from typing import Any

import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from tests.strategies import raw_moe_maps, raw_sensitivity_maps
from tests.unit.conftest import make_map
from vramfit.adapters.outbound.sensitivity_map_json import map_from_dict
from vramfit.domain.model import SensitivityMap
from vramfit.domain.placement import refused_cheapest_stack_moves
from vramfit.domain.scan import is_expert_stack, layer_prefix
from vramfit.domain.solver import group_bytes, solve

overheads = st.floats(
    min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False
)

UP0 = "model.layers.0.mlp.experts.up_proj"
DOWN0 = "model.layers.0.mlp.experts.down_proj"
UP1 = "model.layers.1.mlp.experts.up_proj"
DOWN1 = "model.layers.1.mlp.experts.down_proj"


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


def stacks_at_cheapest(
    map_: SensitivityMap, state: dict[str, int], cheapest: int
) -> dict[str, int]:
    """Count each layer's expert stacks sitting at the cheapest width."""
    counts: dict[str, int] = {}
    for g in map_.groups:
        if is_expert_stack(g.name) and state[g.name] == cheapest:
            layer = layer_prefix(g.name)
            assert layer is not None
            counts[layer] = counts.get(layer, 0) + 1
    return counts


def eligible_movers(
    map_: SensitivityMap, state: dict[str, int], cheapest: int, overhead: float
) -> dict[str, list]:
    """Each layer's stacks that can still move to the cheapest width.

    Mirrors the rule's own eligibility: above the cheapest width and
    freeing bytes. The property runs pass no pins.
    """
    movers: dict[str, list] = {}
    for g in map_.groups:
        if not is_expert_stack(g.name):
            continue
        current = state[g.name]
        if current <= cheapest:
            continue
        freed = group_bytes(g.bytes_fp16, current, overhead) - group_bytes(
            g.bytes_fp16, cheapest, overhead
        )
        if freed <= 0:
            continue
        layer = layer_prefix(g.name)
        assert layer is not None
        movers.setdefault(layer, []).append(g)
    return movers


@pytest.mark.unit
class TestSpreadPlacementProperties:
    @given(raw=raw_moe_maps(), overhead=overheads, data=st.data())
    def test_second_cheapest_stack_in_a_layer_waits_until_no_layer_lacks_one(
        self, raw, overhead, data
    ) -> None:
        floor, ceiling = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        map_ = map_from_dict(raw)
        cheapest = map_.scan.precisions[-1]

        recipe = solve_simple(map_, budget, overhead)

        state = {g.name: map_.scan.precisions[0] for g in map_.groups}
        placements = 0
        for step in recipe.plan.trace:
            if is_expert_stack(step.group) and step.to_bits == cheapest:
                placements += 1
                layer = layer_prefix(step.group)
                counts = stacks_at_cheapest(map_, state, cheapest)
                if counts.get(layer, 0) >= 1:
                    movers = eligible_movers(map_, state, cheapest, overhead)
                    starved = [name for name in movers if not counts.get(name)]
                    assert not starved
            state[step.group] = step.to_bits
        event(f"cheapest placements: {placements}")

    @given(raw=raw_moe_maps(), overhead=overheads, data=st.data())
    def test_cheapest_stack_placement_always_takes_the_cheapest_priced_mover(
        self, raw, overhead, data
    ) -> None:
        floor, ceiling = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        map_ = map_from_dict(raw)
        cheapest = map_.scan.precisions[-1]
        by_name = {g.name: g for g in map_.groups}

        recipe = solve_simple(map_, budget, overhead)

        state = {g.name: map_.scan.precisions[0] for g in map_.groups}
        for step in recipe.plan.trace:
            if is_expert_stack(step.group) and step.to_bits == cheapest:
                layer = layer_prefix(step.group)
                movers = eligible_movers(map_, state, cheapest, overhead)
                chosen = by_name[step.group]
                assert all(
                    chosen.sensitivity[cheapest] <= mover.sensitivity[cheapest]
                    for mover in movers.get(layer, [])
                )
            state[step.group] = step.to_bits

    @given(raw=raw_moe_maps(), overhead=overheads, data=st.data())
    def test_stack_recipes_stay_deterministic_and_input_order_invariant(
        self, raw, overhead, data
    ) -> None:
        floor, ceiling = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        seed = data.draw(st.integers(min_value=0, max_value=2**16))
        shuffled = dict(raw)
        shuffled["groups"] = list(raw["groups"])
        random.Random(seed).shuffle(shuffled["groups"])

        first = solve_simple(map_from_dict(raw), budget, overhead)
        again = solve_simple(map_from_dict(raw), budget, overhead)
        reordered = solve_simple(map_from_dict(shuffled), budget, overhead)

        assert first == again
        assert {a.group: a.bits for a in first.assignments} == {
            a.group: a.bits for a in reordered.assignments
        }
        assert first.plan.trace == reordered.plan.trace

    @given(raw=raw_sensitivity_maps(), overhead=overheads, data=st.data())
    def test_dense_only_map_every_step_minimizes_the_plain_key(
        self, raw, overhead, data
    ) -> None:
        # Dense groups keep the plain damage-per-byte order — every
        # trace step is the global key minimum over the moves open at
        # that point. The final step is exempt: the refinement pass
        # may replace it with a milder same-group step (ADR-0007).
        floor, ceiling = bounds(raw, overhead)
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))
        map_ = map_from_dict(raw)

        recipe = solve_simple(map_, budget, overhead)

        state = {g.name: map_.scan.precisions[0] for g in map_.groups}
        trace = recipe.plan.trace
        for step in trace[:-1]:
            keys = []
            for g in map_.groups:
                current = state[g.name]
                current_bytes = group_bytes(g.bytes_fp16, current, overhead)
                for target in map_.scan.precisions:
                    if target >= current:
                        continue
                    freed = current_bytes - group_bytes(g.bytes_fp16, target, overhead)
                    if freed <= 0:
                        continue
                    delta = g.sensitivity[target] - g.sensitivity[current]
                    keys.append((delta / freed, g.name, -target))
            assert (step.ratio, step.group, -step.to_bits) == min(keys)
            state[step.group] = step.to_bits


@pytest.mark.unit
class TestSpreadPlacementExamples:
    def curve(self, at_cheapest: float, at_reference: float = 0.0) -> dict[int, float]:
        return {8: at_reference, 2: at_cheapest}

    def test_two_cheap_stacks_land_in_two_layers_not_one(self) -> None:
        # Plain damage-per-byte would put both cheapest-width stacks
        # in layer 0 — its damages are far lower. Clause 1 spreads
        # them, and clause 2 picks up_proj in both layers.
        raw = make_map(
            [
                (UP0, 1600, self.curve(0.1)),
                (DOWN0, 1600, self.curve(0.2)),
                (UP1, 1600, self.curve(5.0)),
                (DOWN1, 1600, self.curve(6.0)),
            ],
            precisions=(8, 2),
        )

        recipe = solve_simple(map_from_dict(raw), budget=2000, overhead=0.0)

        bits = {a.group: a.bits for a in recipe.assignments}
        assert bits == {UP0: 2, DOWN0: 8, UP1: 2, DOWN1: 8}
        placed = [layer_prefix(s.group) for s in recipe.plan.trace if s.to_bits == 2]
        assert placed == ["model.layers.0", "model.layers.1"]

    def test_projection_tie_break_overrides_a_better_ratio(self) -> None:
        # down_proj has the smaller damage delta (0.5 against 0.9),
        # so the unconstrained key would take it. The map prices
        # up_proj cheaper at the candidate width (0.9 against 1.0),
        # and the ruled tie-break reads that price, not the delta.
        raw = make_map(
            [
                (UP0, 1600, {8: 0.0, 2: 0.9}),
                (DOWN0, 1600, {8: 0.5, 2: 1.0}),
            ],
            precisions=(8, 2),
        )

        recipe = solve_simple(map_from_dict(raw), budget=1000, overhead=0.0)

        bits = {a.group: a.bits for a in recipe.assignments}
        assert bits == {UP0: 2, DOWN0: 8}

    def test_layer_pinned_above_cheapest_never_blocks_the_spread(self) -> None:
        # Layer 1 can never take a cheapest-width stack — both its
        # stacks are pinned. It must not starve layer 0's second
        # placement, or a prechecked-feasible budget would dead-end.
        raw = make_map(
            [
                (UP0, 1600, self.curve(0.1)),
                (DOWN0, 1600, self.curve(0.2)),
                (UP1, 1600, self.curve(5.0)),
                (DOWN1, 1600, self.curve(6.0)),
            ],
            precisions=(8, 2),
        )

        recipe = solve_simple(
            map_from_dict(raw),
            budget=2000,
            overhead=0.0,
            pins={"model.layers.1.mlp.experts.*": 8},
        )

        bits = {a.group: a.bits for a in recipe.assignments}
        assert bits == {UP0: 2, DOWN0: 2, UP1: 8, DOWN1: 8}

    def test_second_cheapest_stack_lands_after_every_layer_has_one(self) -> None:
        raw = make_map(
            [
                (UP0, 1600, self.curve(0.1)),
                (DOWN0, 1600, self.curve(0.2)),
                (UP1, 1600, self.curve(0.3)),
                (DOWN1, 1600, self.curve(0.4)),
            ],
            precisions=(8, 2),
        )

        recipe = solve_simple(map_from_dict(raw), budget=1400, overhead=0.0)

        bits = {a.group: a.bits for a in recipe.assignments}
        assert bits == {UP0: 2, DOWN0: 2, UP1: 2, DOWN1: 8}
        assert [s.group for s in recipe.plan.trace] == [UP0, UP1, DOWN0]

    def test_dense_groups_in_one_layer_concentrate_freely(self) -> None:
        # The rule reads on expert-stack groups only. Two dense
        # groups of one layer both reach the cheapest width while
        # layer 1 has none.
        raw = make_map(
            [
                ("model.layers.0.self_attn", 1600, self.curve(0.1)),
                ("model.layers.0.mlp", 1600, self.curve(0.2)),
                ("model.layers.1.self_attn", 1600, self.curve(5.0)),
                ("model.layers.1.mlp", 1600, self.curve(6.0)),
            ],
            precisions=(8, 2),
        )

        recipe = solve_simple(map_from_dict(raw), budget=2000, overhead=0.0)

        bits = {a.group: a.bits for a in recipe.assignments}
        assert bits == {
            "model.layers.0.self_attn": 2,
            "model.layers.0.mlp": 2,
            "model.layers.1.self_attn": 8,
            "model.layers.1.mlp": 8,
        }

    def test_intermediate_stack_downgrade_allowed_while_a_layer_has_none(self) -> None:
        # Only cheapest-width moves are constrained. Layer 0 already
        # holds a cheapest stack and layer 1 has none, yet layer 0's
        # other stack still takes the intermediate width.
        raw = make_map(
            [
                (UP0, 1600, {8: 0.0, 4: 0.5, 2: 1.0}),
                (DOWN0, 1600, {8: 0.0, 4: 0.001, 2: 100.0}),
                (UP1, 1600, {8: 0.0, 4: 50.0, 2: 60.0}),
                (DOWN1, 1600, {8: 0.0, 4: 55.0, 2: 65.0}),
            ],
            precisions=(8, 4, 2),
        )

        recipe = solve_simple(
            map_from_dict(raw), budget=2200, overhead=0.0, pins={UP0: 2}
        )

        bits = {a.group: a.bits for a in recipe.assignments}
        assert bits == {UP0: 2, DOWN0: 4, UP1: 8, DOWN1: 8}


@pytest.mark.unit
class TestRefusedCheapestStackMoves:
    def groups(self, raw: dict[str, Any]):
        return map_from_dict(raw).groups

    def size(self, group, bits: int) -> int:
        return group_bytes(group.bytes_fp16, bits, 0.0)

    def refused(
        self,
        raw: dict[str, Any],
        state: dict[str, int],
        pinned: dict[str, int] | None = None,
    ) -> frozenset[str]:
        return refused_cheapest_stack_moves(
            self.groups(raw), pinned or {}, state, 2, self.size
        )

    def two_layer_map(self) -> dict[str, Any]:
        return make_map(
            [
                (UP0, 1600, {8: 0.0, 2: 0.1}),
                (DOWN0, 1600, {8: 0.0, 2: 0.2}),
                (UP1, 1600, {8: 0.0, 2: 5.0}),
                (DOWN1, 1600, {8: 0.0, 2: 6.0}),
            ],
            precisions=(8, 2),
        )

    def test_layer_with_one_refuses_its_second_while_another_has_none(self) -> None:
        state = {UP0: 2, DOWN0: 8, UP1: 8, DOWN1: 8}

        refused = self.refused(self.two_layer_map(), state)

        # DOWN0 by clause 1, DOWN1 by clause 2 (UP1 prices cheaper).
        assert refused == {DOWN0, DOWN1}

    def test_no_layer_lacking_one_admits_every_sole_mover(self) -> None:
        state = {UP0: 2, DOWN0: 8, UP1: 2, DOWN1: 8}

        refused = self.refused(self.two_layer_map(), state)

        assert refused == frozenset()

    def test_pinned_stack_at_cheapest_counts_for_its_layer(self) -> None:
        state = {UP0: 2, DOWN0: 8, UP1: 8, DOWN1: 8}

        refused = self.refused(self.two_layer_map(), state, pinned={UP0: 2})

        assert DOWN0 in refused

    def test_layer_that_cannot_take_one_never_starves_the_rest(self) -> None:
        state = {UP0: 2, DOWN0: 8, UP1: 8, DOWN1: 8}

        refused = self.refused(self.two_layer_map(), state, pinned={UP1: 8, DOWN1: 8})

        assert refused == frozenset()

    def test_equal_prices_keep_both_projections(self) -> None:
        raw = make_map(
            [
                (UP0, 1600, {8: 0.0, 2: 0.5}),
                (DOWN0, 1600, {8: 0.0, 2: 0.5}),
            ],
            precisions=(8, 2),
        )
        state = {UP0: 8, DOWN0: 8}

        refused = self.refused(raw, state)

        assert refused == frozenset()

    def test_dense_group_names_never_enter_the_rule(self) -> None:
        raw = make_map(
            [
                ("model.layers.0.self_attn", 1600, {8: 0.0, 2: 0.1}),
                ("model.layers.0.mlp", 1600, {8: 0.0, 2: 0.2}),
            ],
            precisions=(8, 2),
        )
        state = {"model.layers.0.self_attn": 2, "model.layers.0.mlp": 8}

        refused = self.refused(raw, state)

        assert refused == frozenset()

    def test_move_that_frees_no_bytes_cannot_starve_a_layer(self) -> None:
        # Layer 1's stack is one byte at reference, so its sizes
        # round equal at both widths and the move frees nothing. It
        # must not hold up layer 0's second placement.
        raw = make_map(
            [
                (UP0, 1600, {8: 0.0, 2: 0.1}),
                (DOWN0, 1600, {8: 0.0, 2: 0.2}),
                (UP1, 1, {8: 0.0, 2: 5.0}),
            ],
            precisions=(8, 2),
        )
        state = {UP0: 2, DOWN0: 8, UP1: 8}

        refused = self.refused(raw, state)

        assert refused == frozenset()
