"""Unit tests for equal-byte neighbour generation."""

from __future__ import annotations

import pytest

from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    ScanMeta,
    SensitivityMap,
)
from vramfit.domain.refinement import (
    Candidate,
    Move,
    RefinementError,
    apply_move,
    decline_reason,
    neighbours,
    predicted_delta,
    refuse_unpriced_move,
)


def _map(groups=None, precisions=(8, 4, 2)):
    """Build a map whose stacks share one reference size."""
    groups = groups or [
        ("g0", 1600, {8: 0.0, 4: 0.10, 2: 0.40}),
        ("g1", 1600, {8: 0.0, 4: 0.20, 2: 0.90}),
        ("g2", 1600, {8: 0.0, 4: 0.05, 2: 0.30}),
        ("g3", 1600, {8: 0.0, 4: 0.30, 2: 0.50}),
    ]
    return SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="/work/calibration.txt",
            calibration_tokens=131072,
            precisions=precisions,
            group_by="stack",
            started_at="2026-07-27T00:00:00Z",
        ),
        groups=tuple(
            LayerGroup(
                name=name,
                tensors=(f"{name}.weight",),
                bytes_fp16=size,
                sensitivity=curve,
            )
            for name, size, curve in groups
        ),
    )


def _recipe(bits):
    """Build a recipe whose groups take the given precisions."""
    sizes = {2: 200, 4: 400, 8: 800}
    assignments = tuple(
        Assignment(group=name, bits=b, bytes=sizes[b], damage=0.1)
        for name, b in bits.items()
    )
    plan = PlanMeta(
        vram_budget_bytes=10_000,
        kv_headroom_bytes=0,
        weight_budget_bytes=10_000,
        predicted_total_bytes=sum(a.bytes for a in assignments),
        predicted_damage=0.4,
        solver="greedy-damage-per-byte",
        pins={},
        protections={},
        format_overhead=0.0,
        trace=(),
    )
    return Recipe(
        model_id="test/model",
        plan=plan,
        assignments=assignments,
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


def test_neighbours_of_a_two_level_recipe_keeps_total_bytes_fixed() -> None:
    recipe = _recipe({"g0": 2, "g1": 2, "g2": 4, "g3": 4})
    control = sum(a.bytes for a in recipe.assignments)
    found = neighbours(recipe, _map())
    assert found
    assert all(c.total_bytes() == control for c in found)


def test_neighbours_of_a_two_level_recipe_covers_every_pair() -> None:
    recipe = _recipe({"g0": 2, "g1": 2, "g2": 4, "g3": 4})
    found = neighbours(recipe, _map())
    pairs = {(c.move.promoted, c.move.demoted) for c in found}
    assert pairs == {
        ("g0", "g2"),
        ("g0", "g3"),
        ("g1", "g2"),
        ("g1", "g3"),
    }


def test_neighbours_changes_exactly_two_assignments() -> None:
    recipe = _recipe({"g0": 2, "g1": 2, "g2": 4, "g3": 4})
    before = {a.group: a for a in recipe.assignments}
    for candidate in neighbours(recipe, _map()):
        changed = {a.group for a in candidate.assignments if a != before[a.group]}
        assert changed == {candidate.move.promoted, candidate.move.demoted}


def test_neighbours_swaps_the_two_precisions() -> None:
    recipe = _recipe({"g0": 2, "g1": 2, "g2": 4, "g3": 4})
    candidate = neighbours(recipe, _map())[0]
    after = {a.group: a for a in candidate.assignments}
    assert after[candidate.move.promoted].bits == candidate.move.to_bits
    assert after[candidate.move.demoted].bits == candidate.move.from_bits


def test_neighbours_reads_damage_from_the_map() -> None:
    recipe = _recipe({"g0": 2, "g2": 4})
    candidate = neighbours(recipe, _map())[0]
    after = {a.group: a for a in candidate.assignments}
    assert after["g0"].damage == pytest.approx(0.10)
    assert after["g2"].damage == pytest.approx(0.30)


def test_neighbours_skips_a_pair_of_unequal_reference_sizes() -> None:
    map_ = _map(
        [
            ("g0", 1600, {8: 0.0, 4: 0.10, 2: 0.40}),
            ("g1", 3200, {8: 0.0, 4: 0.20, 2: 0.90}),
        ]
    )
    recipe = _recipe({"g0": 2, "g1": 4})
    assert neighbours(recipe, map_) == ()


def test_neighbours_skips_a_precision_the_map_never_measured() -> None:
    map_ = _map(
        [
            ("g0", 1600, {8: 0.0, 4: 0.10}),
            ("g1", 1600, {8: 0.0, 4: 0.20}),
        ],
        precisions=(8, 4),
    )
    recipe = _recipe({"g0": 4, "g1": 8})
    assert neighbours(recipe, map_) != ()
    recipe_at_two = _recipe({"g0": 2, "g1": 4})
    assert neighbours(recipe_at_two, map_) == ()


def test_predicted_delta_is_recorded_but_does_not_order_the_result() -> None:
    recipe = _recipe({"g0": 2, "g1": 2, "g2": 4, "g3": 4})
    found = neighbours(recipe, _map())
    deltas = [c.predicted_delta for c in found]
    assert deltas != sorted(deltas)


def test_predicted_delta_prices_the_swap_from_the_map() -> None:
    sensitivity = {
        "a": {2: 0.40, 4: 0.10},
        "b": {2: 0.30, 4: 0.05},
    }
    move = Move(promoted="a", demoted="b", from_bits=2, to_bits=4)
    assert predicted_delta(move, sensitivity) == pytest.approx(-0.05)


def test_move_refuses_one_group_swapped_with_itself() -> None:
    with pytest.raises(ValueError, match="two distinct groups"):
        Move(promoted="g0", demoted="g0", from_bits=2, to_bits=4)


def test_move_refuses_a_swap_that_raises_nothing() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        Move(promoted="g0", demoted="g1", from_bits=4, to_bits=4)


def test_refuse_unpriced_move_names_a_group_the_map_omits() -> None:
    reason = refuse_unpriced_move(
        Move(promoted="ghost", demoted="g1", from_bits=2, to_bits=4),
        {"g1": 1600},
        {"g1": {2: 0.9, 4: 0.2}},
    )
    assert reason is not None
    assert "ghost" in reason


def test_apply_move_refuses_a_group_at_the_wrong_precision() -> None:
    recipe = _recipe({"g0": 4, "g2": 4})
    move = Move(promoted="g0", demoted="g2", from_bits=2, to_bits=4)
    with pytest.raises(RefinementError, match="not the 2"):
        apply_move(
            move, recipe.assignments, {g.name: g.sensitivity for g in _map().groups}
        )


def test_apply_move_refuses_a_group_the_recipe_omits() -> None:
    recipe = _recipe({"g0": 2, "g2": 4})
    move = Move(promoted="g1", demoted="g2", from_bits=2, to_bits=4)
    with pytest.raises(RefinementError, match="omits group g1"):
        apply_move(
            move, recipe.assignments, {g.name: g.sensitivity for g in _map().groups}
        )


def test_decline_reason_reports_a_recipe_at_one_precision() -> None:
    recipe = _recipe({"g0": 4, "g1": 4, "g2": 4})
    reason = decline_reason(recipe, _map())
    assert reason is not None
    assert "no swap moves precision" in reason


def test_decline_reason_reports_the_49b_shape() -> None:
    """81 groups at the floor and one that prices differently."""
    groups = [("g0", 3200, {8: 0.0, 3: 0.10})]
    groups += [(f"g{i}", 1600, {8: 0.0, 3: 0.10}) for i in range(1, 82)]
    map_ = _map(groups, precisions=(8, 3))
    bits = {"g0": 8}
    bits.update({f"g{i}": 3 for i in range(1, 82)})
    reason = decline_reason(_recipe_at(bits), map_)
    assert reason is not None
    assert "no byte-neutral neighbour" in reason


def _recipe_at(bits):
    """Build a recipe over arbitrary precisions for the decline shape."""
    sizes = {2: 200, 3: 300, 4: 400, 8: 800}
    assignments = tuple(
        Assignment(group=name, bits=b, bytes=sizes[b], damage=0.1)
        for name, b in bits.items()
    )
    plan = PlanMeta(
        vram_budget_bytes=10**9,
        kv_headroom_bytes=0,
        weight_budget_bytes=10**9,
        predicted_total_bytes=sum(a.bytes for a in assignments),
        predicted_damage=0.4,
        solver="greedy-damage-per-byte",
        pins={},
        protections={},
        format_overhead=0.0,
        trace=(),
    )
    return Recipe(
        model_id="test/model",
        plan=plan,
        assignments=assignments,
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


def test_decline_reason_returns_none_when_a_neighbour_exists() -> None:
    recipe = _recipe({"g0": 2, "g2": 4})
    assert decline_reason(recipe, _map()) is None


def test_candidate_total_bytes_sums_the_assignments() -> None:
    candidate = Candidate(
        move=Move(promoted="a", demoted="b", from_bits=2, to_bits=4),
        assignments=(
            Assignment(group="a", bits=4, bytes=400, damage=0.1),
            Assignment(group="b", bits=2, bytes=200, damage=0.4),
        ),
        predicted_delta=0.0,
    )
    assert candidate.total_bytes() == 600
