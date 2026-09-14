"""Hypothesis properties for refinement and its paired comparison."""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given

from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    ScanMeta,
    SensitivityMap,
)
from vramfit.domain.paired import compare, per_chunk, select
from vramfit.domain.refinement import decline_reason, neighbours

# Every group carries one reference size, which is the condition a
# byte-neutral swap needs. The C4 protocol ran on expert stacks, which
# are the same size as each other.
REFERENCE_BYTES = 1600
PRECISIONS = (8, 4, 2)
SIZE_AT = {8: 800, 4: 400, 2: 200}


@st.composite
def _recipes(draw: st.DrawFn) -> tuple[Recipe, SensitivityMap]:
    """Draw a recipe over equal-size groups and the map that priced it."""
    count = draw(st.integers(min_value=2, max_value=6))
    names = [f"g{i}" for i in range(count)]
    curves = {
        name: {
            bits: draw(st.floats(min_value=0.0, max_value=2.0)) for bits in PRECISIONS
        }
        for name in names
    }
    bits = {name: draw(st.sampled_from(PRECISIONS)) for name in names}
    map_ = SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="/work/calibration.txt",
            calibration_tokens=131072,
            precisions=PRECISIONS,
            group_by="stack",
            started_at="2026-07-27T00:00:00Z",
        ),
        groups=tuple(
            LayerGroup(
                name=name,
                tensors=(f"{name}.weight",),
                bytes_fp16=REFERENCE_BYTES,
                sensitivity=curves[name],
            )
            for name in names
        ),
    )
    assignments = tuple(
        Assignment(
            group=name,
            bits=bits[name],
            bytes=SIZE_AT[bits[name]],
            damage=curves[name][bits[name]],
        )
        for name in names
    )
    plan = PlanMeta(
        vram_budget_bytes=10**9,
        kv_headroom_bytes=0,
        weight_budget_bytes=10**9,
        predicted_total_bytes=sum(a.bytes for a in assignments),
        predicted_damage=sum(a.damage for a in assignments),
        solver="greedy-damage-per-byte",
        pins={},
        protections={},
        format_overhead=0.0,
        trace=(),
    )
    recipe = Recipe(
        model_id="test/model",
        plan=plan,
        assignments=assignments,
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )
    return recipe, map_


@given(_recipes())
def test_every_neighbour_spends_the_recipes_bytes(
    drawn: tuple[Recipe, SensitivityMap],
) -> None:
    """The equal-byte invariant the whole protocol rests on."""
    recipe, map_ = drawn
    total = sum(a.bytes for a in recipe.assignments)
    for candidate in neighbours(recipe, map_):
        assert candidate.total_bytes() == total


@given(_recipes())
def test_every_neighbour_keeps_the_recipes_groups(
    drawn: tuple[Recipe, SensitivityMap],
) -> None:
    recipe, map_ = drawn
    names = [a.group for a in recipe.assignments]
    for candidate in neighbours(recipe, map_):
        assert [a.group for a in candidate.assignments] == names


@given(_recipes())
def test_every_neighbour_moves_exactly_two_assignments(
    drawn: tuple[Recipe, SensitivityMap],
) -> None:
    recipe, map_ = drawn
    before = {a.group: a for a in recipe.assignments}
    for candidate in neighbours(recipe, map_):
        changed = [a for a in candidate.assignments if a != before[a.group]]
        assert len(changed) == 2


@given(_recipes())
def test_decline_speaks_exactly_when_the_neighbourhood_is_empty(
    drawn: tuple[Recipe, SensitivityMap],
) -> None:
    recipe, map_ = drawn
    assert (decline_reason(recipe, map_) is None) == bool(neighbours(recipe, map_))


@st.composite
def _arms(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Draw two arms measured over the same chunks."""
    size = draw(st.integers(min_value=2, max_value=40))
    value = st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
    return (
        draw(st.lists(value, min_size=size, max_size=size)),
        draw(st.lists(value, min_size=size, max_size=size)),
    )


@given(_arms())
def test_compare_reverses_sign_when_the_arms_swap(
    drawn: tuple[list[float], list[float]],
) -> None:
    a, b = drawn
    assert compare(a, b).delta == pytest.approx(-compare(b, a).delta)


@given(_arms())
def test_compare_counts_every_chunk_it_was_given(
    drawn: tuple[list[float], list[float]],
) -> None:
    a, b = drawn
    assert compare(a, b).chunks == len(a)


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
        min_size=1,
        max_size=30,
    )
)
def test_per_chunk_recumulates_to_its_input(values: list[float]) -> None:
    recovered = per_chunk(values)
    running = [sum(recovered[: i + 1]) / (i + 1) for i in range(len(recovered))]
    assert running == pytest.approx(values)


@given(_arms(), st.floats(min_value=0.0, max_value=20.0))
def test_select_never_returns_an_arm_short_of_the_bar(
    drawn: tuple[list[float], list[float]], bar: float
) -> None:
    a, b = drawn
    results = {"only": compare(a, b)}
    winner, refusal = select(results, bar=bar)
    if winner is None:
        assert refusal is not None
    else:
        assert results[winner].improved(bar)
