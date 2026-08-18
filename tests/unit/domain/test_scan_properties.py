from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import precision_sets
from vramfit.domain.model import ScanMeta
from vramfit.domain.scan import (
    GroupSpec,
    Measurement,
    assemble_map,
    plan_measurements,
    select_groups,
)

pytestmark = pytest.mark.unit


@st.composite
def spec_lists(draw: st.DrawFn) -> tuple[GroupSpec, ...]:
    """Draw 1-6 uniquely named group specs."""
    n = draw(st.integers(min_value=1, max_value=6))
    return tuple(
        GroupSpec(
            name=f"model.layers.{i}",
            tensors=(f"model.layers.{i}.w",),
            bytes_fp16=draw(st.integers(min_value=1, max_value=10_000)),
        )
        for i in range(n)
    )


@given(specs=spec_lists(), precisions=precision_sets(), data=st.data())
def test_plan_and_done_always_partition_the_grid(specs, precisions, data) -> None:
    grid = [(spec.name, bits) for spec in specs for bits in precisions]
    done = data.draw(st.lists(st.sampled_from(grid), unique=True))

    todo = plan_measurements(specs, precisions, done)

    assert set(todo) | set(done) == set(grid)
    assert set(todo) & set(done) == set()
    assert len(todo) == len(set(todo))


@given(specs=spec_lists(), precisions=precision_sets(), data=st.data())
def test_todo_preserves_grid_order(specs, precisions, data) -> None:
    grid = [(spec.name, bits) for spec in specs for bits in precisions]
    done = data.draw(st.lists(st.sampled_from(grid), unique=True))

    todo = plan_measurements(specs, precisions, done)

    positions = {cell: i for i, cell in enumerate(grid)}
    assert list(todo) == sorted(todo, key=positions.__getitem__)


@given(specs=spec_lists(), precisions=precision_sets(), data=st.data())
def test_assembly_is_invariant_to_measurement_order(specs, precisions, data) -> None:
    meta = ScanMeta(
        metric="kl_divergence",
        calibration="calib.txt",
        calibration_tokens=1024,
        precisions=precisions,
        group_by="layer",
        started_at="2026-07-28T00:00:00Z",
    )
    measurements = [
        Measurement(
            group=spec.name,
            bits=bits,
            damage=data.draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        )
        for spec in specs
        for bits in precisions
    ]
    shuffled = data.draw(st.permutations(measurements))

    assert assemble_map("m", meta, specs, shuffled) == assemble_map(
        "m", meta, specs, measurements
    )


@st.composite
def name_subsets(draw: st.DrawFn, specs: tuple[GroupSpec, ...]) -> list[str]:
    """Draw a unique subset of a spec list's names, possibly empty."""
    return draw(st.lists(st.sampled_from([spec.name for spec in specs]), unique=True))


@given(specs=spec_lists())
def test_selecting_every_name_returns_every_spec(specs) -> None:
    every = [spec.name for spec in specs]

    assert select_groups(specs, every) == tuple(specs)


@given(specs=spec_lists(), data=st.data())
def test_a_selection_always_keeps_discovery_order(specs, data) -> None:
    names = data.draw(name_subsets(specs))

    kept = select_groups(specs, names)

    position = {spec.name: i for i, spec in enumerate(specs)}
    indices = [position[spec.name] for spec in kept]
    assert indices == sorted(indices)


@given(specs=spec_lists(), data=st.data())
def test_a_selection_keeps_exactly_the_named_groups(specs, data) -> None:
    names = data.draw(name_subsets(specs))

    kept = select_groups(specs, names)

    expected = set(names) if names else {spec.name for spec in specs}
    assert {spec.name for spec in kept} == expected


@given(specs=spec_lists(), precisions=precision_sets(), data=st.data())
def test_a_selections_plan_is_the_full_plan_restricted_to_it(
    specs, precisions, data
) -> None:
    """The invariant the checkpoint-sharing design rests on.

    A narrowed run must plan exactly the cells a wide run would plan
    for those groups, in the same order. Otherwise a shared checkpoint
    could not serve both.
    """
    names = data.draw(name_subsets(specs))
    kept = select_groups(specs, names)
    keep = {spec.name for spec in kept}

    full = plan_measurements(specs, precisions)

    assert plan_measurements(kept, precisions) == tuple(
        cell for cell in full if cell[0] in keep
    )


@given(specs=spec_lists(), data=st.data())
def test_an_unknown_name_always_refuses(specs, data) -> None:
    names = data.draw(name_subsets(specs))

    with pytest.raises(ValueError, match="no discovered group matches"):
        select_groups(specs, [*names, "model.layers.absent"])
