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
