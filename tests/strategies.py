from __future__ import annotations

from typing import Any

import hypothesis.strategies as st

from tests.unit.conftest import make_map


@st.composite
def precision_sets(draw: st.DrawFn) -> tuple[int, ...]:
    """Draw a strictly descending tuple of candidate precisions.

    Single-element sets are legal and deliberately generated: a map
    scanned at one precision leaves the solver nothing to downgrade.
    """
    values = draw(
        st.lists(
            st.integers(min_value=1, max_value=16), min_size=1, max_size=5, unique=True
        )
    )
    return tuple(sorted(values, reverse=True))


@st.composite
def raw_sensitivity_maps(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a valid raw sensitivity-map dict of 1-8 heterogeneous groups."""
    precisions = draw(precision_sets())
    # Mix discrete and continuous damages so equal-ratio ties actually
    # occur — the tie-break is part of the determinism contract.
    damage = st.one_of(
        st.sampled_from([0.0, 0.1, 1.0]),
        st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    sizes = st.one_of(
        st.sampled_from([1, 160, 1600, 100_000]),
        st.integers(min_value=1, max_value=10_000_000),
    )
    n_groups = draw(st.integers(min_value=1, max_value=8))
    groups = []
    for i in range(n_groups):
        bytes_fp16 = draw(sizes)
        curve = {bits: draw(damage) for bits in precisions}
        groups.append((f"model.layers.{i}.g", bytes_fp16, curve))
    return make_map(groups, precisions=precisions)
