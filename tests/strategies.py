from __future__ import annotations

from typing import Any

import hypothesis.strategies as st

from tests.unit.conftest import make_map
from vramfit.domain.budget import KVLayer, ModelShape


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
def raw_protected_maps(draw: st.DrawFn) -> tuple[dict[str, Any], int]:
    """Draw a map of two-tensor groups with sizes, plus a protection floor.

    Every group carries ``tensor_bytes`` so protections can price
    against it (ADR-0022). The floor comes from the map's own
    candidate set, so a plan at the highest precision is never below
    it.
    """
    precisions = draw(precision_sets())
    floor = draw(st.sampled_from(list(precisions)))
    damage = st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    )
    n_groups = draw(st.integers(min_value=1, max_value=6))
    groups = []
    for i in range(n_groups):
        first = draw(st.integers(min_value=1, max_value=1_000_000))
        second = draw(st.integers(min_value=1, max_value=1_000_000))
        curve = {bits: draw(damage) for bits in precisions}
        groups.append((f"model.layers.{i}", first, second, curve))
    raw = make_map(
        [(name, a + b, curve) for name, a, b, curve in groups],
        precisions=precisions,
    )
    for entry, (name, a, b, _) in zip(raw["groups"], groups, strict=True):
        entry["tensors"] = [f"{name}.t0", f"{name}.t1"]
        entry["tensor_bytes"] = {f"{name}.t0": a, f"{name}.t1": b}
    return raw, floor


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


@st.composite
def raw_moe_maps(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a map of MoE layers, two projection stacks each.

    The spread placement rule (ADR-0007, 2026-08-21 amendment) reads
    the layer relation off the stack group names, so every layer
    carries an `up_proj` and a `down_proj` stack. Sizes are equal
    within a layer about half the time — equal sizes are where the
    projection tie-break and the plain ratio order can disagree.
    Zero to two dense groups ride along inside the same layers, so
    the draws cover mixed layers — the real MoE topology.
    """
    precisions = draw(precision_sets())
    damage = st.one_of(
        st.sampled_from([0.0, 0.1, 1.0]),
        st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    sizes = st.one_of(
        st.sampled_from([160, 1600, 100_000]),
        st.integers(min_value=1, max_value=10_000_000),
    )
    n_layers = draw(st.integers(min_value=1, max_value=4))
    groups = []
    for i in range(n_layers):
        up_bytes = draw(sizes)
        down_bytes = up_bytes if draw(st.booleans()) else draw(sizes)
        for projection, bytes_fp16 in (
            ("up_proj", up_bytes),
            ("down_proj", down_bytes),
        ):
            curve = {bits: draw(damage) for bits in precisions}
            groups.append(
                (f"model.layers.{i}.mlp.experts.{projection}", bytes_fp16, curve)
            )
    for i in range(draw(st.integers(min_value=0, max_value=min(2, n_layers)))):
        curve = {bits: draw(damage) for bits in precisions}
        groups.append((f"model.layers.{i}.self_attn", draw(sizes), curve))
    return make_map(groups, precisions=precisions)


@st.composite
def raw_maps_with_discovered_bytes(
    draw: st.DrawFn,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Draw a map plus a checkpoint's group sizes that partly cover it.

    The source always carries every group the map does, at the map's
    own size, plus zero or more groups the map never measured
    (ADR-0029). Those are the uncovered groups the solver holds at
    reference precision.
    """
    raw = draw(raw_sensitivity_maps())
    discovered = {g["name"]: g["bytes_fp16"] for g in raw["groups"]}
    n_uncovered = draw(st.integers(min_value=0, max_value=4))
    for i in range(n_uncovered):
        discovered[f"model.uncovered.{i}"] = draw(
            st.integers(min_value=1, max_value=10_000_000)
        )
    return raw, discovered


@st.composite
def kv_shapes(draw: st.DrawFn) -> ModelShape:
    """Draw a heterogeneous `ModelShape` of 1-8 KV layers.

    Layers mix global and sliding attention, head widths, KV-head
    counts, K=V storage, and shared-KV entries, so the KV arithmetic
    properties hold across every mechanism #421 models.
    """
    n_layers = draw(st.integers(min_value=1, max_value=8))
    layers = tuple(
        KVLayer(
            kv_heads=draw(st.integers(min_value=1, max_value=64)),
            head_dim=draw(st.integers(min_value=1, max_value=512)),
            window=draw(
                st.one_of(st.none(), st.integers(min_value=1, max_value=1 << 16))
            ),
            kv_tensors=draw(st.sampled_from([1, 2])),
            shares_kv=draw(st.booleans()),
        )
        for _ in range(n_layers)
    )
    return ModelShape(kv_layers=layers)
