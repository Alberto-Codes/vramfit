"""Scan orchestration logic: work plan, resume filtering, map assembly.

The scan loop itself lives in the inbound adapter (it drives ports).
This module holds the pure parts: which (group x precision) cells still
need measurement, how a finished pile of measurements becomes a
`SensitivityMap`, the within-group method tokens and the kquant
coverage set (ADR-0006, ADR-0018 — `SCAN_METHOD` re-exported from
[quantfit.domain.model][], where it is the `ScanMeta` default), and
the escaped, method-carrying fingerprint that guards resume against
mixing two different scans' checkpoints.

Examples:
    Plan the remaining work after a partial scan:

    ```python
    from quantfit.domain.scan import GroupSpec, plan_measurements

    specs = (GroupSpec(name="g0", tensors=("w",), bytes_fp16=1000),)
    todo = plan_measurements(specs, precisions=(8, 4), done=[("g0", 8)])
    assert todo == (("g0", 4),)
    ```

See Also:
    - [quantfit.domain.model][]: `SensitivityMap`, the scan's output type.
    - [quantfit.ports.outbound][]: The ports the scan loop drives.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from quantfit.domain.model import (
    SCAN_METHOD as SCAN_METHOD,  # noqa: PLC0414 - re-export: method tokens read from this module
)
from quantfit.domain.model import LayerGroup, ScanMeta, SensitivityMap


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """One discovered layer group, before any measurement.

    Attributes:
        name (str): Unique group name, e.g. ``model.layers.0.self_attn``.
        tensors (tuple[str, ...]): Tensor names quantized together in
            this group.
        bytes_fp16 (int): Group size in bytes at reference precision.

    Examples:
        A one-tensor group:

        ```python
        from quantfit.domain.scan import GroupSpec

        spec = GroupSpec(name="model.embed", tensors=("weight",), bytes_fp16=4096)
        ```
    """

    name: str
    tensors: tuple[str, ...]
    bytes_fp16: int

    def __post_init__(self) -> None:
        """Enforce the spec invariants.

        Raises:
            ValueError: If ``name`` is empty, ``bytes_fp16`` is not
                positive, or ``tensors`` is empty.
        """
        if not self.name:
            raise ValueError("name must not be empty")
        if self.bytes_fp16 <= 0:
            raise ValueError("bytes_fp16 must be positive")
        if not self.tensors:
            raise ValueError("tensors must not be empty")


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured cell of the scan grid.

    Attributes:
        group (str): The measured layer group's name.
        bits (int): The candidate precision the group was quantized to.
        damage (float): Measured damage, per the scan metric.

    Examples:
        A cheap 8-bit measurement:

        ```python
        from quantfit.domain.scan import Measurement

        m = Measurement(group="g0", bits=8, damage=0.0001)
        ```
    """

    group: str
    bits: int
    damage: float

    def __post_init__(self) -> None:
        """Enforce the measurement invariants.

        Raises:
            ValueError: If ``group`` is empty, ``bits`` is not positive,
                or ``damage`` is negative or not finite.
        """
        if not self.group:
            raise ValueError("group must not be empty")
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if not math.isfinite(self.damage) or self.damage < 0.0:
            raise ValueError("damage must be a finite non-negative number")


# The v1 method token SCAN_METHOD (ADR-0006) is defined beside
# `ScanMeta` and re-exported above — a method change is a new scan,
# so the token lives in the fingerprint.
# The K-quant-faithful method (ADR-0018): llama.cpp reference
# quantizers ported to torch — Q2_K, Q3_K, Q4_K, Q8_0.
KQUANT_METHOD = "kquant-ref"
# The precisions the kquant port covers. The scan validates candidate
# precisions against this before it loads a model.
KQUANT_PRECISIONS = (8, 4, 3, 2)


def scan_fingerprint(model_id: str, meta: ScanMeta) -> str:
    """Derive the identity string that guards checkpoint resume.

    Two scans share a fingerprint when their recorded provenance
    matches: model identifier, metric, calibration path and size,
    grouping, candidate precisions, and within-group method. The
    fingerprint identifies provenance, not content — it cannot detect
    weights or calibration text changing under an unchanged path.
    ``started_at`` is excluded — a resumed scan is a new invocation of
    the same scan.

    Args:
        model_id: The scanned model's identifier.
        meta: The scan's provenance, including the within-group
            method token (ADR-0018).

    Returns:
        A stable, human-readable identity string. Field separators
        inside values are escaped, so no two distinct scans collide.

    Examples:
        The fingerprint survives a restart:

        ```python
        from quantfit.domain.scan import scan_fingerprint

        assert scan_fingerprint("m", meta) == scan_fingerprint("m", meta)
        ```
    """
    precisions = ",".join(str(p) for p in meta.precisions)
    fields = (
        model_id,
        meta.metric,
        meta.calibration,
        str(meta.calibration_tokens),
        meta.group_by,
        precisions,
        meta.within_group,
    )
    escaped = (f.replace("\\", "\\\\").replace("|", "\\|") for f in fields)
    return "|".join(escaped)


def plan_measurements(
    specs: Iterable[GroupSpec],
    precisions: Iterable[int],
    done: Collection[tuple[str, int]] = (),
) -> tuple[tuple[str, int], ...]:
    """Compute the ordered (group, bits) cells still to measure.

    The order is deterministic: groups in discovery order, precisions in
    the given order within each group. Determinism makes a resumed scan
    continue exactly where the crashed one stopped.

    Args:
        specs: Discovered layer groups.
        precisions: Candidate precisions. Any iterable works — the
            values are materialized once before the grid is built.
        done: Cells already measured, e.g. from a checkpoint.

    Returns:
        The remaining cells, in measurement order.

    Raises:
        ValueError: If two specs share a name, ``done`` contains a cell
            outside the scan grid, or ``done`` repeats a cell. Each is
            a sign the checkpoint belongs to a different or damaged
            scan — better rejected now than after hours of measuring.

    Examples:
        A fresh scan measures the full grid:

        ```python
        from quantfit.domain.scan import GroupSpec, plan_measurements

        specs = (GroupSpec(name="g0", tensors=("w",), bytes_fp16=8),)
        assert plan_measurements(specs, (8, 4)) == (("g0", 8), ("g0", 4))
        ```
    """
    spec_list = list(specs)
    names = [spec.name for spec in spec_list]
    if len(set(names)) != len(names):
        raise ValueError("group names must be unique")
    # Materialize first — a generator would exhaust after the first
    # group and silently truncate the grid.
    precision_list = list(precisions)
    grid = [(name, bits) for name in names for bits in precision_list]
    grid_set = set(grid)
    seen: set[tuple[str, int]] = set()
    for cell in done:
        if cell not in grid_set:
            raise ValueError(
                f"checkpoint cell {cell!r} is outside the scan grid — "
                "the checkpoint belongs to a different scan"
            )
        if cell in seen:
            raise ValueError(
                f"checkpoint cell {cell!r} appears twice — the checkpoint "
                "is damaged (two scans may have shared one output path)"
            )
        seen.add(cell)
    return tuple(cell for cell in grid if cell not in seen)


def assemble_map(
    model_id: str,
    meta: ScanMeta,
    specs: Iterable[GroupSpec],
    measurements: Iterable[Measurement],
) -> SensitivityMap:
    """Assemble finished measurements into a validated sensitivity map.

    Args:
        model_id: The scanned model's identifier.
        meta: The scan's provenance.
        specs: Discovered layer groups, in map order.
        measurements: One measurement per (group x precision) cell.

    Returns:
        The validated map.

    Raises:
        ValueError: If a measurement duplicates a cell, names an unknown
            group, or any (group x precision) cell is missing — the
            error names the group and lists the missing precisions.

    Examples:
        One group, two precisions:

        ```python
        from quantfit.domain.scan import GroupSpec, Measurement, assemble_map

        specs = (GroupSpec(name="g0", tensors=("w",), bytes_fp16=8),)
        map_ = assemble_map(
            "m",
            meta,
            specs,
            [Measurement("g0", 8, 0.0), Measurement("g0", 4, 0.1)],
        )
        ```
    """
    spec_list = list(specs)
    known = {spec.name for spec in spec_list}
    curves: dict[str, dict[int, float]] = {name: {} for name in known}
    for m in measurements:
        if m.group not in known:
            raise ValueError(f'measurement names unknown group "{m.group}"')
        if m.bits in curves[m.group]:
            raise ValueError(
                f'duplicate measurement for group "{m.group}" at {m.bits}-bit'
            )
        curves[m.group][m.bits] = m.damage
    expected = set(meta.precisions)
    for spec in spec_list:
        missing = expected - set(curves[spec.name])
        if missing:
            listed = ", ".join(f"{bits}-bit" for bits in sorted(missing, reverse=True))
            raise ValueError(f'group "{spec.name}" is missing measurements at {listed}')
    return SensitivityMap(
        model_id=model_id,
        scan=meta,
        groups=tuple(
            LayerGroup(
                name=spec.name,
                tensors=spec.tensors,
                bytes_fp16=spec.bytes_fp16,
                sensitivity=curves[spec.name],
            )
            for spec in spec_list
        ),
    )
