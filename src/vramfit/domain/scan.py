"""Scan orchestration logic: naming, work plan, resume, map assembly.

The scan loop itself lives in the inbound adapter (it drives ports).
This module holds the pure parts: which group a parameter belongs to
under each granularity (`group_key`, the `layer`/`tensor`/`stack`
naming rule that the torch meter applies but does not own), which
group names are routed-expert stacks (`is_expert_stack`, the
predicate the solver prices through — ADR-0028),
which decoder layer a group name carries (`layer_prefix`, the
relation the spread placement rule reads — the 2026-08-21 ADR-0007
amendment),
which discovered groups a caller's selection keeps
(`select_groups`, the subset rule that keeps a narrowed run out of
the fingerprint),
which (group x precision) cells still
need measurement, how a finished pile of measurements becomes a
`SensitivityMap` — per-tensor sizes and imatrix count summaries
riding through from `GroupSpec`
(ADR-0022, ADR-0026 decision 4) — the pooled count reduction
(`summarize_imatrix_counts`), the within-group method tokens and each
method's precision coverage set (`rtn-block32`, `kquant-ref`,
`kquant-imx`, and `q0-ref` since the 2026-08-18 amendment —
ADR-0006, ADR-0018, ADR-0020, with
every token re-exported from [vramfit.domain.model][], where
`ScanMeta` anchors them), and
the escaped fingerprint — carrying the method and the imatrix
path — that guards resume against
mixing two different scans' checkpoints.

Examples:
    Plan the remaining work after a partial scan:

    ```python
    from vramfit.domain.scan import GroupSpec, plan_measurements

    specs = (GroupSpec(name="g0", tensors=("w",), bytes_fp16=1000),)
    todo = plan_measurements(specs, precisions=(8, 4), done=[("g0", 8)])
    assert todo == (("g0", 4),)
    ```

See Also:
    - [vramfit.domain.model][]: `SensitivityMap`, the scan's output type.
    - [vramfit.ports.outbound][]: The ports the scan loop drives.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from vramfit.domain.model import (
    KQUANT_IMX_METHOD as KQUANT_IMX_METHOD,  # noqa: PLC0414 - re-export: method tokens read from this module
)
from vramfit.domain.model import (
    KQUANT_METHOD as KQUANT_METHOD,  # noqa: PLC0414 - re-export: method tokens read from this module
)
from vramfit.domain.model import (
    Q0_REF_METHOD as Q0_REF_METHOD,  # noqa: PLC0414 - re-export: method tokens read from this module
)
from vramfit.domain.model import (
    SCAN_METHOD as SCAN_METHOD,  # noqa: PLC0414 - re-export: method tokens read from this module
)
from vramfit.domain.model import (
    ImatrixCountSummary,
    LayerGroup,
    ScanMeta,
    SensitivityMap,
)

# Decoder-layer prefixes across common naming families: llama-style
# ".layers.N.", GPT-2-style ".h.N.", and ".blocks.N.".
_LAYER_PREFIX = re.compile(r"^(.*\.(?:layers|h|blocks)\.\d+)\.")
# The routed-expert index inside a mixture-of-experts parameter name.
# Every family spells the index the same way, between ".experts." and
# the projection: Mixtral at ".block_sparse_moe.experts.N.w1", Qwen and
# DeepSeek at ".mlp.experts.N.up_proj", and Nemotron 3.5 Lightning at
# "backbone.layers.N.mixer.experts.M.down_proj" (#160).
_EXPERT_INDEX = re.compile(r"\.experts\.\d+\.")
# A collapsed routed-expert-stack group: a layer prefix, then
# ".experts." with the index already collapsed by `group_key`, then
# the projection. Mirrors the GGUF adapter's recognizer — issue #190
# owns folding the duplicated naming rule into one place.
_EXPERT_STACK_GROUP = re.compile(
    r"^.+\.(?:layers|h|blocks)\.\d+\.(?:.*\.)?experts\.[A-Za-z0-9_]+$"
)


def group_key(name: str, group_by: Literal["layer", "tensor", "stack"]) -> str:
    """Derive one parameter's group name under a granularity.

    Pure naming policy, so the meter stays a thin discovery loop and
    the fast suite covers every granularity without torch.

    Args:
        name: Parameter name, e.g. ``model.layers.0.self_attn.q_proj.weight``.
        group_by: Grouping granularity.

    Returns:
        The group this parameter belongs to. ``layer`` returns the
        decoder-layer prefix (`layer_prefix` owns the rule), and
        falls back to the bare tensor name
        when the parameter sits outside any layer. ``tensor`` returns
        the tensor name. ``stack`` returns the tensor name with a
        routed-expert index removed, which fuses one projection's
        experts into the unit a pack addresses (#159, #161).

    Examples:
        Collapse one projection's routed experts:

        ```python
        from vramfit.domain.scan import group_key

        name = "backbone.layers.3.mixer.experts.57.down_proj.weight"
        assert group_key(name, "stack") == ("backbone.layers.3.mixer.experts.down_proj")
        ```
    """
    if group_by == "tensor":
        return name.removesuffix(".weight")
    if group_by == "stack":
        return _EXPERT_INDEX.sub(".experts.", name).removesuffix(".weight")
    return layer_prefix(name) or name.removesuffix(".weight")


def is_expert_stack(group: str) -> bool:
    """Report whether a group name is a routed-expert stack.

    The solver prices such a group through the expert-stack
    effective-bits table (ADR-0028), so the predicate must recognize
    exactly the groups the GGUF backend treats as expert stacks. The
    backend then maps or refuses each by its projection and its
    precision — recognition here, vocabulary there.

    Args:
        group: Group name, as `group_key` produces it.

    Returns:
        True when the name is a collapsed routed-expert stack under
        a decoder-layer prefix.

    Examples:
        ```python
        from vramfit.domain.scan import is_expert_stack

        assert is_expert_stack("backbone.layers.3.mixer.experts.down_proj")
        assert not is_expert_stack("backbone.layers.3")
        ```
    """
    return _EXPERT_STACK_GROUP.match(group) is not None


def layer_prefix(name: str) -> str | None:
    """Derive the decoder-layer prefix a name carries.

    The spread placement rule ([vramfit.domain.placement][]) reads
    the (layer, projection) relation off the expert-stack group name
    — the ADR-0007 amendment derives it and adds no schema field.

    Args:
        name: Parameter or group name.

    Returns:
        The decoder-layer prefix, or None when the name sits outside
        any layer.

    Examples:
        ```python
        from vramfit.domain.scan import layer_prefix

        stack = "backbone.layers.3.mixer.experts.down_proj"
        assert layer_prefix(stack) == "backbone.layers.3"
        assert layer_prefix("lm_head") is None
        ```
    """
    match = _LAYER_PREFIX.match(name)
    return match.group(1) if match else None


def matches_a_layer(name: str) -> bool:
    """Report whether a parameter sits inside a decoder layer.

    The meter refuses ``layer`` grouping on a model where nothing
    matches — degrading to per-tensor groups would misrepresent the
    map.

    Args:
        name: Parameter name.

    Returns:
        True when the name carries a decoder-layer prefix
        (`layer_prefix` owns the rule).

    Examples:
        ```python
        from vramfit.domain.scan import matches_a_layer

        assert matches_a_layer("model.layers.0.mlp.up_proj.weight")
        assert not matches_a_layer("model.embed_tokens.weight")
        ```
    """
    return layer_prefix(name) is not None


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """One discovered layer group, before any measurement.

    Attributes:
        name (str): Unique group name, e.g. ``model.layers.0.self_attn``.
        tensors (tuple[str, ...]): Tensor names quantized together in
            this group.
        bytes_fp16 (int): Group size in bytes at reference precision.
        tensor_bytes (Mapping[str, int]): Each member tensor's bytes
            at reference precision (ADR-0022), or empty when the
            meter predates the field. Rides into the map's groups so
            protections can price against it.
        imatrix_counts (ImatrixCountSummary | None): The pooled count
            distribution of the group's fused expert stacks (ADR-0026
            decision 4), or None for an unassisted meter or a group
            the 2026-08-13 #201 amendment leaves without one. Rides
            into the map's groups unchanged.

    Examples:
        A one-tensor group:

        ```python
        from vramfit.domain.scan import GroupSpec

        spec = GroupSpec(name="model.embed", tensors=("weight",), bytes_fp16=4096)
        ```
    """

    name: str
    tensors: tuple[str, ...]
    bytes_fp16: int
    tensor_bytes: Mapping[str, int] = field(hash=False, default_factory=dict)
    imatrix_counts: ImatrixCountSummary | None = None

    def __post_init__(self) -> None:
        """Enforce the spec invariants and freeze the size record.

        Raises:
            ValueError: If ``name`` is empty, ``bytes_fp16`` is not
                positive, ``tensors`` is empty, or a non-empty
                ``tensor_bytes`` does not cover exactly the group's
                tensors with positive sizes summing to ``bytes_fp16``
                (ADR-0022).
        """
        if not self.name:
            raise ValueError("name must not be empty")
        if self.bytes_fp16 <= 0:
            raise ValueError("bytes_fp16 must be positive")
        if not self.tensors:
            raise ValueError("tensors must not be empty")
        object.__setattr__(
            self, "tensor_bytes", MappingProxyType(dict(self.tensor_bytes))
        )
        if self.tensor_bytes:
            if set(self.tensor_bytes) != set(self.tensors):
                raise ValueError(
                    "tensor_bytes keys must equal the group's tensors (ADR-0022)"
                )
            if any(size <= 0 for size in self.tensor_bytes.values()):
                raise ValueError("tensor_bytes values must all be positive")
            if sum(self.tensor_bytes.values()) != self.bytes_fp16:
                raise ValueError(
                    "tensor_bytes must sum to bytes_fp16 — an inconsistent "
                    "size record would misprice protections (ADR-0022)"
                )


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
        from vramfit.domain.scan import Measurement

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


# The method tokens (ADR-0006/0018/0020) are defined beside
# `ScanMeta` and re-exported above — a method change is a new scan,
# so every token lives in the fingerprint.
# The precisions the kquant port covers. The scan validates candidate
# precisions against this before it loads a model.
KQUANT_PRECISIONS = (8, 4, 3, 2)
# The precisions the q0 port covers (ADR-0018, 2026-08-17
# amendment, token renamed 2026-08-18). Nominal 3 is absent because ADR-0028 decision 2
# refuses it at pack, and 5 and 6 wait for ports.
Q0_REF_PRECISIONS = (8, 4, 2)


def scan_fingerprint(model_id: str, meta: ScanMeta) -> str:
    """Derive the identity string that guards checkpoint resume.

    Two scans share a fingerprint when their recorded provenance
    matches: model identifier, metric, calibration path and size,
    grouping, candidate precisions, within-group method, and the
    imatrix path for assisted scans (empty when unassisted) — two
    assisted scans with different imatrix files must never share a
    checkpoint (ADR-0020). The fingerprint identifies provenance,
    not content — it cannot detect weights, calibration text, or
    imatrix content changing under an unchanged path. ``started_at``
    is excluded — a resumed scan is a new invocation of the same
    scan.

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
        from vramfit.domain.scan import scan_fingerprint

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
        meta.imatrix or "",
    )
    escaped = (f.replace("\\", "\\\\").replace("|", "\\|") for f in fields)
    return "|".join(escaped)


def select_groups(
    specs: Iterable[GroupSpec], names: Collection[str]
) -> tuple[GroupSpec, ...]:
    """Restrict discovered groups to a caller's named subset.

    The result follows discovery order, never the caller's list, so a
    narrowed scan measures its cells in the order a full scan would.
    An empty ``names`` keeps every spec — a caller that names no group
    wants the whole model.

    The selection never reaches the fingerprint. Two scans that differ
    only in selection share one checkpoint on purpose, so a narrow run
    and a wide run reuse each other's finished cells.

    Args:
        specs: Discovered layer groups, in discovery order.
        names: Group names to keep. Empty keeps every group.

    Returns:
        The selected specs, in discovery order.

    Raises:
        ValueError: If a name matches no discovered group. The message
            lists every unmatched name, so one run reports them all
            instead of one per attempt.

    Examples:
        Keep one group of two:

        ```python
        from vramfit.domain.scan import GroupSpec, select_groups

        specs = (
            GroupSpec(name="g0", tensors=("w",), bytes_fp16=8),
            GroupSpec(name="g1", tensors=("w",), bytes_fp16=8),
        )
        kept = select_groups(specs, ["g1"])
        assert tuple(spec.name for spec in kept) == ("g1",)
        ```
    """
    spec_list = tuple(specs)
    if not names:
        return spec_list
    wanted = set(names)
    discovered = {spec.name for spec in spec_list}
    unmatched = sorted(wanted - discovered)
    if unmatched:
        listed = ", ".join(f'"{name}"' for name in unmatched)
        raise ValueError(
            f"no discovered group matches {listed} — the model reports "
            f"{len(discovered)} groups at this granularity"
        )
    return tuple(spec for spec in spec_list if spec.name in wanted)


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
        from vramfit.domain.scan import GroupSpec, plan_measurements

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


def summarize_imatrix_counts(
    vectors: Iterable[Sequence[int]],
) -> ImatrixCountSummary:
    """Pool expert-stack count vectors into one summary (ADR-0026).

    The reduction behind decision 4's map fields, scoped by the
    2026-08-13 #201 amendment: it consumes expert-stack count
    vectors only. The caller selects the vectors — a scalar chunk
    tally never reaches this function.

    Args:
        vectors: One count vector per fused expert stack in the
            group. Each element ``i`` is expert ``i``'s routing
            frequency.

    Returns:
        The pooled count minimum, median, and maximum. The median is
        a float in every case — ``statistics.median`` returns an
        ``int`` for odd-length integer input, and one field must not
        write two JSON types. That conversion is where a count too
        large for a float fails, before the summary's own guard runs
        (#260).

    Raises:
        ValueError: If no counts arrive at all — an empty reduction
            has no distribution to summarize, and the #201 amendment
            leaves such a group's fields absent instead.

    Examples:
        Pool a two-stack group:

        ```python
        from vramfit.domain.scan import summarize_imatrix_counts

        summary = summarize_imatrix_counts([(3, 9), (5, 7)])
        assert (summary.min, summary.median, summary.max) == (3, 6.0, 9)
        ```
    """
    pooled = [count for vector in vectors for count in vector]
    if not pooled:
        raise ValueError(
            "no counts to summarize — a group without a resolved expert "
            "stack records no summary (ADR-0026, #201 amendment)"
        )
    try:
        median = float(statistics.median(pooled))
    except OverflowError as exc:
        # `statistics.median` averages the two middle counts on an
        # even-length pool, and that division overflows before the
        # summary's own guard can run. This is the in-memory route
        # the readers' bound does not cover (#260).
        raise ValueError("counts are too large to summarize") from exc
    return ImatrixCountSummary(min=min(pooled), median=median, max=max(pooled))


def assemble_map(
    model_id: str,
    meta: ScanMeta,
    specs: Iterable[GroupSpec],
    measurements: Iterable[Measurement],
) -> SensitivityMap:
    """Assemble finished measurements into a validated sensitivity map.

    Each spec's per-tensor sizes ride into its map group unchanged
    (ADR-0022) — a meter that reports them makes the map
    protection-ready. A spec's imatrix count summary rides through
    the same way (ADR-0026 decision 4).

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
        from vramfit.domain.scan import GroupSpec, Measurement, assemble_map

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
                tensor_bytes=spec.tensor_bytes,
                imatrix_counts=spec.imatrix_counts,
            )
            for spec in spec_list
        ),
    )
