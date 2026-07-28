"""Domain types for the two pipeline artifacts.

The dataclasses enforce their own structural invariants in
``__post_init__`` (positive sizes, strictly descending precisions,
unique group names, sensitivity keys matching the scan) so an instance
that exists is safe for the solver — however it was constructed.
Serialization and the JSON schema envelope (including the
``quantfit_schema`` version field) live in
[quantfit.adapters.outbound.sensitivity_map_json][] and
[quantfit.adapters.outbound.recipe_json][], whose loaders add JSON-path
error reporting on top of these checks.

Examples:
    Build a one-group map in memory:

    ```python
    from quantfit.domain.model import LayerGroup, ScanMeta, SensitivityMap

    map_ = SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="wikitext",
            calibration_tokens=1024,
            precisions=(8, 4),
            group_by="layer",
            started_at="2026-07-27T00:00:00Z",
        ),
        groups=(
            LayerGroup(
                name="g0",
                tensors=("w",),
                bytes_fp16=1000,
                sensitivity={8: 0.0, 4: 0.1},
            ),
        ),
    )
    ```

See Also:
    - [quantfit.domain.solver][]: Consumes `SensitivityMap`, builds `Recipe`.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True)
class ScanMeta:
    """Provenance of a sensitivity scan.

    Construction validates the invariants the solver relies on — an
    instance with unordered precisions cannot exist.

    Attributes:
        metric (str): Divergence metric name, e.g. ``kl_divergence``.
        calibration (str): Calibration set name or path.
        calibration_tokens (int): Number of calibration tokens measured.
        precisions (tuple[int, ...]): Candidate bit-widths, strictly
            descending.
        group_by (str): Layer grouping granularity (``layer`` or
            ``tensor``).
        started_at (str): ISO-8601 timestamp of the scan start.

    Examples:
        The scan section of a map:

        ```python
        from quantfit.domain.model import ScanMeta

        meta = ScanMeta(
            metric="kl_divergence",
            calibration="wikitext",
            calibration_tokens=131072,
            precisions=(8, 4),
            group_by="layer",
            started_at="2026-07-27T00:00:00Z",
        )
        ```
    """

    metric: str
    calibration: str
    calibration_tokens: int
    precisions: tuple[int, ...]
    group_by: Literal["layer", "tensor"]
    started_at: str

    def __post_init__(self) -> None:
        """Enforce the scan invariants the solver relies on.

        Raises:
            ValueError: If ``calibration_tokens`` is not positive, or
                ``precisions`` is empty, non-positive, or not strictly
                descending.
        """
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if not self.precisions:
            raise ValueError("precisions must not be empty")
        if any(p <= 0 for p in self.precisions):
            raise ValueError("precisions must all be positive")
        if not all(a > b for a, b in itertools.pairwise(self.precisions)):
            raise ValueError("precisions must be strictly descending")


@dataclass(frozen=True, slots=True)
class LayerGroup:
    """One scanned layer group and its damage curve.

    Attributes:
        name (str): Unique group name, e.g. ``model.layers.0.self_attn``.
        tensors (tuple[str, ...]): Tensor names quantized together in this
            group.
        bytes_fp16 (int): Group size in bytes at reference precision.
        sensitivity (Mapping[int, float]): Measured damage per candidate
            precision.

    Examples:
        A group whose damage doubles from 4-bit to 2-bit:

        ```python
        from quantfit.domain.model import LayerGroup

        group = LayerGroup(
            name="model.layers.0.mlp",
            tensors=("gate_proj", "up_proj", "down_proj"),
            bytes_fp16=1000,
            sensitivity={8: 0.0, 4: 0.1, 2: 0.2},
        )
        ```
    """

    name: str
    tensors: tuple[str, ...]
    bytes_fp16: int
    sensitivity: Mapping[int, float] = field(hash=False)

    def __post_init__(self) -> None:
        """Enforce group invariants and freeze the damage curve.

        The sensitivity mapping is defensively copied and wrapped in a
        read-only proxy, so a caller's dict cannot alias or mutate a
        frozen group.

        Raises:
            ValueError: If ``bytes_fp16`` is not positive.
        """
        if self.bytes_fp16 <= 0:
            raise ValueError("bytes_fp16 must be positive")
        object.__setattr__(
            self, "sensitivity", MappingProxyType(dict(self.sensitivity))
        )


@dataclass(frozen=True, slots=True)
class SensitivityMap:
    """The output of ``quantfit scan``: damage curves for every group.

    Attributes:
        model_id (str): The scanned model's identifier.
        scan (ScanMeta): Scan provenance.
        groups (tuple[LayerGroup, ...]): All scanned layer groups.

    Examples:
        Look up one group's damage at 4-bit:

        ```python
        damage = map_.groups[0].sensitivity[4]
        ```
    """

    model_id: str
    scan: ScanMeta
    groups: tuple[LayerGroup, ...]

    def __post_init__(self) -> None:
        """Enforce the cross-group invariants the solver relies on.

        Raises:
            ValueError: If ``groups`` is empty, group names collide, or
                any group's sensitivity keys differ from
                ``scan.precisions``.
        """
        if not self.groups:
            raise ValueError("groups must not be empty")
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("group names must be unique")
        expected = set(self.scan.precisions)
        for g in self.groups:
            if set(g.sensitivity) != expected:
                raise ValueError(
                    f'group "{g.name}" sensitivity keys must equal scan.precisions'
                )


@dataclass(frozen=True, slots=True)
class Assignment:
    """One group's chosen precision in a recipe.

    Attributes:
        group (str): The layer group's name.
        bits (int): Assigned precision.
        bytes (int): Predicted group size at that precision, including
            quantization-format overhead.
        damage (float): Measured damage at the assigned precision.

    Examples:
        A group kept at 8-bit:

        ```python
        from quantfit.domain.model import Assignment

        a = Assignment(group="model.embed", bits=8, bytes=420, damage=0.002)
        ```
    """

    group: str
    bits: int
    bytes: int
    damage: float


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One downgrade step in the solver's explanation trace.

    Attributes:
        step (int): 1-based step number.
        group (str): Name of the downgraded group.
        from_bits (int): Precision before the step.
        to_bits (int): Precision after the step.
        damage_delta (float): Damage added by the step (may be negative).
        bytes_freed (int): Bytes saved by the step.
        ratio (float): ``damage_delta / bytes_freed``, the greedy
            selection key.

    Examples:
        The first downgrade of a solve:

        ```python
        from quantfit.domain.model import TraceStep

        step = TraceStep(
            step=1,
            group="model.layers.7.mlp",
            from_bits=8,
            to_bits=4,
            damage_delta=0.004,
            bytes_freed=2000,
            ratio=0.000002,
        )
        ```
    """

    step: int
    group: str
    from_bits: int
    to_bits: int
    damage_delta: float
    bytes_freed: int
    ratio: float


@dataclass(frozen=True, slots=True)
class PlanMeta:
    """Budget accounting and provenance for a recipe.

    Attributes:
        vram_budget_bytes (int): Total VRAM ceiling the plan was made for.
        kv_headroom_bytes (int): Bytes reserved for KV cache and runtime.
        weight_budget_bytes (int): Bytes available for weights.
        predicted_total_bytes (int): Sum of assignment sizes.
        predicted_damage (float): Sum of assignment damage values.
        solver (str): Name of the solver that produced the recipe.
        pins (Mapping[str, int]): User pin patterns, verbatim; later
            patterns override earlier ones.
        format_overhead (float): Quantization-format overhead fraction
            used for size predictions.
        trace (tuple[TraceStep, ...]): Ordered downgrade steps explaining
            the recipe.

    Examples:
        Minimal plan metadata:

        ```python
        from quantfit.domain.model import PlanMeta

        meta = PlanMeta(
            vram_budget_bytes=100,
            kv_headroom_bytes=10,
            weight_budget_bytes=90,
            predicted_total_bytes=80,
            predicted_damage=0.5,
            solver="greedy-damage-per-byte",
            pins={},
            format_overhead=0.05,
            trace=(),
        )
        ```
    """

    vram_budget_bytes: int
    kv_headroom_bytes: int
    weight_budget_bytes: int
    predicted_total_bytes: int
    predicted_damage: float
    solver: str
    pins: Mapping[str, int] = field(hash=False)
    format_overhead: float = field(hash=False)
    trace: tuple[TraceStep, ...] = field(hash=False)

    def __post_init__(self) -> None:
        """Freeze the pin record so it cannot alias a caller's dict."""
        object.__setattr__(self, "pins", MappingProxyType(dict(self.pins)))


@dataclass(frozen=True, slots=True)
class Recipe:
    """The output of ``quantfit plan``: one precision per layer group.

    Construction validates the invariants the pack step relies on — a
    recipe with duplicate assignment groups cannot exist, however it
    was constructed (the JSON loader adds path-aware reporting on
    top).

    Attributes:
        model_id (str): The target model's identifier.
        plan (PlanMeta): Budget accounting and provenance.
        assignments (tuple[Assignment, ...]): One entry per layer group,
            in sensitivity-map order.
        runtime (str | None): Target runtime the recipe was planned
            for (ADR-0013), or None for an unconstrained plan. No
            default — every constructor states its intent. The
            capability table lives in [quantfit.domain.runtime][].

    Examples:
        Inspect a recipe's predicted size:

        ```python
        print(recipe.plan.predicted_total_bytes)
        ```
    """

    model_id: str
    plan: PlanMeta
    assignments: tuple[Assignment, ...]
    runtime: str | None

    def __post_init__(self) -> None:
        """Enforce the recipe invariants the pack step relies on.

        Raises:
            ValueError: If ``assignments`` is empty or two assignments
                name the same group.
        """
        if not self.assignments:
            raise ValueError("assignments must not be empty")
        names = [a.group for a in self.assignments]
        if len(set(names)) != len(names):
            raise ValueError("assignment groups must be unique")
