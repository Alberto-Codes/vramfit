"""Domain types for the two pipeline artifacts.

The dataclasses enforce their own structural invariants in
``__post_init__``, so an instance that exists is safe for the solver,
however it was constructed. The checks cover positive sizes, strictly
descending precisions, unique group names, sensitivity keys matching
the scan, imatrix provenance pairing with the assisted method token,
tensor sizes covering exactly the group's tensors, protection records
pairing with their resolved pairs (ADR-0022), an ordered imatrix
count summary (ADR-0026 decision 4), and a non-empty derived note
(#136). The
within-group method tokens live here — `SCAN_METHOD` beside the
`ScanMeta` field it is the default for (ADR-0018), the kquant
tokens beside the `imatrix` invariant they anchor (ADR-0020), and
`Q0_REF_METHOD` for the block quantizers (ADR-0018, 2026-08-17
amendment).
Serialization and the JSON schema envelope (including the
``vramfit_schema`` version field) live in
[vramfit.adapters.outbound.sensitivity_map_json][] and
[vramfit.adapters.outbound.recipe_json][], whose loaders add JSON-path
error reporting on top of these checks.

Examples:
    Build a one-group map in memory:

    ```python
    from vramfit.domain.model import LayerGroup, ScanMeta, SensitivityMap

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
    - [vramfit.domain.solver][]: Consumes `SensitivityMap`, builds `Recipe`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

# The v1 within-group method token (ADR-0006): round-to-nearest,
# 32-element scale blocks. Defined beside `ScanMeta`, whose default
# it is. [vramfit.domain.scan][] re-exports every method token so
# they read from one module.
SCAN_METHOD = "rtn-block32"
# The K-quant-faithful method (ADR-0018): llama.cpp reference
# quantizers ported to torch — Q2_K, Q3_K, Q4_K, Q8_0.
KQUANT_METHOD = "kquant-ref"
# The imatrix-assisted K-quant method (ADR-0020): the same port,
# with the pack's imatrix weighting every covered fit. Defined here
# because `ScanMeta` pairs the token with its `imatrix` field.
KQUANT_IMX_METHOD = "kquant-imx"
# The GGUF block-quantizer method (ADR-0018, 2026-08-17 amendment):
# Q2_0, Q4_0, and Q8_0 ported from llama.cpp b10326. It prices the
# rows no K-quant reaches. `q0-imx` stays reserved — its assisted
# path is unbuilt.
Q0_REF_METHOD = "q0-ref"


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
        group_by (str): Grouping granularity — ``layer``, ``tensor``,
            or ``stack``. ``stack`` keys on the unit a pack addresses.
            It collapses a mixture-of-experts layer's *routed* experts
            into one group per projection. Every other weight stays
            separate, including the shared expert and the router.
        started_at (str): ISO-8601 timestamp of the scan start.
        within_group (str): Within-group method token (ADR-0018) —
            `SCAN_METHOD` (``rtn-block32``) unless the scan selected
            another method. `Q0_REF_METHOD` (``q0-ref``) prices
            the rows no K-quant reaches.
        imatrix (str | None): Path of the imatrix that assisted the
            scan (ADR-0020), or None for an unassisted scan. Pairs
            with the `KQUANT_IMX_METHOD` token — a map cannot claim
            assistance without naming its imatrix, or the reverse.

    Examples:
        The scan section of a map:

        ```python
        from vramfit.domain.model import ScanMeta

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
    group_by: Literal["layer", "tensor", "stack"]
    started_at: str
    within_group: str = SCAN_METHOD
    imatrix: str | None = None

    def __post_init__(self) -> None:
        """Enforce the scan invariants the solver relies on.

        Raises:
            ValueError: If ``calibration_tokens`` is not positive,
                ``within_group`` is empty, ``precisions`` is empty,
                non-positive, or not strictly descending, or
                ``imatrix`` does not pair with the assisted method
                token — assisted damages without their imatrix
                provenance are not comparable to anything
                (ADR-0020).
        """
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if not self.within_group:
            raise ValueError("within_group must not be empty")
        if self.imatrix is not None and not self.imatrix:
            raise ValueError("imatrix must not be empty — use None when unassisted")
        if self.within_group == KQUANT_IMX_METHOD and self.imatrix is None:
            raise ValueError(
                f'within_group "{KQUANT_IMX_METHOD}" requires the imatrix '
                "field (ADR-0020)"
            )
        if self.imatrix is not None and self.within_group != KQUANT_IMX_METHOD:
            raise ValueError(
                f'imatrix provenance requires within_group "{KQUANT_IMX_METHOD}", '
                f'got "{self.within_group}" (ADR-0020)'
            )
        if not self.precisions:
            raise ValueError("precisions must not be empty")
        if any(p <= 0 for p in self.precisions):
            raise ValueError("precisions must all be positive")
        if not all(a > b for a, b in itertools.pairwise(self.precisions)):
            raise ValueError("precisions must be strictly descending")


@dataclass(frozen=True, slots=True)
class ImatrixCountSummary:
    """A group's pooled imatrix count distribution (ADR-0026 decision 4).

    The reduction pools the count vectors of the group's fused
    expert stacks and nothing else (ADR-0026, 2026-08-13 #201
    amendment). The numbers are provenance, not a gate — they let a
    later data point challenge decision 1 with evidence instead of
    re-deriving the distribution.

    Attributes:
        min (int): The smallest pooled count.
        median (float): The pooled median. A float in every case —
            ``statistics.median`` returns an ``int`` for odd-length
            integer input, and one field must not write two JSON
            types.
        max (int): The largest pooled count.

    Examples:
        The published matrix's routed-expert distribution:

        ```python
        from vramfit.domain.model import ImatrixCountSummary

        summary = ImatrixCountSummary(min=426, median=18114.0, max=192191)
        ```
    """

    min: int
    median: float
    max: int

    def __post_init__(self) -> None:
        """Enforce the summary invariants and coerce the median.

        The median coerces to ``float`` here, so an in-memory
        constructor cannot reintroduce the two-JSON-types defect the
        attribute documents.

        Raises:
            ValueError: If ``median`` is too large for a float, if
                ``min`` is negative, if ``median`` is not finite, or
                if the three values are not ordered
                ``min <= median <= max`` — an unordered summary
                cannot come from one pooled distribution.
        """
        try:
            object.__setattr__(self, "median", float(self.median))
        except OverflowError as exc:
            # Unreachable through the readers, which bound an integer
            # before it arrives (#260). An in-memory caller building
            # from an unvalidated source is the open route, and
            # `OverflowError` is not a `ValueError` (ADR-0011).
            raise ValueError("median is too large for a float") from exc
        if self.min < 0:
            raise ValueError("min must not be negative")
        if not math.isfinite(self.median):
            raise ValueError("median must be finite")
        if not self.min <= self.median <= self.max:
            raise ValueError("summary must be ordered: min <= median <= max")


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
        tensor_bytes (Mapping[str, int]): Each member tensor's bytes at
            reference precision (ADR-0022), or empty when the map
            predates the field. Protections price against these — a
            plan refuses a protection on a group without them.
        imatrix_counts (ImatrixCountSummary | None): The pooled count
            distribution of the group's fused expert stacks (ADR-0026
            decision 4), or None. The field is all-or-nothing per
            group: it appears only when every expert-stack member
            resolved its full count vector, and a group without an
            expert stack never carries it (the 2026-08-13 #201
            amendment).

    Examples:
        A group whose damage doubles from 4-bit to 2-bit:

        ```python
        from vramfit.domain.model import LayerGroup

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
    tensor_bytes: Mapping[str, int] = field(hash=False, default_factory=dict)
    imatrix_counts: ImatrixCountSummary | None = None

    def __post_init__(self) -> None:
        """Enforce group invariants and freeze the mappings.

        The sensitivity and tensor-size mappings are defensively
        copied and wrapped in read-only proxies, so a caller's dict
        cannot alias or mutate a frozen group.

        Raises:
            ValueError: If ``bytes_fp16`` is not positive, or a
                non-empty ``tensor_bytes`` does not cover exactly the
                group's tensors with positive sizes summing to
                ``bytes_fp16`` — a partial or inconsistent size
                record would misprice protections silently
                (ADR-0022).
        """
        if self.bytes_fp16 <= 0:
            raise ValueError("bytes_fp16 must be positive")
        object.__setattr__(
            self, "sensitivity", MappingProxyType(dict(self.sensitivity))
        )
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
class SensitivityMap:
    """The output of ``vramfit scan``: damage curves for every group.

    Attributes:
        model_id (str): The scanned model's identifier.
        scan (ScanMeta): Scan provenance.
        groups (tuple[LayerGroup, ...]): All scanned layer groups.
        derived (str | None): Why this map is not a scan artifact —
            the edit that produced it and what it is for (#136), or
            None for a map ``vramfit scan`` wrote. A hand-made copy
            (a removed precision column, a backfilled size split)
            reads as a scan artifact without it.

    Examples:
        Look up one group's damage at 4-bit:

        ```python
        damage = map_.groups[0].sensitivity[4]
        ```
    """

    model_id: str
    scan: ScanMeta
    groups: tuple[LayerGroup, ...]
    derived: str | None = None

    def __post_init__(self) -> None:
        """Enforce the map's cross-group invariants and its derived note.

        Raises:
            ValueError: If ``groups`` is empty, group names collide,
                any group's sensitivity keys differ from
                ``scan.precisions``, or ``derived`` is empty — an
                empty note records no provenance, so the map states
                it is derived and then says nothing.
        """
        if not self.groups:
            raise ValueError("groups must not be empty")
        if self.derived == "":
            raise ValueError("derived must not be empty")
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
class ProtectedTensor:
    """One resolved protection: a tensor and the precision it packs at.

    The resolved side of a ``--protect`` rule (ADR-0022): the
    protection floor, recorded only where it exceeds the tensor's
    group assignment. Pack drives these pairs, never the raw floors.
    A floor the assignment already meets resolves to no pair — that
    pair would quantize identically to the unprotected reference and
    falsely fail the reconstruction check.

    Attributes:
        tensor (str): The protected tensor's name, e.g.
            ``model.layers.4.self_attn.v_proj.weight``.
        bits (int): The resolved precision the tensor packs at.
        exclude_imatrix (bool): True when the tensor quantizes
            without its imatrix row (ADR-0023). The imatrix
            exclusion is the fit-collapse remedy that keeps the
            promotion.

    Examples:
        A v_proj held at 5-bit inside a 3-bit group:

        ```python
        from vramfit.domain.model import ProtectedTensor

        pair = ProtectedTensor(tensor="model.layers.4.self_attn.v_proj.weight", bits=5)
        ```
    """

    tensor: str
    bits: int
    exclude_imatrix: bool = False

    def __post_init__(self) -> None:
        """Enforce the pair invariants.

        Raises:
            ValueError: If ``tensor`` is empty or ``bits`` is not
                positive.
        """
        if not self.tensor:
            raise ValueError("tensor must not be empty")
        if self.bits <= 0:
            raise ValueError("bits must be positive")


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
        from vramfit.domain.model import Assignment

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
        from vramfit.domain.model import TraceStep

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
        protections (Mapping[str, int]): User protection patterns,
            verbatim (ADR-0022); later patterns override earlier ones
            for overlapping tensors. The resolved pairs live in
            `Recipe.protected_tensors`.
        format_overhead (float): Overhead fraction used for size
            predictions — a residual over the runtime's effective-bits
            table, or the full quantization-format cost without one
            (ADR-0014).
        trace (tuple[TraceStep, ...]): Ordered downgrade steps explaining
            the recipe.
        imatrix_exclusions (tuple[str, ...]): User exclusion patterns,
            verbatim (ADR-0023). Each marks matched protected tensors
            to quantize without their imatrix rows. The resolved
            tensors carry ``exclude_imatrix`` in
            `Recipe.protected_tensors`.

    Examples:
        Minimal plan metadata:

        ```python
        from vramfit.domain.model import PlanMeta

        meta = PlanMeta(
            vram_budget_bytes=100,
            kv_headroom_bytes=10,
            weight_budget_bytes=90,
            predicted_total_bytes=80,
            predicted_damage=0.5,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
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
    protections: Mapping[str, int] = field(hash=False)
    format_overhead: float = field(hash=False)
    trace: tuple[TraceStep, ...] = field(hash=False)
    imatrix_exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the pin and protection records against aliasing."""
        object.__setattr__(self, "pins", MappingProxyType(dict(self.pins)))
        object.__setattr__(
            self, "protections", MappingProxyType(dict(self.protections))
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    """The output of ``vramfit plan``: one precision per layer group.

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
            capability table lives in [vramfit.domain.runtime][].
        within_group (str | None): The within-group method token of
            the map that priced the recipe (ADR-0019), or None when
            the provenance is unknown — recipes written before the
            field existed. No default — every constructor states
            its intent. The validation pass only checks additivity
            when its method matches this token.
        imatrix (str | None): The imatrix path of the map that
            priced the recipe (ADR-0020), or None for an unassisted
            or unknown-provenance map. Pairs with the
            `KQUANT_IMX_METHOD` token, like `ScanMeta.imatrix` — a
            wrong imatrix file in the validation pass contaminates
            the additivity gap silently.
        protected_tensors (tuple[ProtectedTensor, ...]): Resolved
            protection pairs (ADR-0022), in map order. Empty for a
            recipe without protections. Pack drives these, never the
            raw patterns in ``plan.protections``.

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
    within_group: str | None
    imatrix: str | None
    protected_tensors: tuple[ProtectedTensor, ...]

    def __post_init__(self) -> None:
        """Enforce the recipe invariants the pack step relies on.

        Raises:
            ValueError: If ``assignments`` is empty, two assignments
                name the same group, ``within_group`` or ``imatrix``
                is an empty string — unknown provenance is None,
                never "" — ``imatrix`` does not pair with the
                assisted method token (ADR-0020), two protected
                tensors share a name, resolved pairs exist without a
                protection record (ADR-0022), or the exclusion
                record and the resolved ``exclude_imatrix`` marks
                disagree about whether the recipe excludes imatrix
                rows (ADR-0023). One reverse hole is legal: a
                protection record can resolve to zero pairs when
                every floor is a per-tensor no-op (issue #59). An
                exclusion cannot — the solver refuses a pattern
                whose every pair dropped.
        """
        if self.within_group is not None and not self.within_group:
            raise ValueError("within_group must not be empty — use None for unknown")
        if self.imatrix is not None and not self.imatrix:
            raise ValueError("imatrix must not be empty — use None for unknown")
        if self.within_group == KQUANT_IMX_METHOD and self.imatrix is None:
            raise ValueError(
                f'within_group "{KQUANT_IMX_METHOD}" requires the imatrix '
                "field (ADR-0020)"
            )
        if self.imatrix is not None and self.within_group != KQUANT_IMX_METHOD:
            raise ValueError(
                f'imatrix provenance requires within_group "{KQUANT_IMX_METHOD}", '
                f'got "{self.within_group}" (ADR-0020)'
            )
        if not self.assignments:
            raise ValueError("assignments must not be empty")
        names = [a.group for a in self.assignments]
        if len(set(names)) != len(names):
            raise ValueError("assignment groups must be unique")
        protected = [p.tensor for p in self.protected_tensors]
        if len(set(protected)) != len(protected):
            raise ValueError("protected tensors must be unique")
        if self.protected_tensors and not self.plan.protections:
            raise ValueError(
                "protected_tensors requires plan.protections — resolved "
                "pairs cannot exist without the rules that made them "
                "(ADR-0022)"
            )
        excluded = any(p.exclude_imatrix for p in self.protected_tensors)
        if bool(self.plan.imatrix_exclusions) != excluded:
            raise ValueError(
                "plan.imatrix_exclusions and the exclude_imatrix marks must "
                "both be empty or both be present — the solver refuses an "
                "exclusion whose every pair dropped (ADR-0023, issue #59)"
            )
