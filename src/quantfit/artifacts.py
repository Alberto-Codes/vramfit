"""Load, validate, and save the quantfit JSON artifacts.

The pipeline's two artifacts are the sensitivity map (output of ``scan``,
input to ``plan``) and the recipe (output of ``plan``, input to ``pack``).
Both carry a ``quantfit_schema`` version field. This module reads version
``1`` and rejects everything else.

Attributes:
    SCHEMA_VERSION (int): The artifact schema version this module reads
        and writes.

Examples:
    Round-trip a sensitivity map:

    ```python
    from pathlib import Path

    from quantfit.artifacts import SensitivityMap

    map_ = SensitivityMap.load(Path("sensitivity.json"))
    map_.save(Path("copy.json"))
    assert SensitivityMap.load(Path("copy.json")) == map_
    ```

See Also:
    - [quantfit.solver][]: Consumes a `SensitivityMap`, produces a `Recipe`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final[int] = 1


class ArtifactError(ValueError):
    """Invalid quantfit JSON artifact.

    Attributes:
        json_path (str): Dotted path to the offending element, e.g.
            ``groups[3].sensitivity``.
        message (str): Human-readable description of the problem.

    Examples:
        Catch and report a validation failure:

        ```python
        from quantfit.artifacts import ArtifactError, SensitivityMap

        try:
            SensitivityMap.from_dict({"quantfit_schema": 2})
        except ArtifactError as exc:
            print(exc.json_path, exc.message)
        ```
    """

    def __init__(self, json_path: str, message: str) -> None:
        """Build the error from a JSON path and a message.

        Args:
            json_path: Dotted path to the offending element.
            message: Human-readable description of the problem.
        """
        super().__init__(f"{json_path}: {message}")
        self.json_path = json_path
        self.message = message


def _require(condition: bool, path: str, message: str) -> None:
    """Raise `ArtifactError` at ``path`` unless ``condition`` holds.

    Args:
        condition: The invariant that must be true.
        path: JSON path reported on failure.
        message: Failure description.

    Raises:
        ArtifactError: If ``condition`` is false.
    """
    if not condition:
        raise ArtifactError(path, message)


def _get_dict(obj: Any, path: str) -> dict[str, Any]:
    """Return ``obj`` as a JSON object.

    Args:
        obj: Candidate value.
        path: JSON path for error reporting.

    Returns:
        The value, typed as a dict.

    Raises:
        ArtifactError: If the value is not a JSON object.
    """
    _require(isinstance(obj, dict), path, "expected a JSON object")
    return obj


def _get_list(obj: dict[str, Any], key: str, path: str) -> list[Any]:
    """Return the list stored at ``key``.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The list value.

    Raises:
        ArtifactError: If the key is missing or not a list.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    value = obj[key]
    _require(isinstance(value, list), f"{path}.{key}", "expected a list")
    return value


def _get_str(obj: dict[str, Any], key: str, path: str) -> str:
    """Return the non-empty string stored at ``key``.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The string value.

    Raises:
        ArtifactError: If the key is missing, not a string, or empty.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    value = obj[key]
    _require(isinstance(value, str), f"{path}.{key}", "expected a string")
    _require(value != "", f"{path}.{key}", "must not be empty")
    return value


def _as_int(value: Any, path: str) -> int:
    """Return ``value`` as an int, rejecting JSON booleans.

    Args:
        value: Candidate value.
        path: JSON path for error reporting.

    Returns:
        The integer value.

    Raises:
        ArtifactError: If the value is a bool or not an integer.
    """
    _require(not isinstance(value, bool), path, "expected an integer, got a boolean")
    _require(isinstance(value, int), path, "expected an integer")
    return value


def _get_int(obj: dict[str, Any], key: str, path: str) -> int:
    """Return the integer stored at ``key``.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The integer value.

    Raises:
        ArtifactError: If the key is missing or not an integer.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    return _as_int(obj[key], f"{path}.{key}")


def _as_float(value: Any, path: str) -> float:
    """Return ``value`` as a float, rejecting JSON booleans.

    Args:
        value: Candidate value.
        path: JSON path for error reporting.

    Returns:
        The numeric value as a float.

    Raises:
        ArtifactError: If the value is a bool or not a number.
    """
    _require(not isinstance(value, bool), path, "expected a number, got a boolean")
    _require(isinstance(value, (int, float)), path, "expected a number")
    return float(value)


def _get_float(obj: dict[str, Any], key: str, path: str) -> float:
    """Return the number stored at ``key`` as a float.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The numeric value.

    Raises:
        ArtifactError: If the key is missing or not a number.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    return _as_float(obj[key], f"{path}.{key}")


def _check_schema_version(obj: dict[str, Any], path: str) -> int:
    """Validate the artifact's ``quantfit_schema`` field.

    Args:
        obj: Top-level artifact object.
        path: JSON path of the artifact root.

    Returns:
        The validated schema version.

    Raises:
        ArtifactError: If the version is missing or unsupported.
    """
    version = _get_int(obj, "quantfit_schema", path)
    _require(
        version == SCHEMA_VERSION,
        f"{path}.quantfit_schema",
        f"unsupported schema version {version}; this quantfit reads version "
        f"{SCHEMA_VERSION}",
    )
    return version


def _load_json(path: Path, root: str) -> dict[str, Any]:
    """Read ``path`` and return its top-level JSON object.

    Args:
        path: File to read.
        root: JSON path name for the artifact root in error messages.

    Returns:
        The parsed top-level object.

    Raises:
        ArtifactError: If the file is not valid JSON or the top level is
            not an object.
    """
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(root, f"invalid JSON: {exc}") from exc
    return _get_dict(data, root)


def _save_json(data: dict[str, Any], path: Path) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON.

    Args:
        data: JSON-serializable object.
        path: Destination file.
    """
    path.write_text(json.dumps(data, indent=2) + "\n")


@dataclass(frozen=True, slots=True)
class ScanMeta:
    """Provenance of a sensitivity scan.

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
        from quantfit.artifacts import ScanMeta

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
    group_by: str
    started_at: str


@dataclass(frozen=True, slots=True)
class LayerGroup:
    """One scanned layer group and its damage curve.

    Attributes:
        name (str): Unique group name, e.g. ``model.layers.0.self_attn``.
        tensors (tuple[str, ...]): Tensor names quantized together in this
            group.
        bytes_fp16 (int): Group size in bytes at reference precision.
        sensitivity (dict[int, float]): Measured damage per candidate
            precision.

    Examples:
        A group whose damage doubles from 4-bit to 2-bit:

        ```python
        from quantfit.artifacts import LayerGroup

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
    sensitivity: dict[int, float] = field(hash=False)


@dataclass(frozen=True, slots=True)
class SensitivityMap:
    """The output of ``quantfit scan``: damage curves for every group.

    Attributes:
        quantfit_schema (int): Artifact schema version.
        model_id (str): The scanned model's identifier.
        scan (ScanMeta): Scan provenance.
        groups (tuple[LayerGroup, ...]): All scanned layer groups.

    Examples:
        Build a map from parsed JSON:

        ```python
        from quantfit.artifacts import SensitivityMap

        map_ = SensitivityMap.from_dict(
            {
                "quantfit_schema": 1,
                "model_id": "m",
                "scan": {
                    "metric": "kl_divergence",
                    "calibration": "wikitext",
                    "calibration_tokens": 1024,
                    "precisions": [8, 4],
                    "group_by": "layer",
                    "started_at": "2026-07-27T00:00:00Z",
                },
                "groups": [
                    {
                        "name": "g0",
                        "tensors": ["w"],
                        "bytes_fp16": 1000,
                        "sensitivity": {"8": 0.0, "4": 0.1},
                    }
                ],
            }
        )
        ```
    """

    quantfit_schema: int
    model_id: str
    scan: ScanMeta
    groups: tuple[LayerGroup, ...]

    @classmethod
    def from_dict(cls, data: object) -> SensitivityMap:
        """Validate parsed JSON and build a `SensitivityMap`.

        Args:
            data: Parsed JSON value, expected to be the artifact's
                top-level object.

        Returns:
            The validated map.

        Raises:
            ArtifactError: If any field is missing, mistyped, or violates
                a schema rule (duplicate group names, sensitivity keys not
                matching ``scan.precisions``, and so on).
        """
        root = _get_dict(data, "$")
        version = _check_schema_version(root, "$")
        model_id = _get_str(root, "model_id", "$")
        scan = _parse_scan_meta(_get_dict(root.get("scan"), "$.scan"))
        groups_raw = _get_list(root, "groups", "$")
        _require(len(groups_raw) > 0, "$.groups", "must not be empty")
        expected = set(scan.precisions)
        groups: list[LayerGroup] = []
        seen: set[str] = set()
        for i, raw in enumerate(groups_raw):
            group = _parse_layer_group(raw, f"$.groups[{i}]", expected)
            _require(
                group.name not in seen,
                f"$.groups[{i}].name",
                f'duplicate group name "{group.name}"',
            )
            seen.add(group.name)
            groups.append(group)
        return cls(
            quantfit_schema=version,
            model_id=model_id,
            scan=scan,
            groups=tuple(groups),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns:
            A dict that `from_dict` accepts and round-trips to an equal
            map. Sensitivity keys are stringified in descending-bit order.
        """
        return {
            "quantfit_schema": self.quantfit_schema,
            "model_id": self.model_id,
            "scan": {
                "metric": self.scan.metric,
                "calibration": self.scan.calibration,
                "calibration_tokens": self.scan.calibration_tokens,
                "precisions": list(self.scan.precisions),
                "group_by": self.scan.group_by,
                "started_at": self.scan.started_at,
            },
            "groups": [
                {
                    "name": g.name,
                    "tensors": list(g.tensors),
                    "bytes_fp16": g.bytes_fp16,
                    "sensitivity": {
                        str(bits): g.sensitivity[bits]
                        for bits in sorted(g.sensitivity, reverse=True)
                    },
                }
                for g in self.groups
            ],
        }

    @classmethod
    def load(cls, path: Path) -> SensitivityMap:
        """Read and validate a sensitivity map file.

        Args:
            path: JSON file written by ``quantfit scan`` or `save`.

        Returns:
            The validated map.

        Raises:
            ArtifactError: If the file is not valid JSON or fails
                validation.
        """
        return cls.from_dict(_load_json(path, "$"))

    def save(self, path: Path) -> None:
        """Write the map as pretty-printed JSON.

        Args:
            path: Destination file.
        """
        _save_json(self.to_dict(), path)


def _parse_scan_meta(obj: dict[str, Any]) -> ScanMeta:
    """Validate the ``scan`` section of a sensitivity map.

    Args:
        obj: The ``scan`` JSON object.

    Returns:
        The validated scan provenance.

    Raises:
        ArtifactError: If a field is missing or invalid, or precisions
            are empty, duplicated, or not integers.
    """
    path = "$.scan"
    tokens = _get_int(obj, "calibration_tokens", path)
    _require(tokens > 0, f"{path}.calibration_tokens", "must be positive")
    raw_precisions = _get_list(obj, "precisions", path)
    _require(len(raw_precisions) > 0, f"{path}.precisions", "must not be empty")
    precisions = [
        _as_int(p, f"{path}.precisions[{i}]") for i, p in enumerate(raw_precisions)
    ]
    _require(
        len(set(precisions)) == len(precisions),
        f"{path}.precisions",
        "must not contain duplicates",
    )
    _require(
        all(p > 0 for p in precisions), f"{path}.precisions", "must all be positive"
    )
    return ScanMeta(
        metric=_get_str(obj, "metric", path),
        calibration=_get_str(obj, "calibration", path),
        calibration_tokens=tokens,
        precisions=tuple(sorted(precisions, reverse=True)),
        group_by=_get_str(obj, "group_by", path),
        started_at=_get_str(obj, "started_at", path),
    )


def _parse_layer_group(
    raw: Any, path: str, expected_precisions: set[int]
) -> LayerGroup:
    """Validate one entry of a sensitivity map's ``groups`` list.

    Args:
        raw: The group's JSON value.
        path: JSON path of this group.
        expected_precisions: The scan's candidate precisions; the group's
            sensitivity keys must equal this set.

    Returns:
        The validated group.

    Raises:
        ArtifactError: If a field is missing or invalid, ``bytes_fp16`` is
            not positive, or the sensitivity keys do not match the scan's
            precisions.
    """
    obj = _get_dict(raw, path)
    bytes_fp16 = _get_int(obj, "bytes_fp16", path)
    _require(bytes_fp16 > 0, f"{path}.bytes_fp16", "must be positive")
    tensors_raw = _get_list(obj, "tensors", path)
    tensors = []
    for i, tensor in enumerate(tensors_raw):
        _require(isinstance(tensor, str), f"{path}.tensors[{i}]", "expected a string")
        tensors.append(tensor)
    _require("sensitivity" in obj, path, 'missing required field "sensitivity"')
    sens_obj = _get_dict(obj["sensitivity"], f"{path}.sensitivity")
    sensitivity: dict[int, float] = {}
    for key, value in sens_obj.items():
        try:
            bits = int(key)
        except ValueError:
            raise ArtifactError(
                f"{path}.sensitivity",
                f'key "{key}" is not an integer precision',
            ) from None
        sensitivity[bits] = _as_float(value, f"{path}.sensitivity.{key}")
    _require(
        set(sensitivity) == expected_precisions,
        f"{path}.sensitivity",
        f"keys {sorted(sensitivity, reverse=True)} must equal scan.precisions "
        f"{sorted(expected_precisions, reverse=True)}",
    )
    return LayerGroup(
        name=_get_str(obj, "name", path),
        tensors=tuple(tensors),
        bytes_fp16=bytes_fp16,
        sensitivity=sensitivity,
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
        from quantfit.artifacts import Assignment

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
        from quantfit.artifacts import TraceStep

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
        pins (dict[str, int]): User pin patterns, verbatim.
        format_overhead (float): Quantization-format overhead fraction
            used for size predictions.
        trace (tuple[TraceStep, ...]): Ordered downgrade steps explaining
            the recipe.

    Examples:
        Minimal plan metadata:

        ```python
        from quantfit.artifacts import PlanMeta

        meta = PlanMeta(
            vram_budget_bytes=100,
            kv_headroom_bytes=10,
            weight_budget_bytes=90,
            predicted_total_bytes=80,
            predicted_damage=0.5,
            solver="greedy-damage-per-byte",
            pins={},
        )
        ```
    """

    vram_budget_bytes: int
    kv_headroom_bytes: int
    weight_budget_bytes: int
    predicted_total_bytes: int
    predicted_damage: float
    solver: str
    pins: dict[str, int] = field(hash=False)
    format_overhead: float = 0.05
    trace: tuple[TraceStep, ...] = ()


@dataclass(frozen=True, slots=True)
class Recipe:
    """The output of ``quantfit plan``: one precision per layer group.

    Attributes:
        quantfit_schema (int): Artifact schema version.
        model_id (str): The target model's identifier.
        plan (PlanMeta): Budget accounting and provenance.
        assignments (tuple[Assignment, ...]): One entry per layer group,
            in sensitivity-map order.

    Examples:
        Load a recipe and inspect its budget:

        ```python
        from pathlib import Path

        from quantfit.artifacts import Recipe

        recipe = Recipe.load(Path("recipe.json"))
        print(recipe.plan.predicted_total_bytes)
        ```
    """

    quantfit_schema: int
    model_id: str
    plan: PlanMeta
    assignments: tuple[Assignment, ...]

    @classmethod
    def from_dict(cls, data: object) -> Recipe:
        """Validate parsed JSON and build a `Recipe`.

        Args:
            data: Parsed JSON value, expected to be the artifact's
                top-level object.

        Returns:
            The validated recipe.

        Raises:
            ArtifactError: If any field is missing, mistyped, or violates
                a schema rule.
        """
        root = _get_dict(data, "$")
        version = _check_schema_version(root, "$")
        model_id = _get_str(root, "model_id", "$")
        plan = _parse_plan_meta(_get_dict(root.get("plan"), "$.plan"))
        raw_assignments = _get_list(root, "assignments", "$")
        _require(len(raw_assignments) > 0, "$.assignments", "must not be empty")
        assignments: list[Assignment] = []
        seen: set[str] = set()
        for i, raw in enumerate(raw_assignments):
            path = f"$.assignments[{i}]"
            obj = _get_dict(raw, path)
            assignment = Assignment(
                group=_get_str(obj, "group", path),
                bits=_get_int(obj, "bits", path),
                bytes=_get_int(obj, "bytes", path),
                damage=_get_float(obj, "damage", path),
            )
            _require(
                assignment.group not in seen,
                f"{path}.group",
                f'duplicate group "{assignment.group}"',
            )
            seen.add(assignment.group)
            assignments.append(assignment)
        return cls(
            quantfit_schema=version,
            model_id=model_id,
            plan=plan,
            assignments=tuple(assignments),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns:
            A dict that `from_dict` accepts and round-trips to an equal
            recipe.
        """
        return {
            "quantfit_schema": self.quantfit_schema,
            "model_id": self.model_id,
            "plan": {
                "vram_budget_bytes": self.plan.vram_budget_bytes,
                "kv_headroom_bytes": self.plan.kv_headroom_bytes,
                "weight_budget_bytes": self.plan.weight_budget_bytes,
                "predicted_total_bytes": self.plan.predicted_total_bytes,
                "predicted_damage": self.plan.predicted_damage,
                "solver": self.plan.solver,
                "pins": dict(self.plan.pins),
                "format_overhead": self.plan.format_overhead,
                "trace": [
                    {
                        "step": t.step,
                        "group": t.group,
                        "from_bits": t.from_bits,
                        "to_bits": t.to_bits,
                        "damage_delta": t.damage_delta,
                        "bytes_freed": t.bytes_freed,
                        "ratio": t.ratio,
                    }
                    for t in self.plan.trace
                ],
            },
            "assignments": [
                {
                    "group": a.group,
                    "bits": a.bits,
                    "bytes": a.bytes,
                    "damage": a.damage,
                }
                for a in self.assignments
            ],
        }

    @classmethod
    def load(cls, path: Path) -> Recipe:
        """Read and validate a recipe file.

        Args:
            path: JSON file written by ``quantfit plan`` or `save`.

        Returns:
            The validated recipe.

        Raises:
            ArtifactError: If the file is not valid JSON or fails
                validation.
        """
        return cls.from_dict(_load_json(path, "$"))

    def save(self, path: Path) -> None:
        """Write the recipe as pretty-printed JSON.

        Args:
            path: Destination file.
        """
        _save_json(self.to_dict(), path)


def _parse_plan_meta(obj: dict[str, Any]) -> PlanMeta:
    """Validate the ``plan`` section of a recipe.

    Args:
        obj: The ``plan`` JSON object.

    Returns:
        The validated plan metadata.

    Raises:
        ArtifactError: If a field is missing or invalid.
    """
    path = "$.plan"
    pins_obj = _get_dict(obj.get("pins", {}), f"{path}.pins")
    pins = {
        pattern: _as_int(bits, f"{path}.pins.{pattern}")
        for pattern, bits in pins_obj.items()
    }
    trace_raw = obj.get("trace", [])
    _require(isinstance(trace_raw, list), f"{path}.trace", "expected a list")
    trace: list[TraceStep] = []
    for i, raw in enumerate(trace_raw):
        step_path = f"{path}.trace[{i}]"
        step_obj = _get_dict(raw, step_path)
        trace.append(
            TraceStep(
                step=_get_int(step_obj, "step", step_path),
                group=_get_str(step_obj, "group", step_path),
                from_bits=_get_int(step_obj, "from_bits", step_path),
                to_bits=_get_int(step_obj, "to_bits", step_path),
                damage_delta=_get_float(step_obj, "damage_delta", step_path),
                bytes_freed=_get_int(step_obj, "bytes_freed", step_path),
                ratio=_get_float(step_obj, "ratio", step_path),
            )
        )
    return PlanMeta(
        vram_budget_bytes=_get_int(obj, "vram_budget_bytes", path),
        kv_headroom_bytes=_get_int(obj, "kv_headroom_bytes", path),
        weight_budget_bytes=_get_int(obj, "weight_budget_bytes", path),
        predicted_total_bytes=_get_int(obj, "predicted_total_bytes", path),
        predicted_damage=_get_float(obj, "predicted_damage", path),
        solver=_get_str(obj, "solver", path),
        pins=pins,
        format_overhead=_get_float(obj, "format_overhead", path)
        if "format_overhead" in obj
        else 0.05,
        trace=tuple(trace),
    )
