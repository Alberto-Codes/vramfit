"""JSON file adapter for the sensitivity-map artifact.

Owns (de)serialization and validation of the map schema, including the
``quantfit_schema`` envelope (`MAP_SCHEMA_VERSION` — schema versions
advance per artifact, ADR-0013). One file class serves both directions:
``quantfit scan`` writes through the sink face, ``quantfit plan`` reads
through the source face. Validation is strict: artifacts are rejected,
never normalized — ``scan.precisions`` must arrive strictly descending,
``group_by`` must be a known granularity, and every group's sensitivity
keys must equal it exactly.

Examples:
    Round-trip a map through a file:

    ```python
    from quantfit.adapters.outbound.sensitivity_map_json import (
        load_sensitivity_map,
        save_sensitivity_map,
    )

    save_sensitivity_map(map_, path)
    assert load_sensitivity_map(path) == map_
    ```

See Also:
    - [quantfit.ports.outbound][]: `SensitivityMapSource`, which
      `JsonSensitivityMapFile` satisfies.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from quantfit.adapters.outbound.json_common import (
    ArtifactError,
    _as_float,
    _as_int,
    _check_schema_version,
    _get_dict,
    _get_int,
    _get_list,
    _get_str,
    _load_json,
    _require,
    _save_json,
)
from quantfit.domain.model import LayerGroup, ScanMeta, SensitivityMap

# The sensitivity-map schema version. Versions advance per artifact
# (ADR-0013) — the recipe sits at 2 while the map stays at 1.
MAP_SCHEMA_VERSION: Final[int] = 1


def map_from_dict(data: object) -> SensitivityMap:
    """Validate parsed JSON and build a `SensitivityMap`.

    Args:
        data: Parsed JSON value, expected to be the artifact's top-level
            object.

    Returns:
        The validated map.

    Raises:
        ArtifactError: If the envelope is not `MAP_SCHEMA_VERSION`, or
            any field is missing, mistyped, or violates a schema rule
            (duplicate group names, unknown ``group_by``, sensitivity
            keys not matching ``scan.precisions``, and so on).

    Examples:
        Reject an unsupported schema version:

        ```python
        map_from_dict({"quantfit_schema": 2})  # raises ArtifactError
        ```
    """
    root = _get_dict(data, "$")
    _check_schema_version(root, "$", expected=MAP_SCHEMA_VERSION)
    model_id = _get_str(root, "model_id", "$")
    _require("scan" in root, "$", 'missing required field "scan"')
    scan = _parse_scan_meta(_get_dict(root["scan"], "$.scan"))
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
    return SensitivityMap(model_id=model_id, scan=scan, groups=tuple(groups))


def map_to_dict(map_: SensitivityMap) -> dict[str, Any]:
    """Serialize a map to a JSON-compatible dict with the schema envelope.

    Args:
        map_: The map to serialize.

    Returns:
        A dict that `map_from_dict` accepts and round-trips to an equal
        map, under the `MAP_SCHEMA_VERSION` envelope. Sensitivity keys
        are stringified in descending-bit order.
    """
    return {
        "quantfit_schema": MAP_SCHEMA_VERSION,
        "model_id": map_.model_id,
        "scan": {
            "metric": map_.scan.metric,
            "calibration": map_.scan.calibration,
            "calibration_tokens": map_.scan.calibration_tokens,
            "precisions": list(map_.scan.precisions),
            "group_by": map_.scan.group_by,
            "started_at": map_.scan.started_at,
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
            for g in map_.groups
        ],
    }


def load_sensitivity_map(path: Path) -> SensitivityMap:
    """Read and validate a sensitivity-map file.

    Args:
        path: JSON file written by ``quantfit scan`` or
            `save_sensitivity_map`.

    Returns:
        The validated map.

    Raises:
        ArtifactError: If the file is not valid JSON or fails validation.
    """
    return map_from_dict(_load_json(path, "$"))


def save_sensitivity_map(map_: SensitivityMap, path: Path) -> None:
    """Write a map as pretty-printed JSON.

    Args:
        map_: The map to write.
        path: Destination file.
    """
    _save_json(map_to_dict(map_), path)


@dataclass(frozen=True, slots=True)
class JsonSensitivityMapFile:
    """`SensitivityMapSource` and `SensitivityMapSink` adapter for one file.

    Attributes:
        path (Path): The file to read or write.

    Examples:
        Use as a port implementation:

        ```python
        source = JsonSensitivityMapFile(Path("sensitivity.json"))
        map_ = source.load()
        ```
    """

    path: Path

    def load(self) -> SensitivityMap:
        """Load and validate the sensitivity map from `path`.

        Returns:
            The validated map.

        Raises:
            ArtifactError: If the file is not valid JSON or fails
                validation.
        """
        return load_sensitivity_map(self.path)

    def save(self, map_: SensitivityMap) -> None:
        """Write the map to `path` as pretty-printed JSON.

        Args:
            map_: The map to persist.
        """
        save_sensitivity_map(map_, self.path)


def _parse_scan_meta(obj: dict[str, Any]) -> ScanMeta:
    """Validate the ``scan`` section of a sensitivity map.

    Args:
        obj: The ``scan`` JSON object.

    Returns:
        The validated scan provenance.

    Raises:
        ArtifactError: If a field is missing or invalid, precisions are
            empty, duplicated, not integers, or not strictly descending,
            or ``group_by`` is not ``layer`` or ``tensor``.
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
    _require(
        all(a > b for a, b in itertools.pairwise(precisions)),
        f"{path}.precisions",
        "must be strictly descending",
    )
    group_by = _get_str(obj, "group_by", path)
    _require(
        group_by in ("layer", "tensor"),
        f"{path}.group_by",
        'must be "layer" or "tensor"',
    )
    return ScanMeta(
        metric=_get_str(obj, "metric", path),
        calibration=_get_str(obj, "calibration", path),
        calibration_tokens=tokens,
        precisions=tuple(precisions),
        group_by=cast('Literal["layer", "tensor"]', group_by),
        started_at=_get_str(obj, "started_at", path),
    )


def _parse_sensitivity(obj: dict[str, Any], path: str) -> dict[int, float]:
    """Validate one group's sensitivity mapping.

    Keys are coerced from JSON strings to ints; encodings that collide
    after coercion (``"4"`` and ``"04"``) are rejected rather than
    silently overwritten.

    Args:
        obj: The ``sensitivity`` JSON object.
        path: JSON path of the sensitivity object.

    Returns:
        Mapping of precision to finite damage value.

    Raises:
        ArtifactError: If a key is not an integer, keys collide after
            coercion, or a value is not a finite number.
    """
    sensitivity: dict[int, float] = {}
    for key, value in obj.items():
        try:
            bits = int(key)
        except ValueError:
            raise ArtifactError(
                path, f'key "{key}" is not an integer precision'
            ) from None
        _require(
            bits not in sensitivity,
            path,
            f'duplicate precision key "{key}" after integer coercion',
        )
        sensitivity[bits] = _as_float(value, f"{path}.{key}")
    return sensitivity


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
        ArtifactError: If a field is missing or invalid, ``bytes_fp16``
            is not positive, or the sensitivity keys do not match the
            scan's precisions.
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
    sensitivity = _parse_sensitivity(
        _get_dict(obj["sensitivity"], f"{path}.sensitivity"), f"{path}.sensitivity"
    )
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
