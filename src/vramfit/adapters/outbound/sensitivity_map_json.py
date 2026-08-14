"""JSON file adapter for the sensitivity-map artifact.

Owns (de)serialization and validation of the map schema, including the
``vramfit_schema`` envelope (`MAP_SCHEMA_VERSION` — schema versions
advance per artifact, ADR-0013). The adapter writes version 3 and
also reads version 2, because version 3 only widened ``group_by``
with the ``stack`` value (#161). One file class serves both directions:
``vramfit scan`` writes through the sink face, ``vramfit plan`` reads
through the source face. Validation is strict: artifacts are rejected,
never normalized — ``scan.precisions`` must arrive strictly descending,
``group_by`` must be a known granularity, and every group's sensitivity
keys must equal it exactly. Two fields are additive rather than strict:
``scan.within_group`` (ADR-0018) defaults to ``rtn-block32`` when
absent, because every map written before the field existed measured
with that method, and ``scan.imatrix`` (ADR-0020) defaults to None,
because every map written before the field existed was unassisted.
A group's ``imatrix_counts`` summary (ADR-0026 decision 4) is
additive the same way: absent means the group records no summary.
A present ``imatrix`` must pair with the assisted method token —
the loader rejects a map whose provenance contradicts itself.

Examples:
    Round-trip a map through a file:

    ```python
    from vramfit.adapters.outbound.sensitivity_map_json import (
        load_sensitivity_map,
        save_sensitivity_map,
    )

    save_sensitivity_map(map_, path)
    assert load_sensitivity_map(path) == map_
    ```

See Also:
    - [vramfit.ports.outbound][]: `SensitivityMapSource`, which
      `JsonSensitivityMapFile` satisfies.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from vramfit.adapters.outbound.json_common import (
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
from vramfit.domain.model import (
    ImatrixCountSummary,
    LayerGroup,
    ScanMeta,
    SensitivityMap,
)
from vramfit.domain.scan import KQUANT_IMX_METHOD, SCAN_METHOD

# The sensitivity-map schema version. Versions advance per artifact
# (ADR-0013) — the recipe sits at 6 while the map sits at 3.
MAP_SCHEMA_VERSION: Final[int] = 3
# Older versions this adapter still reads. Version 3 only widened
# ``group_by`` with the ``stack`` value (#161), so every version-2 map
# is already a valid version-3 document. The writer emits 3, which
# tells a reader the producer could have keyed on stacks.
MAP_SCHEMA_ALSO_READS: Final[tuple[int, ...]] = (2,)


def map_from_dict(data: object) -> SensitivityMap:
    """Validate parsed JSON and build a `SensitivityMap`.

    Args:
        data: Parsed JSON value, expected to be the artifact's top-level
            object.

    Returns:
        The validated map.

    Raises:
        ArtifactError: If the envelope is neither `MAP_SCHEMA_VERSION`
            nor a version in `MAP_SCHEMA_ALSO_READS`, or
            any field is missing, mistyped, or violates a schema rule
            (duplicate group names, unknown ``group_by``, sensitivity
            keys not matching ``scan.precisions``, and so on).

    Examples:
        Reject an unsupported schema version:

        ```python
        map_from_dict({"vramfit_schema": 4})  # raises ArtifactError
        ```
    """
    root = _get_dict(data, "$")
    _check_schema_version(
        root,
        "$",
        expected=MAP_SCHEMA_VERSION,
        also_reads=MAP_SCHEMA_ALSO_READS,
    )
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
        are stringified in descending-bit order. The within-group
        method token is always written, even when it is the default,
        and the imatrix path is always written — null when
        unassisted (ADR-0020). Per-tensor sizes are written only
        when known — an absent field means unknown, never zero
        (ADR-0022). A group's imatrix count summary is written only
        when the group records one (ADR-0026 decision 4).
    """
    return {
        "vramfit_schema": MAP_SCHEMA_VERSION,
        "model_id": map_.model_id,
        "scan": {
            "metric": map_.scan.metric,
            "calibration": map_.scan.calibration,
            "calibration_tokens": map_.scan.calibration_tokens,
            "precisions": list(map_.scan.precisions),
            "group_by": map_.scan.group_by,
            "started_at": map_.scan.started_at,
            "within_group": map_.scan.within_group,
            "imatrix": map_.scan.imatrix,
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
                # Written only when known (ADR-0022) — the field is
                # additive and informational, so the schema stays 1.
                **(
                    {"tensor_bytes": {t: g.tensor_bytes[t] for t in g.tensors}}
                    if g.tensor_bytes
                    else {}
                ),
                # Written only when the group's expert stacks all
                # resolved (ADR-0026 decision 4, the #201 amendment)
                # — additive, so the schema holds at 3.
                **(
                    {
                        "imatrix_counts": {
                            "min": g.imatrix_counts.min,
                            "median": g.imatrix_counts.median,
                            "max": g.imatrix_counts.max,
                        }
                    }
                    if g.imatrix_counts is not None
                    else {}
                ),
            }
            for g in map_.groups
        ],
    }


def load_sensitivity_map(path: Path) -> SensitivityMap:
    """Read and validate a sensitivity-map file.

    Args:
        path: JSON file written by ``vramfit scan`` or
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
            ``group_by`` is not ``layer``, ``tensor``, or ``stack``,
            a present
            ``within_group`` is not a non-empty string (absent
            defaults to ``rtn-block32``, ADR-0018), or ``imatrix``
            does not pair with the assisted method token — assisted
            damages without their imatrix provenance are not
            comparable to anything (ADR-0020, absent defaults to
            None).
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
        group_by in ("layer", "tensor", "stack"),
        f"{path}.group_by",
        'must be "layer", "tensor", or "stack"',
    )
    # Optional and additive (ADR-0018): maps written before the field
    # existed are rtn-block32 scans by definition. A present field
    # validates through _get_str, which rejects empty strings.
    within_group = (
        _get_str(obj, "within_group", path) if "within_group" in obj else SCAN_METHOD
    )
    # Optional and additive (ADR-0020): maps written before the field
    # existed were unassisted by definition. The pairing rules mirror
    # ScanMeta's own invariant, re-stated here for JSON-path errors.
    imatrix = _get_str(obj, "imatrix", path) if obj.get("imatrix") is not None else None
    _require(
        not (within_group == KQUANT_IMX_METHOD and imatrix is None),
        f"{path}.imatrix",
        f'within_group "{KQUANT_IMX_METHOD}" requires the imatrix field (ADR-0020)',
    )
    _require(
        not (imatrix is not None and within_group != KQUANT_IMX_METHOD),
        f"{path}.imatrix",
        f'imatrix provenance requires within_group "{KQUANT_IMX_METHOD}", '
        f'got "{within_group}" (ADR-0020)',
    )
    return ScanMeta(
        metric=_get_str(obj, "metric", path),
        calibration=_get_str(obj, "calibration", path),
        calibration_tokens=tokens,
        precisions=tuple(precisions),
        group_by=cast('Literal["layer", "tensor", "stack"]', group_by),
        started_at=_get_str(obj, "started_at", path),
        within_group=within_group,
        imatrix=imatrix,
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
            is not positive, the sensitivity keys do not match the
            scan's precisions, or a present ``tensor_bytes`` does not
            cover exactly the group's tensors with positive sizes
            summing to ``bytes_fp16`` (ADR-0022 — absent means
            unknown, and an explicit null is rejected: the writer
            never emits one), or a present ``imatrix_counts`` fails
            `_parse_imatrix_counts` (ADR-0026 decision 4).
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
    # Optional and additive (ADR-0022): maps written before the field
    # existed carry no per-tensor sizes, and plan refuses protections
    # against them. The writer omits the field when unknown and never
    # writes null, so an explicit null is a hand-edit — rejected, not
    # normalized. The domain type enforces coverage and positivity.
    tensor_bytes: dict[str, int] = {}
    if "tensor_bytes" in obj:
        sizes_obj = _get_dict(obj["tensor_bytes"], f"{path}.tensor_bytes")
        for tensor, size in sizes_obj.items():
            value = _as_int(size, f"{path}.tensor_bytes.{tensor}")
            _require(value > 0, f"{path}.tensor_bytes.{tensor}", "must be positive")
            tensor_bytes[tensor] = value
        _require(
            set(tensor_bytes) == set(tensors),
            f"{path}.tensor_bytes",
            "keys must equal the group's tensors (ADR-0022)",
        )
        _require(
            sum(tensor_bytes.values()) == bytes_fp16,
            f"{path}.tensor_bytes",
            f"values sum to {sum(tensor_bytes.values())} but bytes_fp16 "
            f"is {bytes_fp16} (ADR-0022)",
        )
    return LayerGroup(
        name=_get_str(obj, "name", path),
        tensors=tuple(tensors),
        bytes_fp16=bytes_fp16,
        sensitivity=sensitivity,
        tensor_bytes=tensor_bytes,
        imatrix_counts=_parse_imatrix_counts(obj, path),
    )


def _parse_imatrix_counts(obj: dict[str, Any], path: str) -> ImatrixCountSummary | None:
    """Validate one group's optional imatrix count summary.

    Optional and additive (ADR-0026 decision 4): the writer omits
    the field when the group records no summary — the #201 amendment
    makes it all-or-nothing per group — so an absent field means
    absent, and an explicit null is a hand-edit, rejected rather
    than normalized.

    Args:
        obj: The group's JSON object.
        path: JSON path of this group.

    Returns:
        The validated summary, or None when the field is absent.

    Raises:
        ArtifactError: If a present field is not an object holding
            exactly ``min``, ``median``, and ``max``, a value is
            mistyped or negative, or the three are not ordered
            ``min <= median <= max``.
    """
    if "imatrix_counts" not in obj:
        return None
    field_path = f"{path}.imatrix_counts"
    summary = _get_dict(obj["imatrix_counts"], field_path)
    _require(
        set(summary) == {"min", "median", "max"},
        field_path,
        'must hold exactly "min", "median", and "max" (ADR-0026)',
    )
    minimum = _as_int(summary["min"], f"{field_path}.min")
    median = _as_float(summary["median"], f"{field_path}.median")
    maximum = _as_int(summary["max"], f"{field_path}.max")
    _require(minimum >= 0, f"{field_path}.min", "must not be negative")
    _require(
        minimum <= median <= maximum,
        field_path,
        "must be ordered: min <= median <= max",
    )
    return ImatrixCountSummary(min=minimum, median=median, max=maximum)
