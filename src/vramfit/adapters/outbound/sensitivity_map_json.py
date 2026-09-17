"""JSON file adapter for the sensitivity-map artifact.

Owns (de)serialization and validation of the map schema, including the
``vramfit_schema`` envelope (`MAP_SCHEMA_VERSION` and
`MAP_SCHEMA_ALSO_READS` — schema versions advance per artifact,
ADR-0013). The adapter writes version 5 and
also reads versions 2, 3, and 4, because each later version only
added to the one before it: version 3 widened ``group_by`` with the
``stack`` value (#161), version 4 added the calibration file's
content identity, and version 5 added that digest's provenance mark
with its revision referent, plus each group's measured row width
(issue #558). One file class serves both directions:
``vramfit scan`` writes through the sink face, ``vramfit plan`` reads
through the source face. Validation is strict: artifacts are rejected,
never normalized — ``scan.precisions`` must arrive strictly descending,
``group_by`` must be a known granularity, and every group's sensitivity
keys must equal it exactly. Several fields are additive rather than
strict:
``scan.within_group`` (ADR-0018) defaults to ``rtn-block32`` when
absent, because every map written before the field existed measured
with that method, and ``scan.imatrix`` (ADR-0020) defaults to None,
because every map written before the field existed was unassisted.
A group's ``imatrix_counts`` summary (ADR-0026 decision 4) is
additive the same way: absent means the group records no summary.
``scan.calibration_sha256`` and ``scan.calibration_bytes`` are
additive too, and they pair: absent or null means the scan recorded
no content identity, which stays NOT RECORDED. The loader never
hashes a file to fill them, and a save never invents them.
[vramfit.adapters.outbound.sensitivity_map_scan_json][] owns that
section, including the digest's provenance mark. A group's
``row_width`` is additive as well: absent means the map records no
width, and the plan then reads the checkpoint's or refuses (issue
#558). A save never invents one.
The top-level ``derived`` note (#136) is additive too: absent means
the map is a scan artifact. A present ``imatrix`` must pair with
an assisted method token, ``kquant-imx`` or ``q0-imx`` — the
loader rejects a map whose provenance
contradicts itself. A field the reader does not know warns and loads
(#261, ADR-0013's 2026-08-16 amendment). A save then drops it, and
the warning says so.

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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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
    _warn_unknown_fields,
)
from vramfit.adapters.outbound.sensitivity_map_scan_json import (
    scan_from_dict,
    scan_to_dict,
)
from vramfit.domain.model import (
    ImatrixCountSummary,
    LayerGroup,
    SensitivityMap,
)
from vramfit.domain.runtime import routes_by_row_width, unquantizable_class

# The sensitivity-map schema version. Versions advance per artifact
# (ADR-0013), so this constant moves on its own.
MAP_SCHEMA_VERSION: Final[int] = 5
# Older versions this adapter still reads. Each bump only added:
# version 3 widened ``group_by`` with the ``stack`` value (#161),
# version 4 added the paired calibration digest and byte count, and
# version 5 added that digest's provenance mark with its revision
# referent, plus each group's measured row width (issue #558). So
# every older map is already a valid version-5 document: it records
# no content identity, or a digest whose absent mark reads as
# ``measured``, and it records no row width. The writer emits 5,
# which tells a reader the producer could have recorded both.
MAP_SCHEMA_ALSO_READS: Final[tuple[int, ...]] = (2, 3, 4)

# The first map schema whose producer records a group's measured row
# width. Below it the field did not exist, so a present width is a
# value no scan measured and the reader refuses it (issue #558).
WIDTH_RECORDED_FROM: Final[int] = 5

# Every key the reader carries, per object the schema fixes (#261).
# A key outside these sets warns and loads (ADR-0013, the 2026-08-16
# amendment). ``sensitivity`` and ``tensor_bytes`` are absent here on
# purpose: their keys are precisions and tensor names, and their own
# rules already check them. ``imatrix_counts`` fixes its three keys
# exactly (ADR-0026), so it refuses rather than warns.
MAP_ROOT_FIELDS: Final[frozenset[str]] = frozenset(
    {"vramfit_schema", "model_id", "scan", "groups", "derived"}
)
GROUP_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "tensors",
        "bytes_fp16",
        "sensitivity",
        "tensor_bytes",
        "imatrix_counts",
        "row_width",
    }
)


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
            keys not matching ``scan.precisions``, an empty or
            non-string ``derived`` note, a calibration digest whose
            provenance mark is missing or does not name its referent,
            a ``row_width`` that is not positive or that the
            document's version or the group's shape refuses, and so
            on). A field the
            reader does not know reports and loads instead (#261).

    Examples:
        Reject an unsupported schema version:

        ```python
        map_from_dict({"vramfit_schema": 6})  # raises ArtifactError
        ```
    """
    root = _get_dict(data, "$")
    schema_version = _check_schema_version(
        root,
        "$",
        expected=MAP_SCHEMA_VERSION,
        also_reads=MAP_SCHEMA_ALSO_READS,
    )
    _warn_unknown_fields(root, "$", MAP_ROOT_FIELDS)
    model_id = _get_str(root, "model_id", "$")
    _require("scan" in root, "$", 'missing required field "scan"')
    scan = scan_from_dict(_get_dict(root["scan"], "$.scan"), schema_version)
    groups_raw = _get_list(root, "groups", "$")
    _require(len(groups_raw) > 0, "$.groups", "must not be empty")
    expected = set(scan.precisions)
    groups: list[LayerGroup] = []
    seen: set[str] = set()
    for i, raw in enumerate(groups_raw):
        group = _parse_layer_group(
            raw, f"$.groups[{i}]", expected, schema_version
        )
        _require(
            group.name not in seen,
            f"$.groups[{i}].name",
            f'duplicate group name "{group.name}"',
        )
        seen.add(group.name)
        groups.append(group)
    # Optional and additive (#136): a map without the note is a scan
    # artifact. The writer omits the field then and never writes null,
    # so an explicit null is a hand-edit — rejected, not normalized.
    derived = _get_str(root, "derived", "$") if "derived" in root else None
    return SensitivityMap(
        model_id=model_id, scan=scan, groups=tuple(groups), derived=derived
    )


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
        unassisted (ADR-0020). The calibration digest and byte count
        are always written too, null when the scan recorded no
        content identity, which a reader reads as NOT RECORDED: a
        save never invents a digest. Per-tensor sizes are written only
        when known — an absent field means unknown, never zero
        (ADR-0022). A group's imatrix count summary is written only
        when the group records one (ADR-0026 decision 4). The
        ``derived`` note is written only when the map carries one —
        a save never drops it and never invents it (#136). A group's
        measured row width is written only when the map records one
        (issue #558).
    """
    return {
        "vramfit_schema": MAP_SCHEMA_VERSION,
        "model_id": map_.model_id,
        "scan": scan_to_dict(map_.scan),
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
                # Written only when the scan measured one width for
                # the group (issue #558). A whole-layer group holds
                # classes of several widths and records none, so an
                # absent field means the map states no width, never
                # that the group has none.
                **({"row_width": g.row_width} if g.row_width is not None else {}),
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
        # Written only when the map carries the note (#136) — absent
        # means a scan artifact, so the schema holds at 3. The key
        # trails ``groups`` to match the published maps.
        **({"derived": map_.derived} if map_.derived is not None else {}),
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
    raw: Any, path: str, expected_precisions: set[int], schema_version: int
) -> LayerGroup:
    """Validate one entry of a sensitivity map's ``groups`` list.

    Args:
        raw: The group's JSON value.
        path: JSON path of this group.
        expected_precisions: The scan's candidate precisions; the group's
            sensitivity keys must equal this set.
        schema_version: The version the document declares, which
            `map_from_dict` already validated. `_parse_row_width`
            reads it.

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
            `_parse_imatrix_counts` (ADR-0026 decision 4), or a
            present ``row_width`` is not a positive integer, names a
            group the 256 super-block decision does not reach, or
            appears below `WIDTH_RECORDED_FROM` (issue #558 — absent
            means the map records no width). A field the group does
            not carry reports and loads (#261).
    """
    obj = _get_dict(raw, path)
    _warn_unknown_fields(obj, path, GROUP_FIELDS)
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
    name = _get_str(obj, "name", path)
    return LayerGroup(
        name=name,
        tensors=tuple(tensors),
        bytes_fp16=bytes_fp16,
        sensitivity=sensitivity,
        tensor_bytes=tensor_bytes,
        imatrix_counts=_parse_imatrix_counts(obj, path),
        row_width=_parse_row_width(obj, path, name, schema_version),
    )


def _parse_row_width(
    obj: dict[str, Any], path: str, name: str, schema_version: int
) -> int | None:
    """Validate one group's optional measured row width (issue #558).

    Optional and additive: the writer omits the field when the map
    records no width, so an absent field means absent, and an
    explicit null is a hand-edit, rejected rather than normalized.
    A non-positive width tiles no block, so it would route the 256
    super-block decision off a number no tensor has.

    A document below `WIDTH_RECORDED_FROM` refuses the field
    outright. No producer at those versions wrote it, so its value
    reached the document by hand and no scan measured it. The
    published schema-2 maps are the case: a width hand-added to each
    routed group there would route the ADR-0028 expert-stack table
    with no checkpoint to disagree, so nothing else in the plan
    could catch it.

    The decision reaches a layer-class or routed-expert-stack group
    only, and two other shapes each refuse a width for their own
    reason. A whole-layer group holds classes of several row widths
    and takes the ADR-0012 k-quant table, so no single width
    describes it. A group of a class the quantizer refuses takes
    neither type table — it holds at the F16 passthrough at the
    convert dtype (#409) — so no width ever routes it. Either way
    the field asserts what the group cannot carry, so the reader
    refuses it rather than ignoring it.

    Args:
        obj: The group's JSON object.
        path: JSON path of this group.
        name: The group's name, which decides whether the 256
            super-block decision reaches it.
        schema_version: The version the document declares.

    Returns:
        The measured width, or None when the field is absent.

    Raises:
        ArtifactError: If a present field is not a positive integer,
            the document declares a version below
            `WIDTH_RECORDED_FROM`, or the group is one the
            super-block decision does not reach. The message states
            the reason that document or group earns.
    """
    if "row_width" not in obj:
        return None
    field_path = f"{path}.row_width"
    _require(
        schema_version >= WIDTH_RECORDED_FROM,
        field_path,
        f"no schema-{schema_version} scan recorded a group row width, "
        f"so this value is unmeasured (issue #558)",
    )
    _require(
        unquantizable_class(name) is None,
        field_path,
        f'group "{name}" holds at the F16 passthrough and takes '
        f"neither type table, so no width routes it (#409, issue #558)",
    )
    _require(
        routes_by_row_width(name),
        field_path,
        f'group "{name}" is no layer-class or routed-expert-stack '
        f"group. It takes the ADR-0012 k-quant table, which no row "
        f"width routes (issue #515, issue #558)",
    )
    width = _as_int(obj["row_width"], field_path)
    _require(width > 0, field_path, "must be positive")
    return width


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
            ``min <= median <= max``. A `ValueError` the domain type
            raises translates here too, so a later invariant cannot
            escape the root as a bare `ValueError` (#260).
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
    # The checks above duplicate the domain's, so nothing should reach
    # the constructor and fail. A future invariant would escape as a
    # bare `ValueError` past the CLI's root handler without this, and
    # every sibling parser already translates (#260).
    try:
        return ImatrixCountSummary(min=minimum, median=median, max=maximum)
    except ValueError as exc:
        raise ArtifactError(field_path, str(exc)) from exc
