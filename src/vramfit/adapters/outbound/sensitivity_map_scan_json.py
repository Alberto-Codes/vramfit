"""The sensitivity map's ``scan`` section, read and written.

Split out of [vramfit.adapters.outbound.sensitivity_map_json][] when
schema 5 added the calibration digest's provenance mark (#589's
mechanism) and the file passed its decomposition limit. That module
still owns the envelope, the groups, and the file class. This one
owns one JSON object: the scan's provenance.

Several fields are additive rather than strict, and each defaults to
what was necessarily true before it existed. ``within_group``
(ADR-0018) defaults to ``rtn-block32``, because every map written
before the field measured with that method. ``imatrix`` (ADR-0020)
defaults to None, because every such map was unassisted.

``calibration_provenance`` breaks that pattern on purpose. Below
schema 5 an absent mark beside a digest reads as ``measured``: those
documents predate the field, and ``vramfit scan`` is the only
producer that could have written that digest. At schema 5 and above
an absent mark beside a digest refuses, because the producer could
have recorded one. An explicit null beside a digest refuses at every
version — the writer never pairs the two that way, so the reader
rejects the hand-edit rather than normalizing it.

Examples:
    Round-trip one scan section:

    ```python
    from vramfit.adapters.outbound.sensitivity_map_scan_json import (
        scan_from_dict,
        scan_to_dict,
    )

    assert scan_from_dict(scan_to_dict(meta), 5) == meta
    ```

See Also:
    - [vramfit.domain.model][]: `ScanMeta`, whose invariants these
      checks mirror for JSON-path reporting.
    - [vramfit.domain.provenance][]: The provenance mark's vocabulary
      and its two rules.
"""

from __future__ import annotations

import itertools
from typing import Any, Final, Literal, cast

from vramfit.adapters.outbound.json_common import (
    ArtifactError,
    _as_int,
    _as_str,
    _get_int,
    _get_list,
    _get_str,
    _require,
    _warn_unknown_fields,
)
from vramfit.domain.model import ScanMeta
from vramfit.domain.provenance import MEASURED
from vramfit.domain.scan import ASSISTED_METHODS, SCAN_METHOD

# Every key the ``scan`` object carries. A key outside this set warns
# and loads (ADR-0013, the 2026-08-16 amendment).
SCAN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "metric",
        "calibration",
        "calibration_tokens",
        "calibration_sha256",
        "calibration_bytes",
        "calibration_provenance",
        "calibration_revision",
        "precisions",
        "group_by",
        "started_at",
        "within_group",
        "imatrix",
    }
)

SCAN_PATH: Final[str] = "$.scan"

# The first map schema whose producer could record the calibration
# digest's provenance mark. From here up, an absent mark beside a
# digest is a missing claim, never a default (issue #589's mechanism).
MARK_REQUIRED_FROM: Final[int] = 5


def scan_to_dict(meta: ScanMeta) -> dict[str, Any]:
    """Serialize the scan's provenance.

    The calibration digest, byte count, provenance mark, and revision
    are always written, null when the scan recorded none. A reader
    reads a null as NOT RECORDED: a save never invents a digest and
    never invents the mark that says who established one.

    Args:
        meta: The scan provenance to serialize.

    Returns:
        The ``scan`` object's JSON form.

    Examples:
        ```python
        from vramfit.adapters.outbound.sensitivity_map_scan_json import (
            scan_to_dict,
        )

        assert scan_to_dict(meta)["metric"] == meta.metric
        ```
    """
    return {
        "metric": meta.metric,
        "calibration": meta.calibration,
        "calibration_tokens": meta.calibration_tokens,
        "calibration_sha256": meta.calibration_sha256,
        "calibration_bytes": meta.calibration_bytes,
        "calibration_provenance": meta.calibration_provenance,
        "calibration_revision": meta.calibration_revision,
        "precisions": list(meta.precisions),
        "group_by": meta.group_by,
        "started_at": meta.started_at,
        "within_group": meta.within_group,
        "imatrix": meta.imatrix,
    }


def _parse_precisions(obj: dict[str, Any]) -> list[int]:
    """Validate the candidate precision list.

    Args:
        obj: The ``scan`` JSON object.

    Returns:
        The precisions, in document order.

    Raises:
        ArtifactError: If the list is empty, holds a non-integer or a
            non-positive value, duplicates a precision, or is not
            strictly descending.
    """
    raw = _get_list(obj, "precisions", SCAN_PATH)
    path = f"{SCAN_PATH}.precisions"
    _require(len(raw) > 0, path, "must not be empty")
    precisions = [_as_int(p, f"{path}[{i}]") for i, p in enumerate(raw)]
    _require(
        len(set(precisions)) == len(precisions), path, "must not contain duplicates"
    )
    _require(all(p > 0 for p in precisions), path, "must all be positive")
    _require(
        all(a > b for a, b in itertools.pairwise(precisions)),
        path,
        "must be strictly descending",
    )
    return precisions


def _parse_calibration_mark(
    obj: dict[str, Any], digest: str | None, schema_version: int
) -> str | None:
    """Read the calibration digest's provenance mark.

    The default is version-gated. Below `MARK_REQUIRED_FROM`, an
    absent field beside a digest reads as `MEASURED`: those documents
    predate the field, and ``vramfit scan`` is the only producer that
    could have written that digest, hashing the calibration file as
    it read it. From `MARK_REQUIRED_FROM` up, an absent field beside
    a digest refuses. The producer could have recorded a mark, so the
    absence is a missing claim, not a default.

    The ungated default reproduced the defect the mark exists to
    prevent. The back-fill this schema was raised for writes a
    schema-5 map with a digest hashed today. Omitting the mark read
    back as `MEASURED` with no error, so the artifact asserted that
    the process which produced its damage numbers hashed those bytes
    as it read them. That is false. The referent rules cannot catch
    it, because `MEASURED` is the one mark that needs no referent, so
    the hole sat exactly where the mechanism had to hold.

    The additive precedent of ``within_group`` and ``imatrix`` does
    not transfer, however alike the three fields look. Those two
    describe what the scan did, so a default states a setting. This
    field describes who established a value, and provenance is the
    one thing a reader must never infer.

    An absent field with no digest reads as None at every version. An
    explicit null is never normalized — `ScanMeta` then refuses it
    beside a digest, because the writer never writes that pair.

    Args:
        obj: The ``scan`` JSON object.
        digest: The parsed calibration digest, or None.
        schema_version: The version the document declares, which
            `map_from_dict` already validated.

    Returns:
        The mark, or None.

    Raises:
        ArtifactError: If a present field is neither null nor a
            non-empty string, or the document declares
            `MARK_REQUIRED_FROM` or above and carries a digest with
            no mark.
    """
    if "calibration_provenance" in obj:
        raw = obj["calibration_provenance"]
        if raw is None:
            return None
        return _as_str(raw, f"{SCAN_PATH}.calibration_provenance")
    if digest is None:
        return None
    _require(
        schema_version < MARK_REQUIRED_FROM,
        f"{SCAN_PATH}.calibration_provenance",
        f"schema {schema_version} records who established the "
        f"calibration digest, so a digest with no mark states a claim "
        f"no reader can check (#589)",
    )
    return MEASURED


def _parse_imatrix(obj: dict[str, Any], within_group: str) -> str | None:
    """Read the imatrix path and check it pairs with the method token.

    The pairing mirrors `ScanMeta`'s own invariant, re-stated here so
    a contradiction reports at its own JSON path (ADR-0020).

    Args:
        obj: The ``scan`` JSON object.
        within_group: The parsed within-group method token.

    Returns:
        The imatrix path, or None for an unassisted scan.

    Raises:
        ArtifactError: If an assisted token carries no imatrix, or an
            imatrix accompanies an unassisted token.
    """
    imatrix = (
        _get_str(obj, "imatrix", SCAN_PATH) if obj.get("imatrix") is not None else None
    )
    path = f"{SCAN_PATH}.imatrix"
    _require(
        not (within_group in ASSISTED_METHODS and imatrix is None),
        path,
        f'within_group "{within_group}" requires the imatrix field (ADR-0020)',
    )
    _require(
        not (imatrix is not None and within_group not in ASSISTED_METHODS),
        path,
        "imatrix provenance requires an assisted within_group "
        f'({", ".join(ASSISTED_METHODS)}), got "{within_group}" (ADR-0020)',
    )
    return imatrix


def scan_from_dict(obj: dict[str, Any], schema_version: int) -> ScanMeta:
    """Validate the ``scan`` section of a sensitivity map.

    Args:
        obj: The ``scan`` JSON object.
        schema_version: The version the document declares.
            `map_from_dict` validates the envelope and passes the
            version here, so this reader never re-reads it. The
            calibration mark's default is gated on it.

    Returns:
        The validated scan provenance.

    Raises:
        ArtifactError: If a field is missing or invalid, precisions
            are empty, duplicated, not integers, or not strictly
            descending, ``group_by`` is not ``layer``, ``tensor``, or
            ``stack``, a present ``within_group`` is not a non-empty
            string (absent defaults to ``rtn-block32``, ADR-0018),
            ``imatrix`` does not pair with an assisted method token
            (ADR-0020), the calibration content identity is malformed
            — the digest must hold 64 lowercase hex digits, the byte
            count must be positive, and the two must pair — the
            digest and its provenance mark do not pair, or a mark
            asserting outside the digest does not name its referent
            (``re_derived`` without ``calibration_revision``). A
            mistyped field reports at its own JSON path. The domain
            invariants report at the section's path. A field the
            section does not carry reports and loads (#261).

    Examples:
        ```python
        from vramfit.adapters.outbound.sensitivity_map_scan_json import (
            scan_from_dict,
        )

        meta = scan_from_dict(raw["scan"], 5)
        ```
    """
    _warn_unknown_fields(obj, SCAN_PATH, SCAN_FIELDS)
    tokens = _get_int(obj, "calibration_tokens", SCAN_PATH)
    _require(tokens > 0, f"{SCAN_PATH}.calibration_tokens", "must be positive")
    precisions = _parse_precisions(obj)
    group_by = _get_str(obj, "group_by", SCAN_PATH)
    _require(
        group_by in ("layer", "tensor", "stack"),
        f"{SCAN_PATH}.group_by",
        'must be "layer", "tensor", or "stack"',
    )
    # Optional and additive (ADR-0018): maps written before the field
    # existed are rtn-block32 scans by definition. A present field
    # validates through _get_str, which rejects empty strings.
    within_group = (
        _get_str(obj, "within_group", SCAN_PATH)
        if "within_group" in obj
        else SCAN_METHOD
    )
    imatrix = _parse_imatrix(obj, within_group)
    # Optional, additive, and paired (ScanMeta enforces the pairing):
    # absent or null means the scan recorded no content identity,
    # which stays NOT RECORDED. The loader never hashes a file to
    # fill a gap — a digest taken today proves nothing about the
    # bytes an earlier run measured.
    raw_digest = obj.get("calibration_sha256")
    raw_bytes = obj.get("calibration_bytes")
    digest = (
        None
        if raw_digest is None
        else _as_str(raw_digest, f"{SCAN_PATH}.calibration_sha256")
    )
    n_bytes = (
        None
        if raw_bytes is None
        else _as_int(raw_bytes, f"{SCAN_PATH}.calibration_bytes")
    )
    mark = _parse_calibration_mark(obj, digest, schema_version)
    raw_revision = obj.get("calibration_revision")
    revision = (
        None
        if raw_revision is None
        else _as_str(raw_revision, f"{SCAN_PATH}.calibration_revision")
    )
    # The calibration content rules live in the domain, so the reader
    # states them once. A `ValueError` translates here, as the
    # imatrix count summary's does (#260). Only the constructor sits
    # inside the try, or a field's own `ArtifactError` would come back
    # out relabelled at this block's path.
    metric = _get_str(obj, "metric", SCAN_PATH)
    calibration = _get_str(obj, "calibration", SCAN_PATH)
    started_at = _get_str(obj, "started_at", SCAN_PATH)
    try:
        return ScanMeta(
            metric=metric,
            calibration=calibration,
            calibration_tokens=tokens,
            precisions=tuple(precisions),
            group_by=cast('Literal["layer", "tensor", "stack"]', group_by),
            started_at=started_at,
            within_group=within_group,
            imatrix=imatrix,
            calibration_sha256=digest,
            calibration_bytes=n_bytes,
            calibration_provenance=mark,
            calibration_revision=revision,
        )
    except ValueError as exc:
        raise ArtifactError(SCAN_PATH, str(exc)) from exc
