"""JSON file adapter for scan checkpoints: incremental, resumable.

A scan takes hours (`docs/how-to/scan-a-model.md`), so every finished
measurement lands on disk immediately. The file carries the scan's
fingerprint — loading or appending with a different fingerprint fails
instead of mixing two scans' numbers. The checkpoint has its own
schema version, so a breaking pipeline-artifact change cannot strand
an in-flight scan. The tool rename (#118) is the one exception: the
loader rejects the pre-rename envelope key with a message that names
the new key. Writes go through the same atomic-replace path as
the artifact adapters, so a crash mid-write never corrupts the
checkpoint.

Examples:
    Resume a scan and record one new cell:

    ```python
    from vramfit.adapters.outbound.scan_checkpoint_json import (
        JsonScanCheckpointFile,
    )

    store = JsonScanCheckpointFile(path)
    done = store.load(fingerprint)
    store.append(fingerprint, measurement)
    ```

See Also:
    - [vramfit.ports.outbound][]: `ScanCheckpointStore`, which
      `JsonScanCheckpointFile` satisfies.
    - [vramfit.domain.scan][]: `scan_fingerprint` and `Measurement`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.json_common import (
    ArtifactError,
    _get_dict,
    _get_float,
    _get_int,
    _get_list,
    _get_str,
    _load_json,
    _reject_renamed_envelope_key,
    _require,
    _save_json,
)
from vramfit.domain.scan import Measurement

# The checkpoint versions independently of the pipeline artifacts: a
# breaking recipe or map schema change must not strand in-flight scans.
# The tool rename (#118) is the one coupled bump — every envelope key
# renamed at once, so version-1 checkpoints do not resume.
CHECKPOINT_SCHEMA_VERSION: Final[int] = 2


@dataclass(frozen=True, slots=True)
class JsonScanCheckpointFile:
    """`ScanCheckpointStore` adapter backed by a JSON file.

    Attributes:
        path (Path): The checkpoint file. A missing file means an empty
            checkpoint.

    Examples:
        A fresh path loads as empty:

        ```python
        store = JsonScanCheckpointFile(tmp_path / "scan.checkpoint.json")
        assert store.load("fp") == ()
        ```
    """

    path: Path

    def load(self, fingerprint: str) -> tuple[Measurement, ...]:
        """Load prior measurements for this scan.

        Args:
            fingerprint: The scan's identity string.

        Returns:
            All checkpointed measurements, empty when the file does not
            exist.

        Raises:
            ArtifactError: If the file exists but is corrupt or carries
                a different fingerprint.
            OSError: If the file exists but cannot be read — an
                unreadable checkpoint must not pass for an absent one.
        """
        try:
            self.path.stat()
        except FileNotFoundError:
            return ()
        return _parse_checkpoint(_load_json(self.path, "$"), fingerprint)

    def append(self, fingerprint: str, measurement: Measurement) -> None:
        """Record one finished measurement on disk.

        The whole checkpoint is rewritten through an atomic replace, so
        a crashed process never leaves a truncated file. Checkpoints
        stay small (one entry per grid cell), so the rewrite cost is
        noise next to a measurement's calibration pass.

        Args:
            fingerprint: The scan's identity string.
            measurement: The finished cell.

        Raises:
            ArtifactError: If the existing file is corrupt or carries a
                different fingerprint.
            OSError: If the write fails.
        """
        measurements = [*self.load(fingerprint), measurement]
        _save_json(
            {
                "vramfit_schema": CHECKPOINT_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "measurements": [
                    {"group": m.group, "bits": m.bits, "damage": m.damage}
                    for m in measurements
                ],
            },
            self.path,
        )


def _parse_checkpoint(data: object, fingerprint: str) -> tuple[Measurement, ...]:
    """Validate a checkpoint object against the expected fingerprint.

    Args:
        data: Parsed JSON value, expected to be the checkpoint's
            top-level object.
        fingerprint: The running scan's identity string.

    Returns:
        The checkpointed measurements.

    Raises:
        ArtifactError: If the checkpoint carries the pre-rename
            envelope key (#118), the schema version is unsupported,
            a field is missing or mistyped, a damage value is invalid,
            or the stored fingerprint differs. The pre-rename message
            names `CHECKPOINT_SCHEMA_VERSION` too, because an archived
            checkpoint fails the version next (#154).
    """
    root = _get_dict(data, "$")
    _reject_renamed_envelope_key(root, "$", CHECKPOINT_SCHEMA_VERSION)
    version = _get_int(root, "vramfit_schema", "$")
    _require(
        version == CHECKPOINT_SCHEMA_VERSION,
        "$.vramfit_schema",
        f"unsupported schema version {version} — this vramfit reads "
        f"checkpoint version {CHECKPOINT_SCHEMA_VERSION}",
    )
    stored = _get_str(root, "fingerprint", "$")
    _require(
        stored == fingerprint,
        "$.fingerprint",
        f'checkpoint belongs to a different scan ("{stored}" != "{fingerprint}")',
    )
    measurements: list[Measurement] = []
    for i, raw in enumerate(_get_list(root, "measurements", "$")):
        measurements.append(_parse_measurement(raw, f"$.measurements[{i}]"))
    return tuple(measurements)


def _parse_measurement(raw: Any, path: str) -> Measurement:
    """Validate one checkpointed measurement.

    Args:
        raw: The measurement's JSON value.
        path: JSON path of this entry, for error reporting.

    Returns:
        The validated measurement.

    Raises:
        ArtifactError: If a field is missing or mistyped, or the values
            violate `Measurement` invariants.
    """
    obj = _get_dict(raw, path)
    try:
        return Measurement(
            group=_get_str(obj, "group", path),
            bits=_get_int(obj, "bits", path),
            damage=_get_float(obj, "damage", path),
        )
    except ArtifactError:
        raise
    except ValueError as exc:
        raise ArtifactError(path, str(exc)) from exc
