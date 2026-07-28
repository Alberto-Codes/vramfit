"""JSON file adapter for scan checkpoints: incremental, resumable.

A scan takes hours (`docs/how-to/scan-a-model.md`), so every finished
measurement lands on disk immediately. The file carries the scan's
fingerprint — loading or appending with a different fingerprint fails
instead of mixing two scans' numbers. The checkpoint has its own
schema version, so a breaking pipeline-artifact change cannot strand
an in-flight scan. Writes go through the same atomic-replace path as
the artifact adapters, so a crash mid-write never corrupts the
checkpoint.

Examples:
    Resume a scan and record one new cell:

    ```python
    from quantfit.adapters.outbound.scan_checkpoint_json import (
        JsonScanCheckpointFile,
    )

    store = JsonScanCheckpointFile(path)
    done = store.load(fingerprint)
    store.append(fingerprint, measurement)
    ```

See Also:
    - [quantfit.ports.outbound][]: `ScanCheckpointStore`, which
      `JsonScanCheckpointFile` satisfies.
    - [quantfit.domain.scan][]: `scan_fingerprint` and `Measurement`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from quantfit.adapters.outbound.json_common import (
    ArtifactError,
    _get_dict,
    _get_float,
    _get_int,
    _get_list,
    _get_str,
    _load_json,
    _require,
    _save_json,
)
from quantfit.domain.scan import Measurement

# The checkpoint versions independently of the pipeline artifacts: a
# breaking recipe or map schema change must not strand in-flight scans.
CHECKPOINT_SCHEMA_VERSION: Final[int] = 1


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
                "quantfit_schema": CHECKPOINT_SCHEMA_VERSION,
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
        ArtifactError: If the checkpoint schema version is unsupported,
            a field is missing or mistyped, a damage value is invalid,
            or the stored fingerprint differs.
    """
    root = _get_dict(data, "$")
    version = _get_int(root, "quantfit_schema", "$")
    _require(
        version == CHECKPOINT_SCHEMA_VERSION,
        "$.quantfit_schema",
        f"unsupported schema version {version} — this quantfit reads "
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
