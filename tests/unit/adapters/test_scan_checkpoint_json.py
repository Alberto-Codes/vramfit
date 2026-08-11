from __future__ import annotations

import json

import pytest

from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from vramfit.domain.model import ScanMeta
from vramfit.domain.scan import Measurement, scan_fingerprint

pytestmark = pytest.mark.unit

# The real fingerprint shape the CLI persists, method token included.
FP = scan_fingerprint(
    "test/model",
    ScanMeta(
        metric="kl_divergence",
        calibration="calib.txt",
        calibration_tokens=1024,
        precisions=(8, 4),
        group_by="layer",
        started_at="unused",
    ),
)


def test_missing_file_loads_as_empty_checkpoint(tmp_path) -> None:
    store = JsonScanCheckpointFile(tmp_path / "absent.checkpoint.json")

    assert store.load(FP) == ()


def test_appended_measurements_read_back_in_order(tmp_path) -> None:
    store = JsonScanCheckpointFile(tmp_path / "scan.checkpoint.json")
    first = Measurement(group="g0", bits=8, damage=0.001)
    second = Measurement(group="g0", bits=4, damage=0.02)

    store.append(FP, first)
    store.append(FP, second)

    assert store.load(FP) == (first, second)


def test_load_with_different_fingerprint_raises(tmp_path) -> None:
    store = JsonScanCheckpointFile(tmp_path / "scan.checkpoint.json")
    store.append(FP, Measurement(group="g0", bits=8, damage=0.001))

    with pytest.raises(ArtifactError, match="different scan"):
        store.load("other|fingerprint")


def test_append_with_different_fingerprint_raises(tmp_path) -> None:
    store = JsonScanCheckpointFile(tmp_path / "scan.checkpoint.json")
    store.append(FP, Measurement(group="g0", bits=8, damage=0.001))

    with pytest.raises(ArtifactError, match="different scan"):
        store.append("other|fingerprint", Measurement(group="g0", bits=4, damage=0.1))


def test_corrupt_json_raises_artifact_error(tmp_path) -> None:
    path = tmp_path / "scan.checkpoint.json"
    path.write_text("{not json")

    with pytest.raises(ArtifactError, match="invalid JSON"):
        JsonScanCheckpointFile(path).load(FP)


def test_unsupported_schema_version_raises(tmp_path) -> None:
    path = tmp_path / "scan.checkpoint.json"
    path.write_text(json.dumps({"vramfit_schema": 99, "fingerprint": FP}))

    with pytest.raises(ArtifactError, match="unsupported schema version"):
        JsonScanCheckpointFile(path).load(FP)


def test_pre_rename_envelope_key_rejected(tmp_path) -> None:
    path = tmp_path / "scan.checkpoint.json"
    path.write_text(json.dumps({"quantfit_schema": 1, "fingerprint": FP}))

    with pytest.raises(ArtifactError, match='renamed to "vramfit_schema"'):
        JsonScanCheckpointFile(path).load(FP)


def test_invalid_stored_damage_raises_with_json_path(tmp_path) -> None:
    path = tmp_path / "scan.checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "vramfit_schema": 2,
                "fingerprint": FP,
                "measurements": [{"group": "g0", "bits": 8, "damage": -0.5}],
            }
        )
    )

    with pytest.raises(ArtifactError, match=r"\$\.measurements\[0\]"):
        JsonScanCheckpointFile(path).load(FP)


def test_append_leaves_no_temp_file_behind(tmp_path) -> None:
    store = JsonScanCheckpointFile(tmp_path / "scan.checkpoint.json")

    store.append(FP, Measurement(group="g0", bits=8, damage=0.001))

    assert [p.name for p in tmp_path.iterdir()] == ["scan.checkpoint.json"]
