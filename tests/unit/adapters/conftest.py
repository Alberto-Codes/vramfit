"""Shared helpers for the scan-command suites.

`test_cli_scan` and `test_cli_scan_groups` both drive `vramfit scan`
through the typer runner against `MemoryDamageMeter`. The meter seam,
the invocation, and the two-group fixture live here so neither suite
imports the other.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import MemoryDamageMeter
from vramfit.adapters.inbound import cli_scan
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.calibration_digest import calibration_identity
from vramfit.domain.scan import GroupSpec

SPECS = (
    GroupSpec(name="model.layers.0", tensors=("model.layers.0.w",), bytes_fp16=1000),
    GroupSpec(name="model.layers.1", tensors=("model.layers.1.w",), bytes_fp16=2000),
)
DAMAGES = {
    ("model.layers.0", 8): 0.001,
    ("model.layers.0", 4): 0.01,
    ("model.layers.1", 8): 0.0,
    ("model.layers.1", 4): 0.2,
}

runner = CliRunner()

# The calibration text every scan invocation measures against.
CALIBRATION_TEXT = "calibration text"


def install_meter(monkeypatch: pytest.MonkeyPatch, meter: object) -> dict:
    """Replace the meter-building seam and capture its options.

    Args:
        monkeypatch: The patching fixture.
        meter: The meter the command receives.

    Returns:
        The options dict the command passed, filled on invocation.
    """
    captured: dict = {}

    def build(model, calibration, **options):
        captured.update(options)
        return meter

    monkeypatch.setattr(cli_scan, "_build_meter", build)
    return captured


def write_calibration(tmp_path: Path, text: str = CALIBRATION_TEXT) -> Path:
    """Write the calibration text a scan invocation measures.

    Args:
        tmp_path: The test's temporary directory.
        text: The calibration text. A different text stands for the
            corpus being re-issued behind an unchanged path.

    Returns:
        The calibration file's path.
    """
    path = tmp_path / "calib.txt"
    path.write_text(text)
    return path


def calibration_content(tmp_path: Path) -> tuple[str, int]:
    """Read the content identity `invoke_scan`'s calibration file has.

    A test that rebuilds the scan's fingerprint must fold the same
    digest the command folds, or the run halts on the fingerprint
    instead of the behaviour under test. The file may not exist yet,
    so this writes it first, exactly as `invoke_scan` does.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The SHA-256 hex digest and the byte count.
    """
    return calibration_identity(write_calibration(tmp_path))


def invoke_scan(tmp_path: Path, *extra: str, calibration_text: str = CALIBRATION_TEXT):
    """Run ``vramfit scan`` against a temporary calibration file.

    Args:
        tmp_path: The test's temporary directory.
        *extra: Extra command-line arguments.
        calibration_text: The calibration file's text. A different
            text stands for the corpus being re-issued behind an
            unchanged path.

    Returns:
        The runner result and the map path.
    """
    calibration = write_calibration(tmp_path, calibration_text)
    out = tmp_path / "sensitivity.json"
    args = [
        "scan",
        "test/model",
        "--calibration",
        str(calibration),
        "--out",
        str(out),
        "--precisions",
        "8,4",
        *extra,
    ]
    return runner.invoke(app, args), out


def full_meter() -> MemoryDamageMeter:
    """Build a meter that measures every cell of the two-group grid.

    Returns:
        The meter.
    """
    return MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)


def barren_meter() -> MemoryDamageMeter:
    """Build a meter that raises on any `measure` call.

    A run that exits 0 with this meter measured no cell, so the
    checkpoint served every cell the run needed.

    Returns:
        The meter.
    """
    return MemoryDamageMeter(specs=SPECS, damages={}, tokens=64)
