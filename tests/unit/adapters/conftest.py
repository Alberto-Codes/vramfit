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


def invoke_scan(tmp_path: Path, *extra: str):
    """Run ``vramfit scan`` against a temporary calibration file.

    Args:
        tmp_path: The test's temporary directory.
        *extra: Extra command-line arguments.

    Returns:
        The runner result and the map path.
    """
    calibration = tmp_path / "calib.txt"
    calibration.write_text("calibration text")
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
