from __future__ import annotations

import pytest
from typer.testing import CliRunner

from quantfit.adapters.inbound import cli_scan
from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from quantfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map
from quantfit.domain.scan import GroupSpec, Measurement
from tests.fakes import MemoryDamageMeter

runner = CliRunner()

pytestmark = pytest.mark.unit

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


def install_meter(monkeypatch, meter) -> None:
    monkeypatch.setattr(cli_scan, "_build_meter", lambda *args: meter)


def invoke_scan(tmp_path, *extra: str):
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


def test_scan_writes_a_loadable_map_and_checkpoint(tmp_path, monkeypatch) -> None:
    meter = MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    install_meter(monkeypatch, meter)

    result, out = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    map_ = load_sensitivity_map(out)
    assert map_.model_id == "test/model"
    assert map_.scan.calibration_tokens == 64
    assert [g.name for g in map_.groups] == ["model.layers.0", "model.layers.1"]
    assert map_.groups[1].sensitivity == {8: 0.0, 4: 0.2}
    assert (tmp_path / "sensitivity.checkpoint.json").exists()
    assert "scanned 2 groups x 2 precisions" in result.output


def test_failed_measurement_keeps_checkpoint_and_resumes(tmp_path, monkeypatch) -> None:
    partial = dict(DAMAGES)
    del partial[("model.layers.1", 8)]
    failing = MemoryDamageMeter(specs=SPECS, damages=partial, tokens=64)
    install_meter(monkeypatch, failing)

    first, out = invoke_scan(tmp_path)

    assert first.exit_code == 1
    assert "checkpoint keeps 2 cells" in first.output
    assert not out.exists()

    healthy = MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    install_meter(monkeypatch, healthy)

    second, out = invoke_scan(tmp_path)

    assert second.exit_code == 0, second.output
    assert "resuming: 2 of 4 cells done" in second.output
    assert healthy.calls == [("model.layers.1", 8), ("model.layers.1", 4)]
    assert load_sensitivity_map(out).groups[0].sensitivity == {8: 0.001, 4: 0.01}


def test_checkpoint_from_a_different_scan_fails_with_hint(
    tmp_path, monkeypatch
) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )
    stale = JsonScanCheckpointFile(tmp_path / "sensitivity.checkpoint.json")
    stale.append("other|scan", Measurement(group="model.layers.0", bits=8, damage=0.5))

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "different scan" in result.output
    assert "--no-resume" in result.output


def test_no_resume_discards_a_stale_checkpoint(tmp_path, monkeypatch) -> None:
    meter = MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    install_meter(monkeypatch, meter)
    stale = JsonScanCheckpointFile(tmp_path / "sensitivity.checkpoint.json")
    stale.append("other|scan", Measurement(group="model.layers.0", bits=8, damage=0.5))

    result, out = invoke_scan(tmp_path, "--no-resume")

    assert result.exit_code == 0, result.output
    assert len(meter.calls) == 4
    assert load_sensitivity_map(out).model_id == "test/model"


def test_missing_scan_extra_reports_install_hint(tmp_path, monkeypatch) -> None:
    def raise_import_error(*args):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(cli_scan, "_build_meter", raise_import_error)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "quantfit[scan]" in result.output


def test_meter_build_failure_reports_error(tmp_path, monkeypatch) -> None:
    def raise_value_error(*args):
        raise ValueError("calibration text yields 0 tokens")

    monkeypatch.setattr(cli_scan, "_build_meter", raise_value_error)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "error: calibration text yields 0 tokens" in result.output


@pytest.mark.parametrize(
    "precisions",
    ["4,8", "8,8", "abc", "", "0,-1"],
    ids=["ascending", "duplicate", "not-int", "empty", "non-positive"],
)
def test_invalid_precisions_exit_with_usage_error(
    tmp_path, monkeypatch, precisions: str
) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--precisions", precisions)

    assert result.exit_code == 2
    assert "--precisions" in result.output


def test_invalid_group_by_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--group-by", "block")

    assert result.exit_code == 2
    assert "--group-by" in result.output


def test_completed_checkpoint_reassembles_without_new_measurements(
    tmp_path, monkeypatch
) -> None:
    meter = MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    install_meter(monkeypatch, meter)
    first, _ = invoke_scan(tmp_path)
    assert first.exit_code == 0, first.output
    calls_after_first = list(meter.calls)

    second, out = invoke_scan(tmp_path)

    assert second.exit_code == 0, second.output
    assert meter.calls == calls_after_first
    assert load_sensitivity_map(out).model_id == "test/model"
