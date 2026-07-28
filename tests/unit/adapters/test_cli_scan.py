from __future__ import annotations

import pytest
from typer.testing import CliRunner

from quantfit.adapters.inbound import cli_scan
from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from quantfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map
from quantfit.domain.model import ScanMeta
from quantfit.domain.scan import GroupSpec, Measurement, scan_fingerprint
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
    def build(
        model,
        calibration,
        *,
        max_tokens,
        group_by,
        device,
        trust_remote_code,
        gpu_memory,
    ):
        return meter

    monkeypatch.setattr(cli_scan, "_build_meter", build)


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
    def raise_extra_missing(*args, **kwargs):
        raise cli_scan.ScanExtraMissingError(cli_scan.INSTALL_HINT)

    monkeypatch.setattr(cli_scan, "_build_meter", raise_extra_missing)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "quantfit[scan]" in result.output


def test_backend_import_error_surfaces_as_itself(tmp_path, monkeypatch) -> None:
    def raise_backend_error(*args, **kwargs):
        raise ImportError("requires the SentencePiece library")

    monkeypatch.setattr(cli_scan, "_build_meter", raise_backend_error)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "SentencePiece" in result.output
    assert "quantfit[scan]" not in result.output


def test_meter_build_failure_reports_error(tmp_path, monkeypatch) -> None:
    def raise_value_error(*args, **kwargs):
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


def cli_fingerprint(tmp_path) -> str:
    meta = ScanMeta(
        metric="kl_divergence",
        calibration=str(tmp_path / "calib.txt"),
        calibration_tokens=64,
        precisions=(8, 4),
        group_by="layer",
        started_at="unused",
    )
    return scan_fingerprint("test/model", meta)


def test_nan_damage_fails_cleanly_and_keeps_the_checkpoint(
    tmp_path, monkeypatch
) -> None:
    damages = dict(DAMAGES)
    damages[("model.layers.1", 8)] = float("nan")
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages, tokens=64)
    )

    result, out = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "scan halted at model.layers.1 8-bit" in result.output
    assert "finite" in result.output
    assert "checkpoint keeps 2 cells" in result.output
    assert not out.exists()


def test_checkpoint_write_failure_reports_a_clean_error(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    def refuse(self, fingerprint, measurement) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(JsonScanCheckpointFile, "append", refuse)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "scan halted at model.layers.0 8-bit" in result.output
    assert "No space left on device" in result.output


def test_missing_out_directory_exits_before_loading_the_model(
    tmp_path, monkeypatch
) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("the meter must not be built")

    monkeypatch.setattr(cli_scan, "_build_meter", explode)

    result, _ = invoke_scan(tmp_path, "--out", str(tmp_path / "nope" / "s.json"))

    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_duplicated_checkpoint_cell_fails_upfront_with_hint(
    tmp_path, monkeypatch
) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )
    store = JsonScanCheckpointFile(tmp_path / "sensitivity.checkpoint.json")
    cell = Measurement(group="model.layers.0", bits=8, damage=0.5)
    store.append(cli_fingerprint(tmp_path), cell)
    store.append(cli_fingerprint(tmp_path), cell)

    result, out = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "appears twice" in result.output
    assert "--no-resume" in result.output
    assert not out.exists()


def test_precisions_below_two_bits_exit_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--precisions", "8,4,1")

    assert result.exit_code == 2
    assert "floors at 2 bits" in result.output


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


def test_gpu_memory_reaches_the_meter_as_bytes(tmp_path, monkeypatch) -> None:
    received = {}

    def record(model, calibration, *, gpu_memory, **kwargs):
        received["gpu_memory"] = gpu_memory
        return MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)

    monkeypatch.setattr(cli_scan, "_build_meter", record)

    result, _ = invoke_scan(tmp_path, "--gpu-memory", "17GiB")

    assert result.exit_code == 0, result.output
    assert received["gpu_memory"] == 17 * 2**30


def test_gpu_memory_defaults_to_no_cap(tmp_path, monkeypatch) -> None:
    received = {}

    def record(model, calibration, *, gpu_memory, **kwargs):
        received["gpu_memory"] = gpu_memory
        return MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)

    monkeypatch.setattr(cli_scan, "_build_meter", record)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    assert received["gpu_memory"] is None


def test_malformed_gpu_memory_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--gpu-memory", "lots")

    assert result.exit_code == 2
    assert "--gpu-memory" in result.output


def test_gpu_memory_without_auto_device_exits_with_usage_error(
    tmp_path, monkeypatch
) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--gpu-memory", "17GiB", "--device", "cpu")

    assert result.exit_code == 2
    assert "requires --device auto" in result.output


def test_scan_writes_a_run_log_with_the_full_event_story(tmp_path, monkeypatch) -> None:
    from quantfit.adapters.outbound.run_log_jsonl import read_run_log

    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _out = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    names = [e["event"] for e in events]
    assert names[0] == "scan_started"
    assert names[1] == "meter_built"
    assert names.count("cell_measured") == 4
    assert names[-1] == "scan_finished"
    cell = next(e for e in events if e["event"] == "cell_measured")
    assert {"group", "bits", "damage", "seconds", "rss_hwm_gb", "ts"} <= set(cell)
    assert all(e["quantfit_runlog"] == 1 for e in events)


def test_halted_scan_logs_the_failing_cell(tmp_path, monkeypatch) -> None:
    from quantfit.adapters.outbound.run_log_jsonl import read_run_log

    damages = dict(DAMAGES)
    damages[("model.layers.1", 8)] = float("nan")
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages, tokens=64)
    )

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    halted = events[-1]
    assert halted["event"] == "scan_halted"
    assert halted["stage"] == "measure"
    assert halted["group"] == "model.layers.1"
    assert halted["cells_kept"] == 2


def test_runlog_option_overrides_the_default_path(tmp_path, monkeypatch) -> None:
    from quantfit.adapters.outbound.run_log_jsonl import read_run_log

    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )
    custom = tmp_path / "elsewhere.jsonl"

    result, _ = invoke_scan(tmp_path, "--runlog", str(custom))

    assert result.exit_code == 0, result.output
    assert read_run_log(custom)[0]["event"] == "scan_started"
    assert not (tmp_path / "sensitivity.runlog.jsonl").exists()
