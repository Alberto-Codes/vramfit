from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tests.fakes import MemoryDamageMeter
from vramfit.adapters.inbound import cli_scan
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from vramfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map
from vramfit.domain.model import ScanMeta
from vramfit.domain.scan import GroupSpec, Measurement, scan_fingerprint

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


def install_meter(monkeypatch, meter) -> dict:
    captured: dict = {}

    def build(model, calibration, **options):
        captured.update(options)
        return meter

    monkeypatch.setattr(cli_scan, "_build_meter", build)
    return captured


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
    assert "vramfit[scan]" in result.output


def test_backend_import_error_surfaces_as_itself(tmp_path, monkeypatch) -> None:
    def raise_backend_error(*args, **kwargs):
        raise ImportError("requires the SentencePiece library")

    monkeypatch.setattr(cli_scan, "_build_meter", raise_backend_error)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 1
    assert "SentencePiece" in result.output
    assert "vramfit[scan]" not in result.output


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


def test_invalid_within_group_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--within-group", "gptq")

    assert result.exit_code == 2
    assert "--within-group" in result.output


def test_kquant_with_uncovered_precisions_exits_with_usage_error(
    tmp_path, monkeypatch
) -> None:
    # 6-bit has no K-quant port yet (ADR-0018). Rejected before the
    # model load burns an hour.
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--precisions", "8,6", "--within-group", "kquant")

    assert result.exit_code == 2
    assert "kquant covers" in result.output


def test_gguf_with_uncovered_precisions_exits_with_usage_error(
    tmp_path, monkeypatch
) -> None:
    # ADR-0028 refuses nominal 3 at pack, so the gguf method has no
    # 3-bit port. Rejected before the model load burns an hour.
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--precisions", "8,3", "--within-group", "gguf")

    assert result.exit_code == 2
    assert "gguf covers" in result.output


def test_gguf_scan_records_the_method_in_the_map(tmp_path, monkeypatch) -> None:
    damages = {(spec.name, bits): 0.1 for spec in SPECS for bits in (4, 2)}
    captured = install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages, tokens=64)
    )

    result, out = invoke_scan(tmp_path, "--precisions", "4,2", "--within-group", "gguf")

    assert result.exit_code == 0
    map_ = load_sensitivity_map(out)
    assert map_.scan.within_group == "gguf-ref"
    # The map's claim must match what the meter measured with — a
    # map that says gguf-ref over kquant damages is corrupted
    # provenance.
    assert captured["within_group"] == "gguf"


def test_imatrix_with_gguf_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    # gguf-imx is reserved and unbuilt: quantize_q2_0 accepts an
    # importance matrix and ignores it (ADR-0018).
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )
    imatrix = tmp_path / "imatrix.gguf"
    imatrix.write_bytes(b"GGUF")

    result, _ = invoke_scan(
        tmp_path,
        "--precisions",
        "4,2",
        "--within-group",
        "gguf",
        "--imatrix",
        str(imatrix),
    )

    assert result.exit_code == 2
    assert "--imatrix requires --within-group kquant" in result.output


def test_kquant_scan_records_the_method_in_the_map(tmp_path, monkeypatch) -> None:
    damages = {(spec.name, bits): 0.1 for spec in SPECS for bits in (3, 2)}
    captured = install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages, tokens=64)
    )

    result, out = invoke_scan(
        tmp_path, "--precisions", "3,2", "--within-group", "kquant"
    )

    assert result.exit_code == 0
    map_ = load_sensitivity_map(out)
    assert map_.scan.within_group == "kquant-ref"
    # The map's claim must match what the meter measured with — a map
    # that says kquant-ref over rtn damages is corrupted provenance.
    assert captured["within_group"] == "kquant"


def test_imatrix_without_kquant_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )
    imatrix = tmp_path / "im.gguf"
    imatrix.write_bytes(b"GGUF")

    result, _ = invoke_scan(tmp_path, "--imatrix", str(imatrix))

    assert result.exit_code == 2
    assert "--imatrix requires --within-group kquant" in result.output


def test_missing_imatrix_file_exits_before_loading_the_model(
    tmp_path, monkeypatch
) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("the meter must not be built")

    monkeypatch.setattr(cli_scan, "_build_meter", explode)

    result, _ = invoke_scan(
        tmp_path,
        "--within-group",
        "kquant",
        "--imatrix",
        str(tmp_path / "no-such.gguf"),
    )

    assert result.exit_code == 2
    assert "is not a file" in result.output


def test_assisted_scan_records_token_and_imatrix_in_the_map(
    tmp_path, monkeypatch
) -> None:
    damages = {(spec.name, bits): 0.1 for spec in SPECS for bits in (3, 2)}
    captured = install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages, tokens=64)
    )
    imatrix = tmp_path / "im.gguf"
    imatrix.write_bytes(b"GGUF")

    result, out = invoke_scan(
        tmp_path,
        "--precisions",
        "3,2",
        "--within-group",
        "kquant",
        "--imatrix",
        str(imatrix),
    )

    assert result.exit_code == 0, result.output
    map_ = load_sensitivity_map(out)
    assert map_.scan.within_group == "kquant-imx"
    assert map_.scan.imatrix == str(imatrix)
    # The builder must receive the path — a map that claims assisted
    # pricing over unassisted damages is corrupted provenance.
    assert captured["imatrix"] == imatrix


def test_assisted_checkpoint_refuses_an_unassisted_rerun(tmp_path, monkeypatch) -> None:
    damages = {(spec.name, bits): 0.1 for spec in SPECS for bits in (3, 2)}
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages, tokens=64)
    )
    imatrix = tmp_path / "im.gguf"
    imatrix.write_bytes(b"GGUF")
    first, _ = invoke_scan(
        tmp_path,
        "--precisions",
        "3,2",
        "--within-group",
        "kquant",
        "--imatrix",
        str(imatrix),
    )
    assert first.exit_code == 0, first.output

    second, _ = invoke_scan(tmp_path, "--precisions", "3,2", "--within-group", "kquant")

    assert second.exit_code == 1
    assert "different scan" in second.output


def test_assisted_scan_echoes_the_count_and_logs_the_names(
    tmp_path, monkeypatch
) -> None:
    # The console states the count only — a model with fused expert
    # stacks would land 181 names on one line (#191). The run log's
    # meter_built event carries the full list.
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

    class ImatrixAwareFake(MemoryDamageMeter):
        imatrix_covered_count = 1
        imatrix_uncovered = ("model.layers.1.w", "model.layers.2.w")

    damages = {(spec.name, bits): 0.1 for spec in SPECS for bits in (3, 2)}
    install_meter(
        monkeypatch, ImatrixAwareFake(specs=SPECS, damages=damages, tokens=64)
    )
    imatrix = tmp_path / "im.gguf"
    imatrix.write_bytes(b"GGUF")

    result, _ = invoke_scan(
        tmp_path,
        "--precisions",
        "3,2",
        "--within-group",
        "kquant",
        "--imatrix",
        str(imatrix),
    )

    assert result.exit_code == 0, result.output
    assert (
        "imatrix covers 1 of 3 parameters (2 uncovered — the run log names them)"
        in result.output
    )
    assert "model.layers.1.w" not in result.output
    assert "model.layers.2.w" not in result.output
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    built = next(e for e in events if e["event"] == "meter_built")
    assert built["imatrix_covered"] == 1
    assert built["imatrix_uncovered"] == ["model.layers.1.w", "model.layers.2.w"]
    started = events[0]
    assert started["within_group"] == "kquant-imx"
    assert started["imatrix"] == str(imatrix)


def test_assisted_scan_without_a_coverage_split_warns(tmp_path, monkeypatch) -> None:
    # The coverage contract rides on attribute names — a meter that
    # stops reporting the split while an imatrix was given must not
    # pass silently as "unassisted" (ADR-0020).
    damages = {(spec.name, bits): 0.1 for spec in SPECS for bits in (3, 2)}
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages, tokens=64)
    )
    imatrix = tmp_path / "im.gguf"
    imatrix.write_bytes(b"GGUF")

    result, _ = invoke_scan(
        tmp_path,
        "--precisions",
        "3,2",
        "--within-group",
        "kquant",
        "--imatrix",
        str(imatrix),
    )

    assert result.exit_code == 0, result.output
    assert "warning" in result.output
    assert "no coverage split" in result.output


def test_meter_built_reports_null_imatrix_for_meters_without_the_notion(
    tmp_path, monkeypatch
) -> None:
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    built = next(e for e in events if e["event"] == "meter_built")
    assert built["imatrix_covered"] is None
    assert built["imatrix_uncovered"] is None
    assert events[0]["imatrix"] is None


def test_rtn_checkpoint_refuses_a_kquant_rerun(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )
    first, _ = invoke_scan(tmp_path)
    assert first.exit_code == 0

    second, _ = invoke_scan(tmp_path, "--within-group", "kquant")

    assert second.exit_code == 1
    assert "different scan" in second.output


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
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

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
    assert all(e["vramfit_runlog"] == 2 for e in events)


def test_meter_built_reports_null_offload_for_meters_without_the_notion(
    tmp_path, monkeypatch
) -> None:
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    built = next(e for e in events if e["event"] == "meter_built")
    # Null, not zero: the fake has no offload notion, and a renamed
    # attribute on the torch meter must show as absence, never as
    # "nothing offloaded".
    assert built["offloaded_groups"] is None


def test_meter_built_reports_the_offloaded_group_count(tmp_path, monkeypatch) -> None:
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

    class OffloadAwareFake(MemoryDamageMeter):
        offloaded_group_count = 1

    meter = OffloadAwareFake(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    install_meter(monkeypatch, meter)

    result, _ = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    built = next(e for e in events if e["event"] == "meter_built")
    assert built["offloaded_groups"] == 1


def test_halted_scan_logs_the_failing_cell(tmp_path, monkeypatch) -> None:
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

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
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )
    custom = tmp_path / "elsewhere.jsonl"

    result, _ = invoke_scan(tmp_path, "--runlog", str(custom))

    assert result.exit_code == 0, result.output
    assert read_run_log(custom)[0]["event"] == "scan_started"
    assert not (tmp_path / "sensitivity.runlog.jsonl").exists()


def test_every_event_carries_one_run_id_per_invocation(tmp_path, monkeypatch) -> None:
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log

    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    first, _ = invoke_scan(tmp_path)
    second, _ = invoke_scan(tmp_path)

    assert first.exit_code == 0 and second.exit_code == 0
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    run_ids = {e["run_id"] for e in events}
    assert len(run_ids) == 2
    starts = [e for e in events if e["event"] == "scan_started"]
    assert len({s["run_id"] for s in starts}) == 2


def test_run_log_failure_warns_once_and_the_scan_continues(
    tmp_path, monkeypatch
) -> None:
    from vramfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile

    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    def refuse(self, event, fields):
        raise OSError("No space left on device")

    monkeypatch.setattr(JsonlRunLogFile, "emit", refuse)

    result, out = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    assert result.output.count("warning: run log") == 1
    assert "sensitivity.runlog.jsonl" in result.output
    assert load_sensitivity_map(out).model_id == "test/model"


def test_missing_runlog_directory_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(
        monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)
    )

    result, _ = invoke_scan(tmp_path, "--runlog", str(tmp_path / "nope" / "x.jsonl"))

    assert result.exit_code == 2
    assert "--runlog" in result.output
