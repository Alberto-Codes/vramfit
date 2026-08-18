"""``vramfit scan --groups``: the group-subset filter and its resume rules.

The selection stays out of the fingerprint (#282), so a narrowed run and a
wide run share one checkpoint. These tests pin that both ways: a narrow run
reuses a wide run's cells, and a wide run keeps a narrow run's cells.
"""

from __future__ import annotations

import pytest

from tests.fakes import MemoryDamageMeter
from tests.unit.adapters.test_cli_scan import (
    DAMAGES,
    SPECS,
    install_meter,
    invoke_scan,
)
from vramfit.adapters.outbound.run_log_jsonl import read_run_log
from vramfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map

pytestmark = pytest.mark.unit

FIRST = "model.layers.0"
SECOND = "model.layers.1"


def full_meter() -> MemoryDamageMeter:
    return MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES), tokens=64)


def barren_meter() -> MemoryDamageMeter:
    """A meter that raises on any `measure` — proves nothing measured."""
    return MemoryDamageMeter(specs=SPECS, damages={}, tokens=64)


def test_groups_narrows_the_map_to_the_named_group(tmp_path, monkeypatch) -> None:
    install_meter(monkeypatch, full_meter())

    result, out = invoke_scan(tmp_path, "--groups", SECOND)

    assert result.exit_code == 0, result.output
    map_ = load_sensitivity_map(out)
    assert [g.name for g in map_.groups] == [SECOND]
    assert "scanned 1 groups x 2 precisions" in result.output


def test_groups_measures_only_the_selected_cells(tmp_path, monkeypatch) -> None:
    meter = full_meter()
    install_meter(monkeypatch, meter)

    result, _out = invoke_scan(tmp_path, "--groups", SECOND)

    assert result.exit_code == 0, result.output
    assert meter.calls == [(SECOND, 8), (SECOND, 4)]


def test_groups_keeps_discovery_order_regardless_of_the_listed_order(
    tmp_path, monkeypatch
) -> None:
    install_meter(monkeypatch, full_meter())

    result, out = invoke_scan(tmp_path, "--groups", f"{SECOND},{FIRST}")

    assert result.exit_code == 0, result.output
    assert [g.name for g in load_sensitivity_map(out).groups] == [FIRST, SECOND]


def test_unmatched_group_name_halts_before_any_measurement(
    tmp_path, monkeypatch
) -> None:
    meter = full_meter()
    install_meter(monkeypatch, meter)

    result, out = invoke_scan(tmp_path, "--groups", "model.layers.9")

    assert result.exit_code == 1
    assert "model.layers.9" in result.output
    assert meter.calls == []
    assert not out.exists()


def test_unmatched_group_name_logs_the_group_select_stage(
    tmp_path, monkeypatch
) -> None:
    install_meter(monkeypatch, full_meter())

    result, _out = invoke_scan(tmp_path, "--groups", "model.layers.9")

    assert result.exit_code == 1
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    halt = next(e for e in events if e["event"] == "scan_halted")
    assert halt["stage"] == "group_select"
    assert halt["cells_kept"] is None


def test_unmatched_name_beside_a_known_one_still_halts(tmp_path, monkeypatch) -> None:
    install_meter(monkeypatch, full_meter())

    result, _out = invoke_scan(tmp_path, "--groups", f"{FIRST},model.layers.9")

    assert result.exit_code == 1
    assert "model.layers.9" in result.output


def test_empty_groups_entry_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(monkeypatch, full_meter())

    result, _out = invoke_scan(tmp_path, "--groups", f"{FIRST},")

    assert result.exit_code == 2
    assert "an entry is empty" in result.output


def test_repeated_group_name_exits_with_usage_error(tmp_path, monkeypatch) -> None:
    install_meter(monkeypatch, full_meter())

    result, _out = invoke_scan(tmp_path, "--groups", f"{FIRST},{FIRST}")

    assert result.exit_code == 2
    assert "repeats" in result.output


def test_narrow_run_reuses_a_wide_runs_cells_without_refusing(
    tmp_path, monkeypatch
) -> None:
    install_meter(monkeypatch, full_meter())
    first, _out = invoke_scan(tmp_path)
    assert first.exit_code == 0, first.output

    # A meter with no configured damages raises on any measure, so
    # exit 0 proves every selected cell came from the checkpoint.
    install_meter(monkeypatch, barren_meter())
    second, out = invoke_scan(tmp_path, "--groups", SECOND)

    assert second.exit_code == 0, second.output
    assert [g.name for g in load_sensitivity_map(out).groups] == [SECOND]


def test_wide_run_keeps_a_narrow_runs_cells_and_measures_the_rest(
    tmp_path, monkeypatch
) -> None:
    install_meter(monkeypatch, full_meter())
    first, _out = invoke_scan(tmp_path, "--groups", SECOND)
    assert first.exit_code == 0, first.output

    meter = full_meter()
    install_meter(monkeypatch, meter)
    second, out = invoke_scan(tmp_path)

    assert second.exit_code == 0, second.output
    assert meter.calls == [(FIRST, 8), (FIRST, 4)]
    assert [g.name for g in load_sensitivity_map(out).groups] == [FIRST, SECOND]


def test_narrow_run_leaves_the_unselected_cells_in_the_checkpoint(
    tmp_path, monkeypatch
) -> None:
    install_meter(monkeypatch, full_meter())
    invoke_scan(tmp_path)

    install_meter(monkeypatch, barren_meter())
    narrow, _out = invoke_scan(tmp_path, "--groups", SECOND)
    assert narrow.exit_code == 0, narrow.output

    # The narrow run wrote a one-group map. The dropped cells survive,
    # so a wide re-run still measures nothing.
    install_meter(monkeypatch, barren_meter())
    wide, out = invoke_scan(tmp_path)

    assert wide.exit_code == 0, wide.output
    assert [g.name for g in load_sensitivity_map(out).groups] == [FIRST, SECOND]


def test_narrow_run_resumes_and_reports_the_selected_grid(
    tmp_path, monkeypatch
) -> None:
    install_meter(monkeypatch, full_meter())
    invoke_scan(tmp_path)

    install_meter(monkeypatch, barren_meter())
    result, _out = invoke_scan(tmp_path, "--groups", SECOND)

    assert "resuming: 2 of 2 cells done" in result.output


def test_run_log_records_the_requested_selection(tmp_path, monkeypatch) -> None:
    install_meter(monkeypatch, full_meter())

    result, _out = invoke_scan(tmp_path, "--groups", SECOND)

    assert result.exit_code == 0, result.output
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    started = next(e for e in events if e["event"] == "scan_started")
    assert started["groups"] == [SECOND]


def test_run_log_records_no_selection_as_null(tmp_path, monkeypatch) -> None:
    install_meter(monkeypatch, full_meter())

    result, _out = invoke_scan(tmp_path)

    assert result.exit_code == 0, result.output
    events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
    started = next(e for e in events if e["event"] == "scan_started")
    assert started["groups"] is None


def test_map_of_a_narrowed_run_records_the_full_group_by(tmp_path, monkeypatch) -> None:
    install_meter(monkeypatch, full_meter())

    result, out = invoke_scan(tmp_path, "--groups", SECOND)

    assert result.exit_code == 0, result.output
    # The selection is not provenance, so the map records only the
    # granularity it keyed on.
    assert load_sensitivity_map(out).scan.group_by == "layer"
