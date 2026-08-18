"""``vramfit scan --groups``: the group-subset filter and its resume rules.

The selection stays out of the fingerprint (#282), so a narrowed run and a
wide run share one checkpoint. These tests pin that both ways: a narrow run
reuses a wide run's cells, and a wide run keeps a narrow run's cells.

They also pin what a selection must never do. The checkpoint validates
against the whole model before any narrowing, so a foreign or damaged file
halts the run whatever `--groups` names.
"""

from __future__ import annotations

import pytest

from tests.unit.adapters.conftest import (
    DAMAGES,
    SPECS,
    barren_meter,
    full_meter,
    install_meter,
    invoke_scan,
)
from vramfit.adapters.outbound.run_log_jsonl import read_run_log
from vramfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from vramfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map
from vramfit.domain.model import ScanMeta
from vramfit.domain.scan import SCAN_METHOD, Measurement, scan_fingerprint

pytestmark = pytest.mark.unit

FIRST = "model.layers.0"
SECOND = "model.layers.1"
SECOND_CURVE = {8: 0.0, 4: 0.2}
FIRST_CURVE = {8: 0.001, 4: 0.01}


def live_fingerprint(tmp_path) -> str:
    """Build the fingerprint `invoke_scan` produces, for seeding a file.

    A seeded checkpoint must carry this scan's own identity. A mismatch
    would halt on the fingerprint instead of the grid, which passes the
    test for the wrong reason.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The identity string of the default invocation.
    """
    return scan_fingerprint(
        "test/model",
        ScanMeta(
            metric="kl_divergence",
            calibration=str(tmp_path / "calib.txt"),
            calibration_tokens=64,
            precisions=(8, 4),
            group_by="layer",
            started_at="2026-08-18T00:00:00Z",
            within_group=SCAN_METHOD,
        ),
    )


class TestSelection:
    def test_groups_narrows_the_map_to_the_named_group(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())

        result, out = invoke_scan(tmp_path, "--groups", SECOND)

        assert result.exit_code == 0, result.output
        map_ = load_sensitivity_map(out)
        assert [g.name for g in map_.groups] == [SECOND]
        assert map_.groups[0].sensitivity == SECOND_CURVE

    def test_groups_measures_only_the_selected_cells(
        self, tmp_path, monkeypatch
    ) -> None:
        meter = full_meter()
        install_meter(monkeypatch, meter)

        result, _out = invoke_scan(tmp_path, "--groups", SECOND)

        assert result.exit_code == 0, result.output
        assert meter.calls == [(SECOND, 8), (SECOND, 4)]

    def test_listed_order_does_not_reorder_the_map(self, tmp_path, monkeypatch) -> None:
        install_meter(monkeypatch, full_meter())

        result, out = invoke_scan(tmp_path, "--groups", f"{SECOND},{FIRST}")

        assert result.exit_code == 0, result.output
        assert [g.name for g in load_sensitivity_map(out).groups] == [FIRST, SECOND]

    def test_narrowed_map_records_the_granularity_it_keyed_on(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())

        result, out = invoke_scan(tmp_path, "--groups", SECOND)

        assert result.exit_code == 0, result.output
        # The selection is not provenance, so the map records only the
        # granularity it keyed on.
        assert load_sensitivity_map(out).scan.group_by == "layer"


class TestUnmatchedNames:
    def test_unmatched_name_halts_before_any_measurement(
        self, tmp_path, monkeypatch
    ) -> None:
        meter = full_meter()
        install_meter(monkeypatch, meter)

        result, out = invoke_scan(tmp_path, "--groups", "model.layers.9")

        assert result.exit_code == 1
        assert "model.layers.9" in result.output
        assert meter.calls == []
        assert not out.exists()

    def test_unmatched_name_logs_the_group_select_stage(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())

        result, _out = invoke_scan(tmp_path, "--groups", "model.layers.9")

        assert result.exit_code == 1
        events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
        halt = next(e for e in events if e["event"] == "scan_halted")
        assert halt["stage"] == "group_select"
        assert halt["cells_kept"] is None

    def test_unmatched_name_beside_a_known_one_still_halts(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())

        result, _out = invoke_scan(tmp_path, "--groups", f"{FIRST},model.layers.9")

        assert result.exit_code == 1
        assert "model.layers.9" in result.output


class TestFlagParsing:
    @pytest.mark.parametrize(
        "value",
        [f"{FIRST},", f",{FIRST}", "", " "],
        ids=["trailing", "leading", "empty", "blank"],
    )
    def test_blank_entry_exits_with_usage_error(
        self, tmp_path, monkeypatch, value: str
    ) -> None:
        install_meter(monkeypatch, full_meter())

        result, _out = invoke_scan(tmp_path, "--groups", value)

        assert result.exit_code == 2
        assert "an entry is empty" in result.output

    @pytest.mark.parametrize(
        "value",
        [f"{FIRST},{FIRST}", f"{FIRST}, {FIRST}"],
        ids=["exact", "spaced"],
    )
    def test_repeated_name_exits_with_usage_error(
        self, tmp_path, monkeypatch, value: str
    ) -> None:
        install_meter(monkeypatch, full_meter())

        result, _out = invoke_scan(tmp_path, "--groups", value)

        assert result.exit_code == 2
        assert "repeats" in result.output

    def test_space_after_a_comma_is_accepted(self, tmp_path, monkeypatch) -> None:
        install_meter(monkeypatch, full_meter())

        result, out = invoke_scan(tmp_path, "--groups", f"{FIRST}, {SECOND}")

        assert result.exit_code == 0, result.output
        assert [g.name for g in load_sensitivity_map(out).groups] == [FIRST, SECOND]


class TestCheckpointReuse:
    def test_narrow_run_reuses_a_wide_runs_cells_without_refusing(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())
        first, _out = invoke_scan(tmp_path)
        assert first.exit_code == 0, first.output

        install_meter(monkeypatch, barren_meter())
        second, out = invoke_scan(tmp_path, "--groups", SECOND)

        assert second.exit_code == 0, second.output
        map_ = load_sensitivity_map(out)
        assert [g.name for g in map_.groups] == [SECOND]
        assert map_.groups[0].sensitivity == SECOND_CURVE

    def test_wide_run_keeps_a_narrow_runs_cells_and_measures_the_rest(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())
        first, _out = invoke_scan(tmp_path, "--groups", SECOND)
        assert first.exit_code == 0, first.output

        meter = full_meter()
        install_meter(monkeypatch, meter)
        second, out = invoke_scan(tmp_path)

        assert second.exit_code == 0, second.output
        assert meter.calls == [(FIRST, 8), (FIRST, 4)]
        map_ = load_sensitivity_map(out)
        assert {g.name: g.sensitivity for g in map_.groups} == {
            FIRST: FIRST_CURVE,
            SECOND: SECOND_CURVE,
        }

    def test_narrow_run_that_measures_keeps_the_deselected_cells(
        self, tmp_path, monkeypatch
    ) -> None:
        # A wide run halts after FIRST's cells, so SECOND is unmeasured.
        partial = {cell: d for cell, d in DAMAGES.items() if cell[0] == FIRST}
        install_meter(
            monkeypatch, full_meter().__class__(specs=SPECS, damages=partial, tokens=64)
        )
        halted, _out = invoke_scan(tmp_path)
        assert halted.exit_code == 1

        # The narrow run appends SECOND's cells. FIRST's must survive.
        meter = full_meter()
        install_meter(monkeypatch, meter)
        narrow, _out = invoke_scan(tmp_path, "--groups", SECOND)
        assert narrow.exit_code == 0, narrow.output
        assert meter.calls == [(SECOND, 8), (SECOND, 4)]

        install_meter(monkeypatch, barren_meter())
        wide, out = invoke_scan(tmp_path)

        assert wide.exit_code == 0, wide.output
        assert {g.name: g.sensitivity for g in load_sensitivity_map(out).groups} == {
            FIRST: FIRST_CURVE,
            SECOND: SECOND_CURVE,
        }

    def test_partial_selection_measures_only_its_missing_cells(
        self, tmp_path, monkeypatch
    ) -> None:
        partial = {cell: d for cell, d in DAMAGES.items() if cell != (SECOND, 4)}
        install_meter(
            monkeypatch, full_meter().__class__(specs=SPECS, damages=partial, tokens=64)
        )
        halted, _out = invoke_scan(tmp_path)
        assert halted.exit_code == 1

        meter = full_meter()
        install_meter(monkeypatch, meter)
        result, _out = invoke_scan(tmp_path, "--groups", SECOND)

        assert result.exit_code == 0, result.output
        assert meter.calls == [(SECOND, 4)]
        assert "resuming: 1 of 2 cells done" in result.output

    def test_narrow_run_reports_the_cells_it_ignored(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())
        invoke_scan(tmp_path)

        install_meter(monkeypatch, barren_meter())
        result, _out = invoke_scan(tmp_path, "--groups", SECOND)

        assert "ignoring 2 checkpoint cells outside the selection" in result.output
        events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
        resumed = next(e for e in events if e["event"] == "resume_loaded")
        assert (resumed["cells"], resumed["remaining"], resumed["dropped"]) == (2, 0, 2)

    def test_wide_run_reports_no_ignored_cells(self, tmp_path, monkeypatch) -> None:
        install_meter(monkeypatch, full_meter())
        invoke_scan(tmp_path)

        install_meter(monkeypatch, barren_meter())
        result, _out = invoke_scan(tmp_path)

        assert "ignoring" not in result.output
        events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
        resumed = next(e for e in events if e["event"] == "resume_loaded")
        assert resumed["dropped"] == 0


class TestSelectionNeverHidesABadCheckpoint:
    """A selection narrows what a run measures, never what it checks."""

    def seed(self, tmp_path, cells: list[Measurement]) -> None:
        store = JsonScanCheckpointFile(tmp_path / "sensitivity.checkpoint.json")
        for cell in cells:
            store.append(live_fingerprint(tmp_path), cell)

    @pytest.mark.parametrize(
        "extra", [(), ("--groups", SECOND)], ids=["wide", "narrow"]
    )
    def test_a_cell_outside_the_model_halts_the_run(
        self, tmp_path, monkeypatch, extra: tuple[str, ...]
    ) -> None:
        self.seed(
            tmp_path,
            [
                Measurement(group=SECOND, bits=8, damage=0.0),
                Measurement(group="model.layers.7", bits=8, damage=0.5),
            ],
        )
        meter = full_meter()
        install_meter(monkeypatch, meter)

        result, out = invoke_scan(tmp_path, *extra)

        assert result.exit_code == 1, result.output
        assert "different scan" in result.output
        assert "--no-resume" in result.output
        assert meter.calls == []
        assert not out.exists()

    @pytest.mark.parametrize(
        "extra", [(), ("--groups", SECOND)], ids=["wide", "narrow"]
    )
    def test_a_repeated_cell_halts_the_run(
        self, tmp_path, monkeypatch, extra: tuple[str, ...]
    ) -> None:
        # The repeat sits in FIRST, which the narrow run deselects.
        self.seed(
            tmp_path,
            [
                Measurement(group=FIRST, bits=8, damage=0.001),
                Measurement(group=FIRST, bits=8, damage=0.001),
            ],
        )
        install_meter(monkeypatch, full_meter())

        result, out = invoke_scan(tmp_path, *extra)

        assert result.exit_code == 1, result.output
        assert "damaged" in result.output
        assert not out.exists()

    def test_a_bad_checkpoint_logs_the_checkpoint_load_stage(
        self, tmp_path, monkeypatch
    ) -> None:
        self.seed(tmp_path, [Measurement(group="model.layers.7", bits=8, damage=0.5)])
        install_meter(monkeypatch, full_meter())

        result, _out = invoke_scan(tmp_path, "--groups", SECOND)

        assert result.exit_code == 1
        events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
        halt = next(e for e in events if e["event"] == "scan_halted")
        assert halt["stage"] == "checkpoint_load"


class TestRunLog:
    def test_run_log_records_the_requested_selection(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(monkeypatch, full_meter())

        result, _out = invoke_scan(tmp_path, "--groups", SECOND)

        assert result.exit_code == 0, result.output
        events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
        started = next(e for e in events if e["event"] == "scan_started")
        assert started["groups"] == [SECOND]

    def test_run_log_records_no_selection_as_null(self, tmp_path, monkeypatch) -> None:
        install_meter(monkeypatch, full_meter())

        result, _out = invoke_scan(tmp_path)

        assert result.exit_code == 0, result.output
        events = read_run_log(tmp_path / "sensitivity.runlog.jsonl")
        started = next(e for e in events if e["event"] == "scan_started")
        assert started["groups"] is None
