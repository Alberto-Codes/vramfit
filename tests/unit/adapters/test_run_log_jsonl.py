from __future__ import annotations

import pytest

from quantfit.adapters.outbound.run_log_jsonl import (
    RUNLOG_VERSION,
    JsonlRunLogFile,
    read_run_log,
)

pytestmark = pytest.mark.unit


def test_every_line_carries_the_envelope_and_timestamp(tmp_path) -> None:
    path = tmp_path / "scan.runlog.jsonl"

    JsonlRunLogFile(path).emit("scan_started", {"model": "m"})

    (event,) = read_run_log(path)
    assert event["event"] == "scan_started"
    assert event["quantfit_runlog"] == RUNLOG_VERSION
    assert event["model"] == "m"
    assert event["ts"].endswith("Z") or "+" in event["ts"]


def test_events_append_across_adapter_instances(tmp_path) -> None:
    path = tmp_path / "scan.runlog.jsonl"

    JsonlRunLogFile(path).emit("scan_started", {})
    JsonlRunLogFile(path).emit("scan_finished", {})

    assert [e["event"] for e in read_run_log(path)] == [
        "scan_started",
        "scan_finished",
    ]


def test_emit_into_a_missing_directory_raises_os_error(tmp_path) -> None:
    with pytest.raises(OSError):
        JsonlRunLogFile(tmp_path / "nope" / "x.jsonl").emit("scan_started", {})


def test_events_reach_disk_before_the_handle_closes(tmp_path) -> None:
    # The black-box guarantee: a crash must not lose emitted events.
    # structlog's PrintLogger flushes per line — pin it.
    path = tmp_path / "scan.runlog.jsonl"
    sink = JsonlRunLogFile(path)

    sink.emit("scan_started", {})

    assert path.stat().st_size > 0
    assert read_run_log(path)[0]["event"] == "scan_started"


def test_non_serializable_field_raises_instead_of_degrading(tmp_path) -> None:
    sink = JsonlRunLogFile(tmp_path / "x.jsonl")

    with pytest.raises(TypeError):
        sink.emit("scan_started", {"bad": object()})


def test_non_finite_field_raises_instead_of_corrupting_json(tmp_path) -> None:
    sink = JsonlRunLogFile(tmp_path / "x.jsonl")

    with pytest.raises(ValueError, match=r"[Nn]a[Nn]|allow_nan"):
        sink.emit("cell_measured", {"damage": float("nan")})


def test_reader_drops_a_torn_final_line(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    JsonlRunLogFile(path).emit("scan_started", {})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "cell_measu')  # crash mid-write

    events = read_run_log(path)

    assert [e["event"] for e in events] == ["scan_started"]


def test_reader_raises_on_a_torn_middle_line(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"broken\n{"event": "scan_finished", "quantfit_runlog": 1}\n')

    with pytest.raises(ValueError):
        read_run_log(path)
