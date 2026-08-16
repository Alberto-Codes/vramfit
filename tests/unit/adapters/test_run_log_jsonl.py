from __future__ import annotations

import pytest

from vramfit.adapters.outbound.run_log_jsonl import (
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
    assert event["vramfit_runlog"] == RUNLOG_VERSION
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
    path.write_text('{"broken\n{"event": "scan_finished", "vramfit_runlog": 2}\n')

    with pytest.raises(ValueError):
        read_run_log(path)


def test_emit_cannot_write_a_duplicate_envelope_key(tmp_path) -> None:
    # The premise behind the refusals below: vramfit never produces the
    # defect, because a colliding field is a duplicate keyword argument
    # at the `msg` call. So a duplicate key means a hand edit or a
    # foreign writer (#283).
    sink = JsonlRunLogFile(tmp_path / "x.jsonl")

    with pytest.raises(TypeError, match="vramfit_runlog"):
        sink.emit("scan_started", {"vramfit_runlog": 99})
    with pytest.raises(TypeError, match="event"):
        sink.emit("scan_started", {"event": "other"})


def test_reader_raises_on_a_duplicate_key_in_a_middle_line(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"event": "cell_measured", "damage": 1.0, "damage": 2.0}\n'
        '{"event": "scan_finished", "vramfit_runlog": 2}\n'
    )

    with pytest.raises(ValueError, match='line 1: duplicate key "damage"'):
        read_run_log(path)


def test_reader_names_the_file_line_past_a_blank_line(tmp_path) -> None:
    # The locator must survive the blank-line filter. A hand edit writes
    # the duplicate key, and a hand edit leaves the blank line.
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"event": "scan_started"}\n'
        "\n"
        "\n"
        '{"event": "cell_measured", "damage": 1.0, "damage": 2.0}\n'
    )

    with pytest.raises(ValueError, match='line 4: duplicate key "damage"'):
        read_run_log(path)


def test_reader_raises_on_a_duplicate_key_in_the_final_line(tmp_path) -> None:
    # ADR-0011 decision 2 drops a torn final line, which is the crash
    # signature. A duplicate key parses and no crash writes one, so the
    # drop rule does not reach it (#283).
    path = tmp_path / "x.jsonl"
    JsonlRunLogFile(path).emit("scan_started", {})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "cell_measured", "damage": 1.0, "damage": 2.0}\n')

    with pytest.raises(ValueError, match='line 2: duplicate key "damage"'):
        read_run_log(path)
