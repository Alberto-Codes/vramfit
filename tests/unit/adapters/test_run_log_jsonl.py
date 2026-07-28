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
