from __future__ import annotations

from pathlib import Path

import pytest

from quantfit.adapters.inbound.run_log import SafeRunLog

pytestmark = pytest.mark.unit


class RefusingSink:
    """`RunLogSink` double whose every write fails like a full disk."""

    def __init__(self) -> None:
        """Start the write-attempt counter."""
        self.attempts = 0

    def emit(self, event: str, fields) -> None:
        """Refuse the write, counting the attempt."""
        self.attempts += 1
        raise OSError("No space left on device")


class TestSafeRunLog:
    def test_failed_write_warning_names_file_and_event(self, capsys) -> None:
        path = Path("out/scan.runlog.jsonl")
        safe = SafeRunLog(RefusingSink(), path=path)

        safe.emit("scan_started", {})

        err = capsys.readouterr().err
        assert "scan.runlog.jsonl" in err
        assert '"scan_started"' in err

    def test_failed_write_disables_later_emits(self, capsys) -> None:
        sink = RefusingSink()
        safe = SafeRunLog(sink, path=Path("x.runlog.jsonl"))

        safe.emit("scan_started", {})
        safe.emit("cell_measured", {})

        assert sink.attempts == 1
        assert capsys.readouterr().err.count("warning:") == 1
