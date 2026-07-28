"""JSONL run-log adapter: structlog-rendered events beside the artifacts.

Implements the machine channel from ADR-0011. Every line is one JSON
object carrying the ``quantfit_runlog`` version, a UTC ISO timestamp,
the event name, and the event's fields. Lines append — a crash keeps
everything emitted before it, and a resumed run appends to the same
file.

Examples:
    Record a scan's first event:

    ```python
    from quantfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile

    sink = JsonlRunLogFile(path)
    sink.emit("scan_started", {"model": "test/model"})
    ```

See Also:
    - [quantfit.ports.outbound][]: `RunLogSink`, which this satisfies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import structlog

RUNLOG_VERSION: Final[int] = 1


def _build_logger(handle: Any) -> Any:
    """Wrap a writable handle in a JSON-rendering structlog logger.

    Args:
        handle: An open text file handle in append mode.

    Returns:
        A bound logger whose ``msg`` writes one JSON line per event.
    """
    return structlog.wrap_logger(
        structlog.PrintLogger(file=handle),
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
    )


@dataclass
class JsonlRunLogFile:
    """`RunLogSink` adapter appending JSON lines to one file.

    Attributes:
        path (Path): The run-log file. Created on first emit, appended
            to afterwards.

    Examples:
        Events survive across adapter instances:

        ```python
        JsonlRunLogFile(path).emit("scan_started", {})
        JsonlRunLogFile(path).emit("scan_finished", {})
        ```
    """

    path: Path
    _logger: Any = field(default=None, repr=False)

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        """Append one event line, flushed immediately.

        Args:
            event: Past-tense event name, e.g. ``cell_measured``.
            fields: JSON-representable payload for the event.

        Raises:
            OSError: If the file cannot be opened or written.
        """
        if self._logger is None:
            handle = self.path.open("a", encoding="utf-8")
            self._logger = _build_logger(handle)
        self._logger.msg(event, quantfit_runlog=RUNLOG_VERSION, **fields)


def read_run_log(path: Path) -> list[dict[str, Any]]:
    """Read a run-log file back into event dicts.

    A convenience for tests and analysis — the file is plain JSONL and
    needs no special reader.

    Args:
        path: The run-log file.

    Returns:
        One dict per line, in file order.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a line is not valid JSON.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
