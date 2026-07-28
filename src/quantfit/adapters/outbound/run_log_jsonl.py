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
    """Wrap a writable handle in a strict JSON-rendering logger.

    Args:
        handle: An open text file handle in append mode.

    Returns:
        A bound logger whose ``msg`` writes one JSON line per event.
        Non-serializable or non-finite fields raise at emit time.
    """
    return structlog.wrap_logger(
        structlog.PrintLogger(file=handle),
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            # Strict rendering: a non-serializable field raises instead
            # of degrading to repr, and NaN/Infinity raise instead of
            # emitting invalid JSON that only Python parsers accept.
            structlog.processors.JSONRenderer(
                sort_keys=True, allow_nan=False, default=None
            ),
        ],
    )


@dataclass
class JsonlRunLogFile:
    """`RunLogSink` adapter appending JSON lines to one file.

    Rendering is strict: bad fields raise instead of degrading the
    machine contract silently. Identity is the path — the cached
    handle stays out of equality and construction.

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
    _logger: Any = field(default=None, repr=False, init=False, compare=False)

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        """Append one event line, flushed immediately.

        Args:
            event: Past-tense event name, e.g. ``cell_measured``.
            fields: JSON-representable payload for the event.

        Raises:
            OSError: If the file cannot be opened or written.
            TypeError: If a field is not JSON-serializable.
            ValueError: If a field is NaN or infinite — invalid JSON.
        """
        if self._logger is None:
            handle = self.path.open("a", encoding="utf-8")
            self._logger = _build_logger(handle)
        self._logger.msg(event, quantfit_runlog=RUNLOG_VERSION, **fields)


def read_run_log(path: Path) -> list[dict[str, Any]]:
    """Read a run-log file back into event dicts.

    A convenience for tests and analysis — the file is plain JSONL and
    needs no special reader. A torn final line (the signature of a
    crash mid-write) is dropped, honoring the crash-tolerance rule of
    ADR-0011. A torn line anywhere else is corruption and raises.

    Args:
        path: The run-log file.

    Returns:
        One dict per line, in file order.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a non-final line is not valid JSON.
    """
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    events: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            if i == len(lines) - 1:
                break
            raise
    return events
