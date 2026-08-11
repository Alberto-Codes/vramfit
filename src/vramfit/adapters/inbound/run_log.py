"""Run-log wiring shared by the long-running commands.

`SafeRunLog` enforces the ADR-0011 failure policy for every command
that emits run-log events: a write failure warns once on the human
channel and disables further events — pipeline work outlives its
telemetry, and a halt event can never displace the error it reports.
Every event carries a ``run_id``, so reruns and resumes stay
separable in one file. Scan-specific event helpers stay in
[vramfit.adapters.inbound.scan_events][].

Examples:
    Wrap a sink for one run:

    ```python
    safe = SafeRunLog(JsonlRunLogFile(path), path=path)
    safe.emit("pack_started", {"recipe": "recipe.json"})
    ```

See Also:
    - [vramfit.adapters.outbound.run_log_jsonl][]: The sink this wraps.
"""

from __future__ import annotations

import resource
import uuid
from collections.abc import Mapping
from pathlib import Path

import typer

from vramfit.ports.outbound import RunLogSink


def rss_hwm_gb() -> float:
    """Report the process resident-set high-water mark in GB.

    Returns:
        ``ru_maxrss`` converted from KiB (the Linux unit) to GB,
        rounded to two decimals.
    """
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 976_562.5, 2)


class SafeRunLog:
    """`RunLogSink` wrapper implementing the run-log failure policy.

    The first failed write echoes one ``warning:`` line and disables
    the run log for the rest of the process. Nothing is silent, no
    pipeline work dies for its telemetry, and an emit inside an error
    handler cannot displace the error it reports (ADR-0011).

    Attributes:
        run_id (str): Twelve hex characters stamped on every event.

    Examples:
        A dead sink swallows nothing silently:

        ```python
        safe = SafeRunLog(sink, path=path)
        safe.emit("scan_started", {})  # warns once if the sink fails
        ```
    """

    def __init__(self, sink: RunLogSink, *, path: Path) -> None:
        """Wrap a sink and mint the run identity.

        Args:
            sink: The real sink to protect.
            path: The run-log file the sink writes, named in the
                disable warning.
        """
        self._sink = sink
        self._path = path
        self._dead = False
        self.run_id = uuid.uuid4().hex[:12]

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        """Record one event, warning once and disabling on failure.

        The warning quotes the run-log file and the event that died,
        so the one line the user sees points at the sink and the emit
        that broke.

        Args:
            event: Past-tense event name.
            fields: JSON-representable payload. ``run_id`` is added.
        """
        if self._dead:
            return
        try:
            self._sink.emit(event, {"run_id": self.run_id, **fields})
        except (OSError, TypeError, ValueError) as exc:
            self._dead = True
            typer.echo(
                f'warning: run log "{self._path}" disabled at "{event}": {exc}',
                err=True,
            )
