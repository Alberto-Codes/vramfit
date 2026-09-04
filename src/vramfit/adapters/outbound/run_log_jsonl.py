"""JSONL run-log adapter: structlog-rendered events beside the artifacts.

Implements the machine channel from ADR-0011. Every line is one JSON
object carrying the ``vramfit_runlog`` version, a UTC ISO timestamp,
the event name, and the event's fields. Lines append — a crash keeps
everything emitted before it, and a resumed run appends to the same
file.

`read_run_log` refuses a line that defines one key twice (#283). The
writer cannot produce one, because a colliding field is a duplicate
keyword argument at the `msg` call. So a repeat means a hand edit or a
foreign writer, and it raises at every line position. ADR-0011's
crash-tolerance rule stays scoped to a decode failure on the final
line. `read_run_log` raises every refusal as `RunLogError`, which sits
under the `VramfitError` root and keeps `ValueError` as a base (#346).
`emit` raises the same class for a field the renderer refuses (#475).

`emit` opens the file, appends one line, and closes it (#468). A
cached handle would need an owner to close it, and every composition
root would carry that lifecycle. One open per event keeps the port
and the roots unchanged.

Examples:
    Record a scan's first event:

    ```python
    from vramfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile

    sink = JsonlRunLogFile(path)
    sink.emit("scan_started", {"model": "test/model"})
    ```

See Also:
    - [vramfit.ports.outbound][]: `RunLogSink`, which this satisfies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import structlog

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.errors import VramfitError

RUNLOG_VERSION: Final[int] = 2


class RunLogError(VramfitError, ValueError):
    """A run-log line the reader or the writer refuses.

    `read_run_log` raises it for a repeated key, a non-final line that
    is not valid JSON, and a number literal the parser refuses. The
    message names the file and the line. `emit` raises it for a field
    the renderer refuses: a value that is not JSON-serializable, a
    NaN or infinite number, or a key that collides with the envelope
    (#475). That message names the file and the event.

    The class sits under the `VramfitError` root per ADR-0011 decision
    5. It keeps `ValueError` as a base, so a caller that catches the
    historical type still does. Before #346 the reader raised a plain
    `ValueError`, which escaped the root.

    Examples:
        Catch every refusal the reader makes through the root:

        ```python
        from vramfit.domain.errors import VramfitError

        try:
            read_run_log(path)
        except VramfitError as exc:
            print(exc)
        ```
    """


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
    machine contract silently. Every emit opens the file, appends one
    line, and closes it, so no handle outlives the call (#468).

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

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        """Append one event line and close the file.

        The renderer's `TypeError` and `ValueError` are foreign to the
        port. Both become `RunLogError`, under the root per ADR-0011
        decision 5 (#475). `OSError` passes through unchanged, as the
        port names it.

        Args:
            event: Past-tense event name, e.g. ``cell_measured``.
            fields: JSON-representable payload for the event.

        Raises:
            OSError: If the file cannot be opened or written.
            RunLogError: If a field is not JSON-serializable, is NaN or
                infinite, or collides with an envelope key.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            try:
                _build_logger(handle).msg(
                    event, vramfit_runlog=RUNLOG_VERSION, **fields
                )
            except (TypeError, ValueError) as exc:
                raise RunLogError(f"{self.path}: event {event!r}: {exc}") from exc


def read_run_log(path: Path) -> list[dict[str, Any]]:
    """Read a run-log file back into event dicts.

    A convenience for tests and analysis — the file is plain JSONL and
    needs no special reader. A torn final line (the signature of a
    crash mid-write) is dropped, honoring the crash-tolerance rule of
    ADR-0011. A torn line anywhere else is corruption and raises.

    The drop covers every JSON decode failure on the final line, not
    truncation alone. A tear and a bad prefix do not separate. A hand
    edit that deletes the end of a line writes the same bytes a crash
    writes, so no test of the content tells them apart.

    One sub-case does separate, and the reader still drops it. A
    complete value followed by trailing text decodes cleanly under
    `json.JSONDecoder.raw_decode`, which a torn prefix never does.
    Measured 2026-08-18: two crash artifacts take that shape. A
    complete line padded with NUL bytes follows a power loss under
    delayed allocation. A line that lost its newline before a restart
    appended, ``{"a":1}{"b":2``, follows a crash. Refusing the shape
    would refuse both, which ADR-0011 decision 2 forbids.

    Every other failure raises, at every position (#315). A line
    carrying an integer literal past ``sys.get_int_max_str_digits``
    parses to no number, and that is not a crash signature. Dropping it
    would report one event where the file records two. That refusal
    drops the parser's closing advice, which recommends raising the
    limit this reader enforces. Any other failure keeps its whole
    text.

    A line that repeats a key raises at every position, including the
    last (#283). `emit` cannot write one: a `fields` entry named
    ``event`` or ``vramfit_runlog`` collides with the keyword argument
    and raises `TypeError` at the call. So a repeat means a hand edit or
    a foreign writer, never a crash. Every refusal names the file line,
    counting blank lines that the reader itself skips. #283 gave the
    duplicate-key refusal that locator, and #315 extends it to the
    decode refusals, which reported the line inside the parsed string.
    Every refusal raises `RunLogError`, under the error root (#346).

    Args:
        path: The run-log file.

    Returns:
        One dict per line, in file order.

    Raises:
        OSError: If the file cannot be read.
        RunLogError: If a non-final line is not valid JSON. If any line
            defines the same key twice. If any line carries a number
            literal the parser refuses.
    """
    # Each line keeps its file number, because the blank-line filter
    # would otherwise make the refusal below name the wrong line. A hand
    # edit is what writes a duplicate key, and it is what leaves a stray
    # blank line too.
    lines = [
        (number, line)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    ]
    events: list[dict[str, Any]] = []
    for i, (number, line) in enumerate(lines):
        try:
            events.append(json.loads(line, object_pairs_hook=object_from_pairs))
        except DuplicateKeyError as exc:
            raise RunLogError(f"{path}: line {number}: {exc.message}") from exc
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                break
            raise RunLogError(
                f"{path}: line {number}: invalid JSON: {exc.msg}: column {exc.colno}"
            ) from exc
        except ValueError as exc:
            # The syntax parsed and the value conversion failed. An
            # integer literal past `sys.get_int_max_str_digits` lands
            # here (#260, #315). A crash cannot write one, so the final
            # line earns no drop. No `RunLogError` originates inside
            # the try, so this clause cannot relabel one (#262).
            #
            # The clause after the first semicolon advises raising
            # that limit, which would parse the value this reader
            # refuses. Strict mode bans the semicolon too, so the
            # message drops that clause. The test guards the wording,
            # and an unrelated failure keeps its whole text.
            reason = str(exc)
            if "for integer string conversion" in reason:
                reason = reason.split(";", 1)[0]
            raise RunLogError(
                f"{path}: line {number}: cannot parse JSON: {reason}"
            ) from exc
    return events
