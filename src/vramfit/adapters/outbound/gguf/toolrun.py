"""Shared subprocess plumbing for the GGUF toolchain adapters.

One place owns how a llama.cpp tool runs and fails (ADR-0011,
ADR-0012). `run_tool` merges stderr into stdout so the failure tail
is the tool's real last words, and names a signal death. `sized_file`
rejects a zero-exit tool that wrote no usable file. The pack and
smoke adapters both drive tools through here, so their error
messages cannot drift apart.

Every failure here raises `PackError` (ADR-0011). `_signal_name`
exists to hold that line. `signal.Signals` has no member between
`SIGRTMIN` and `SIGRTMAX`, so naming a realtime signal death raised
`ValueError` straight through the boundary (#253).

`run_tool` replaces undecodable bytes instead of refusing them.
llama.cpp truncates its metadata previews mid-character on a
byte-level BPE tokenizer. A strict decode raised `UnicodeDecodeError`
out of a successful quantize and skipped every guard that reads the
output (#247). Replacement never invents a scanner match. It can
delete one, but only inside the line that carried the bad bytes.
#252 carries that residual risk. `surrogateescape` instead writes
unpaired surrogates into the run log. RFC 8259 section 8.2 leaves a
reader's handling of those undefined.

Examples:
    Run one tool and keep its output for inspection:

    ```python
    output = run_tool([str(bin_), "-m", str(model)], stage="smoke")
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: The convert and
      quantize stages.
    - [vramfit.adapters.outbound.gguf.smoke][]: The smoke stage.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.gguf.types import PackError

_TAIL_LINES: Final[int] = 15


def _signal_name(number: int) -> str:
    """Name a signal, falling back to its number.

    `signal.Signals` carries the standard signals and `SIGRTMIN` and
    `SIGRTMAX`. It carries no member between them, so a realtime
    signal raises `ValueError` on lookup. That escaped `run_tool` as a
    `ValueError` and defeated every `except PackError` above it
    (#253).

    Args:
        number: A positive signal number.

    Returns:
        The signal's enum name. A signal outside the enum returns its
        number and the word ``unnamed``, so a reader can tell the two
        apart.

    Examples:
        A standard signal names itself:

        ```python
        assert _signal_name(signal.SIGKILL) == "SIGKILL"
        ```
    """
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"{number} (unnamed)"


def tail_of(output: str) -> str:
    """Keep a tool's last output lines for an error message.

    Args:
        output: The tool's merged stdout and stderr.

    Returns:
        The last lines, or a placeholder when there are none.
    """
    tail = "\n".join(output.splitlines()[-_TAIL_LINES:])
    return tail or "(no output captured)"


def run_tool(
    command: list[str], stage: str, timeout_seconds: float | None = None
) -> str:
    """Run one toolchain subprocess, translating failure to `PackError`.

    stderr merges into stdout so the failure tail is the tool's real
    last words, whichever stream carried them. A killed tool's
    partial output decodes here, because CPython hands that stream
    over as raw bytes.

    Args:
        command: Argument vector. The first element is the tool path.
        stage: Stage name for the error message (``convert``,
            ``quantize``, or ``smoke``).
        timeout_seconds: Kill the tool after this long. None waits
            forever — a 49B conversion legitimately runs for a long
            time, and a wrong guess kills real work.

    Returns:
        The tool's merged output. Undecodable bytes become U+FFFD.

    Raises:
        PackError: If the tool cannot start, exits nonzero, dies to a
            signal, or exceeds the timeout. A signal death names the
            signal, or reports its number when the enum has no name.
            The message carries the tool's last output lines.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed tool paths from the composition root, no shell
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except OSError as exc:
        raise PackError(f"{stage}: cannot run {command[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        # CPython builds this exception from the raw read chunks on
        # POSIX, so the encoding above never reaches it. Decode here
        # or the killed tool's last words reach nobody.
        captured = exc.stdout or b""
        output = (
            captured
            if isinstance(captured, str)
            else captured.decode("utf-8", errors="replace")
        )
        raise PackError(
            f"{stage} exceeded {timeout_seconds:g} s and was killed:\n{tail_of(output)}"
        ) from exc
    if completed.returncode != 0:
        code = completed.returncode
        died = (
            f"killed by signal {_signal_name(-code)}"
            if code < 0
            else f"failed with exit code {code}"
        )
        raise PackError(f"{stage} {died}:\n{tail_of(completed.stdout or '')}")
    return completed.stdout or ""


def sized_file(path: Path, stage: str) -> int:
    """Measure a tool's output file, rejecting absent or empty results.

    Args:
        path: The file the tool reported writing.
        stage: Stage name for the error message.

    Returns:
        The file size in bytes, always positive.

    Raises:
        PackError: If the file is absent, empty, or unreadable — a
            zero-exit tool that wrote nothing must not pass as
            success.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PackError(f"{stage} output {path} cannot be inspected: {exc}") from exc
    if size == 0:
        raise PackError(f"{stage} exited 0 but wrote an empty file at {path}")
    return size
