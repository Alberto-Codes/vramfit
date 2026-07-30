"""Shared subprocess plumbing for the GGUF toolchain adapters.

One place owns how a llama.cpp tool runs and fails: stderr merges
into stdout so the failure tail is the tool's real last words, a
signal death is named, and a zero-exit tool that wrote nothing must
not pass as success (ADR-0011, ADR-0012). The pack and smoke
adapters both drive tools through here, so their error messages
cannot drift apart.

Examples:
    Run one tool and keep its output for inspection:

    ```python
    output = run_tool([str(bin_), "-m", str(model)], stage="smoke")
    ```

See Also:
    - [quantfit.adapters.outbound.gguf.pack][]: The convert and
      quantize stages.
    - [quantfit.adapters.outbound.gguf.smoke][]: The smoke stage.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Final

from quantfit.adapters.outbound.gguf.types import PackError

_TAIL_LINES: Final[int] = 15


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
    last words, whichever stream carried them.

    Args:
        command: Argument vector. The first element is the tool path.
        stage: Stage name for the error message (``convert``,
            ``quantize``, or ``smoke``).
        timeout_seconds: Kill the tool after this long. None waits
            forever — a 49B conversion legitimately runs for a long
            time, and a wrong guess kills real work.

    Returns:
        The tool's merged output.

    Raises:
        PackError: If the tool cannot start, exits nonzero, dies to a
            signal (named in the message), or exceeds the timeout.
            The message carries the tool's last output lines.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed tool paths from the composition root, no shell
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except OSError as exc:
        raise PackError(f"{stage}: cannot run {command[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        raise PackError(
            f"{stage} exceeded {timeout_seconds:g} s and was killed:\n{tail_of(output)}"
        ) from exc
    if completed.returncode != 0:
        code = completed.returncode
        died = (
            f"killed by signal {signal.Signals(-code).name}"
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
