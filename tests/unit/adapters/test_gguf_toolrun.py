from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

from vramfit.adapters.outbound.gguf.pack import _TYPE_FALLBACK
from vramfit.adapters.outbound.gguf.toolrun import (
    _TAIL_LINES,
    run_tool,
    signal_name,
    sized_file,
    tail_of,
)
from vramfit.adapters.outbound.gguf.types import PackError

pytestmark = pytest.mark.unit

# A realtime signal one above SIGRTMIN. `signal.Signals` names
# SIGRTMIN and SIGRTMAX and nothing between them, so this number has
# no enum member (#253). Windows and macOS ship no realtime signals.
_HAS_REALTIME_SIGNALS = hasattr(signal, "SIGRTMIN")
UNNAMED_SIGNAL = int(signal.SIGRTMIN) + 1 if _HAS_REALTIME_SIGNALS else 0
no_realtime_signals = pytest.mark.skipif(
    not _HAS_REALTIME_SIGNALS, reason="platform has no realtime signals"
)

# llama.cpp previews the merges array and truncates the string inside
# a character, so the stream carries a lead byte with no continuation
# byte (#247). U+0120 is the byte-level BPE space marker.
TRUNCATED_MERGES = (
    b"llama_model_loader: - kv  54: tokenizer.ggml.merges arr[str,269443] = "
    b'["\xc4\xa0 \xc4\xa0", "\xc4\xa0 t", "e r", "\xc4'
)

# The real ADR-0028 decision 3 warning, one merged output line. It
# matches `_TYPE_FALLBACK`, which is the guard #247 stopped from
# running.
FALLBACK_WARNING = (
    b"\nwarning: blk.1.ffn_up_exps.weight            - ncols   2688 not "
    b"divisible by 256 (required for type    q3_K) -> falling back to    q4_0"
)
FALLBACK_REWRITE = ("blk.1.ffn_up_exps.weight", "q3_K", "q4_0")


def emitter(
    tmp_path: Path,
    payload: bytes,
    *,
    exit_code: int = 0,
    then: str = "pass",
) -> list[str]:
    """Build a command that writes exact bytes, runs `then`, and exits."""
    script = tmp_path / "emit.py"
    script.write_text(
        "import os, signal, sys, time\n"
        f"sys.stdout.buffer.write({payload!r})\n"
        "sys.stdout.buffer.flush()\n"
        f"{then}\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_run_tool_undecodable_output_returns_replacement_characters(
    tmp_path: Path,
) -> None:
    output = run_tool(emitter(tmp_path, TRUNCATED_MERGES), stage="quantize")

    assert "�" in output
    assert "tokenizer.ggml.merges" in output


def test_run_tool_undecodable_output_still_matches_the_type_fallback_scan(
    tmp_path: Path,
) -> None:
    payload = TRUNCATED_MERGES + FALLBACK_WARNING

    output = run_tool(emitter(tmp_path, payload), stage="quantize")

    # The ADR-0028 decision 3 halt reads this return value. Bad bytes
    # earlier in the stream must not cost it the match.
    assert _TYPE_FALLBACK.findall(output) == [FALLBACK_REWRITE]


def test_run_tool_undecodable_output_on_failure_raises_pack_error(
    tmp_path: Path,
) -> None:
    command = emitter(tmp_path, TRUNCATED_MERGES, exit_code=3)

    with pytest.raises(PackError, match="quantize failed with exit code 3"):
        run_tool(command, stage="quantize")


def test_run_tool_signal_death_names_the_signal(tmp_path: Path) -> None:
    command = emitter(
        tmp_path, b"loading model\n", then="os.kill(os.getpid(), signal.SIGKILL)"
    )

    with pytest.raises(PackError, match="quantize killed by signal SIGKILL"):
        run_tool(command, stage="quantize")


@no_realtime_signals
def test_run_tool_realtime_signal_death_reports_the_number(tmp_path: Path) -> None:
    # `signal.Signals` carries SIGRTMIN and SIGRTMAX and no member
    # between them, so this lookup used to raise `ValueError` straight
    # past every `except PackError` handler above (#253).
    command = emitter(
        tmp_path,
        b"loading model\n",
        then="os.kill(os.getpid(), signal.SIGRTMIN + 1)",
    )

    with pytest.raises(PackError, match=f"quantize killed by signal {UNNAMED_SIGNAL}"):
        run_tool(command, stage="quantize")


def test_signal_name_standard_signal_returns_the_enum_name() -> None:
    assert signal_name(signal.SIGKILL) == "SIGKILL"


@no_realtime_signals
def test_signal_name_realtime_signal_returns_the_number() -> None:
    assert signal_name(UNNAMED_SIGNAL) == str(UNNAMED_SIGNAL)


def test_run_tool_timeout_carries_the_output_the_tool_wrote(tmp_path: Path) -> None:
    command = emitter(tmp_path, b"the tool's real last words\n", then="time.sleep(30)")

    # The killed tool's tail is the operator's only diagnostic, and
    # CPython hands it over as raw bytes on POSIX.
    with pytest.raises(PackError, match="real last words"):
        run_tool(command, stage="smoke", timeout_seconds=0.5)


def test_tail_of_long_output_keeps_the_last_lines() -> None:
    lines = [f"line {index}" for index in range(_TAIL_LINES * 2)]

    tail = tail_of("\n".join(lines))

    assert tail.splitlines() == lines[-_TAIL_LINES:]


def test_tail_of_empty_output_returns_placeholder() -> None:
    assert tail_of("") == "(no output captured)"


def test_sized_file_empty_file_raises_pack_error(tmp_path: Path) -> None:
    empty = tmp_path / "model.gguf"
    empty.touch()

    with pytest.raises(PackError, match="wrote an empty file"):
        sized_file(empty, stage="quantize")
