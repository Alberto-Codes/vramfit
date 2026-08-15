from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vramfit.adapters.outbound.gguf.toolrun import run_tool, sized_file, tail_of
from vramfit.adapters.outbound.gguf.types import PackError

pytestmark = pytest.mark.unit

# llama.cpp previews the merges array and truncates the string inside
# a character, so the stream carries a lead byte with no continuation
# byte (#247). U+0120 is the byte-level BPE space marker.
TRUNCATED_MERGES = (
    b"llama_model_loader: - kv  54: tokenizer.ggml.merges arr[str,269443] = "
    b'["\xc4\xa0 \xc4\xa0", "\xc4\xa0 t", "e r", "\xc4'
)
TYPE_FALLBACK = b"\nllama_model_loader: converting to q4_0 .. cannot be quantized"


def emitter(tmp_path: Path, payload: bytes, exit_code: int = 0) -> list[str]:
    """Build a command that writes exact bytes to stdout and exits."""
    script = tmp_path / "emit.py"
    script.write_text(
        "import sys\n"
        f"sys.stdout.buffer.write({payload!r})\n"
        "sys.stdout.buffer.flush()\n"
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


def test_run_tool_undecodable_output_keeps_the_scanned_lines_intact(
    tmp_path: Path,
) -> None:
    payload = TRUNCATED_MERGES + TYPE_FALLBACK

    output = run_tool(emitter(tmp_path, payload), stage="quantize")

    # The ADR-0028 and ADR-0016 scans read this return value. A
    # replacement character earlier in the stream must not reach them.
    assert output.endswith("cannot be quantized")


def test_run_tool_undecodable_output_on_failure_raises_pack_error(
    tmp_path: Path,
) -> None:
    command = emitter(tmp_path, TRUNCATED_MERGES, exit_code=3)

    with pytest.raises(PackError, match="quantize failed with exit code 3"):
        run_tool(command, stage="quantize")


def test_tail_of_empty_output_returns_placeholder() -> None:
    assert tail_of("") == "(no output captured)"


def test_sized_file_empty_file_raises_pack_error(tmp_path: Path) -> None:
    empty = tmp_path / "model.gguf"
    empty.touch()

    with pytest.raises(PackError, match="wrote an empty file"):
        sized_file(empty, stage="quantize")
