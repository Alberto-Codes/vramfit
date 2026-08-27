"""Wrap calibration prose in a fixed model-turn frame.

A channel-locked instruct checkpoint prices raw prose at degenerate
perplexity and the same prose inside its serving frame at sane values
(vramfit issue #423). This script builds the framed calibration file
for such a target. It wraps the prose in repeated blocks. Each block
renders one complete conversation: a fixed user turn, then the model
turn's generation prompt, then a prose chunk as the answer, then the
turn close.

The block targets a fixed token count, 512 by default, so every
512-token instrument window contains a frame. Alignment stays
approximate: instruments slice a raw token stream, so windows cross
block boundaries. State that convention beside every published
number.

The script verifies its output. It re-encodes the framed file with a
plain ``tokenizer(text)`` call — the same call the scan meter makes —
and reports block count, token totals, and frame-token presence.

Examples:
    Build a framed file and verify it:

    ```console
    $ uv run python scripts/frame_calibration.py --model ./model
        --text calibration.txt --out calibration-framed.txt
    ```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

FRAME_PREFIX = (
    "<bos><|turn>user\n"
    "Continue the passage.<turn|>\n"
    "<|turn>model\n"
    "<|channel>thought\n"
    "<channel|>"
)
FRAME_SUFFIX = "<turn|>\n"
DEFAULT_BLOCK_TOKENS = 512


def build_framed_text(tokenizer: Any, prose: str, block_tokens: int) -> str:
    """Assemble framed blocks that each target ``block_tokens`` tokens.

    Args:
        tokenizer: The target model's tokenizer.
        prose: The raw calibration text.
        block_tokens: Token count each block targets.

    Returns:
        The framed calibration text.

    Raises:
        ValueError: If the prose contains a frame marker, or the frame
            alone reaches ``block_tokens``.
    """
    for marker in ("<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"):
        if marker in prose:
            raise ValueError(f"prose contains frame marker {marker!r}")

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    frame_len = len(encode(FRAME_PREFIX)) + len(encode(FRAME_SUFFIX))
    chunk_len = block_tokens - frame_len
    if chunk_len < 2:  # noqa: PLR2004 - one next-token prediction needs two
        raise ValueError(f"frame ({frame_len} tokens) leaves no room in {block_tokens}")
    prose_ids = encode(prose)
    blocks = []
    for start in range(0, len(prose_ids), chunk_len):
        chunk = tokenizer.decode(prose_ids[start : start + chunk_len])
        blocks.append(f"{FRAME_PREFIX}{chunk}{FRAME_SUFFIX}")
    return "".join(blocks)


def main() -> int:
    """Build the framed file, verify the re-encode, and report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="tokenizer checkpoint path")
    parser.add_argument("--text", required=True, type=Path, help="raw calibration text")
    parser.add_argument("--out", required=True, type=Path, help="framed output path")
    parser.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS)
    args = parser.parse_args()

    # Import after argparse errors, so `--help` needs no scan extra.
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prose = args.text.read_text(encoding="utf-8")
    framed = build_framed_text(tokenizer, prose, args.block_tokens)
    args.out.write_text(framed, encoding="utf-8")

    ids = tokenizer(framed, add_special_tokens=False).input_ids
    bos_id = tokenizer.bos_token_id
    n_blocks = framed.count(FRAME_PREFIX)
    n_bos = sum(1 for i in ids if i == bos_id)
    prose_tokens = len(tokenizer(prose, add_special_tokens=False).input_ids)
    print(f"blocks: {n_blocks}")
    print(f"prose tokens in: {prose_tokens}")
    print(f"framed tokens out (plain re-encode): {len(ids)}")
    print(f"bos ids in re-encode: {n_bos}")
    print(f"mean block tokens: {len(ids) / n_blocks:.1f} (target {args.block_tokens})")
    if n_bos != n_blocks:
        print("FAIL: bos count differs from block count — specials did not parse")
        return 1
    print("OK: every block's frame re-encodes to special ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
