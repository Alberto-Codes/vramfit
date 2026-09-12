"""Wrap calibration prose in the checkpoint's own model-turn frame.

A channel-locked instruct checkpoint prices raw prose at degenerate
perplexity and the same prose inside its serving frame at sane values
(vramfit issue #423). This script builds the framed calibration file
for such a target. It wraps the prose in repeated blocks. Each block
renders one complete conversation: a fixed user turn, then the
model-turn opening the template answers in, then a prose chunk as the
answer, then the turn close. The block closes every channel it
opens.

The script reads the frame from the checkpoint's own chat template, so
each checkpoint gets the frame that checkpoint defines. A checkpoint
that carries no chat template is refused, not guessed at.

The block targets a fixed token count, 512 by default, so a
512-token instrument window contains a frame boundary when blocks
stay at or under the target. Alignment stays approximate:
instruments slice a raw token stream, so windows cross block
boundaries. State that convention beside every published number.

The script verifies its output. It checks each frame marker encodes
to one control id, refuses a frame that writes a marker the
vocabulary lost, re-encodes the framed file, and reports block count
and token totals.

Examples:
    Build a framed file and verify it:

    ```console
    $ uv run python scripts/frame_calibration.py --model ./model
        --text calibration.txt --out calibration-framed.txt
    ```
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

FRAME_USER_TURN = "Continue the passage."
PROSE_SLOT = "VRAMFIT_PROSE_SLOT"
DEFAULT_BLOCK_TOKENS = 512


def encode(tokenizer: Any, text: str) -> list[int]:
    """Encode ``text`` without the tokenizer's own added specials."""
    return tokenizer(text, add_special_tokens=False).input_ids


def build_frame(tokenizer: Any) -> tuple[str, str]:
    """Render the checkpoint's own model-turn frame.

    The template renders the model turn two ways. The generation
    prompt asks the model to answer. The completed render shows the
    template's own finished turn, which the function splits at the
    prose slot. The function reads the control tokens of each, then
    takes the completed render's half when that half carries a
    control token the generation prompt lacks. Such a token is the
    template closing what the generation prompt left open.

    The suffix always comes from the completed render. Both halves
    come from the chat template, never from a table of per-family
    marker strings. A template that ignores ``add_generation_prompt``
    renders the same text either way, which opens no model turn, so
    the function refuses it.

    Args:
        tokenizer: The target model's tokenizer.

    Returns:
        The frame prefix and the frame suffix. The prose goes between
        them.

    Raises:
        ValueError: If the checkpoint carries no chat template, the
            template fails to render, the template writes no
            generation prompt, or the render drops the answer.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "the checkpoint carries no chat template, so it defines no frame."
            " A framed scan needs chat_template.jinja, or a chat_template"
            " entry in tokenizer_config.json"
        )
    user_turn = {"role": "user", "content": FRAME_USER_TURN}
    answer_turn = {"role": "assistant", "content": PROSE_SLOT}
    try:
        generation = tokenizer.apply_chat_template(
            [user_turn], tokenize=False, add_generation_prompt=True
        )
        user_only = tokenizer.apply_chat_template(
            [user_turn], tokenize=False, add_generation_prompt=False
        )
        rendered = tokenizer.apply_chat_template(
            [user_turn, answer_turn], tokenize=False
        )
    except Exception as err:  # any template failure refuses the checkpoint
        raise ValueError(f"the chat template did not render: {err}") from err
    if not generation or generation == user_only:
        raise ValueError(
            "the chat template wrote no model-turn generation prompt, so the"
            " frame would close a model turn it never opened"
        )
    answered, found, suffix = rendered.partition(PROSE_SLOT)
    if not found:
        raise ValueError("the chat template dropped the assistant answer")
    closes = set(frame_markers(tokenizer, answered, "")) - set(
        frame_markers(tokenizer, generation, "")
    )
    return (answered if closes else generation), suffix


def control_tokens(tokenizer: Any) -> dict[str, int]:
    """Map every control token in the vocabulary to its id.

    A control token is one the tokenizer holds in its added-token
    table, or names as a special token. Both kinds encode atomically.
    Some checkpoints omit a frame marker from ``all_special_ids``
    while the added-token table still carries it, so read both.

    Args:
        tokenizer: The target model's tokenizer.

    Returns:
        Each control token's text against its id.
    """
    tokens: dict[str, int] = {}
    for token_id, token in getattr(tokenizer, "added_tokens_decoder", {}).items():
        content = getattr(token, "content", token)
        if isinstance(content, str) and content:
            tokens[content] = token_id
    specials = zip(
        tokenizer.all_special_tokens, tokenizer.all_special_ids, strict=False
    )
    tokens.update({c: i for c, i in specials if isinstance(c, str) and c})
    return tokens


def frame_markers(tokenizer: Any, prefix: str, suffix: str) -> tuple[str, ...]:
    """List the control tokens the frame emits.

    The tokenizer decides. Encoding the frame and reading back the
    control ids reports what the frame emits, where a text scan would
    also report a vocabulary token the frame only contains.

    Args:
        tokenizer: The target model's tokenizer.
        prefix: The frame prefix.
        suffix: The frame suffix.

    Returns:
        Each control token the frame emits, longest first, without
        duplicates. Longest first so a marker that contains a shorter
        marker matches ahead of it.
    """
    by_id = {i: t for t, i in control_tokens(tokenizer).items()}
    ids = encode(tokenizer, prefix) + encode(tokenizer, suffix)
    found = {by_id[i] for i in ids if i in by_id}
    return tuple(sorted(found, key=lambda t: (-len(t), t)))


def control_shape(tokenizer: Any) -> re.Pattern[str] | None:
    """Build the pattern this checkpoint spells its control tokens in.

    The shape comes from the checkpoint's own control tokens: each one
    contributes the pair of delimiters it opens and closes with. A
    token that opens or closes on a letter or a digit contributes
    nothing, because that pair also matches ordinary prose.

    Args:
        tokenizer: The target model's tokenizer.

    Returns:
        A pattern that matches control-token spellings, or ``None``
        when the vocabulary spells none of them with delimiters.
    """
    pairs = {
        (token[0], token[-1])
        for token in control_tokens(tokenizer)
        if len(token) > 1 and not token[0].isalnum() and not token[-1].isalnum()
    }
    if not pairs:
        return None
    return re.compile(
        "|".join(
            f"{re.escape(opener)}[^{re.escape(closer)}]*{re.escape(closer)}"
            for opener, closer in sorted(pairs)
        )
    )


def unparsed_markers(
    tokenizer: Any, markers: tuple[str, ...], prefix: str, suffix: str
) -> list[str]:
    """Find frame markers the tokenizer never emitted as control ids.

    ``markers`` holds what the tokenizer did emit. Removing those from
    the frame text leaves whatever the template wrote and the
    tokenizer read as prose. Any control-shaped text left over is a
    marker this vocabulary lost.

    Args:
        tokenizer: The target model's tokenizer.
        markers: The control tokens the frame emitted, longest first.
        prefix: The frame prefix.
        suffix: The frame suffix.

    Returns:
        Each control-shaped string the frame writes but does not
        encode, in the order it appears.
    """
    shape = control_shape(tokenizer)
    if shape is None:
        return []
    leftover = prefix + suffix
    for marker in markers:
        leftover = leftover.replace(marker, " ")
    return shape.findall(leftover)


def control_core(token: str) -> str:
    """Strip a control token down to the word it names.

    Args:
        token: One control token's text.

    Returns:
        The token without its leading and trailing delimiters.
    """
    start, end = 0, len(token)
    while start < end and not token[start].isalnum():
        start += 1
    while end > start and not token[end - 1].isalnum():
        end -= 1
    return token[start:end]


def control_pairs(tokenizer: Any) -> tuple[tuple[str, str], ...]:
    """Pair the control tokens that open and close the same channel.

    Two control tokens pair when they name the same word and spell
    their delimiters differently. The checkpoint's own vocabulary
    supplies both spellings, so no family table is needed. A word
    with one spelling opens nothing, and a word with three is
    ambiguous, so both stay unpaired.

    Args:
        tokenizer: The target model's tokenizer.

    Returns:
        Each pair of control tokens, sorted inside the pair.
    """
    cores: dict[str, set[str]] = {}
    for token in control_tokens(tokenizer):
        core = control_core(token)
        if core and core != token:
            cores.setdefault(core, set()).add(token)
    pairs = [tuple(sorted(m)) for m in cores.values() if len(m) == 2]  # noqa: PLR2004
    return tuple(sorted(pairs))  # ty: ignore[invalid-return-type]


def unbalanced_pair(
    tokenizer: Any, prefix: str, suffix: str
) -> tuple[str, str, int, int] | None:
    """Find a channel the assembled block opens but never closes.

    Args:
        tokenizer: The target model's tokenizer.
        prefix: The frame prefix.
        suffix: The frame suffix.

    Returns:
        The first pair the block spells an unequal number of times,
        with each count, or ``None`` when every pair balances.
    """
    pairs = control_pairs(tokenizer)
    if not pairs:
        return None
    known = sorted(control_tokens(tokenizer), key=lambda t: (-len(t), t))
    pattern = re.compile("|".join(re.escape(t) for t in known))
    counts = Counter(pattern.findall(prefix + suffix))
    for opener, closer in pairs:
        if counts[opener] != counts[closer]:
            return opener, closer, counts[opener], counts[closer]
    return None


def verify_frame(
    tokenizer: Any, markers: tuple[str, ...], prefix: str, suffix: str
) -> None:
    """Check the frame against the vocabulary.

    Each marker must encode to exactly one id, and that id must be the
    control token's own. The frame must leave no control-shaped text
    the tokenizer read as prose, and must close every channel it
    opens. A marker that splits into pieces is prose to the model, not
    a frame.

    Args:
        tokenizer: The target model's tokenizer.
        markers: The frame's control tokens.
        prefix: The frame prefix, which carries any opening bos token.
        suffix: The frame suffix.

    Raises:
        ValueError: If the frame carries no control token, a marker is
            not one control id, the frame writes a marker the
            vocabulary lost, the block leaves a channel open, or the
            bos marker disagrees with ``bos_token_id``.
    """
    if not markers:
        raise ValueError("the frame carries no control token, so it frames nothing")
    known = control_tokens(tokenizer)
    for marker in markers:
        ids = encode(tokenizer, marker)
        if len(ids) != 1 or ids[0] != known.get(marker):
            raise ValueError(
                f"frame marker {marker!r} is not one control id in this vocabulary"
            )
    unparsed = unparsed_markers(tokenizer, markers, prefix, suffix)
    if unparsed:
        raise ValueError(
            f"the chat template writes {unparsed[0]!r}, which this vocabulary"
            " reads as prose rather than as one control id"
        )
    unbalanced = unbalanced_pair(tokenizer, prefix, suffix)
    if unbalanced:
        opener, closer, opened, closed = unbalanced
        raise ValueError(
            f"the frame writes {opener!r} {opened} times against {closer!r}"
            f" {closed} times, so the block leaves a channel open"
        )
    bos = getattr(tokenizer, "bos_token", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos and bos_id is not None and bos in prefix:
        encoded = encode(tokenizer, bos)[0]
        if encoded != bos_id:
            raise ValueError(f"{bos!r} encodes to {encoded}, bos_token_id is {bos_id}")


def build_framed_text(
    tokenizer: Any, prose: str, block_tokens: int, prefix: str, suffix: str
) -> str:
    """Assemble framed blocks that each target ``block_tokens`` tokens.

    Args:
        tokenizer: The target model's tokenizer.
        prose: The raw calibration text.
        block_tokens: Token count each block targets.
        prefix: The frame prefix.
        suffix: The frame suffix.

    Returns:
        The framed calibration text.

    Raises:
        ValueError: If the prose encodes to a control id, the prose is
            empty, or the frame alone reaches ``block_tokens``.
    """
    frame_len = len(encode(tokenizer, prefix)) + len(encode(tokenizer, suffix))
    chunk_len = block_tokens - frame_len
    if chunk_len < 2:  # noqa: PLR2004 - one next-token prediction needs two
        raise ValueError(f"frame ({frame_len} tokens) leaves no room in {block_tokens}")
    prose_ids = encode(tokenizer, prose)
    if not prose_ids:
        raise ValueError("prose is empty")
    stray = sorted(set(prose_ids) & set(control_tokens(tokenizer).values()))
    if stray:
        raise ValueError(f"prose encodes to control ids {stray}")
    blocks = []
    for start in range(0, len(prose_ids), chunk_len):
        chunk = tokenizer.decode(prose_ids[start : start + chunk_len])
        blocks.append(f"{prefix}{chunk}{suffix}")
    return "".join(blocks)


def check_reencode(
    tokenizer: Any, framed: str, markers: tuple[str, ...], ids: list[int]
) -> str | None:
    """Compare each marker's text count against its id count.

    Args:
        tokenizer: The target model's tokenizer.
        framed: The framed calibration text.
        markers: The frame's control tokens, longest first.
        ids: The plain re-encode of ``framed``.

    Returns:
        The first disagreement as a message, or ``None`` when every
        marker parsed.
    """
    # One left-to-right pass, longest marker first, so a marker that
    # contains a shorter marker does not count it twice.
    pattern = re.compile("|".join(re.escape(m) for m in markers))
    in_text = Counter(pattern.findall(framed))
    in_ids = Counter(ids)
    known = control_tokens(tokenizer)
    for marker in markers:
        want = in_text[marker]
        got = in_ids[known[marker]]
        if want != got:
            return (
                f"{got} {marker!r} ids against {want} in the text"
                " — controls did not parse"
            )
    return None


def _parse_args() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", required=True, help="tokenizer checkpoint path")
    parser.add_argument("--text", required=True, type=Path, help="raw calibration text")
    parser.add_argument("--out", required=True, type=Path, help="framed output path")
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=DEFAULT_BLOCK_TOKENS,
        help="tokens each framed block targets",
    )
    return parser.parse_args()


def main() -> int:
    """Build the framed file, verify the re-encode, and report."""
    args = _parse_args()

    # Import after argparse errors, so `--help` needs no scan extra.
    from transformers import AutoTokenizer  # noqa: PLC0415

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prefix, suffix = build_frame(tokenizer)
        markers = frame_markers(tokenizer, prefix, suffix)
        verify_frame(tokenizer, markers, prefix, suffix)
        prose = args.text.read_text(encoding="utf-8")
        framed = build_framed_text(tokenizer, prose, args.block_tokens, prefix, suffix)
    except (ValueError, OSError) as err:
        print(f"error: {args.model}: {err}", file=sys.stderr)
        return 1

    # Verify the re-encode before any write, so a failure leaves no file.
    ids = encode(tokenizer, framed)
    problem = check_reencode(tokenizer, framed, markers, ids)
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    try:
        args.out.write_text(framed, encoding="utf-8")
    except OSError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    n_blocks = framed.count(prefix)
    prose_tokens = len(encode(tokenizer, prose))
    print(f"frame prefix: {prefix!r}")
    print(f"frame suffix: {suffix!r}")
    print(f"frame markers: {', '.join(markers)}")
    print(f"blocks: {n_blocks}")
    print(f"prose tokens in: {prose_tokens}")
    print(f"framed tokens out (plain re-encode): {len(ids)}")
    print(f"mean block tokens: {len(ids) / n_blocks:.1f} (target {args.block_tokens})")
    print("OK: every block's frame re-encodes to control ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
