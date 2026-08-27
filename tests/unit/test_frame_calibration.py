"""Hermetic tests for ``scripts/frame_calibration.py`` (#423, #437).

The script frames calibration prose for a channel-locked target. The
real tokenizer needs the scan extra, so these tests drive
``build_framed_text`` with a fake that models the parts the script
reads: plain encode, decode, ``all_special_ids``, ``bos_token_id``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    """Import the script by path — ``scripts/`` is not an installed package."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "frame_calibration.py"
    spec = importlib.util.spec_from_file_location("frame_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fc = _load_script()

pytestmark = pytest.mark.unit


class _Batch:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


class FakeTokenizer:
    """Word-level tokenizer with single-id frame markers.

    Markers encode to one id each. Every other whitespace-separated
    word gets its own id on first sight. ``special`` controls which
    marker ids count as special.
    """

    def __init__(
        self,
        markers: tuple[str, ...] = fc.FRAME_MARKERS,
        special: tuple[str, ...] | None = None,
        extra_special_words: tuple[str, ...] = (),
    ) -> None:
        """Build the vocabulary from markers and extra special words."""
        self._vocab: dict[str, int] = {}
        for marker in markers:
            self._vocab[marker] = len(self._vocab)
        for word in extra_special_words:
            self._vocab[word] = len(self._vocab)
        special_tokens = markers if special is None else special
        self.all_special_ids = [
            self._vocab[t] for t in (*special_tokens, *extra_special_words)
        ]
        self.bos_token_id = self._vocab.get("<bos>")
        self._pattern = re.compile(
            "(" + "|".join(re.escape(m) for m in self._vocab) + ")"
        )

    def __call__(self, text: str, add_special_tokens: bool = True) -> _Batch:
        ids: list[int] = []
        for part in self._pattern.split(text):
            if part in self._vocab:
                ids.append(self._vocab[part])
                continue
            for word in part.split():
                if word not in self._vocab:
                    self._vocab[word] = len(self._vocab)
                ids.append(self._vocab[word])
        return _Batch(ids)

    def decode(self, ids: list[int]) -> str:
        rev = {i: t for t, i in self._vocab.items()}
        return " ".join(rev[i] for i in ids)


def test_build_framed_text_prose_survives_in_order() -> None:
    tok = FakeTokenizer()
    prose = " ".join(f"w{i}" for i in range(40))
    framed = fc.build_framed_text(tok, prose, block_tokens=20)
    ids = tok(framed).input_ids
    frame_ids = set(tok(fc.FRAME_PREFIX).input_ids) | set(
        tok(fc.FRAME_SUFFIX).input_ids
    )
    prose_ids = tok(prose).input_ids
    kept = [i for i in ids if i not in frame_ids]
    assert kept == prose_ids


def test_build_framed_text_every_block_carries_the_frame() -> None:
    tok = FakeTokenizer()
    prose = " ".join(f"w{i}" for i in range(40))
    frame_len = len(tok(fc.FRAME_PREFIX).input_ids) + len(
        tok(fc.FRAME_SUFFIX).input_ids
    )
    chunk_len = 20 - frame_len
    framed = fc.build_framed_text(tok, prose, block_tokens=20)
    expected_blocks = -(-40 // chunk_len)
    assert framed.count(fc.FRAME_PREFIX) == expected_blocks
    # The user turn inside the prefix also closes with the suffix text.
    assert framed.count(fc.FRAME_SUFFIX) == 2 * expected_blocks
    assert framed.endswith(fc.FRAME_SUFFIX)


def test_build_framed_text_marker_not_special_raises() -> None:
    tok = FakeTokenizer(special=tuple(m for m in fc.FRAME_MARKERS if m != "<|turn>"))
    with pytest.raises(ValueError, match="not one special id"):
        fc.build_framed_text(tok, "some prose", block_tokens=64)


def test_build_framed_text_marker_multi_id_raises() -> None:
    tok = FakeTokenizer(markers=tuple(m for m in fc.FRAME_MARKERS if m != "<|channel>"))
    with pytest.raises(ValueError, match="not one special id"):
        fc.build_framed_text(tok, "some prose", block_tokens=64)


def test_build_framed_text_empty_prose_raises() -> None:
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="prose is empty"):
        fc.build_framed_text(tok, "   ", block_tokens=64)


def test_build_framed_text_prose_with_special_id_raises() -> None:
    tok = FakeTokenizer(extra_special_words=("<eos>",))
    with pytest.raises(ValueError, match="special ids"):
        fc.build_framed_text(tok, "prose then <eos> more", block_tokens=64)


def test_build_framed_text_frame_fills_block_raises() -> None:
    tok = FakeTokenizer()
    frame_len = len(tok(fc.FRAME_PREFIX).input_ids) + len(
        tok(fc.FRAME_SUFFIX).input_ids
    )
    with pytest.raises(ValueError, match="leaves no room"):
        fc.build_framed_text(tok, "some prose", block_tokens=frame_len + 1)
