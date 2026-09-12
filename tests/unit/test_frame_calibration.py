"""Hermetic tests for ``scripts/frame_calibration.py`` (#423, #437).

The script frames calibration prose for a channel-locked target. It
reads the frame from the checkpoint's own chat template, so these
tests drive it with fakes of three shapes: a Gemma-shaped template, a
ChatML-shaped template, and a checkpoint with no template. The real
tokenizer needs the scan extra, so each fake models the parts the
script reads: plain encode, decode, ``all_special_tokens``,
``all_special_ids``, ``added_tokens_decoder``, ``bos_token``,
``bos_token_id``, ``chat_template``, and ``apply_chat_template``
under both ``add_generation_prompt`` settings.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable, Sequence
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

Messages = Sequence[dict[str, str]]
Renderer = Callable[..., str]

GEMMA_SPECIALS = ("<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>")
CHATML_SPECIALS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


GEMMA_GENERATION_PROMPT = "<|turn>model\n<|channel>thought\n<channel|>"


def render_gemma(messages: Messages, add_generation_prompt: bool = False) -> str:
    """Render a Gemma-shaped conversation.

    The generation prompt opens the thought channel, the way
    google/gemma-4-31B-it-qat-q4_0-unquantized does.
    """
    turns = "".join(f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages)
    tail = GEMMA_GENERATION_PROMPT if add_generation_prompt else ""
    return f"<bos>{turns}{tail}"


def render_chatml(messages: Messages, add_generation_prompt: bool = False) -> str:
    """Render a ChatML-shaped conversation."""
    turns = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )
    tail = "<|im_start|>assistant\n" if add_generation_prompt else ""
    return f"{turns}{tail}"


def _chatml_thinking(messages: Messages, answered: str) -> str:
    """Render ChatML turns whose model turn carries a thought block."""
    return "".join(
        f"<|im_start|>{m['role']}\n"
        + (answered if m["role"] == "assistant" else "")
        + f"{m['content']}<|im_end|>\n"
        for m in messages
    )


def render_thinking_chatml(
    messages: Messages, add_generation_prompt: bool = False
) -> str:
    """Render a thinking ChatML template that closes what it opens.

    The generation prompt opens the thought channel, the way
    nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 does. The
    completed turn instead shows the channel opened and closed.
    """
    tail = "<|im_start|>assistant\n<think>\n" if add_generation_prompt else ""
    return _chatml_thinking(messages, "<think></think>") + tail


def render_open_channel_chatml(
    messages: Messages, add_generation_prompt: bool = False
) -> str:
    """Render a thinking template that closes the channel in neither form."""
    tail = "<|im_start|>assistant\n<think>\n" if add_generation_prompt else ""
    return _chatml_thinking(messages, "<think>") + tail


class _AddedToken:
    def __init__(self, content: str, special: bool) -> None:
        self.content = content
        self.special = special


class _Batch:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


class FakeTokenizer:
    """Word-level tokenizer with single-id special tokens.

    Special tokens encode to one id each. Every other
    whitespace-separated word gets its own id on first sight.
    """

    def __init__(
        self,
        specials: tuple[str, ...] = GEMMA_SPECIALS,
        render: Renderer | None = render_gemma,
        added_non_special: tuple[str, ...] = (),
        plain: tuple[str, ...] = (),
        extra_special_words: tuple[str, ...] = (),
        bos: str | None = "<bos>",
        bos_token_id: int | None = -1,
    ) -> None:
        """Build the vocabulary from the special tokens and extra words."""
        self._vocab: dict[str, int] = {}
        for token in (*specials, *added_non_special, *plain, *extra_special_words):
            self._vocab[token] = len(self._vocab)
        self.all_special_tokens = [*specials, *extra_special_words]
        self.all_special_ids = [self._vocab[t] for t in self.all_special_tokens]
        # `plain` tokens stay out of the added-token table: the
        # vocabulary holds them, but they are prose, not controls.
        self.added_tokens_decoder = {
            self._vocab[t]: _AddedToken(t, t in self.all_special_tokens)
            for t in (*specials, *added_non_special, *extra_special_words)
        }
        self.bos_token = bos
        self.bos_token_id = (
            self._vocab.get(bos or "") if bos_token_id == -1 else bos_token_id
        )
        self.chat_template = "{# fake #}" if render is not None else None
        self._render = render
        self._pattern = re.compile(
            "(" + "|".join(re.escape(m) for m in self._vocab) + ")"
        )

    def apply_chat_template(
        self,
        messages: Messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        assert self._render is not None
        return self._render(messages, add_generation_prompt=add_generation_prompt)

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


def _frame(tok: FakeTokenizer) -> tuple[str, str]:
    """Build and verify a frame the way ``main`` does."""
    prefix, suffix = fc.build_frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    fc.verify_frame(tok, markers, prefix, suffix)
    return prefix, suffix


def _framed(tok: FakeTokenizer, prose: str, block_tokens: int) -> str:
    prefix, suffix = _frame(tok)
    return fc.build_framed_text(tok, prose, block_tokens, prefix, suffix)


# --- the frame comes from the checkpoint's own template ---------------


def test_build_frame_gemma_template_yields_gemma_markers() -> None:
    tok = FakeTokenizer()
    prefix, suffix = _frame(tok)
    assert prefix == (
        "<bos><|turn>user\nContinue the passage.<turn|>\n"
        "<|turn>model\n<|channel>thought\n<channel|>"
    )
    assert suffix == "<turn|>\n"
    assert set(fc.frame_markers(tok, prefix, suffix)) == set(GEMMA_SPECIALS)


def test_build_frame_keeps_the_generation_prompt_not_the_turn_header() -> None:
    """The frame opens the channel the checkpoint opens (#423)."""
    tok = FakeTokenizer()
    prefix, _ = _frame(tok)
    assert prefix.endswith(GEMMA_GENERATION_PROMPT)
    assert not prefix.endswith("<|turn>assistant\n")


def test_build_frame_template_ignoring_generation_prompt_refuses() -> None:
    """A conversion whose template never reads ``add_generation_prompt``.

    Such a template renders the same text under both settings, so the
    prefix would be the user turn alone and every block would close a
    model turn it never opened.
    """

    def render(messages: Messages, add_generation_prompt: bool = False) -> str:
        return render_gemma(messages)

    hi = [{"role": "user", "content": "hi"}]
    tok = FakeTokenizer(render=render)
    assert tok.apply_chat_template(hi, add_generation_prompt=True) == (
        tok.apply_chat_template(hi, add_generation_prompt=False)
    )
    with pytest.raises(ValueError, match="no model-turn generation prompt"):
        fc.build_frame(tok)


def test_build_frame_chatml_template_yields_chatml_markers() -> None:
    tok = FakeTokenizer(specials=CHATML_SPECIALS, render=render_chatml, bos=None)
    prefix, suffix = _frame(tok)
    assert prefix == (
        "<|im_start|>user\nContinue the passage.<|im_end|>\n<|im_start|>assistant\n"
    )
    assert suffix == "<|im_end|>\n"
    assert "<|turn>" not in prefix


def test_frame_markers_lists_only_the_markers_the_frame_uses() -> None:
    tok = FakeTokenizer(specials=CHATML_SPECIALS, render=render_chatml, bos=None)
    prefix, suffix = fc.build_frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    # `<think>` and `</think>` belong to the vocabulary, not to this frame.
    assert set(markers) == {"<|im_start|>", "<|im_end|>"}


def test_build_frame_thinking_template_takes_the_closed_model_turn() -> None:
    """The completed render closes the thought the generation prompt opens.

    Nemotron 3.5 Lightning 30B-A3B renders exactly this disagreement:
    its generation prompt opens the thought channel and its completed
    turn shows that channel opened and closed. The block must price the prose as the
    answer, not as reasoning in a channel nothing closes.
    """
    tok = FakeTokenizer(
        specials=CHATML_SPECIALS, render=render_thinking_chatml, bos=None
    )
    generation = tok.apply_chat_template(
        [{"role": "user", "content": fc.FRAME_USER_TURN}], add_generation_prompt=True
    )
    prefix, suffix = _frame(tok)
    assert generation.endswith("<think>\n")
    assert prefix != generation
    assert prefix.endswith("<think></think>")
    block = fc.build_framed_text(tok, "w0 w1 w2 w3", 64, prefix, suffix)
    assert block.count("<think>") == block.count("</think>")


def test_build_frame_closed_thought_keeps_the_generation_prompt() -> None:
    """Gemma's generation prompt carries an empty, closed thought block.

    The completed render carries no thought block at all, so the
    generation prompt adds control tokens rather than missing any.
    That is the frame behind the published #423 figure, so it stays.
    """
    tok = FakeTokenizer()
    generation = tok.apply_chat_template(
        [{"role": "user", "content": fc.FRAME_USER_TURN}], add_generation_prompt=True
    )
    prefix, suffix = _frame(tok)
    assert prefix == generation
    assert prefix.endswith(GEMMA_GENERATION_PROMPT)
    assert fc.unbalanced_pair(tok, prefix, suffix) is None


def test_build_frame_agreeing_renders_keep_the_generation_prompt() -> None:
    """Qwen3-Coder renders the same model-turn opening either way."""
    tok = FakeTokenizer(specials=CHATML_SPECIALS, render=render_chatml, bos=None)
    user = [{"role": "user", "content": fc.FRAME_USER_TURN}]
    generation = tok.apply_chat_template(user, add_generation_prompt=True)
    answered, _, _ = tok.apply_chat_template(
        [*user, {"role": "assistant", "content": fc.PROSE_SLOT}]
    ).partition(fc.PROSE_SLOT)
    prefix, _ = _frame(tok)
    assert generation == answered
    assert prefix == generation


def test_verify_frame_channel_open_in_both_renders_refuses() -> None:
    """Neither render closes the thought, so no prefix choice saves it."""
    tok = FakeTokenizer(
        specials=CHATML_SPECIALS, render=render_open_channel_chatml, bos=None
    )
    prefix, suffix = fc.build_frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    assert "<think>" in prefix
    assert "</think>" not in prefix + suffix
    with pytest.raises(ValueError, match="leaves that channel unbalanced") as caught:
        fc.verify_frame(tok, markers, prefix, suffix)
    message = str(caught.value)
    assert message.index("'<think>'") < message.index("'</think>'")
    assert fc.unbalanced_pair(tok, prefix, suffix) == ("<think>", "</think>", 1, 0)


def test_build_frame_no_chat_template_refuses_with_the_cause() -> None:
    tok = FakeTokenizer(render=None)
    with pytest.raises(ValueError, match="carries no chat template"):
        fc.build_frame(tok)


def test_build_frame_template_drops_the_answer_refuses() -> None:
    tok = FakeTokenizer(
        render=lambda messages, add_generation_prompt=False: (
            "<bos><|turn>user\n<turn|>\n"
            + ("<|turn>model\n" if add_generation_prompt else "")
        )
    )
    with pytest.raises(ValueError, match="dropped the assistant answer"):
        fc.build_frame(tok)


def test_build_frame_template_raises_refuses() -> None:
    def explode(messages: Messages, add_generation_prompt: bool = False) -> str:
        raise RuntimeError("unknown role")

    tok = FakeTokenizer(render=explode)
    with pytest.raises(ValueError, match="did not render"):
        fc.build_frame(tok)


# --- the one-special-id rule still holds ------------------------------


def test_verify_frame_marker_typed_as_prose_raises() -> None:
    """A conversion that ships a marker as a normal token (#423)."""
    tok = FakeTokenizer(
        specials=tuple(m for m in GEMMA_SPECIALS if m != "<|turn>"),
        plain=("<|turn>",),
    )
    prefix, suffix = fc.build_frame(tok)
    markers = (*fc.frame_markers(tok, prefix, suffix), "<|turn>")
    with pytest.raises(ValueError, match="not one control id"):
        fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_added_token_absent_from_special_ids_passes() -> None:
    """Nemotron 30B-A3B ships `<|im_start|>` outside `all_special_ids`."""
    tok = FakeTokenizer(
        specials=("<|im_end|>",),
        added_non_special=("<|im_start|>", "<think>", "</think>"),
        render=render_chatml,
        bos=None,
    )
    prefix, suffix = fc.build_frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    assert set(markers) == {"<|im_start|>", "<|im_end|>"}
    fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_template_marker_missing_from_vocabulary_raises() -> None:
    """A re-upload whose ChatML template outlived its added-token table.

    Discovery never returns ``<|im_start|>`` here, so only the residue
    check stands between this checkpoint and a calibration file whose
    turn headers are spelled out as prose.
    """
    tok = FakeTokenizer(specials=("<|im_end|>",), render=render_chatml, bos=None)
    prefix, suffix = fc.build_frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    assert "<|im_start|>" in prefix
    assert "<|im_start|>" not in markers
    with pytest.raises(ValueError, match=r"reads as prose rather than"):
        fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_healthy_chatml_frame_leaves_no_residue() -> None:
    """The same shape, with the added-token table intact, passes."""
    tok = FakeTokenizer(
        specials=("<|im_end|>",),
        added_non_special=("<|im_start|>",),
        render=render_chatml,
        bos=None,
    )
    prefix, suffix = fc.build_frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    assert fc.unparsed_markers(tok, markers, prefix, suffix) == []
    fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_marker_of_several_tokens_raises() -> None:
    """Property 1: a frame marker must encode to exactly one token."""
    tok = FakeTokenizer()
    prefix, suffix = fc.build_frame(tok)
    split_marker = "<|turn> <turn|>"
    assert len(fc.encode(tok, split_marker)) > 1
    markers = (*fc.frame_markers(tok, prefix, suffix), split_marker)
    with pytest.raises(ValueError, match="not one control id"):
        fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_marker_of_one_unregistered_id_raises() -> None:
    """Property 2: that one token must be a control the checkpoint registers."""
    tok = FakeTokenizer(plain=("<|thought|>",))
    prefix, suffix = fc.build_frame(tok)
    assert len(fc.encode(tok, "<|thought|>")) == 1
    assert "<|thought|>" not in fc.control_tokens(tok)
    markers = (*fc.frame_markers(tok, prefix, suffix), "<|thought|>")
    with pytest.raises(ValueError, match="not one control id"):
        fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_marker_absent_from_vocabulary_raises() -> None:
    tok = FakeTokenizer(specials=tuple(m for m in GEMMA_SPECIALS if m != "<bos>"))
    prefix, suffix = fc.build_frame(tok)
    markers = (*fc.frame_markers(tok, prefix, suffix), "<bos>")
    with pytest.raises(ValueError, match="not one control id"):
        fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_bos_id_disagreement_raises() -> None:
    tok = FakeTokenizer(bos_token_id=999)
    prefix, suffix = fc.build_frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    with pytest.raises(ValueError, match="bos_token_id is 999"):
        fc.verify_frame(tok, markers, prefix, suffix)


def test_verify_frame_without_special_markers_raises() -> None:
    tok = FakeTokenizer(
        render=lambda messages, add_generation_prompt=False: (
            f"user: {messages[-1]['content']}"
            + ("\nmodel: " if add_generation_prompt else "")
        )
    )
    prefix, suffix = fc.build_frame(tok)
    with pytest.raises(ValueError, match="frames nothing"):
        fc.verify_frame(tok, fc.frame_markers(tok, prefix, suffix), prefix, suffix)


# --- block assembly ---------------------------------------------------


@pytest.mark.parametrize(
    ("specials", "render", "bos"),
    [
        (GEMMA_SPECIALS, render_gemma, "<bos>"),
        (CHATML_SPECIALS, render_chatml, None),
    ],
    ids=["gemma", "chatml"],
)
def test_build_framed_text_prose_survives_in_order(
    specials: tuple[str, ...],
    render: Renderer,
    bos: str | None,
) -> None:
    tok = FakeTokenizer(specials=specials, render=render, bos=bos)
    prefix, suffix = _frame(tok)
    prose = " ".join(f"w{i}" for i in range(40))
    framed = fc.build_framed_text(tok, prose, 20, prefix, suffix)
    frame_ids = set(tok(prefix).input_ids) | set(tok(suffix).input_ids)
    kept = [i for i in tok(framed).input_ids if i not in frame_ids]
    assert kept == tok(prose).input_ids


@pytest.mark.parametrize(
    ("specials", "render", "bos"),
    [
        (GEMMA_SPECIALS, render_gemma, "<bos>"),
        (CHATML_SPECIALS, render_chatml, None),
    ],
    ids=["gemma", "chatml"],
)
def test_build_framed_text_every_block_carries_the_frame(
    specials: tuple[str, ...],
    render: Renderer,
    bos: str | None,
) -> None:
    tok = FakeTokenizer(specials=specials, render=render, bos=bos)
    prefix, suffix = _frame(tok)
    frame_len = len(tok(prefix).input_ids) + len(tok(suffix).input_ids)
    chunk_len = 20 - frame_len
    framed = fc.build_framed_text(
        tok, " ".join(f"w{i}" for i in range(40)), 20, *_frame(tok)
    )
    expected_blocks = -(-40 // chunk_len)
    assert framed.count(prefix) == expected_blocks
    # The user turn inside the prefix also closes with the suffix text.
    assert framed.count(suffix) == 2 * expected_blocks
    assert framed.endswith(suffix)


def test_build_framed_text_empty_prose_raises() -> None:
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="prose is empty"):
        _framed(tok, "   ", 64)


def test_build_framed_text_prose_with_special_id_raises() -> None:
    tok = FakeTokenizer(extra_special_words=("<eos>",))
    with pytest.raises(ValueError, match="control ids"):
        _framed(tok, "prose then <eos> more", 64)


def test_build_framed_text_prose_with_added_non_special_id_raises() -> None:
    """Nemotron 30B-A3B keeps `<|im_start|>` out of `all_special_ids` (#423)."""
    tok = FakeTokenizer(
        specials=("<|im_end|>",),
        added_non_special=("<|im_start|>", "<think>", "</think>"),
        render=render_chatml,
        bos=None,
    )
    assert fc.encode(tok, "<|im_start|>")[0] not in tok.all_special_ids
    with pytest.raises(ValueError, match="control ids"):
        _framed(tok, "prose then <|im_start|> more", 64)


def test_build_framed_text_frame_fills_block_raises() -> None:
    tok = FakeTokenizer()
    prefix, suffix = _frame(tok)
    frame_len = len(tok(prefix).input_ids) + len(tok(suffix).input_ids)
    with pytest.raises(ValueError, match="leaves no room"):
        _framed(tok, "some prose", frame_len + 1)


# --- the re-encode check ----------------------------------------------


@pytest.mark.parametrize(
    ("specials", "render", "bos"),
    [
        (GEMMA_SPECIALS, render_gemma, "<bos>"),
        (CHATML_SPECIALS, render_chatml, None),
    ],
    ids=["gemma", "chatml"],
)
def test_check_reencode_clean_frame_reports_no_problem(
    specials: tuple[str, ...],
    render: Renderer,
    bos: str | None,
) -> None:
    tok = FakeTokenizer(specials=specials, render=render, bos=bos)
    prefix, suffix = _frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    framed = fc.build_framed_text(
        tok, " ".join(f"w{i}" for i in range(40)), 20, prefix, suffix
    )
    assert fc.check_reencode(tok, framed, markers, tok(framed).input_ids) is None


def test_check_reencode_missing_marker_id_reports_the_marker() -> None:
    tok = FakeTokenizer()
    prefix, suffix = _frame(tok)
    markers = fc.frame_markers(tok, prefix, suffix)
    framed = fc.build_framed_text(
        tok, " ".join(f"w{i}" for i in range(40)), 20, prefix, suffix
    )
    ids = [i for i in tok(framed).input_ids if i != tok("<bos>").input_ids[0]]
    problem = fc.check_reencode(tok, framed, markers, ids)
    assert problem is not None
    assert "controls did not parse" in problem
