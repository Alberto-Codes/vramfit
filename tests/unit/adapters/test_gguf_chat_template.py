"""The chat-template stamp against written GGUFs (#617).

The base is a real GGUF gguf-py writes, and the quantizer is a stub
that copies its input to its output. The pack adapter's own header
reads, the file-type relabel, and the chat-template stamp all run
for real. gguf-py reads the output back, so the template's bytes are
checked by a reader the stamp does not share.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="gguf extra not installed")
pytest.importorskip("gguf", reason="gguf extra not installed")

from gguf import GGUFReader, GGUFWriter

from vramfit.adapters.outbound.gguf.chat_template import (
    CHAT_TEMPLATE_KEY,
    read_source_chat_template,
    stamp_chat_template,
)
from vramfit.adapters.outbound.gguf.header import GgufHeader, read_header
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.model import Assignment, PlanMeta, Recipe
from vramfit.domain.pack import PackResult

pytestmark = pytest.mark.unit

# Multi-line, with non-ASCII text, and a trailing newline the stamp
# must keep.
TEMPLATE = (
    "{%- for message in messages %}\n"
    "<start_of_turn>{{ message['role'] }} — «héllo» 你好\n"
    "{{ message['content'] }}<end_of_turn>\n"
    "{%- endfor %}\n"
)
OTHER_TEMPLATE = "{{ messages[0]['content'] }}"

QUANTIZE_STUB = """\
#!{python}
import shutil, sys
shutil.copyfile(sys.argv[-4], sys.argv[-3])
"""


def write_base(path: Path, *, template: str | None = None) -> None:
    writer = GGUFWriter(path, "llama")
    writer.add_block_count(2)
    writer.add_file_type(1)
    if template is not None:
        writer.add_chat_template(template)
    rng = np.random.default_rng(7)
    writer.add_tensor(
        "blk.0.attn_q.weight", rng.standard_normal((4, 64)).astype(np.float16)
    )
    writer.add_tensor("blk.1.attn_q.weight", rng.standard_normal(96).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def recipe() -> Recipe:
    return Recipe(
        model_id="model",
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=3_000,
            predicted_total_bytes=2_500,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.layers.0", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="model.layers.1", bits=4, bytes=500, damage=0.01),
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    quantize = tmp_path / "llama-quantize"
    quantize.write_text(QUANTIZE_STUB.format(python=sys.executable))
    quantize.chmod(0o700)
    model = tmp_path / "model"
    model.mkdir()
    return {
        "base": tmp_path / "base.gguf",
        "quantize": quantize,
        "model": model,
        "out": tmp_path / "out.gguf",
    }


def pack(workspace: dict[str, Path]) -> PackResult:
    return LlamaCppPacker(
        model_dir=workspace["model"],
        base_gguf=workspace["base"],
        out_path=workspace["out"],
        convert_script=workspace["base"],
        quantize_bin=workspace["quantize"],
        python_bin=Path(sys.executable),
        threads=1,
    ).pack(recipe())


def template_bytes(path: Path) -> bytes | None:
    field = GGUFReader(str(path)).get_field(CHAT_TEMPLATE_KEY)
    if field is None:
        return None
    return bytes(field.parts[field.data[0]])


def tensors(path: Path) -> list[tuple[str, tuple[int, ...], int, bytes]]:
    return [
        (
            tensor.name,
            tuple(int(dim) for dim in tensor.shape),
            int(tensor.tensor_type),
            tensor.data.tobytes(),
        )
        for tensor in GGUFReader(str(path)).tensors
    ]


class TestPackEmbedsTheChatTemplate:
    def test_pack_with_jinja_source_embeds_the_template_byte_for_byte(
        self, workspace: dict[str, Path]
    ) -> None:
        jinja = workspace["model"] / "chat_template.jinja"
        jinja.write_bytes(TEMPLATE.encode("utf-8"))
        write_base(workspace["base"])

        result = pack(workspace)

        assert template_bytes(workspace["out"]) == jinja.read_bytes()
        assert tensors(workspace["out"]) == tensors(workspace["base"])
        assert result.chat_template_source == "chat_template.jinja"
        assert result.chat_template_embedded

    def test_pack_with_tokenizer_config_string_embeds_it(
        self, workspace: dict[str, Path]
    ) -> None:
        config = {"chat_template": TEMPLATE, "eos_token": "<eos>"}
        (workspace["model"] / "tokenizer_config.json").write_text(json.dumps(config))
        write_base(workspace["base"])

        result = pack(workspace)

        assert template_bytes(workspace["out"]) == TEMPLATE.encode("utf-8")
        assert result.chat_template_source == "tokenizer_config.json"
        assert result.chat_template_embedded

    def test_pack_with_tokenizer_config_list_embeds_the_default_entry(
        self, workspace: dict[str, Path]
    ) -> None:
        config = {
            "chat_template": [
                {"name": "tool_use", "template": OTHER_TEMPLATE},
                {"name": "default", "template": TEMPLATE},
            ]
        }
        (workspace["model"] / "tokenizer_config.json").write_text(json.dumps(config))
        write_base(workspace["base"])

        result = pack(workspace)

        assert template_bytes(workspace["out"]) == TEMPLATE.encode("utf-8")
        assert result.chat_template_source == "tokenizer_config.json"

    def test_pack_without_source_embeds_nothing_and_continues(
        self, workspace: dict[str, Path]
    ) -> None:
        write_base(workspace["base"])

        result = pack(workspace)

        assert template_bytes(workspace["out"]) is None
        assert result.chat_template_source is None
        assert not result.chat_template_embedded

    def test_pack_with_equal_existing_template_rewrites_nothing(
        self, workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (workspace["model"] / "chat_template.jinja").write_text(
            TEMPLATE, encoding="utf-8"
        )
        write_base(workspace["base"], template=TEMPLATE)
        stamped: list[Path] = []
        monkeypatch.setattr(
            "vramfit.adapters.outbound.gguf.chat_template.stamp_chat_template",
            lambda path, template: stamped.append(path),
        )

        result = pack(workspace)

        assert stamped == []
        assert template_bytes(workspace["out"]) == TEMPLATE.encode("utf-8")
        assert result.chat_template_source == "chat_template.jinja"
        assert result.chat_template_embedded

    def test_pack_with_different_existing_template_refuses(
        self, workspace: dict[str, Path]
    ) -> None:
        (workspace["model"] / "chat_template.jinja").write_text(
            TEMPLATE, encoding="utf-8"
        )
        write_base(workspace["base"], template=OTHER_TEMPLATE)

        with pytest.raises(PackError, match=r"chat_template\.jinja") as info:
            pack(workspace)

        assert str(workspace["out"]) in str(info.value)
        assert template_bytes(workspace["out"]) == OTHER_TEMPLATE.encode("utf-8")

    def test_pack_without_source_records_an_existing_template_as_embedded(
        self, workspace: dict[str, Path]
    ) -> None:
        write_base(workspace["base"], template=TEMPLATE)

        result = pack(workspace)

        assert result.chat_template_source is None
        assert result.chat_template_embedded


class TestReadSourceChatTemplate:
    def test_jinja_wins_over_tokenizer_config(self, tmp_path: Path) -> None:
        (tmp_path / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": OTHER_TEMPLATE})
        )

        source = read_source_chat_template(tmp_path)

        assert source is not None
        assert (source.text, source.source) == (TEMPLATE, "chat_template.jinja")

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"chat_template": None},
            {"chat_template": 3},
            {"chat_template": [{"name": "tool_use", "template": OTHER_TEMPLATE}]},
        ],
        ids=["absent", "null", "number", "list-without-default"],
    )
    def test_tokenizer_config_without_usable_template_returns_none(
        self, tmp_path: Path, config: dict[str, object]
    ) -> None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))

        assert read_source_chat_template(tmp_path) is None

    def test_missing_model_dir_returns_none(self, tmp_path: Path) -> None:
        assert read_source_chat_template(tmp_path / "absent") is None

    def test_invalid_tokenizer_config_raises_naming_the_path(
        self, tmp_path: Path
    ) -> None:
        config = tmp_path / "tokenizer_config.json"
        config.write_text("{not json")

        with pytest.raises(PackError, match=r"tokenizer_config\.json"):
            read_source_chat_template(tmp_path)


class TestStampChatTemplate:
    def test_stamp_adds_the_key_and_removes_the_temporary(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed.gguf"
        write_base(packed)
        before = read_header(packed)

        stamp_chat_template(packed, TEMPLATE)

        after = read_header(packed)
        assert after.chat_template == TEMPLATE
        assert after.tensors == before.tensors
        assert not (tmp_path / "packed.chat-template.gguf").exists()
        assert sorted(path.name for path in tmp_path.iterdir()) == ["packed.gguf"]

    def test_stamp_keeps_every_other_key_value(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed.gguf"
        write_base(packed)
        before = GGUFReader(str(packed)).fields

        stamp_chat_template(packed, TEMPLATE)

        after = GGUFReader(str(packed)).fields
        assert list(after) == [*before, CHAT_TEMPLATE_KEY]
        assert after["GGUF.kv_count"].contents() == (
            before["GGUF.kv_count"].contents() + 1
        )
        for key, field in before.items():
            if key != "GGUF.kv_count":
                assert after[key].contents() == field.contents()

    def test_failed_verification_refuses_and_removes_the_temporary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        packed = tmp_path / "packed.gguf"
        write_base(packed)
        original = packed.read_bytes()
        real_read = read_header

        def drifted(path: Path) -> GgufHeader:
            header = real_read(path)
            if path.name.endswith(".chat-template.gguf"):
                return replace(header, chat_template="drift")
            return header

        monkeypatch.setattr(
            "vramfit.adapters.outbound.gguf.chat_template.read_header", drifted
        )

        with pytest.raises(PackError, match="verify"):
            stamp_chat_template(packed, TEMPLATE)

        assert packed.read_bytes() == original
        assert not (tmp_path / "packed.chat-template.gguf").exists()


class TestReadHeaderChatTemplate:
    def test_header_reports_the_template_when_present(self, tmp_path: Path) -> None:
        path = tmp_path / "with.gguf"
        write_base(path, template=TEMPLATE)

        assert read_header(path).chat_template == TEMPLATE

    def test_header_reports_none_when_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "without.gguf"
        write_base(path)

        assert read_header(path).chat_template is None


class TestStampRefusesANameCollision:
    """The stamp never deletes a file it did not create (#617)."""

    def test_stamp_with_an_existing_temporary_refuses_and_keeps_both_files(
        self, tmp_path: Path
    ) -> None:
        packed = tmp_path / "packed.gguf"
        packed.write_bytes(b"not a gguf")
        temporary = tmp_path / "packed.chat-template.gguf"
        temporary.write_bytes(b"USER DATA")
        with pytest.raises(PackError, match="already exists"):
            stamp_chat_template(packed, "{{ messages }}")
        assert temporary.read_bytes() == b"USER DATA"
        assert packed.read_bytes() == b"not a gguf"
