"""Checks of the owned GGUF header parse, the Q2_0 block decoder, and the mixed rewrite.

The parse and the rewrite import no gguf-py, but the fixtures here
are written with gguf-py's writer so the parse is held against a
real writer's layout (ADR-0032).
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports
from __future__ import annotations

import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="gguf extra not installed")
pytest.importorskip("gguf", reason="gguf extra not installed")

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from vramfit.adapters.outbound.gguf.header import (
    GgufHeader,
    TensorInfo,
    read_header,
    write_tensor_infos,
)
from vramfit.adapters.outbound.gguf.mixed_gguf import write_mixed_gguf
from vramfit.adapters.outbound.gguf.q2_0_blocks import (
    Q2_0_BLOCK_BYTES,
    Q2_0_TYPE_ID,
    dequantize_q2_0,
    q2_0_payload_bytes,
)
from vramfit.adapters.outbound.gguf.types import PackError

pytestmark = pytest.mark.unit


def write_base(path: Path, *, alignment: int | None = None) -> dict[str, np.ndarray]:
    """Write a three-tensor f16/f32 base the way convert would."""
    writer = GGUFWriter(path, "llama")
    writer.add_block_count(1)
    writer.add_file_type(1)
    if alignment is not None:
        writer.add_custom_alignment(alignment)
    rng = np.random.default_rng(3)
    tensors = {
        "blk.0.ffn_down_exps.weight": rng.standard_normal((2, 4, 128)).astype(
            np.float16
        ),
        "blk.0.attn_norm.weight": np.ones(64, dtype=np.float32),
        "blk.0.attn_q.weight": rng.standard_normal((16, 64)).astype(np.float16),
    }
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


class TestReadHeader:
    def test_reads_names_dims_types_and_offsets_in_header_order(
        self, tmp_path: Path
    ) -> None:
        base = tmp_path / "base.gguf"
        write_base(base)
        header = read_header(base)
        assert [info.name for info in header.tensors] == [
            "blk.0.ffn_down_exps.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
        ]
        stack = header.tensors[0]
        assert stack.dims == (128, 4, 2)
        assert stack.type_id == GGMLQuantizationType.F16
        assert stack.elements == 1024
        assert header.tensors[1].type_id == GGMLQuantizationType.F32
        assert header.file_type == 1
        assert header.alignment == 32
        assert header.data_start % 32 == 0
        reader = GGUFReader(base)
        for info, tensor in zip(header.tensors, reader.tensors, strict=True):
            assert info.offset == tensor.data_offset - header.data_start

    def test_spans_bound_each_tensor_by_the_next_offset(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_base(base)
        header = read_header(base)
        spans = header.spans(base.stat().st_size)
        stack_start, stack_end = spans["blk.0.ffn_down_exps.weight"]
        assert stack_end - stack_start == 1024 * 2
        assert spans["blk.0.attn_q.weight"][1] == base.stat().st_size

    def test_custom_alignment_is_read(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_base(base, alignment=64)
        assert read_header(base).alignment == 64

    def test_non_gguf_refuses(self, tmp_path: Path) -> None:
        bogus = tmp_path / "x.gguf"
        bogus.write_bytes(b"nope")
        with pytest.raises(PackError, match="no GGUF magic"):
            read_header(bogus)

    def test_truncated_header_refuses(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_base(base)
        base.write_bytes(base.read_bytes()[:60])
        with pytest.raises(PackError, match="header ends"):
            read_header(base)

    def test_missing_file_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(PackError, match="cannot read the GGUF"):
            read_header(tmp_path / "absent.gguf")

    def test_write_tensor_infos_round_trips_through_the_parser(
        self, tmp_path: Path
    ) -> None:
        base = tmp_path / "base.gguf"
        write_base(base)
        header = read_header(base)
        rewritten = tmp_path / "infos.bin"
        with rewritten.open("wb") as handle:
            write_tensor_infos(handle, header.tensors)
        original = base.read_bytes()[header.infos_offset : header.data_start]
        assert original.startswith(rewritten.read_bytes())


class TestQ2_0Blocks:
    def test_payload_bytes_counts_eighteen_per_block(self) -> None:
        assert q2_0_payload_bytes(64) == Q2_0_BLOCK_BYTES
        assert q2_0_payload_bytes(2048) == 32 * 18

    @pytest.mark.parametrize("elements", [0, 32, 100])
    def test_partial_block_refuses(self, elements: int) -> None:
        with pytest.raises(PackError, match="do not divide"):
            q2_0_payload_bytes(elements)

    def test_decodes_codes_minus_one_times_the_fp16_scale(self) -> None:
        scale = np.float16(0.5).tobytes()
        # Element codes 3, 2, 1, 0 in the first byte: levels 2, 1, 0, -1.
        codes = bytes([0b00011011]) + bytes(15)
        values = dequantize_q2_0(scale + codes, 64)
        assert values[:4].tolist() == [1.0, 0.5, 0.0, -0.5]
        assert np.all(values[4:] == -0.5)

    def test_wrong_payload_size_refuses(self) -> None:
        with pytest.raises(PackError, match="holds 18 bytes, got 17"):
            dequantize_q2_0(bytes(17), 64)

    def test_type_id_matches_the_file_type_table(self) -> None:
        from vramfit.adapters.outbound.gguf.file_type import TENSOR_TYPE_NAMES

        assert TENSOR_TYPE_NAMES[Q2_0_TYPE_ID] == "Q2_0"


class TestWriteMixedGguf:
    def test_replaces_the_named_tensor_and_keeps_every_other_byte(
        self, tmp_path: Path
    ) -> None:
        base = tmp_path / "base.gguf"
        tensors = write_base(base)
        payload = tmp_path / "stack.q2_0"
        payload.write_bytes(
            bytes(range(256)) * (1024 // 64 * 18 // 256) + bytes(1024 // 64 * 18 % 256)
        )
        mixed = tmp_path / "mixed.gguf"
        written = write_mixed_gguf(
            base, mixed, {"blk.0.ffn_down_exps.weight": (payload, Q2_0_TYPE_ID)}
        )
        assert written == mixed.stat().st_size
        header = read_header(mixed)
        stack = header.tensors[0]
        assert stack.type_id == Q2_0_TYPE_ID
        assert stack.dims == (128, 4, 2)
        spans = header.spans(written)
        start, _ = spans[stack.name]
        with mixed.open("rb") as handle:
            handle.seek(start)
            assert handle.read(payload.stat().st_size) == payload.read_bytes()
            for name in ("blk.0.attn_norm.weight", "blk.0.attn_q.weight"):
                info = next(t for t in header.tensors if t.name == name)
                handle.seek(header.data_start + info.offset)
                assert handle.read(tensors[name].nbytes) == tensors[name].tobytes()
        for info in header.tensors:
            assert info.offset % header.alignment == 0
        # The key-value section is the base's, byte for byte.
        base_header = read_header(base)
        assert (
            mixed.read_bytes()[: base_header.infos_offset]
            == base.read_bytes()[: base_header.infos_offset]
        )

    def test_a_replacement_gguf_py_can_name_reads_back_through_the_reader(
        self, tmp_path: Path
    ) -> None:
        base = tmp_path / "base.gguf"
        tensors = write_base(base)
        payload = tmp_path / "q.q8_0"
        # Q8_0 stores 34 bytes per 32 elements; gguf-py knows the type.
        payload.write_bytes(bytes(1024 // 32 * 34))
        mixed = tmp_path / "mixed.gguf"
        write_mixed_gguf(
            base,
            mixed,
            {"blk.0.attn_q.weight": (payload, int(GGMLQuantizationType.Q8_0))},
        )
        reader = GGUFReader(mixed)
        by_name = {t.name: t for t in reader.tensors}
        assert by_name["blk.0.attn_q.weight"].tensor_type == GGMLQuantizationType.Q8_0
        assert by_name["blk.0.attn_q.weight"].data.nbytes == 1024 // 32 * 34
        assert np.array_equal(
            by_name["blk.0.ffn_down_exps.weight"].data,
            tensors["blk.0.ffn_down_exps.weight"],
        )
        assert reader.fields["general.file_type"].contents() == 1

    def test_unknown_tensor_refuses(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_base(base)
        payload = tmp_path / "x"
        payload.write_bytes(b"x")
        with pytest.raises(
            PackError, match="carries no tensor for 1 pre-encoded payload"
        ):
            write_mixed_gguf(
                base, tmp_path / "m.gguf", {"blk.9.x.weight": (payload, 42)}
            )

    def test_missing_payload_refuses(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_base(base)
        with pytest.raises(PackError, match="cannot size the pre-encoded payload"):
            write_mixed_gguf(
                base,
                tmp_path / "m.gguf",
                {"blk.0.attn_q.weight": (tmp_path / "absent", 42)},
            )


class TestGgufHeaderRecord:
    def test_spans_refuse_an_offset_past_the_file(self) -> None:
        header = GgufHeader(
            version=3,
            alignment=32,
            file_type_offset=None,
            file_type=0,
            infos_offset=64,
            data_start=64,
            tensors=(TensorInfo("a", (8,), 0, 0), TensorInfo("b", (8,), 0, 1024)),
        )
        with pytest.raises(PackError, match="runs past the file"):
            header.spans(64 + 512)

    def test_elements_multiplies_the_dims(self) -> None:
        assert TensorInfo("a", (128, 4, 2), 1, 0).elements == 1024
        assert struct.calcsize("<Q") == 8
