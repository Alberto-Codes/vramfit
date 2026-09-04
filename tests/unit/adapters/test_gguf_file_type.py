"""The file-type relabel against written GGUFs (#413, #414).

The pure modal rule runs in `tests/unit/domain/test_pack.py`. This
module proves the header parse and the in-place write against files
gguf-py wrote, so it carries the same guard as
`test_override_match_gguf.py`. One file holds a tensor type id
gguf-py 0.19.0 cannot name (``Q2_0``, 42), which is the case the
parser exists for.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

np = pytest.importorskip("numpy", reason="pack extra not installed")
pytest.importorskip("gguf", reason="pack extra not installed")

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from vramfit.adapters.outbound.gguf.file_type import (
    FILE_TYPE_KEY,
    FTYPE_BY_TENSOR_TYPE,
    TENSOR_TYPE_NAMES,
    declared_file_type,
    read_layout,
    stamp_modal_file_type,
)
from vramfit.adapters.outbound.gguf.types import PackError

pytestmark = pytest.mark.unit

Q2_K_FTYPE = 10
Q4_0_FTYPE = 2
# The upstream ids gguf-py 0.19.0 lacks (ggml/include/ggml.h).
Q2_0_TYPE_ID = 42
# One Q4_0 block: 32 weights in 18 bytes.
Q4_0_BLOCK_BYTES = 18


def write_gguf(
    path: Path,
    tensors: tuple[tuple[str, int, int], ...],
    *,
    file_type: int | None = Q2_K_FTYPE,
    dtype: type = np.int8,
) -> None:
    """Write ``(name, type id, byte count)`` tensors under one ftype.

    int8 keeps the writer from deriving a block shape, so a type id
    it cannot name still writes. A test that reads the file back
    through gguf-py passes uint8 instead, so the writer records the
    element shape the reader checks.
    """
    writer = GGUFWriter(path, arch="llama")
    if file_type is not None:
        writer.add_file_type(file_type)
    for name, type_id, n_bytes in tensors:
        raw_dtype = cast("GGMLQuantizationType", type_id)
        writer.add_tensor(name, np.zeros(n_bytes, dtype=dtype), raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def declared_by_gguf_py(path: Path) -> int:
    field = GGUFReader(str(path)).get_field(FILE_TYPE_KEY)
    assert field is not None
    return int(field.contents())


class TestReadLayout:
    def test_bytes_sum_per_type_with_alignment_padding(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed.gguf"
        write_gguf(
            packed,
            (
                ("blk.0.ffn_up_exps.weight", GGMLQuantizationType.Q4_0, 72),
                ("blk.1.ffn_down_exps.weight", Q2_0_TYPE_ID, 20),
                ("output_norm.weight", GGMLQuantizationType.F32, 16),
            ),
        )

        layout = read_layout(packed)

        # Each tensor pads to the 32-byte alignment, and the padding
        # counts toward the tensor before it.
        assert layout.bytes_by_type == {"Q4_0": 96, "Q2_0": 32, "F32": 32}
        assert layout.file_type == Q2_K_FTYPE

    def test_file_declaring_no_file_type_refuses(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed.gguf"
        write_gguf(packed, (("a", GGMLQuantizationType.F32, 16),), file_type=None)

        with pytest.raises(PackError, match=r"declares no general\.file_type"):
            read_layout(packed)

    def test_unnamed_tensor_type_refuses_naming_the_id(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed.gguf"
        write_gguf(packed, (("a", 99, 16),))

        with pytest.raises(PackError, match="type id 99"):
            read_layout(packed)

    def test_non_gguf_file_refuses(self, tmp_path: Path) -> None:
        bogus = tmp_path / "packed.gguf"
        bogus.write_bytes(b"not a gguf")

        with pytest.raises(PackError, match="cannot read the packed GGUF"):
            read_layout(bogus)

    def test_truncated_header_refuses(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed.gguf"
        write_gguf(packed, (("a", GGMLQuantizationType.F32, 16),))
        packed.write_bytes(packed.read_bytes()[:40])

        with pytest.raises(PackError, match="cannot read the packed GGUF"):
            read_layout(packed)

    def test_missing_file_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(PackError, match="cannot read the packed GGUF"):
            read_layout(tmp_path / "absent.gguf")


class TestStampModalFileType:
    def test_stamp_writes_the_modal_type_gguf_py_reads_back(
        self, tmp_path: Path
    ) -> None:
        packed = tmp_path / "packed.gguf"
        write_gguf(
            packed,
            (
                (
                    "blk.0.ffn_up_exps.weight",
                    GGMLQuantizationType.Q4_0,
                    4 * Q4_0_BLOCK_BYTES,
                ),
                ("blk.0.attn_q.weight", GGMLQuantizationType.Q8_0, 34),
            ),
            dtype=np.uint8,
        )
        assert declared_by_gguf_py(packed) == Q2_K_FTYPE

        assert stamp_modal_file_type(packed) == "Q4_0"

        assert declared_by_gguf_py(packed) == Q4_0_FTYPE

    def test_stamp_changes_only_the_four_value_bytes(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed.gguf"
        write_gguf(packed, (("a", GGMLQuantizationType.Q8_0, 34),))
        before = packed.read_bytes()
        offset = read_layout(packed).file_type_offset

        stamp_modal_file_type(packed)

        after = packed.read_bytes()
        assert len(after) == len(before)
        changed = [
            i for i, (x, y) in enumerate(zip(before, after, strict=True)) if x != y
        ]
        assert changed and all(offset <= i < offset + 4 for i in changed)

    def test_stamp_over_the_30b_shape_names_q4_0_beside_q2_0(
        self, tmp_path: Path
    ) -> None:
        # The #413 shape: a Q2_K label over Q4_0, Q8_0, and Q2_0
        # tensors, where gguf-py cannot even name the Q2_0 id.
        packed = tmp_path / "packed.gguf"
        write_gguf(
            packed,
            (
                ("blk.0.ffn_up_exps.weight", GGMLQuantizationType.Q4_0, 720),
                ("blk.0.attn_q.weight", GGMLQuantizationType.Q8_0, 136),
                ("blk.0.ffn_down_exps.weight", Q2_0_TYPE_ID, 112),
            ),
        )

        assert stamp_modal_file_type(packed) == "Q4_0"

        assert read_layout(packed).file_type == Q4_0_FTYPE

    def test_stamp_matching_the_declared_type_leaves_the_file_unchanged(
        self, tmp_path: Path
    ) -> None:
        packed = tmp_path / "packed.gguf"
        write_gguf(
            packed, (("a", GGMLQuantizationType.Q4_0, 72),), file_type=Q4_0_FTYPE
        )
        before = packed.read_bytes()

        assert stamp_modal_file_type(packed) == "Q4_0"

        assert packed.read_bytes() == before


class TestDeclaredFileType:
    def test_k_quant_declares_its_s_ftype(self) -> None:
        assert declared_file_type({"Q4_K": 10, "Q8_0": 1}) == ("Q4_K_S", 14)

    def test_q2_0_declares_the_upstream_ftype(self) -> None:
        assert declared_file_type({"Q2_0": 10}) == ("Q2_0", 41)

    def test_empty_table_refuses(self) -> None:
        with pytest.raises(PackError, match="cannot pick a file type"):
            declared_file_type({})

    def test_every_named_tensor_type_has_an_ftype(self) -> None:
        assert set(TENSOR_TYPE_NAMES.values()) == set(FTYPE_BY_TENSOR_TYPE)

    def test_named_ids_agree_with_gguf_py_where_it_knows_them(self) -> None:
        known = {member.value: member.name for member in GGMLQuantizationType}
        for type_id, name in TENSOR_TYPE_NAMES.items():
            if type_id in known:
                assert known[type_id] == name
