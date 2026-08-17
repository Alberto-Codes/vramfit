"""`imatrix_entry_names` and the refusal against a written imatrix (#309).

The matching rule itself runs without gguf-py in
`test_exclusion_match.py`. Only the read needs a real file, so only
this module carries the guard.

The module is marked `unit` rather than `contract`, for the reason
`test_override_match_gguf.py` records: ADR-0009 reserves `contract`
for verified-fake port suites and #207 owns whether that widens.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="pack extra not installed")
pytest.importorskip("gguf", reason="pack extra not installed")

from gguf import GGUFWriter

from vramfit.adapters.outbound.gguf.exclusion_match import check_exclusion_match
from vramfit.adapters.outbound.gguf.imatrix_counts import imatrix_entry_names
from vramfit.adapters.outbound.gguf.types import PackError

pytestmark = pytest.mark.unit


def write_imatrix(path: Path, names: tuple[str, ...]) -> None:
    """Write an imatrix carrying one dense entry per name."""
    writer = GGUFWriter(path, arch="imatrix")
    writer.add_type("imatrix")
    for name in names:
        writer.add_tensor(f"{name}.in_sum2", np.ones((1, 4), dtype=np.float32))
        writer.add_tensor(f"{name}.counts", np.array([[5.0]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestImatrixEntryNames:
    def test_the_suffixes_come_off_the_written_names(self, tmp_path: Path) -> None:
        path = tmp_path / "model.imatrix.gguf"
        write_imatrix(path, ("blk.0.attn_v.weight", "blk.1.attn_v.weight"))
        assert set(imatrix_entry_names(path)) == {
            "blk.0.attn_v.weight",
            "blk.1.attn_v.weight",
        }

    def test_a_non_imatrix_gguf_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / "model.imatrix.gguf"
        writer = GGUFWriter(path, arch="llama")
        writer.add_tensor("blk.0.attn_v.weight", np.zeros((2, 2), dtype=np.float16))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with pytest.raises(PackError, match="is not an imatrix GGUF"):
            imatrix_entry_names(path)

    def test_a_non_gguf_file_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / "model.imatrix.gguf"
        path.write_bytes(b"not a gguf")
        with pytest.raises(PackError, match=r"imatrix .* is not a GGUF"):
            imatrix_entry_names(path)


class TestCheckExclusionMatchAgainstAFile:
    def test_an_exclusion_the_matrix_carries_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "model.imatrix.gguf"
        write_imatrix(path, ("blk.0.attn_v.weight", "blk.1.attn_v.weight"))
        check_exclusion_match(("blk.1.attn_v.weight",), path)

    def test_an_exclusion_no_entry_carries_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / "model.imatrix.gguf"
        write_imatrix(path, ("blk.0.attn_v.weight",))
        with pytest.raises(PackError, match="carries no row for 1 of 1") as exc:
            check_exclusion_match(("blk.9.attn_v.weight",), path)
        assert "blk.9.attn_v.weight" in str(exc.value)
        assert str(path) in str(exc.value)

    def test_the_refusal_names_every_unreached_exclusion(self, tmp_path: Path) -> None:
        path = tmp_path / "model.imatrix.gguf"
        write_imatrix(path, ("blk.0.attn_v.weight",))
        names = ("blk.0.attn_v.weight", "blk.8.attn_v.weight", "blk.9.attn_v.weight")
        with pytest.raises(PackError, match="carries no row for 2 of 3") as exc:
            check_exclusion_match(names, path)
        assert "blk.8.attn_v.weight" in str(exc.value)
        assert "blk.9.attn_v.weight" in str(exc.value)
