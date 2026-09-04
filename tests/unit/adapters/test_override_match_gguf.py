"""`base_tensor_names` against a written GGUF (#303, #307).

The rest of the override matching runs without gguf-py in
`test_override_match.py`. Only the header read needs a real file, so
only this module carries the guard.

The module is marked `unit` rather than `contract`. ADR-0009 reserves
`contract` for verified-fake port suites, and #207 owns whether that
marker widens. The CI test job installs the gguf extra, so these run
there and skip on a dev box synced without it.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="gguf extra not installed")
pytest.importorskip("gguf", reason="gguf extra not installed")

from gguf import GGUFWriter

from vramfit.adapters.outbound.gguf.override_match import (
    SPLIT_COUNT_KEY,
    SPLIT_NO_KEY,
    base_tensor_names,
    check_base_coverage,
)
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import TypeOverride

pytestmark = pytest.mark.unit


def write_gguf(
    path: Path,
    names: tuple[str, ...],
    *,
    split: tuple[int, int] | None = None,
) -> None:
    writer = GGUFWriter(path, arch="llama")
    if split is not None:
        index, count = split
        # The two keys `llama-gguf-split` writes, at the types
        # gguf-py's own shard writer uses (uint16 for both).
        writer.add_uint16(SPLIT_NO_KEY, index)
        writer.add_uint16(SPLIT_COUNT_KEY, count)
    for name in names:
        writer.add_tensor(name, np.zeros((2, 2), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestBaseTensorNames:
    def test_written_names_read_back_in_file_order(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight", "blk.1.attn_v.weight"))
        assert base_tensor_names(base) == (
            "blk.0.attn_v.weight",
            "blk.1.attn_v.weight",
        )

    def test_non_gguf_file_refuses(self, tmp_path: Path) -> None:
        bogus = tmp_path / "base.gguf"
        bogus.write_bytes(b"not a gguf")
        with pytest.raises(PackError, match="cannot read the base GGUF"):
            base_tensor_names(bogus)

    @pytest.mark.parametrize("length", [4, 24, 32, 33, 34])
    def test_truncated_header_refuses_as_pack_error(
        self, tmp_path: Path, length: int
    ) -> None:
        # `GGUFReader` reads each header field as `self._get(...)[0]`,
        # so a short file raises `IndexError` from an empty numpy
        # slice. `IndexError` is neither `VramfitError` nor anything
        # the CLI catches, so it would surface as a traceback with no
        # `pack_halted` event (ADR-0011).
        full = tmp_path / "full.gguf"
        write_gguf(full, ("blk.0.attn_v.weight",))
        short = tmp_path / "short.gguf"
        short.write_bytes(full.read_bytes()[:length])

        with pytest.raises(PackError, match="cannot read the base GGUF"):
            base_tensor_names(short)

    def test_interrupted_convert_names_the_remedy(self, tmp_path: Path) -> None:
        # `convert` reuses any existing base GGUF, so a partial file
        # from a killed convert reaches the next pack.
        short = tmp_path / "base.gguf"
        short.write_bytes(b"GGUF")

        with pytest.raises(PackError) as caught:
            base_tensor_names(short)
        assert "convert again" in str(caught.value)

    def test_gguf_holding_no_tensors_returns_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "base.gguf"
        write_gguf(empty, ())
        assert base_tensor_names(empty) == ()


def _write_gguf_with_kv(path: Path, fields: dict[str, tuple[str, int | str]]) -> None:
    """Write a one-tensor GGUF carrying hand-typed KV fields.

    The split keys carry a type, and the reader's gate reads it. So a
    case pinning a wrong type writes the field itself rather than
    going through `write_gguf`.
    """
    writer = GGUFWriter(path, arch="llama")
    for key, (kind, value) in fields.items():
        if kind == "uint16" and isinstance(value, int):
            writer.add_uint16(key, value)
        elif kind == "int16" and isinstance(value, int):
            writer.add_int16(key, value)
        elif kind == "string" and isinstance(value, str):
            writer.add_string(key, value)
        else:
            raise AssertionError(f"{key}: no writer for {kind} at {value!r}")
    writer.add_tensor("blk.0.attn_v.weight", np.zeros((2, 2), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestSplitBaseGguf:
    """A shard of a split GGUF refuses, and names itself (#308)."""

    def test_first_shard_refuses_and_names_its_position(self, tmp_path: Path) -> None:
        # unsloth and bartowski both publish the #158 target's bf16
        # base as `…-00001-of-00002.gguf`, measured 2026-08-18. The
        # shard carries blk.0 alone, so a recipe covering blk.0
        # through blk.51 reports 51 patterns unmatched.
        base = tmp_path / "base-00001-of-00003.gguf"
        write_gguf(base, ("blk.0.attn_v.weight",), split=(0, 3))

        with pytest.raises(PackError) as caught:
            base_tensor_names(base)
        assert "shard 1 of 3 of a split file" in str(caught.value)

    def test_later_shard_refuses_at_its_own_index(self, tmp_path: Path) -> None:
        base = tmp_path / "base-00003-of-00003.gguf"
        write_gguf(base, ("blk.51.attn_v.weight",), split=(2, 3))

        with pytest.raises(PackError, match="shard 3 of 3"):
            base_tensor_names(base)

    def test_refusal_names_the_merge_remedy(self, tmp_path: Path) -> None:
        # The pre-#308 message read "Check the recipe's group names
        # against the base GGUF's tensor names", which blames a
        # correct recipe. The remedy is a merge, not a recipe edit.
        base = tmp_path / "base-00001-of-00002.gguf"
        write_gguf(base, ("blk.0.attn_v.weight",), split=(0, 2))

        with pytest.raises(PackError) as caught:
            base_tensor_names(base)
        message = str(caught.value)
        assert "llama-gguf-split --merge" in message
        assert "Check the recipe's group names" not in message

    def test_split_count_of_one_reads_as_a_whole_file(self, tmp_path: Path) -> None:
        # `llama_model_loader` gates on `n_split > 1`, so a file
        # declaring one shard is the whole model.
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight",), split=(0, 1))
        assert base_tensor_names(base) == ("blk.0.attn_v.weight",)

    @pytest.mark.parametrize("count", [0, -1], ids=["zero", "negative"])
    def test_split_count_below_one_reads_as_a_whole_file(
        self, tmp_path: Path, count: int
    ) -> None:
        # The gate is `count <= 1` and not `count != 1`, so a count
        # the format cannot mean reads as a whole file rather than
        # refusing at an impossible chain length.
        base = tmp_path / "base.gguf"
        _write_gguf_with_kv(base, {SPLIT_COUNT_KEY: ("int16", count)})
        assert base_tensor_names(base) == ("blk.0.attn_v.weight",)

    def test_non_integer_split_count_reads_as_a_whole_file(
        self, tmp_path: Path
    ) -> None:
        # `get_key` throws "key %s has wrong type" on such a file, so
        # the quantizer names that defect itself.
        base = tmp_path / "base.gguf"
        _write_gguf_with_kv(base, {SPLIT_COUNT_KEY: ("string", "3")})
        assert base_tensor_names(base) == ("blk.0.attn_v.weight",)

    def test_split_no_absent_refuses_as_the_first_shard(self, tmp_path: Path) -> None:
        # `llama-gguf-split` writes both keys. A file carrying the
        # count alone still refuses, and reports the shard it can
        # justify rather than guessing.
        base = tmp_path / "base.gguf"
        _write_gguf_with_kv(base, {SPLIT_COUNT_KEY: ("uint16", 4)})
        with pytest.raises(PackError, match="shard 1 of 4"):
            base_tensor_names(base)

    def test_split_no_outside_the_chain_refuses_at_a_possible_shard(
        self, tmp_path: Path
    ) -> None:
        # An index of 7 across 3 shards states an impossible
        # position. Reporting it back would read "shard 8 of 3",
        # which names a shard the chain cannot hold.
        base = tmp_path / "base.gguf"
        _write_gguf_with_kv(
            base,
            {SPLIT_NO_KEY: ("uint16", 7), SPLIT_COUNT_KEY: ("uint16", 3)},
        )
        with pytest.raises(PackError) as caught:
            base_tensor_names(base)
        message = str(caught.value)
        assert "shard 1 of 3" in message
        assert "shard 8 of 3" not in message

    def test_coverage_check_refuses_the_shard_not_the_recipe(
        self, tmp_path: Path
    ) -> None:
        # The regression #308 records: a correct recipe against the
        # first shard reported unmatched patterns and told the
        # operator to check the recipe.
        base = tmp_path / "base-00001-of-00002.gguf"
        write_gguf(base, ("blk.0.attn_v.weight",), split=(0, 2))
        overrides = (
            TypeOverride(r"blk\.0\.", "q4_k"),
            TypeOverride(r"blk\.1\.", "q4_k"),
        )

        with pytest.raises(PackError) as caught:
            check_base_coverage(overrides, base)
        message = str(caught.value)
        assert "shard 1 of 2" in message
        assert "no tensor for" not in message


class TestCheckAgainstARealFile:
    def test_override_reaching_a_written_tensor_passes(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight",))
        check_base_coverage((TypeOverride(r"blk\.0\.", "q4_k"),), base)

    def test_override_naming_an_absent_layer_refuses(self, tmp_path: Path) -> None:
        # A recipe packed against the wrong checkpoint, or one naming
        # more layers than it carries.
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight", "blk.1.attn_v.weight"))
        overrides = (
            TypeOverride(r"blk\.7\.", "q4_k"),
            TypeOverride(r"blk\.8\.", "q2_k"),
        )
        with pytest.raises(PackError, match="no tensor for 2 of 2"):
            check_base_coverage(overrides, base)

    def test_prefixed_tree_matches_and_passes(self, tmp_path: Path) -> None:
        # `blk\.0\.` is a substring of `v.blk.0.attn_v.weight`, so the
        # quantizer's own search matches it. The check passes, and
        # #236 owns the resulting mis-application.
        base = tmp_path / "base.gguf"
        write_gguf(base, ("v.blk.0.attn_v.weight",))
        assert check_base_coverage((TypeOverride(r"blk\.0\.", "q4_k"),), base) == ()

    def test_layer_the_recipe_never_addressed_reports_from_the_file(
        self, tmp_path: Path
    ) -> None:
        # #307 read from a real header: the recipe covers blk.0, the
        # file also numbers blk.52, and the quantizer would drop it to
        # the --pure floor on a zero exit.
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight", "blk.52.attn_v.weight"))
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        assert check_base_coverage(overrides, base) == ("blk.52.",)
