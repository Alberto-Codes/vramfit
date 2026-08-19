"""TensorSizeSource contract: the safetensors adapter and the fake agree.

The real side reads shards this suite writes byte for byte — a
little-endian u64 header length then that many bytes of JSON — so the
suite stays hermetic and needs no safetensors library (ADR-0009).

The malformed-header refusals are real-only pins below the shared
contract: the fake holds no file to malform.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tests.fakes import MemoryTensorSizeSource
from vramfit.adapters.outbound.safetensors_sizes import SafetensorsSizes
from vramfit.domain.sizes import SizeSourceError, TensorSize
from vramfit.ports.outbound import TensorSizeSource

pytestmark = pytest.mark.contract

DENSE = "backbone.layers.0.mlp.up_proj.weight"
EXPERT = "backbone.layers.1.mixer.experts.0.up_proj.weight"
MTP = "mtp.layers.0.mlp.up_proj.weight"
NORM = "backbone.layers.0.input_layernorm.weight"

DENSE_BYTES = 4 * 8 * 2
EXPERT_BYTES = 2 * 3 * 2


def write_shard(path: Path, entries: dict[str, dict]) -> None:
    """Write one safetensors shard carrying only a header."""
    offset = 0
    header: dict[str, object] = {}
    for name, record in entries.items():
        entry = {key: value for key, value in record.items() if key != "bytes"}
        if "bytes" in record:
            span = record["bytes"]
            entry["data_offsets"] = [offset, offset + span]
            offset += span
        header[name] = entry
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def _real_source(tmp_path: Path) -> TensorSizeSource:
    write_shard(
        tmp_path / "model-00001-of-00002.safetensors",
        {
            DENSE: {"dtype": "BF16", "shape": [4, 8], "bytes": DENSE_BYTES},
            NORM: {"dtype": "BF16", "shape": [4], "bytes": 8},
        },
    )
    write_shard(
        tmp_path / "model-00002-of-00002.safetensors",
        {
            EXPERT: {"dtype": "BF16", "shape": [2, 3], "bytes": EXPERT_BYTES},
            MTP: {"dtype": "BF16", "shape": [4, 8], "bytes": DENSE_BYTES},
        },
    )
    return SafetensorsSizes(tmp_path)


def _fake_source(tmp_path: Path) -> TensorSizeSource:
    return MemoryTensorSizeSource(
        sizes={
            DENSE: TensorSize(dtype="BF16", bytes=DENSE_BYTES),
            EXPERT: TensorSize(dtype="BF16", bytes=EXPERT_BYTES),
            MTP: TensorSize(dtype="BF16", bytes=DENSE_BYTES),
        }
    )


def _failing_real(tmp_path: Path) -> TensorSizeSource:
    return SafetensorsSizes(tmp_path)


def _failing_fake(tmp_path: Path) -> TensorSizeSource:
    return MemoryTensorSizeSource(fail=True)


@pytest.mark.parametrize(
    "build", [_real_source, _fake_source], ids=["real-safetensors", "fake-memory"]
)
class TestTensorSizeSourceContract:
    def test_a_tensor_reports_its_stored_bytes(self, build, tmp_path) -> None:
        source = build(tmp_path)

        sizes = source.tensor_sizes()

        assert sizes[DENSE].bytes == DENSE_BYTES

    def test_a_tensor_reports_its_header_dtype_verbatim(self, build, tmp_path) -> None:
        source = build(tmp_path)

        sizes = source.tensor_sizes()

        assert sizes[DENSE].dtype == "BF16"

    def test_keys_are_the_checkpoint_names(self, build, tmp_path) -> None:
        source = build(tmp_path)

        sizes = source.tensor_sizes()

        assert EXPERT in sizes

    def test_the_mtp_block_stays_out(self, build, tmp_path) -> None:
        source = build(tmp_path)

        sizes = source.tensor_sizes()

        assert MTP not in sizes


@pytest.mark.parametrize(
    "build", [_failing_real, _failing_fake], ids=["real-safetensors", "fake-memory"]
)
class TestTensorSizeSourceRefusal:
    def test_an_unreadable_checkpoint_raises_size_source_error(
        self, build, tmp_path
    ) -> None:
        source = build(tmp_path)

        with pytest.raises(SizeSourceError, match="safetensors"):
            source.tensor_sizes()


class TestRealReaderReads:
    def test_a_one_dimensional_tensor_stays_out(self, tmp_path) -> None:
        # The torch meter's `discover_groups` skips it too. The two
        # must agree on what the model's groups are.
        sizes = _real_source(tmp_path).tensor_sizes()

        assert NORM not in sizes

    def test_every_shard_is_read(self, tmp_path) -> None:
        sizes = _real_source(tmp_path).tensor_sizes()

        assert set(sizes) == {DENSE, EXPERT}

    def test_bytes_come_from_the_data_range_not_the_shape(self, tmp_path) -> None:
        # An unknown dtype still reports a size, so the domain refuses
        # it rather than the checkpoint reading as a smaller one.
        write_shard(
            tmp_path / "model.safetensors",
            {DENSE: {"dtype": "I8", "shape": [4, 8], "data_offsets": [0, 32]}},
        )

        sizes = SafetensorsSizes(tmp_path).tensor_sizes()

        assert sizes[DENSE] == TensorSize(dtype="I8", bytes=32)


class TestRealReaderRefusals:
    """Malformed shards, pinned case by case. The fake has no file."""

    def test_no_shards_refuses(self, tmp_path) -> None:
        with pytest.raises(SizeSourceError, match=r"no \*\.safetensors shards"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_a_short_shard_refuses(self, tmp_path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"\x00\x01")

        with pytest.raises(SizeSourceError, match="too short"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_a_non_json_header_refuses(self, tmp_path) -> None:
        blob = b"not json"
        (tmp_path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(blob)) + blob
        )

        with pytest.raises(SizeSourceError, match="not valid JSON"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_a_repeated_header_key_refuses(self, tmp_path) -> None:
        blob = b'{"a": {"dtype": "BF16"}, "a": {"dtype": "BF16"}}'
        (tmp_path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(blob)) + blob
        )

        with pytest.raises(SizeSourceError, match="twice"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_two_shards_defining_one_tensor_refuse(self, tmp_path) -> None:
        entry = {DENSE: {"dtype": "BF16", "shape": [4, 8], "bytes": DENSE_BYTES}}
        write_shard(tmp_path / "a.safetensors", dict(entry))
        write_shard(tmp_path / "b.safetensors", dict(entry))

        with pytest.raises(SizeSourceError, match="already defined by"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_an_entry_without_a_dtype_refuses(self, tmp_path) -> None:
        write_shard(
            tmp_path / "model.safetensors",
            {DENSE: {"shape": [4, 8], "data_offsets": [0, 64]}},
        )

        with pytest.raises(SizeSourceError, match="no dtype string"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_an_entry_without_a_shape_refuses(self, tmp_path) -> None:
        write_shard(
            tmp_path / "model.safetensors",
            {DENSE: {"dtype": "BF16", "data_offsets": [0, 64]}},
        )

        with pytest.raises(SizeSourceError, match="non-negative integers"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_an_entry_without_data_offsets_refuses(self, tmp_path) -> None:
        write_shard(
            tmp_path / "model.safetensors", {DENSE: {"dtype": "BF16", "shape": [4, 8]}}
        )

        with pytest.raises(SizeSourceError, match="data_offsets"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_an_empty_data_range_refuses(self, tmp_path) -> None:
        write_shard(
            tmp_path / "model.safetensors",
            {DENSE: {"dtype": "BF16", "shape": [0, 8], "data_offsets": [0, 0]}},
        )

        with pytest.raises(SizeSourceError, match="spans no bytes"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_an_entry_that_is_not_an_object_refuses(self, tmp_path) -> None:
        blob = json.dumps({DENSE: "nonsense"}).encode("utf-8")
        (tmp_path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(blob)) + blob
        )

        with pytest.raises(SizeSourceError, match="not an object"):
            SafetensorsSizes(tmp_path).tensor_sizes()

    def test_the_metadata_entry_stays_out(self, tmp_path) -> None:
        blob = json.dumps(
            {
                "__metadata__": {"format": "pt"},
                DENSE: {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]},
            }
        ).encode("utf-8")
        (tmp_path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(blob)) + blob
        )

        sizes = SafetensorsSizes(tmp_path).tensor_sizes()

        assert set(sizes) == {DENSE}
