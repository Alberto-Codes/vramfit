"""Refusals the shared shard-header reader owes both of its callers.

`scripts/backfill_tensor_sizes.py` and `vramfit plan` read one reader
since ADR-0029 decision 1, so a header defect reaches the plan step too.
The doors below escaped it: two raise something other than `ValueError`,
and the third renders a publisher-written name whole (#335, #287).
"""

from __future__ import annotations

import json
import os
import stat
import struct
from pathlib import Path

import pytest

from vramfit.adapters.outbound.safetensors_sizes import (
    HEADER_PREFIX_BYTES,
    NAME_LIMIT,
    SafetensorsSizes,
    bounded,
    read_safetensors_header,
)
from vramfit.domain.sizes import SizeSourceError

pytestmark = pytest.mark.unit

TENSOR = "backbone.layers.0.mlp.up_proj.weight"


def write_header(path: Path, blob: bytes) -> Path:
    """Write one shard carrying `blob` as its header, prefix included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


def test_header_nested_past_the_recursion_limit_refuses_and_names_the_shard(
    tmp_path,
) -> None:
    # `json.loads` raises `RecursionError` on deep nesting, and that is
    # no `ValueError`. Both callers catch `ValueError`, so it reached a
    # traceback through the script and past `SafetensorsSizes`.
    shard = write_header(
        tmp_path / "model-00001-of-00001.safetensors",
        b"[" * 200_000 + b"]" * 200_000,
    )

    with pytest.raises(ValueError, match="nests too deeply") as caught:
        read_safetensors_header(shard)

    assert str(shard) in str(caught.value)


def test_header_past_the_digit_limit_refuses_and_names_the_shard(tmp_path) -> None:
    # `json.loads` raises a plain `ValueError` past
    # `sys.get_int_max_str_digits`, which the two named clauses miss. It
    # reported with no locator before the catch-all (#287).
    shard = write_header(
        tmp_path / "model-00001-of-00001.safetensors",
        f'{{"{TENSOR}": {"1" * 5000}}}'.encode(),
    )

    with pytest.raises(ValueError, match="cannot parse header JSON") as caught:
        read_safetensors_header(shard)

    assert str(shard) in str(caught.value)


def test_a_name_within_the_limit_renders_whole() -> None:
    # The cap must not touch an ordinary checkpoint name.
    assert bounded(TENSOR) == TENSOR


def test_a_name_past_the_limit_renders_cut() -> None:
    assert bounded("A" * 100_000) == f"{'A' * NAME_LIMIT}..."


def test_a_refusal_quoting_a_header_key_stays_bounded(tmp_path) -> None:
    # A header key is publisher-written and unbounded. The record
    # refusals embedded it whole, so one shard could render 100,000
    # characters of stderr through either caller.
    long_name = "A" * 100_000
    write_header(
        tmp_path / "model-00001-of-00001.safetensors",
        json.dumps({long_name: {"dtype": "BF16", "shape": [4, 2]}}).encode(),
    )

    with pytest.raises(SizeSourceError) as caught:
        SafetensorsSizes(tmp_path).tensor_sizes()

    message = str(caught.value)
    assert len(message) < 1000
    assert f"{'A' * NAME_LIMIT}..." in message


def test_a_repeated_tensor_name_across_shards_stays_bounded(tmp_path) -> None:
    # The collision refusal quotes the name too, and it names two paths
    # beside it.
    long_name = "A" * 100_000
    record = {"dtype": "BF16", "shape": [4, 2], "data_offsets": [0, 16]}
    for shard_name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        write_header(tmp_path / shard_name, json.dumps({long_name: record}).encode())

    with pytest.raises(SizeSourceError) as caught:
        SafetensorsSizes(tmp_path).tensor_sizes()

    message = str(caught.value)
    assert len(message) < 1000
    assert "is already defined by" in message


def test_a_read_shorter_than_the_prefix_promised_refuses(tmp_path, monkeypatch) -> None:
    # `stat` and `read` see the file at two moments, so a writer that
    # truncates between them returns a short read. Reporting a larger
    # `st_size` reproduces that window: the size guard passes and the
    # read comes back short. Raised by the Copilot review on PR #361.
    blob = json.dumps({TENSOR: {"dtype": "BF16", "shape": [4, 2]}}).encode()
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", 5_000) + blob)

    def generous_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        fields = list(os.stat(str(self)))
        fields[stat.ST_SIZE] = HEADER_PREFIX_BYTES + 5_000
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "stat", generous_stat)

    with pytest.raises(ValueError, match="truncated while reading") as caught:
        read_safetensors_header(shard)

    assert str(shard) in str(caught.value)
