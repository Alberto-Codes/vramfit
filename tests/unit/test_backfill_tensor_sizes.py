from __future__ import annotations

import json
import struct
from pathlib import Path

import backfill_tensor_sizes as script
import pytest

pytestmark = pytest.mark.unit

TENSOR = "model.layers.0.mlp.up_proj.weight"
OTHER_TENSOR = "model.layers.1.mlp.up_proj.weight"
# A (4, 2) tensor holds 8 elements, which the script prices at 2 bytes each.
TENSOR_BYTES = 16
RECORD = {"dtype": "BF16", "shape": [4, 2], "data_offsets": [0, TENSOR_BYTES]}

# One duplicate per position a real map could carry. The hook fires on
# every object in the document, so a nested repeat must refuse too.
DUPLICATE_MAPS = [
    pytest.param('{"groups": [], "groups": []}', "groups", id="top-level"),
    pytest.param(
        '{"groups": [{"name": "g", "tensors": [], "bytes_fp16": 0, "bytes_fp16": 99}]}',
        "bytes_fp16",
        id="inside-a-group",
    ),
]


def write_shard(model_dir: Path, header: str | bytes | None = None) -> Path:
    """Write one safetensors shard, header only, no payload."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = json.dumps(
            {"__metadata__": {"format": "pt"}, TENSOR: RECORD, OTHER_TENSOR: RECORD}
        )
    blob = header.encode("utf-8") if isinstance(header, str) else header
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return shard


def write_map(path: Path) -> None:
    """Write a two-group map the default shard covers exactly."""
    path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "name": "model.layers.0.mlp",
                        "tensors": [TENSOR],
                        "bytes_fp16": TENSOR_BYTES,
                    },
                    {
                        "name": "model.layers.1.mlp",
                        "tensors": [OTHER_TENSOR],
                        "bytes_fp16": TENSOR_BYTES,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def run(monkeypatch, map_path: Path, model_dir: Path, out: Path) -> int:
    """Invoke the script's entry point with a built argument vector."""
    monkeypatch.setattr(
        "sys.argv",
        ["backfill_tensor_sizes.py", str(map_path), str(model_dir), "--out", str(out)],
    )
    return script.main()


@pytest.mark.parametrize(("body", "key"), DUPLICATE_MAPS)
def test_map_with_a_duplicate_key_refuses_and_names_the_file(
    tmp_path, monkeypatch, capsys, body: str, key: str
) -> None:
    # `json.loads` keeps the last value and reports nothing, so the
    # annotated copy would carry one of two conflicting records (#262).
    # The checkpoint is well-formed, so only the map can refuse the run.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(body, encoding="utf-8")
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(map_path) in stderr
    assert f'duplicate key "{key}"' in stderr
    assert not out.exists()


def test_shard_header_with_a_duplicate_key_refuses_and_names_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # A repeated tensor name in a header means two shapes at once, so the
    # script cannot price the tensor. #283 set this rule for the shard
    # index, and a header is the same kind of publisher-written JSON.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    record = json.dumps(RECORD)
    shard = write_shard(
        model_dir, header=f'{{"{TENSOR}": {record}, "{TENSOR}": {record}}}'
    )
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert f'duplicate key "{TENSOR}"' in stderr
    assert not out.exists()


def test_shard_header_that_is_not_utf8_refuses_and_names_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # `json.loads` decodes the header bytes itself, so this door sits
    # beside the JSON one and must carry the same locator.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, header=b'{"a": "\xff\xfe"}')
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert "not valid UTF-8" in stderr
    assert not out.exists()


def test_map_without_a_duplicate_key_backfills_every_group(
    tmp_path, monkeypatch, capsys
) -> None:
    # The hook must leave a well-formed document alone, at both reads.
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 0
    assert capsys.readouterr().err == ""
    groups = json.loads(out.read_text(encoding="utf-8"))["groups"]
    assert [g["tensor_bytes"] for g in groups] == [
        {TENSOR: TENSOR_BYTES},
        {OTHER_TENSOR: TENSOR_BYTES},
    ]
