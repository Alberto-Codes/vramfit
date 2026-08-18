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


def group(**overrides: object) -> dict:
    """Build one well-formed group, overriding named fields."""
    return {"name": "g", "tensors": [TENSOR], "bytes_fp16": TENSOR_BYTES} | overrides


# Every document the script must refuse before it annotates anything.
# Two of these exited 0 and wrote a file. The rest reached a traceback.
# Both are #298, at the document level and at the group level.
NON_MAPS = [
    pytest.param(
        {"vramfit_schema": 1, "assignments": {}}, 'no "groups" key', id="recipe"
    ),
    pytest.param([1, 2, 3], 'no "groups" key', id="list"),
    pytest.param(None, 'no "groups" key', id="null"),
    pytest.param(
        {"groups": {"name": "x"}}, '"groups" is not a list', id="groups-object"
    ),
    pytest.param({"groups": []}, '"groups" is empty', id="empty-groups"),
    pytest.param({"groups": ["x"]}, "groups[0] is not an object", id="group-string"),
    # The index must track the element, not read 0 for every group.
    pytest.param(
        {"groups": [group(), "x"]}, "groups[1] is not an object", id="group-at-index-1"
    ),
]

# One group-level malformation per field the annotation loop reads. Each
# reached a traceback, except the missing `name`, which exited 0 and
# wrote an annotated file — #298's own symptom one level down.
BAD_GROUPS = [
    pytest.param({"name": "g", "bytes_fp16": 0}, 'no "tensors"', id="no-tensors"),
    pytest.param(
        {"tensors": [TENSOR], "bytes_fp16": TENSOR_BYTES}, 'no "name"', id="no-name"
    ),
    pytest.param({"name": "g", "tensors": []}, 'no "bytes_fp16"', id="no-bytes-fp16"),
    pytest.param({}, 'no "name"', id="empty-group"),
    pytest.param(group(name=5), '"name" is not a string', id="name-not-a-string"),
    pytest.param(
        group(bytes_fp16="16"), '"bytes_fp16" is not an integer', id="bytes-a-string"
    ),
    # A string iterates per character, so the loop blamed the checkpoint
    # for a one-letter tensor the map never named.
    pytest.param(
        group(tensors=TENSOR), '"tensors" is not a list of names', id="tensors-a-string"
    ),
    pytest.param(
        group(tensors=[7]), '"tensors" is not a list of names', id="tensors-not-names"
    ),
    pytest.param(
        group(tensors=[], bytes_fp16=0), '"tensors" is empty', id="tensors-empty"
    ),
]


def write_shard(
    model_dir: Path,
    header: str | bytes | None = None,
    name: str = "model-00001-of-00001.safetensors",
) -> Path:
    """Write one safetensors shard, header only, no payload."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = json.dumps(
            {"__metadata__": {"format": "pt"}, TENSOR: RECORD, OTHER_TENSOR: RECORD}
        )
    blob = header.encode("utf-8") if isinstance(header, str) else header
    shard = model_dir / name
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return shard


def write_one_group_map(path: Path) -> None:
    """Write a map naming `TENSOR` alone.

    A shard-collision test must reach the collision. The two-group map
    below is not covered by a shard defining `TENSOR` alone, so the
    coverage refusal would fire first and the test would pass for the
    wrong reason.
    """
    path.write_text(
        json.dumps({"groups": [group(name="model.layers.0.mlp")]}), encoding="utf-8"
    )


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
    write_one_group_map(map_path)
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


@pytest.mark.parametrize(("document", "reason"), NON_MAPS)
def test_input_that_is_not_a_sensitivity_map_refuses_and_writes_no_output(
    tmp_path, monkeypatch, capsys, document: object, reason: str
) -> None:
    # A recipe and an empty map iterated zero times through
    # `raw.get("groups", [])` and reported success (#298). The other
    # shapes reached a traceback. The guard converts both to one
    # `error:` line. The checkpoint is well-formed, so only the input
    # document can refuse the run.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(json.dumps(document), encoding="utf-8")
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(map_path) in stderr
    assert reason in stderr
    assert not out.exists()


@pytest.mark.parametrize(("bad", "reason"), BAD_GROUPS)
def test_group_the_annotation_loop_cannot_read_refuses_and_writes_no_output(
    tmp_path, monkeypatch, capsys, bad: dict, reason: str
) -> None:
    # The loop reads `name`, `tensors`, and `bytes_fp16` off every group.
    # Guarding the document alone left every one of these reaching a
    # traceback or, for a missing `name`, a zero exit over a written
    # file. Both are #298 one level down.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(json.dumps({"groups": [bad]}), encoding="utf-8")
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(map_path) in stderr
    assert "groups[0]" in stderr
    assert reason in stderr
    assert not out.exists()


def test_one_tensor_name_across_two_shards_refuses_and_names_both(
    tmp_path, monkeypatch, capsys
) -> None:
    # Merging the headers kept whichever shard sorted last and reported
    # nothing, so a wrong size reached `tensor_bytes` in a new artifact
    # (#297). The operator needs both paths to tell which file to remove.
    map_path = tmp_path / "sensitivity.json"
    write_one_group_map(map_path)
    model_dir = tmp_path / "model"
    record = json.dumps(RECORD)
    first = write_shard(
        model_dir,
        header=f'{{"{TENSOR}": {record}}}',
        name="model-00001-of-00002.safetensors",
    )
    bigger = json.dumps({"dtype": "BF16", "shape": [8, 2], "data_offsets": [0, 32]})
    second = write_shard(
        model_dir,
        header=f'{{"{TENSOR}": {bigger}}}',
        name="model-00002-of-00002.safetensors",
    )
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(second) in stderr
    assert str(first) in stderr
    assert f'tensor "{TENSOR}" is already defined' in stderr
    assert not out.exists()


def test_an_equal_size_repeated_across_two_shards_refuses_too(
    tmp_path, monkeypatch, capsys
) -> None:
    # `object_from_pairs` refuses a repeated key whatever the values, and
    # a collision across two files is that rule one level up (#297). Two
    # shards claiming one tensor is ambiguous even at an equal size.
    map_path = tmp_path / "sensitivity.json"
    write_one_group_map(map_path)
    model_dir = tmp_path / "model"
    record = json.dumps(RECORD)
    write_shard(
        model_dir,
        header=f'{{"{TENSOR}": {record}}}',
        name="model-00001-of-00002.safetensors",
    )
    write_shard(
        model_dir,
        header=f'{{"{TENSOR}": {record}}}',
        name="model-00002-of-00002.safetensors",
    )
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    assert f'tensor "{TENSOR}" is already defined' in capsys.readouterr().err
    assert not out.exists()


def test_two_shards_defining_different_tensors_backfill_normally(
    tmp_path, monkeypatch, capsys
) -> None:
    # The refusal must key on the repeated name and never on shard count,
    # because a real checkpoint of this target ships many shards. Every
    # real shard header also carries `__metadata__`, so both shards carry
    # it here — `read_safetensors_header` pops it, and a move of that pop
    # past the merge would refuse every real multi-shard checkpoint.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    record = json.dumps(RECORD)
    meta = json.dumps({"format": "pt"})
    write_shard(
        model_dir,
        header=f'{{"__metadata__": {meta}, "{TENSOR}": {record}}}',
        name="model-00001-of-00002.safetensors",
    )
    write_shard(
        model_dir,
        header=f'{{"__metadata__": {meta}, "{OTHER_TENSOR}": {record}}}',
        name="model-00002-of-00002.safetensors",
    )
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 0
    assert capsys.readouterr().err == ""
    groups = json.loads(out.read_text(encoding="utf-8"))["groups"]
    assert [g["tensor_bytes"] for g in groups] == [
        {TENSOR: TENSOR_BYTES},
        {OTHER_TENSOR: TENSOR_BYTES},
    ]


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
