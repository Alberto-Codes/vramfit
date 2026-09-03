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
    declared: int | None = None,
) -> Path:
    """Write one safetensors shard, header only, no payload.

    `declared` overrides the u64 length prefix, so a test can write a
    shard whose prefix disagrees with the bytes behind it.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = json.dumps(
            {"__metadata__": {"format": "pt"}, TENSOR: RECORD, OTHER_TENSOR: RECORD}
        )
    blob = header.encode("utf-8") if isinstance(header, str) else header
    shard = model_dir / name
    length = len(blob) if declared is None else declared
    shard.write_bytes(struct.pack("<Q", length) + blob)
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


# Every map read failure that reached a bare traceback, one per door
# `read_map_document` now converts (#287, #335). The digit-limit row is
# the plain `ValueError` `json.loads` raises past
# `sys.get_int_max_str_digits`, which no named clause catches.
BAD_MAP_FILES = [
    pytest.param(b"{oops", "invalid JSON", id="not-json"),
    pytest.param(b'{"groups": "\xff\xfe"}', "not valid UTF-8", id="not-utf8"),
    pytest.param(
        b'{"groups": [{"name": "g", "tensors": ["t"], "bytes_fp16": '
        + b"1" * 5000
        + b"}]}",
        "cannot parse JSON",
        id="digit-limit",
    ),
]

# Every header record shape the pricing arithmetic cannot take. The
# first three reached a traceback. `shape-holds-a-string` multiplied
# into a longer string and raised several frames away, at the group sum.
# `negative-dimension` priced clean and wrote a negative size.
BAD_RECORDS = [
    pytest.param("oops", "record is not an object", id="record-is-a-string"),
    pytest.param({"dtype": "BF16"}, 'no "shape"', id="no-shape"),
    pytest.param({"shape": "42"}, '"shape" is not a list', id="shape-is-a-string"),
    pytest.param(
        {"shape": [4, "2"]},
        '"shape" holds a str, not a dimension',
        id="shape-holds-a-string",
    ),
    pytest.param(
        {"shape": [4, True]},
        '"shape" holds a bool, not a dimension',
        id="shape-holds-a-bool",
    ),
    pytest.param(
        {"shape": [-4, 2]},
        '"shape" holds a dimension below 1',
        id="negative-dimension",
    ),
    # A zero dimension prices the tensor at zero bytes, and
    # `sensitivity_map_json.py:485` requires a positive value. Two
    # negatives multiply positive, so the size alone cannot carry the
    # rule and every dimension is checked on its own.
    pytest.param(
        {"shape": [0, 2]},
        '"shape" holds a dimension below 1',
        id="zero-dimension",
    ),
]


@pytest.mark.parametrize(("body", "reason"), BAD_MAP_FILES)
def test_map_file_the_reader_cannot_parse_refuses_and_names_the_file(
    tmp_path, monkeypatch, capsys, body: bytes, reason: str
) -> None:
    # `read_safetensors_header` named the shard for each of these doors
    # and the map read beside it named nothing, so every one arrived as
    # a traceback (#287). The checkpoint is well-formed, so only the map
    # can refuse the run.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_bytes(body)
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert stderr.startswith("error: ")
    assert str(map_path) in stderr
    assert reason in stderr
    assert not out.exists()


def test_map_path_that_does_not_exist_refuses_and_names_the_file(
    tmp_path, monkeypatch, capsys
) -> None:
    # `read_text` raised `FileNotFoundError` out of `main`, so a typo in
    # the first argument printed a traceback rather than a refusal.
    map_path = tmp_path / "absent.json"
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(map_path) in stderr
    assert "cannot read" in stderr
    assert not out.exists()


def test_out_equal_to_the_input_map_refuses_before_reading_anything(
    tmp_path, monkeypatch, capsys
) -> None:
    # The guard runs before either read, so the input survives whatever
    # the checkpoint holds. This case passes one path twice, so plain
    # equality satisfies it. The symlink case below is what pins the
    # resolve.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    before = map_path.read_text(encoding="utf-8")
    model_dir = tmp_path / "model"
    write_shard(model_dir)

    code = run(monkeypatch, map_path, model_dir, map_path)

    assert code == 1
    assert (
        f"--out must differ from the input map: {map_path}" in capsys.readouterr().err
    )
    assert map_path.read_text(encoding="utf-8") == before


def test_model_directory_without_shards_refuses_and_names_the_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    # A checkpoint held as `.bin` pickles, or a path off by one
    # directory, both land here. Merging zero headers would refuse every
    # group for a missing tensor instead, blaming the map.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(model_dir) in stderr
    assert "no *.safetensors shards found" in stderr
    assert not out.exists()


def test_shard_shorter_than_the_length_prefix_refuses_and_names_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # An empty or part-written file reaches `struct.unpack` with fewer
    # than eight bytes, which would raise `struct.error` outside the
    # `ValueError` root `main` catches.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"\x00\x01\x02\x03")
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert "too short for a safetensors header" in stderr
    assert not out.exists()


def test_shard_header_that_is_not_json_refuses_and_names_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # The prefix agrees with the bytes behind it, so the size guard
    # passes and the parse is what refuses.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, header="{oops")
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert "header is not valid JSON" in stderr
    assert not out.exists()


def test_shard_header_past_the_digit_limit_refuses_and_names_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # `json.loads` raises a plain `ValueError` past
    # `sys.get_int_max_str_digits`, which the two named clauses miss.
    # The refusal reported with no locator before the catch-all (#287).
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, header=f'{{"{TENSOR}": {"1" * 5000}}}')
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert "cannot parse header JSON" in stderr
    assert not out.exists()


def test_shard_header_that_is_not_an_object_refuses_and_names_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # A JSON array parses clean and reaches `header.pop`, which raised
    # `AttributeError` outside the `ValueError` root `main` catches.
    # The refusal is the shared reader's since #358 moved it, so this
    # pins the script's route to it rather than the wording.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, header="[1, 2, 3]")
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert "a safetensors header is an object" in stderr
    assert not out.exists()


def test_shard_declaring_a_header_longer_than_the_file_refuses(
    tmp_path, monkeypatch, capsys
) -> None:
    # The quiet row of #335. `read` returned short, the truncated bytes
    # parsed clean, and the script priced the tensors and exited 0 over
    # a written file. The header here is complete and the prefix lies,
    # so nothing but the size comparison can catch it.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, declared=10_000)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert "header declares 10000 bytes" in stderr
    assert not out.exists()


def test_shard_declaring_a_header_of_gigabytes_refuses_without_allocating(
    tmp_path, monkeypatch, capsys
) -> None:
    # The guard compares against `st_size` rather than the length `read`
    # returns. Reading first would ask for this many bytes up front.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, declared=2**40)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert f"header declares {2**40} bytes" in stderr
    assert not out.exists()


@pytest.mark.parametrize(("record", "reason"), BAD_RECORDS)
def test_header_record_the_pricing_cannot_take_refuses_and_names_the_tensor(
    tmp_path, monkeypatch, capsys, record: object, reason: str
) -> None:
    # A publisher writes the header, so its records are untrusted like
    # the document around them. `math.prod(record["shape"]) * 2` took
    # whatever the file held (#335). The map here names one tensor, so
    # the record refusal fires before the coverage comparison.
    map_path = tmp_path / "sensitivity.json"
    write_one_group_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, header=json.dumps({TENSOR: record}))
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert f'tensor "{TENSOR}"' in stderr
    assert reason in stderr
    assert not out.exists()


def test_checkpoint_without_a_group_tensor_refuses_and_names_both(
    tmp_path, monkeypatch, capsys
) -> None:
    # The map and the checkpoint disagree, and the operator needs the
    # group and the tensor to tell which pair is wrong.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    write_shard(model_dir, header=json.dumps({TENSOR: RECORD}))
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert 'group "model.layers.1.mlp"' in stderr
    assert f'checkpoint has no tensor "{OTHER_TENSOR}"' in stderr
    assert not out.exists()


def test_checkpoint_sizes_disagreeing_with_bytes_fp16_refuse_with_both_figures(
    tmp_path, monkeypatch, capsys
) -> None:
    # This comparison stands between a wrong checkpoint and a corrupt
    # `tensor_bytes` annotation. A revision that reshaped one tensor
    # covers every name and still prices the group wrong.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"groups": [group(name="model.layers.0.mlp", bytes_fp16=999)]}),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    write_shard(model_dir, header=json.dumps({TENSOR: RECORD}))
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert 'group "model.layers.0.mlp"' in stderr
    assert f"checkpoint sizes sum to {TENSOR_BYTES}" in stderr
    assert "bytes_fp16 999" in stderr
    assert not out.exists()


def test_output_in_a_directory_that_does_not_exist_refuses_and_names_the_path(
    tmp_path, monkeypatch, capsys
) -> None:
    # `write_text` raised `FileNotFoundError` after every read had
    # succeeded, so the whole run's work reported as a traceback naming
    # no artifact.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "absent" / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(out) in stderr
    assert "cannot write" in stderr
    assert not out.exists()


def test_out_symlinked_to_the_input_map_refuses(tmp_path, monkeypatch, capsys) -> None:
    # The guard compares resolved paths, and the identical-path case
    # above passes under plain equality too. Only a second name for one
    # file separates the two. Without the resolve the script reads the
    # map, then writes the annotated copy back through the link.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    before = map_path.read_text(encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(map_path)
    model_dir = tmp_path / "model"
    write_shard(model_dir)

    code = run(monkeypatch, map_path, model_dir, link)

    assert code == 1
    assert (
        f"--out must differ from the input map: {map_path}" in capsys.readouterr().err
    )
    assert map_path.read_text(encoding="utf-8") == before


def test_out_naming_a_checkpoint_shard_refuses_and_leaves_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # The map guard covered one input and the shards are input too. The
    # script read this shard, priced it, then truncated it to JSON and
    # exited 0. That destroys checkpoint data on a success report.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir)
    before = shard.read_bytes()

    code = run(monkeypatch, map_path, model_dir, shard)

    assert code == 1
    stderr = capsys.readouterr().err
    assert "--out must not name a checkpoint shard" in stderr
    assert str(shard) in stderr
    assert shard.read_bytes() == before


def test_out_that_does_not_resolve_refuses_and_names_the_path(
    tmp_path, monkeypatch, capsys
) -> None:
    # `Path.resolve` raises `RuntimeError` on a symlink loop, which is
    # no `ValueError`. The comparison it feeds ran above the try block,
    # so the loop reached a traceback.
    #
    # The raise is patched rather than provoked with a real loop. 3.12
    # raises `RuntimeError` on a self-symlink and 3.13 resolves it and
    # fails later at the write, so the live input pins two different
    # messages. CI runs both interpreters.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "loop.json"

    def raise_loop(self: Path, *args: object, **kwargs: object) -> Path:
        raise RuntimeError(f"Symlink loop from {str(self)!r}")

    monkeypatch.setattr(Path, "resolve", raise_loop)
    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(out) in stderr
    assert "cannot resolve" in stderr
    assert not out.exists()


def test_a_real_symlink_loop_at_out_never_reaches_a_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    # The interpreters disagree on where a self-symlink fails, so this
    # pins only what the contract promises: one `error:` line naming the
    # path, and no output. 3.12 refuses at the resolve, 3.13 at the
    # write.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    loop = tmp_path / "loop.json"
    loop.symlink_to(loop)

    code = run(monkeypatch, map_path, model_dir, loop)

    assert code == 1
    stderr = capsys.readouterr().err
    assert stderr.startswith("error: ")
    assert str(loop) in stderr


def test_map_nested_past_the_recursion_limit_refuses_and_names_the_file(
    tmp_path, monkeypatch, capsys
) -> None:
    # `json.loads` raises `RecursionError` on deep nesting, and that is
    # no `ValueError`. It escaped every clause in both readers and in
    # `main`, so it falsified the docstring's contract outright.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_bytes(b"[" * 200_000 + b"]" * 200_000)
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(map_path) in stderr
    assert "nests too deeply" in stderr
    assert not out.exists()


def test_shard_header_nested_past_the_recursion_limit_refuses_and_names_it(
    tmp_path, monkeypatch, capsys
) -> None:
    # The header carries the same door as the map document.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir, header=b"[" * 200_000 + b"]" * 200_000)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(shard) in stderr
    assert "nests too deeply" in stderr
    assert not out.exists()


def test_model_directory_that_does_not_exist_refuses_as_unlistable(
    tmp_path, monkeypatch, capsys
) -> None:
    # `Path.glob` swallows every `OSError`, so an absent directory read
    # as an empty one and the operator was told the checkpoint holds no
    # shards. It holds nothing because it is not there.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "absent"
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(model_dir) in stderr
    assert "cannot list" in stderr
    assert "no *.safetensors shards found" not in stderr
    assert not out.exists()


def test_refusal_quoting_a_tensor_name_bounds_the_message(
    tmp_path, monkeypatch, capsys
) -> None:
    # A header key is publisher-written and unbounded, and the record
    # refusal quotes it. The collision case below carries the same rule
    # for the message that already quoted a name before this change.
    long_name = "A" * 100_000
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"groups": [group(tensors=[long_name])]}), encoding="utf-8"
    )
    model_dir = tmp_path / "model"
    write_shard(model_dir, header=json.dumps({long_name: {"shape": [4, "2"]}}))
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert len(stderr) < 1000
    assert f'tensor "{"A" * script.NAME_LIMIT}..."' in stderr
    assert not out.exists()


def test_shard_collision_refusal_bounds_the_tensor_name(
    tmp_path, monkeypatch, capsys
) -> None:
    # This message quoted a header key before this change, and a 100,000
    # character name rendered 100,409 bytes of stderr. It is the
    # pre-existing instance of the bound, against ADR-0011's 2026-08-16
    # rule that a refusal message stays bounded.
    long_name = "A" * 100_000
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"groups": [group(tensors=[long_name])]}), encoding="utf-8"
    )
    model_dir = tmp_path / "model"
    record = json.dumps(RECORD)
    for shard_name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        write_shard(model_dir, header=f'{{"{long_name}": {record}}}', name=shard_name)
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert len(stderr) < 1000
    assert "is already defined by" in stderr
    assert f'tensor "{"A" * script.NAME_LIMIT}..."' in stderr
    assert not out.exists()


def test_document_nested_past_the_encoder_limit_refuses_and_writes_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    # `json.dumps` with an indent uses the pure-Python encoder, whose
    # recursion budget is smaller than the C scanner's. A document the
    # read accepted reached a traceback at the render, which is the one
    # door the write guard's `except OSError` cannot see.
    #
    # The raise is patched rather than provoked. The depth that exhausts
    # the encoder moves between interpreters — 1,200 levels refuse on
    # 3.12 and backfill on 3.13 — so a live document pins a version
    # rather than the handler. CI runs both.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    def raise_recursion(*args: object, **kwargs: object) -> str:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(script.json, "dumps", raise_recursion)
    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(out) in stderr
    assert "cannot render" in stderr
    assert not out.exists()


def test_out_hardlinked_to_a_shard_refuses_and_leaves_the_shard(
    tmp_path, monkeypatch, capsys
) -> None:
    # A resolved path catches a symlink and a `..` traversal. It cannot
    # catch a hardlink, which gives one inode a second name, so the
    # guard let the write through and destroyed the shard.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    shard = write_shard(model_dir)
    before = shard.read_bytes()
    hardlink = tmp_path / "hardout.json"
    hardlink.hardlink_to(shard)

    code = run(monkeypatch, map_path, model_dir, hardlink)

    assert code == 1
    assert "--out must not name a checkpoint shard" in capsys.readouterr().err
    assert shard.read_bytes() == before


def test_out_hardlinked_to_the_input_map_refuses(tmp_path, monkeypatch, capsys) -> None:
    # The map guard carried the same gap as the shard guard.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    before = map_path.read_text(encoding="utf-8")
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    hardlink = tmp_path / "hardout.json"
    hardlink.hardlink_to(map_path)

    code = run(monkeypatch, map_path, model_dir, hardlink)

    assert code == 1
    assert (
        f"--out must differ from the input map: {map_path}" in capsys.readouterr().err
    )
    assert map_path.read_text(encoding="utf-8") == before


def test_a_size_too_wide_to_render_refuses_and_still_names_the_map(
    tmp_path, monkeypatch, capsys
) -> None:
    # `int.__str__` raises past `sys.get_int_max_str_digits`. Building
    # the mismatch message raised there, so `main` printed the
    # interpreter's own text and named no file, group, or tensor.
    map_path = tmp_path / "sensitivity.json"
    write_one_group_map(map_path)
    model_dir = tmp_path / "model"
    wide = 10**4290
    write_shard(
        model_dir, header=json.dumps({TENSOR: {"dtype": "BF16", "shape": [wide, wide]}})
    )
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert str(map_path) in stderr
    assert "bits" in stderr
    assert "sys.set_int_max_str_digits" not in stderr
    assert not out.exists()


def test_annotate_refusals_bound_the_group_and_tensor_names(
    tmp_path, monkeypatch, capsys
) -> None:
    # Both names reach the map from the checkpoint, so a publisher
    # writes them. The coverage refusal rendered 200,080 bytes of stderr
    # against a pair of 100,000-character names.
    long_name = "A" * 100_000
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"groups": [group(name=long_name, tensors=[long_name])]}),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    write_shard(model_dir, header=json.dumps({TENSOR: RECORD}))
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert len(stderr) < 1000
    assert str(map_path) in stderr
    assert "checkpoint has no tensor" in stderr
    assert not out.exists()


def test_a_scalar_tensor_prices_at_two_bytes(tmp_path, monkeypatch, capsys) -> None:
    # `shape` of `[]` is a scalar and `math.prod([])` is 1, so it prices
    # at the reference's 2 bytes. The dimension guard must not reach it:
    # a scalar carries no dimension to fail.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(json.dumps({"groups": [group(bytes_fp16=2)]}), encoding="utf-8")
    model_dir = tmp_path / "model"
    write_shard(model_dir, header=json.dumps({TENSOR: {"dtype": "F32", "shape": []}}))
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 0
    assert capsys.readouterr().err == ""
    groups = json.loads(out.read_text(encoding="utf-8"))["groups"]
    assert groups[0]["tensor_bytes"] == {TENSOR: 2}


def test_map_with_an_integer_past_the_reader_bound_refuses_and_writes_no_output(
    tmp_path, monkeypatch, capsys
) -> None:
    # `_save_json` bounds every artifact integer to the signed 64-bit
    # range. This script wrote with a bare `json.dumps`. A map recording
    # a `bytes_fp16` of 2 * 10^30 against a shard declaring that shape
    # backfilled on a zero exit. The map reader then refused the copy at
    # the same field (#317).
    dim = 10**30
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"groups": [group(bytes_fp16=2 * dim)]}), encoding="utf-8"
    )
    header = json.dumps(
        {TENSOR: {"dtype": "BF16", "shape": [dim], "data_offsets": [0, 2 * dim]}}
    )
    write_shard(tmp_path / "model", header)
    out = tmp_path / "sized.json"

    assert run(monkeypatch, map_path, tmp_path / "model", out) == 1

    err = capsys.readouterr().err
    assert err.startswith(f"error: {map_path}: $.groups[0].bytes_fp16: ")
    assert "outside the signed 64-bit range" in err
    assert not out.exists()


def test_size_at_the_top_of_the_reader_bound_still_backfills(
    tmp_path, monkeypatch, capsys
) -> None:
    # The bound is the reader's and no narrower. The largest even size
    # inside the range backfills, and the written value survives.
    dim = 2**62 - 1
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"groups": [group(bytes_fp16=2 * dim)]}), encoding="utf-8"
    )
    header = json.dumps(
        {TENSOR: {"dtype": "BF16", "shape": [dim], "data_offsets": [0, 2 * dim]}}
    )
    write_shard(tmp_path / "model", header)
    out = tmp_path / "sized.json"

    assert run(monkeypatch, map_path, tmp_path / "model", out) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["groups"][0]["tensor_bytes"] == {TENSOR: 2 * dim}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(2**63, 1, id="past-the-top"),
        pytest.param(-(2**63) - 1, 1, id="past-the-bottom"),
        pytest.param(2**63 - 1, 0, id="at-the-top"),
    ],
)
def test_integer_outside_groups_is_held_to_the_reader_bound(
    tmp_path, monkeypatch, capsys, value: int, expected: int
) -> None:
    # The walk is `_save_json`'s and covers the whole document, so a
    # field the script never reads refuses under its own path. The
    # pricing cannot reach 2^63 - 1, so the exact top of the range is
    # pinned here instead.
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"scan": {"calibration_tokens": value}, "groups": [group()]}),
        encoding="utf-8",
    )
    write_shard(tmp_path / "model")
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, tmp_path / "model", out)

    assert code == expected
    err = capsys.readouterr().err
    if expected:
        assert err.startswith(f"error: {map_path}: $.scan.calibration_tokens: ")
        assert not out.exists()
    else:
        assert err == ""
        assert json.loads(out.read_text(encoding="utf-8"))["scan"] == {
            "calibration_tokens": value
        }


def test_integer_refusal_bounds_a_publisher_written_key(
    tmp_path, monkeypatch, capsys
) -> None:
    # The walk renders the key inside its JSON path. A 100,000-character
    # key rendered 100,267 bytes of stderr before the cap.
    key = "k" * 100_000
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps({"scan": {key: 2**63}, "groups": [group()]}), encoding="utf-8"
    )
    write_shard(tmp_path / "model")
    out = tmp_path / "sized.json"

    code = run(monkeypatch, map_path, tmp_path / "model", out)

    assert code == 1
    err = capsys.readouterr().err
    kept = "k" * (script.NAME_LIMIT - len("$.scan."))
    assert f": $.scan.{kept}...: integer is outside" in err
    assert "k" * (script.NAME_LIMIT + 1) not in err


def test_document_nested_past_the_walk_limit_refuses_and_writes_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    # `_check_writable_ints` recurses in Python. A document nested past
    # the interpreter's limit raised `RecursionError` there, ahead of
    # the render guard, and reached a traceback. Patched rather than
    # provoked, for the reason the render test gives.
    map_path = tmp_path / "sensitivity.json"
    write_map(map_path)
    model_dir = tmp_path / "model"
    write_shard(model_dir)
    out = tmp_path / "sized.json"

    def raise_recursion(*args: object, **kwargs: object) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(script, "_check_writable_ints", raise_recursion)
    code = run(monkeypatch, map_path, model_dir, out)

    assert code == 1
    stderr = capsys.readouterr().err
    assert stderr.startswith(f"error: {map_path}: JSON nests too deeply")
    assert not out.exists()
