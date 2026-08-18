r"""Backfill per-tensor sizes into a sensitivity map (ADR-0022).

Reads each safetensors shard header of a model checkpoint — a JSON
parse of the first bytes, no torch — and writes an annotated copy of
the map with ``tensor_bytes`` on every group (the map-copy mechanism
from ADR-0021 decision 4). Sizes are element counts at 2 bytes per
parameter, matching the scan's bf16 reference convention.

The script reports each refusal below as ``error:`` on stderr and exits
1. It refuses a document that is not a sensitivity map (#298). It
refuses a map whose tensors the checkpoint does not cover. It refuses a
group whose checkpoint sizes disagree with its recorded ``bytes_fp16``.
It refuses two shards that define one tensor name (#297). It refuses to
overwrite the input. It refuses a shard it cannot parse. It refuses a
JSON document that defines the same key twice, under the rule #262 set
for every vramfit reader. Both reads apply that rule: the map and each
shard header.

Seven inputs outside the map still escape that contract and reach a
traceback. #335 carries them.

Examples:
    Annotate an existing map:

    ```console
    $ uv run python scripts/backfill_tensor_sizes.py \\
        sensitivity.json /models/nemotron --out sensitivity-sized.json
    backfilled 82 groups (438 tensors) -> sensitivity-sized.json
    ```
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)

# A safetensors file opens with a little-endian u64 header length.
HEADER_PREFIX_BYTES = 8


class MapRefusal(ValueError):
    """The input document is not a sensitivity map this script can read.

    `main` catches `ValueError` to print one ``error:`` line and exit 1,
    so this subclass reaches that path unchanged. It exists because the
    guards behind it test types, and a bare `ValueError` there fights
    ruff's `TRY004`. Raising `TypeError` instead would force `main` to
    catch `TypeError` as well, and that wider clause would relabel a
    real one from `checkpoint_tensor_bytes` as an operator mistake.
    """


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Parse one shard's header: tensor name to dtype/shape record.

    Args:
        path: A ``.safetensors`` file.

    Returns:
        The header mapping, metadata entry removed.

    Raises:
        ValueError: If the file is too short, the header is not valid
            UTF-8 or valid JSON, or the header defines the same key
            twice.
    """
    with path.open("rb") as handle:
        prefix = handle.read(HEADER_PREFIX_BYTES)
        if len(prefix) < HEADER_PREFIX_BYTES:
            raise ValueError(f"{path}: too short for a safetensors header")
        (header_bytes,) = struct.unpack("<Q", prefix)
        try:
            header = json.loads(
                handle.read(header_bytes), object_pairs_hook=object_from_pairs
            )
        except DuplicateKeyError as exc:
            raise ValueError(f"{path}: {exc.message}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: header is not valid JSON: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: header is not valid UTF-8: {exc}") from exc
    header.pop("__metadata__", None)
    return header


def checkpoint_tensor_bytes(model_dir: Path) -> dict[str, int]:
    """Collect every checkpoint tensor's size at the bf16 reference.

    Two shards that declare one tensor name mean two shapes at once, so
    the script cannot price the tensor. It refuses rather than keeping
    whichever shard sorts last (#297). An equal repeated size refuses
    too, matching `object_from_pairs`, which reads the repeat and never
    the values.

    Args:
        model_dir: Checkpoint directory holding ``*.safetensors``.

    Returns:
        Mapping of tensor name to ``numel * 2`` bytes.

    Raises:
        ValueError: If the directory holds no safetensors shards, or two
            shards define one tensor name.
    """
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{model_dir}: no *.safetensors shards found")
    sizes: dict[str, int] = {}
    origin: dict[str, Path] = {}
    for shard in shards:
        for name, record in read_safetensors_header(shard).items():
            if name in origin:
                raise ValueError(
                    f'{shard}: tensor "{name}" is already defined by '
                    f"{origin[name]} — two copies of one checkpoint here?"
                )
            sizes[name] = math.prod(record["shape"]) * 2
            origin[name] = shard
    return sizes


def _require_group_fields(group: dict, where: str) -> None:
    """Refuse a group the annotation loop cannot read.

    The loop reads three fields off every group. It reports a refusal
    under ``name``, it tests membership over ``tensors``, and it compares
    a sum against ``bytes_fp16``. A group missing one reaches a bare
    `KeyError`. A group carrying the wrong type is worse. A string
    ``tensors`` iterates per character, so the loop blames the checkpoint
    for a tensor the map never named.

    Args:
        group: One element of the map's ``groups`` list.
        where: The file and index, for the refusal message.

    Raises:
        MapRefusal: If a field is absent, or carries the wrong type.
    """
    for field in ("name", "tensors", "bytes_fp16"):
        if field not in group:
            raise MapRefusal(f'{where}: no "{field}" — not a sensitivity map')
    if not isinstance(group["name"], str):
        raise MapRefusal(f'{where}: "name" is not a string')
    if not isinstance(group["bytes_fp16"], int) or isinstance(
        group["bytes_fp16"], bool
    ):
        raise MapRefusal(f'{where}: "bytes_fp16" is not an integer')
    tensors = group["tensors"]
    if not isinstance(tensors, list) or not all(isinstance(t, str) for t in tensors):
        raise MapRefusal(f'{where}: "tensors" is not a list of names')
    if not tensors:
        raise MapRefusal(f'{where}: "tensors" is empty — a group names one tensor')


def map_groups(raw: object, path: Path) -> list[dict]:
    """Return the map's groups, refusing a document that is not a map.

    The script reaches for ``groups`` and annotates each element. Given a
    recipe, it iterated zero times and wrote an unannotated copy on a
    zero exit (#298). That report also worded a genuinely empty map, so
    the two cases read alike. Handing the script the wrong artifact is
    the most likely operator mistake, so it refuses here.

    An empty ``groups`` list refuses too. `map_from_dict` already
    requires a map to carry one group, so a zero-group document is not a
    map the project accepts.

    This guard checks the fields the annotation loop dereferences and
    stops there. `vramfit.adapters.outbound.sensitivity_map_json` owns
    schema validation, and this script never grows a second copy of it.

    Args:
        raw: The parsed input document.
        path: The file it came from, for the refusal message.

    Returns:
        The ``groups`` list, every element a readable group.

    Raises:
        MapRefusal: If the document is not a sensitivity map.
    """
    if not isinstance(raw, dict) or "groups" not in raw:
        raise MapRefusal(f'{path}: not a sensitivity map — no "groups" key')
    groups = raw["groups"]
    if not isinstance(groups, list):
        raise MapRefusal(f'{path}: "groups" is not a list — not a sensitivity map')
    if not groups:
        raise MapRefusal(f'{path}: "groups" is empty — a map carries one group')
    for i, group in enumerate(groups):
        where = f"{path}: groups[{i}]"
        if not isinstance(group, dict):
            raise MapRefusal(f"{where} is not an object — not a sensitivity map")
        _require_group_fields(group, where)
    return groups


def main() -> int:
    """Annotate a sensitivity map with per-tensor sizes.

    Returns:
        Process exit code: 0 on success, 1 on refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_path", type=Path, help="sensitivity map to annotate")
    parser.add_argument("model_dir", type=Path, help="checkpoint with safetensors")
    parser.add_argument("--out", type=Path, required=True, help="annotated map copy")
    args = parser.parse_args()

    if args.out.resolve() == args.map_path.resolve():
        print("error: --out must differ from the input map", file=sys.stderr)
        return 1
    try:
        raw = json.loads(
            args.map_path.read_text(encoding="utf-8"),
            object_pairs_hook=object_from_pairs,
        )
    except DuplicateKeyError as exc:
        print(f"error: {args.map_path}: {exc.message}", file=sys.stderr)
        return 1
    try:
        groups = map_groups(raw, args.map_path)
        sizes = checkpoint_tensor_bytes(args.model_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tensor_count = 0
    for group in groups:
        missing = [t for t in group["tensors"] if t not in sizes]
        if missing:
            print(
                f'error: group "{group["name"]}": checkpoint has no tensor '
                f'"{missing[0]}" — wrong checkpoint for this map?',
                file=sys.stderr,
            )
            return 1
        tensor_bytes = {t: sizes[t] for t in group["tensors"]}
        if sum(tensor_bytes.values()) != group["bytes_fp16"]:
            print(
                f'error: group "{group["name"]}": checkpoint sizes sum to '
                f"{sum(tensor_bytes.values())} but the map records "
                f"bytes_fp16 {group['bytes_fp16']} — wrong checkpoint for "
                "this map?",
                file=sys.stderr,
            )
            return 1
        group["tensor_bytes"] = tensor_bytes
        tensor_count += len(group["tensors"])

    args.out.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(f"backfilled {len(groups)} groups ({tensor_count} tensors) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
