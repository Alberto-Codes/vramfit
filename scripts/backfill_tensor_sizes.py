r"""Backfill per-tensor sizes into a sensitivity map (ADR-0022).

Reads each safetensors shard header of a model checkpoint — a JSON
parse of the first bytes, no torch — and writes an annotated copy of
the map with ``tensor_bytes`` on every group (the map-copy mechanism
from ADR-0021 decision 4). Sizes are element counts at 2 bytes per
parameter, matching the scan's bf16 reference convention.

The script reports every refusal as ``error:`` on stderr and exits 1.
It refuses a map whose tensors the checkpoint does not cover. It
refuses a group whose checkpoint sizes disagree with its recorded
``bytes_fp16``. It refuses to overwrite the input. It refuses a shard
it cannot parse. It refuses a JSON document that defines the same key
twice, under the rule #262 set for every vramfit reader. Both reads
apply that rule: the map and each shard header.

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

    Args:
        model_dir: Checkpoint directory holding ``*.safetensors``.

    Returns:
        Mapping of tensor name to ``numel * 2`` bytes.

    Raises:
        ValueError: If the directory holds no safetensors shards.
    """
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{model_dir}: no *.safetensors shards found")
    sizes: dict[str, int] = {}
    for shard in shards:
        for name, record in read_safetensors_header(shard).items():
            sizes[name] = math.prod(record["shape"]) * 2
    return sizes


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
        sizes = checkpoint_tensor_bytes(args.model_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tensor_count = 0
    for group in raw.get("groups", []):
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
    print(
        f"backfilled {len(raw.get('groups', []))} groups "
        f"({tensor_count} tensors) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
