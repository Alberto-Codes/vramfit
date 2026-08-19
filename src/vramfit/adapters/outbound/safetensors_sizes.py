"""`TensorSizeSource` over a checkpoint's safetensors shard headers.

The adapter ADR-0029 decision 1 names. A ``.safetensors`` file opens
with a little-endian u64 header length, then that many bytes of JSON
describing every tensor's dtype, shape, and byte range. Reading the
header is a JSON parse and needs no torch, so the plan step stays
importable under ADR-0005. The safetensors *index* carries no shapes,
which is why the source reads the shards themselves.

`read_safetensors_header` is the reader ADR-0022's Consequences
promised would earn a second consumer:
`scripts/backfill_tensor_sizes.py` imports it from here rather than
carrying its own copy.

Two filters shape what the source returns. The MTP block stays out
(decision 2) — GGUF numbers one layer stack, so backbone and MTP
cannot pack together, and a source summing both would overstate the
weight budget by the block's 1.335B parameters. Tensors below two
dimensions stay out because they are not quantizable, which is the
same rule
[vramfit.adapters.outbound.scan.discovery][] applies when the torch
meter discovers groups. The two must agree, or the source would price
groups the map can never carry. The residual overhead fraction covers
those tensors, per ADR-0014.

The adapter reports each header's dtype verbatim and computes no
convention of its own (decision 5). Converting a stored size to
reference bytes is domain arithmetic
([vramfit.domain.sizes][]), and so is group aggregation.

Examples:
    Read the 30B target's checkpoint:

    ```python
    from pathlib import Path

    from vramfit.adapters.outbound.safetensors_sizes import SafetensorsSizes

    source = SafetensorsSizes(Path("/models/nemotron-30b"))
    sizes = source.tensor_sizes()
    ```

See Also:
    - [vramfit.ports.outbound][]: The `TensorSizeSource` port.
    - [vramfit.domain.sizes][]: Reference-byte and grouping arithmetic.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.sizes import SizeSourceError, TensorSize

# A safetensors file opens with a little-endian u64 header length.
HEADER_PREFIX_BYTES: Final[int] = 8

# The multi-token-prediction block's root. GGUF numbers one layer
# stack, so a pack carries the backbone or the MTP block and never
# both (ADR-0029 decision 2).
MTP_ROOT: Final[str] = "mtp."

# A quantizable parameter has at least this many dimensions. The
# torch meter's `discover_groups` applies the same rule, and the two
# must agree on what the model's groups are.
MIN_QUANTIZABLE_DIMENSIONS: Final[int] = 2


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Parse one shard's header: tensor name to dtype/shape record.

    Args:
        path: A ``.safetensors`` file.

    Returns:
        The header mapping, metadata entry removed.

    Raises:
        ValueError: If the file is too short, the header is not valid
            UTF-8 or valid JSON, or the header defines the same key
            twice (#262).
        OSError: If the file cannot be read.

    Examples:
        ```python
        from pathlib import Path

        from vramfit.adapters.outbound.safetensors_sizes import (
            read_safetensors_header,
        )

        header = read_safetensors_header(Path("model-00001-of-00014.safetensors"))
        ```
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


def _stored_bytes(shard: Path, name: str, record: dict) -> int:
    """Read one header entry's stored byte count from its data range.

    The byte count comes from ``data_offsets`` rather than from
    ``shape`` times an element size. The header states the range, so
    the adapter reads it instead of owning a dtype-to-width table
    (ADR-0029 decision 5).

    Args:
        shard: The shard the entry came from, named in every refusal.
        name: The tensor's name.
        record: The header's entry for it.

    Returns:
        The tensor's stored size in bytes.

    Raises:
        SizeSourceError: If ``data_offsets`` is not a pair of
            integers describing a non-empty range.
    """
    offsets = record.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2  # noqa: PLR2004 - a byte range is a begin and an end
        or not all(
            isinstance(edge, int) and not isinstance(edge, bool) for edge in offsets
        )
    ):
        raise SizeSourceError(
            f'{shard}: entry "{name}" has no data_offsets pair of integers'
        )
    begin, end = offsets
    if end <= begin:
        raise SizeSourceError(
            f'{shard}: entry "{name}" spans no bytes, at data_offsets [{begin}, {end}]'
        )
    return end - begin


def _record_size(shard: Path, name: str, record: object) -> TensorSize | None:
    """Read one header entry into a size record, or skip it.

    Args:
        shard: The shard the entry came from, named in every refusal.
        name: The tensor's name.
        record: The header's entry for it.

    Returns:
        The tensor's stored size, or None when the entry has fewer
        than `MIN_QUANTIZABLE_DIMENSIONS` dimensions. A dtype outside
        the domain's table is not skipped — the domain refuses it, so
        an unpriceable checkpoint never reads as a smaller one.

    Raises:
        SizeSourceError: If the entry is not an object carrying a
            ``dtype`` string, a ``shape`` list of non-negative
            integers, and a ``data_offsets`` range.
    """
    if not isinstance(record, dict):
        raise SizeSourceError(f'{shard}: entry "{name}" is not an object')
    dtype = record.get("dtype")
    shape = record.get("shape")
    if not isinstance(dtype, str) or not dtype:
        raise SizeSourceError(f'{shard}: entry "{name}" has no dtype string')
    if not isinstance(shape, list) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
    ):
        raise SizeSourceError(
            f'{shard}: entry "{name}" has no shape of non-negative integers'
        )
    if len(shape) < MIN_QUANTIZABLE_DIMENSIONS:
        return None
    return TensorSize(dtype=dtype, bytes=_stored_bytes(shard, name, record))


@dataclass(frozen=True, slots=True)
class SafetensorsSizes:
    """Reads per-tensor sizes from a checkpoint's safetensors shards.

    Attributes:
        model_dir (Path): The checkpoint directory holding
            ``*.safetensors`` shards.

    Examples:
        ```python
        from pathlib import Path

        from vramfit.adapters.outbound.safetensors_sizes import SafetensorsSizes

        source = SafetensorsSizes(Path("/models/nemotron-30b"))
        ```
    """

    model_dir: Path

    def tensor_sizes(self) -> dict[str, TensorSize]:
        """Read every quantizable tensor's stored size and dtype.

        Two shards that declare one tensor name mean two shapes at
        once, so the source cannot price the tensor. It refuses
        rather than keeping whichever shard sorts last, matching
        `scripts/backfill_tensor_sizes.py` (#297).

        Returns:
            One record per quantizable tensor outside the MTP block,
            keyed by the checkpoint's own tensor name.

        Raises:
            SizeSourceError: If the directory holds no shards, a
                shard cannot be parsed, or two shards define one
                tensor name.
            OSError: If a shard cannot be read.
        """
        shards = sorted(self.model_dir.glob("*.safetensors"))
        if not shards:
            raise SizeSourceError(
                f"{self.model_dir}: no *.safetensors shards found — plan reads "
                f"tensor sizes from the checkpoint (ADR-0029)"
            )
        sizes: dict[str, TensorSize] = {}
        origin: dict[str, Path] = {}
        for shard in shards:
            try:
                header = read_safetensors_header(shard)
            except ValueError as exc:
                raise SizeSourceError(str(exc)) from exc
            for name, record in header.items():
                if name in origin:
                    raise SizeSourceError(
                        f'{shard}: tensor "{name}" is already defined by '
                        f"{origin[name]} — two copies of one checkpoint here?"
                    )
                origin[name] = shard
                if name.startswith(MTP_ROOT):
                    continue
                size = _record_size(shard, name, record)
                if size is not None:
                    sizes[name] = size
        return sizes
