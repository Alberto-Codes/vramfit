"""`TensorSizeSource` over a checkpoint's safetensors shard headers.

The adapter ADR-0029 decision 1 names. A ``.safetensors`` file opens
with a little-endian u64 header length, then that many bytes of JSON
describing every tensor's dtype, shape, and byte range. Reading the
header is a JSON parse and needs no torch, so the plan step stays
importable under ADR-0005. The safetensors *index* carries no shapes,
which is why the source reads the shards themselves.

ADR-0022's Consequences said the backfill script's header reader
"earns a CLI command when a second consumer appears". This source is
that second consumer, and ADR-0029 decision 1 rules that the two
share the reader instead. So `read_safetensors_header` lives here and
`scripts/backfill_tensor_sizes.py` imports it rather than carrying
its own copy.

Two filters shape what the source returns. The MTP block stays out
(decision 2). GGUF numbers one layer stack, so backbone and MTP cannot
pack together, and a source summing both would overstate the weight
budget by the block's 1.335B parameters. Tensors below two dimensions
stay out because they are not quantizable, which is the rank half of
the rule [vramfit.adapters.outbound.scan.discovery][] applies when the
torch meter discovers groups. The two must agree, or the source would
price groups the map can never carry. ADR-0014's residual overhead
fraction covers the tensors both drop.

The meter's other halves diverge on purpose. `discover_groups` also
skips a parameter that is not floating point, because it cannot
perturb one. This source keeps it and lets
[vramfit.domain.sizes][] refuse the dtype. A 2-D integer tensor is
still weight bytes on the card, and skipping it would understate the
budget — the direction ADR-0029 exists to stop. `discover_groups`
also skips a class the quantizer refuses (#204). This source keeps
that too. [vramfit.domain.sizes][] keys such a tensor by its own name
under every granularity, so it reaches the recipe uncovered and the
solver prices it at the convert dtype (#409).

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
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.sizes import (
    DTYPE_ELEMENT_BYTES,
    SizeSourceError,
    TensorSize,
)

# A safetensors file opens with a little-endian u64 header length.
HEADER_PREFIX_BYTES: Final[int] = 8

# The multi-token-prediction block's root. GGUF numbers one layer
# stack, so a pack carries the backbone or the MTP block and never
# both (ADR-0029 decision 2).
MTP_ROOT: Final[str] = "mtp."

# A quantizable parameter has at least this many dimensions. The
# torch meter's `discover_groups` applies the same rule, so the source
# and the meter discover the same groups.
MIN_QUANTIZABLE_DIMENSIONS: Final[int] = 2

# The longest tensor name a refusal quotes. A header key is
# publisher-written and unbounded, so a refusal that embeds one whole
# renders whatever the file holds (#335). No record fixes the width.
NAME_LIMIT: Final[int] = 80


def bounded(name: str) -> str:
    """Cut one tensor name to the length a refusal may quote.

    Args:
        name: A tensor name from a publisher-written header.

    Returns:
        The name, or its first `NAME_LIMIT` characters and an ellipsis.

    Examples:
        ```python
        from vramfit.adapters.outbound.safetensors_sizes import bounded

        assert bounded("blk.0.attn_q.weight") == "blk.0.attn_q.weight"
        ```
    """
    if len(name) <= NAME_LIMIT:
        return name
    return f"{name[:NAME_LIMIT]}..."


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Parse one shard's header: tensor name to dtype/shape record.

    Args:
        path: A ``.safetensors`` file.

    Returns:
        The header mapping, metadata entry removed.

    Raises:
        ValueError: If the file is too short, the header declares more
            bytes than the file holds, the read returns fewer bytes
            than the prefix promised, the header is not valid UTF-8 or
            valid JSON, the header nests past the recursion limit, the
            header carries an integer past the digit limit, the header
            is not a JSON object, or the header defines the same key
            twice (#262).
        OSError: If the file cannot be read or stat'd.

    Examples:
        ```python
        from pathlib import Path

        from vramfit.adapters.outbound.safetensors_sizes import (
            read_safetensors_header,
        )

        header = read_safetensors_header(Path("model-00001-of-00014.safetensors"))
        ```
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(HEADER_PREFIX_BYTES)
        if len(prefix) < HEADER_PREFIX_BYTES:
            raise ValueError(f"{path}: too short for a safetensors header")
        (header_bytes,) = struct.unpack("<Q", prefix)
        # A u64 spans further than any file, and `read` would try to
        # allocate whatever it declares. That raises `OverflowError` or
        # `MemoryError`, neither of which a caller catching `ValueError`
        # sees. Compare against the file first (#335).
        if header_bytes > size - HEADER_PREFIX_BYTES:
            raise ValueError(
                f"{path}: header declares {header_bytes} bytes, and the file "
                f"holds {size - HEADER_PREFIX_BYTES} after the prefix"
            )
        blob = handle.read(header_bytes)
        # `stat` and `read` see the file at two moments. A writer that
        # truncates between them returns a short read, and truncated
        # JSON that still parsed would price a partial checkpoint. The
        # length is what the prefix promised, so check it (#335).
        if len(blob) < header_bytes:
            raise ValueError(
                f"{path}: header declares {header_bytes} bytes, and the read "
                f"returned {len(blob)} — truncated while reading?"
            )
        try:
            header = json.loads(blob, object_pairs_hook=object_from_pairs)
        except DuplicateKeyError as exc:
            raise ValueError(f"{path}: {exc.message}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: header is not valid JSON: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: header is not valid UTF-8: {exc}") from exc
        except RecursionError as exc:
            # Deep nesting exhausts the decoder's stack. `RecursionError`
            # is no `ValueError`, so it escapes both callers (#335).
            raise ValueError(f"{path}: header JSON nests too deeply: {exc}") from exc
        except ValueError as exc:
            # An integer literal past `sys.get_int_max_str_digits`, 4300
            # by default, raises the scanner's plain `ValueError`. It
            # reported with no locator before this clause (#287).
            # `DuplicateKeyError` is no `ValueError`, so the structural
            # refusal cannot land here whatever the order.
            raise ValueError(f"{path}: cannot parse header JSON: {exc}") from exc
    # A top-level array or number parses cleanly and carries no
    # `pop`, so the next line would raise `TypeError` past every
    # caller's catch.
    if not isinstance(header, dict):
        # The reader's whole refusal contract is `ValueError`, and
        # both callers catch exactly that. A `TypeError` here would
        # escape them, which is the defect this guard closes.
        raise ValueError(  # noqa: TRY004 - the reader refuses through ValueError by contract
            f"{path}: header is a JSON {type(header).__name__}, and a "
            f"safetensors header is an object"
        )
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
            f'{shard}: entry "{bounded(name)}" has no data_offsets pair of integers'
        )
    begin, end = offsets
    if end <= begin:
        raise SizeSourceError(
            f'{shard}: entry "{bounded(name)}" spans no bytes, at data_offsets '
            f"[{begin}, {end}]"
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
            integers, and a ``data_offsets`` range. Also if the range
            and the shape disagree about the tensor's size, for a
            dtype the domain can price — a header that contradicts
            itself would understate the weight budget.
    """
    if not isinstance(record, dict):
        raise SizeSourceError(f'{shard}: entry "{bounded(name)}" is not an object')
    dtype = record.get("dtype")
    shape = record.get("shape")
    if not isinstance(dtype, str) or not dtype:
        raise SizeSourceError(f'{shard}: entry "{bounded(name)}" has no dtype string')
    if not isinstance(shape, list) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
    ):
        raise SizeSourceError(
            f'{shard}: entry "{bounded(name)}" has no shape of non-negative integers'
        )
    dims: list[int] = [dim for dim in shape if isinstance(dim, int)]
    if len(dims) < MIN_QUANTIZABLE_DIMENSIONS:
        return None
    stored = _stored_bytes(shard, name, record)
    element = DTYPE_ELEMENT_BYTES.get(dtype)
    if element is not None and math.prod(dims) * element != stored:
        raise SizeSourceError(
            f'{shard}: entry "{bounded(name)}" spans {stored} bytes, and its shape '
            f"{dims} at {dtype} needs {math.prod(dims) * element}"
        )
    return TensorSize(dtype=dtype, bytes=stored)


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
                        f'{shard}: tensor "{bounded(name)}" is already defined by '
                        f"{origin[name]} — two copies of one checkpoint here?"
                    )
                origin[name] = shard
                if name.startswith(MTP_ROOT):
                    continue
                size = _record_size(shard, name, record)
                if size is not None:
                    sizes[name] = size
        return sizes
