"""Write the temporary mixed GGUF the preprocessor hands to the quantizer.

ADR-0032 decision 1: the preprocessor reads an immutable
floating-point base and writes a temporary mixed GGUF beside it,
replacing only the tensors the encoder pre-encoded. This module owns
that rewrite, and it imports no torch and no gguf-py: gguf-py 0.19.0
cannot write a ``Q2_0`` tensor, because its type table does not name
one. The header parse comes from
[vramfit.adapters.outbound.gguf.header][].

The rewrite copies the base's key-value section byte for byte, so
every metadata value the base states survives unchanged. It rewrites
the tensor infos with the replaced tensors' new type id and every
tensor's recomputed offset, pads to the base's alignment, and streams
the data section in the base's own data order: a kept tensor's span
copies verbatim, and a replaced tensor's payload file takes its
place, padded to the alignment. Nothing loads a whole tensor into
memory, so the peak is one copy chunk.

Examples:
    Replace one expert stack with its pre-encoded payload:

    ```python
    written = write_mixed_gguf(
        base=Path("model-f16.gguf"),
        out=Path("model.mixed.gguf"),
        payloads={"blk.0.ffn_down_exps.weight": (Path("down.q2_0"), 42)},
    )
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pre_encode][]: The stage that
      selects the replacements and verifies them after packing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, Final

from vramfit.adapters.outbound.gguf.header import (
    GgufHeader,
    TensorInfo,
    read_header,
    write_tensor_infos,
)
from vramfit.adapters.outbound.gguf.types import PackError

# One streamed copy chunk. The rewrite never holds more than this
# of tensor data at once, whatever the base's size.
_COPY_CHUNK: Final[int] = 16 << 20


def _pad(handle: BinaryIO, alignment: int) -> None:
    """Zero-pad the file to the next alignment boundary.

    Args:
        handle: The open output, at its current end.
        alignment: The data-section alignment.
    """
    remainder = handle.tell() % alignment
    if remainder:
        handle.write(bytes(alignment - remainder))


def _copy_span(source: BinaryIO, target: BinaryIO, start: int, end: int) -> None:
    """Stream ``source[start:end]`` into ``target``.

    Args:
        source: The open base file.
        target: The open output file.
        start: First byte to copy.
        end: Past-the-end byte.

    Raises:
        PackError: If the base ends before ``end``.
    """
    source.seek(start)
    remaining = end - start
    while remaining:
        chunk = source.read(min(_COPY_CHUNK, remaining))
        if not chunk:
            raise PackError(f"the base GGUF ends before byte {end}")
        target.write(chunk)
        remaining -= len(chunk)


def _copy_file(path: Path, target: BinaryIO) -> int:
    """Stream a payload file into ``target``.

    Args:
        path: The payload file.
        target: The open output file.

    Returns:
        The bytes copied.

    Raises:
        PackError: If the payload cannot be read.
    """
    copied = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(_COPY_CHUNK):
                target.write(chunk)
                copied += len(chunk)
    except OSError as exc:
        raise PackError(f"cannot read the pre-encoded payload {path}: {exc}") from exc
    return copied


def _new_infos(
    header: GgufHeader,
    spans: Mapping[str, tuple[int, int]],
    payloads: Mapping[str, tuple[Path, int]],
) -> tuple[TensorInfo, ...]:
    """Recompute every tensor's type id and data offset.

    Offsets follow the base's data order, so the rewrite streams the
    base forward once. A replaced tensor takes its payload's size,
    padded to the alignment.

    Args:
        header: The base's header.
        spans: The base's absolute data spans per tensor.
        payloads: ``(payload path, type id)`` per replaced tensor.

    Returns:
        The infos in header order, with new offsets.

    Raises:
        PackError: If a replaced tensor is not in the base, or its
            payload cannot be sized.
    """
    missing = sorted(set(payloads) - {info.name for info in header.tensors})
    if missing:
        raise PackError(
            f"the base GGUF carries no tensor for {len(missing)} pre-encoded "
            f"payload{'' if len(missing) == 1 else 's'}: {', '.join(missing)}"
        )
    offsets: dict[str, int] = {}
    cursor = 0
    for info in sorted(header.tensors, key=lambda info: info.offset):
        offsets[info.name] = cursor
        if info.name in payloads:
            path, _ = payloads[info.name]
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise PackError(
                    f"cannot size the pre-encoded payload {path}: {exc}"
                ) from exc
            cursor += -(-size // header.alignment) * header.alignment
        else:
            start, end = spans[info.name]
            cursor += end - start
    return tuple(
        replace(
            info,
            type_id=payloads[info.name][1] if info.name in payloads else info.type_id,
            offset=offsets[info.name],
        )
        for info in header.tensors
    )


def write_mixed_gguf(
    base: Path, out: Path, payloads: Mapping[str, tuple[Path, int]]
) -> int:
    """Write the base GGUF with the named tensors replaced.

    Args:
        base: The floating-point base GGUF. Never modified.
        out: The mixed GGUF to write. An existing file is replaced.
        payloads: ``(payload path, ggml type id)`` per tensor to
            replace. The payload holds the tensor's stored bytes in
            row-major order.

    Returns:
        The mixed file's size in bytes.

    Raises:
        PackError: If the base cannot be read, a replaced tensor is
            not in it, a payload cannot be read, or the output
            cannot be written.

    Examples:
        Replace one tensor and hand the result to the quantizer:

        ```python
        write_mixed_gguf(base, mixed, {"blk.0.ffn_down_exps.weight": (payload, 42)})
        ```
    """
    header = read_header(base)
    try:
        spans = header.spans(base.stat().st_size)
    except OSError as exc:
        raise PackError(f"cannot read the base GGUF {base}: {exc}") from exc
    infos = _new_infos(header, spans, payloads)
    try:
        with base.open("rb") as source, out.open("wb") as target:
            _copy_span(source, target, 0, header.infos_offset)
            write_tensor_infos(target, infos)
            _pad(target, header.alignment)
            for info in sorted(header.tensors, key=lambda info: info.offset):
                if info.name in payloads:
                    _copy_file(payloads[info.name][0], target)
                    _pad(target, header.alignment)
                else:
                    start, end = spans[info.name]
                    _copy_span(source, target, start, end)
            return target.tell()
    except OSError as exc:
        raise PackError(f"cannot write the mixed GGUF {out}: {exc}") from exc
