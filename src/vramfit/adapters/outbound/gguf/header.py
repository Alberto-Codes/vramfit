"""One owned GGUF header parse for the pack path (ADR-0012, ADR-0032).

The pack path reads GGUF headers itself where gguf-py cannot serve
it. gguf-py's reader names every tensor type through its enum, and
the PyPI release can lag llama.cpp's type table: 0.19.0 carries no
``Q2_0``, which every expert-stack pack at nominal 2 holds
(ADR-0028) and which the preprocessor now writes (ADR-0032). This
parser walks the header once and records what its callers need: the
``general.file_type`` value and its position for the relabel
([vramfit.adapters.outbound.gguf.file_type][]), and each tensor's
name, shape, type id, and data offset for the mixed-GGUF rewrite and
the payload verification
([vramfit.adapters.outbound.gguf.pre_encode][]). It measures bytes
by offset and never needs a block-size table, so a type it cannot
name still counts.

The format is llama.cpp's ``gguf.h``: magic, version, tensor and
key-value counts, the key-value pairs, then one info per tensor. The
data section starts at the first ``general.alignment`` boundary
after the last info.

Examples:
    Read a packed file's composition:

    ```python
    header = read_header(Path("packed.gguf"))
    names = [tensor.name for tensor in header.tensors]
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.file_type][]: The relabel that
      reads `GgufHeader.file_type_offset`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from vramfit.adapters.outbound.gguf.types import PackError

FILE_TYPE_KEY: Final[str] = "general.file_type"
ALIGNMENT_KEY: Final[str] = "general.alignment"

_MAGIC: Final[bytes] = b"GGUF"
_VERSIONS: Final[frozenset[int]] = frozenset({2, 3})
_DEFAULT_ALIGNMENT: Final[int] = 32

# GGUF metadata value types (ggml/include/gguf.h). Scalars carry
# their struct format. String and array are the two variable widths.
_SCALAR_FORMATS: Final[dict[int, str]] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING_TYPE: Final[int] = 8
_ARRAY_TYPE: Final[int] = 9
_UINT32_TYPE: Final[int] = 4


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """One tensor's header entry.

    Attributes:
        name (str): The tensor name.
        dims (tuple[int, ...]): The shape as the file states it,
            ``ne[0]`` first, so ``dims[0]`` is the row width.
        type_id (int): The ggml type id.
        offset (int): The data offset, relative to the data section.

    Examples:
        The row width of an expert stack:

        ```python
        width = info.dims[0]
        ```
    """

    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def elements(self) -> int:
        """Count the tensor's elements.

        Returns:
            The product of the dims.
        """
        count = 1
        for dim in self.dims:
            count *= dim
        return count


@dataclass(frozen=True, slots=True)
class GgufHeader:
    """What one header parse learned about a GGUF.

    Attributes:
        version (int): The GGUF version, 2 or 3.
        alignment (int): The data-section alignment.
        file_type_offset (int | None): Absolute offset of the
            ``general.file_type`` uint32 value, or None when the file
            declares none.
        file_type (int): The ftype value the file declares, 0 when
            absent.
        infos_offset (int): Absolute offset of the first tensor info,
            right after the last key-value pair.
        data_start (int): Absolute offset of the data section.
        tensors (tuple[TensorInfo, ...]): Every tensor in header
            order.

    Examples:
        Locate one tensor's bytes:

        ```python
        start = header.data_start + info.offset
        ```
    """

    version: int
    alignment: int
    file_type_offset: int | None
    file_type: int
    infos_offset: int
    data_start: int
    tensors: tuple[TensorInfo, ...]

    def spans(self, file_size: int) -> dict[str, tuple[int, int]]:
        """Bound each tensor's data by the next tensor's offset.

        Args:
            file_size: The whole file's size in bytes.

        Returns:
            Absolute ``(start, end)`` per tensor name. Alignment
            padding counts toward the tensor before it, and the last
            tensor runs to the end of the file.

        Raises:
            PackError: If a tensor's offset lies past the data
                section.
        """
        data_bytes = file_size - self.data_start
        if data_bytes < 0:
            raise PackError(
                f"the data section would start past the end, at {self.data_start}"
            )
        ordered = sorted(self.tensors, key=lambda info: info.offset)
        spans: dict[str, tuple[int, int]] = {}
        for index, info in enumerate(ordered):
            end = ordered[index + 1].offset if index + 1 < len(ordered) else data_bytes
            if end < info.offset:
                raise PackError(
                    f"tensor data at offset {info.offset} runs past the file"
                )
            spans[info.name] = (self.data_start + info.offset, self.data_start + end)
        return spans


class _Parser:
    """A forward-only reader over the GGUF header.

    Attributes:
        handle (BinaryIO): The open file, positioned at the next
            unread byte.
        offset (int): Bytes consumed so far, so a metadata value's
            absolute position is known when it is read.

    Examples:
        Read the magic and the version:

        ```python
        parser = _Parser(handle)
        magic = parser.handle.read(4)
        version = parser.read("<I")
        ```
    """

    def __init__(self, handle: BinaryIO) -> None:
        """Start at the file's first byte.

        Args:
            handle: The open file, at offset 0.
        """
        self.handle = handle
        self.offset = 0

    def read(self, fmt: str) -> int | float | bool:
        """Read one packed scalar.

        Args:
            fmt: The struct format of the scalar.

        Returns:
            The unpacked value.

        Raises:
            PackError: If the file ends inside the value.
        """
        size = struct.calcsize(fmt)
        data = self.handle.read(size)
        if len(data) != size:
            raise PackError(f"header ends at byte {self.offset + len(data)}")
        self.offset += size
        return struct.unpack(fmt, data)[0]

    def read_string(self) -> str:
        """Read one length-prefixed string.

        Returns:
            The decoded string. Undecodable bytes become U+FFFD.

        Raises:
            PackError: If the file ends inside the string.
        """
        length = int(self.read("<Q"))
        data = self.handle.read(length)
        if len(data) != length:
            raise PackError(f"header ends at byte {self.offset + len(data)}")
        self.offset += length
        return data.decode("utf-8", errors="replace")

    def skip_value(self, value_type: int) -> None:
        """Consume one metadata value of the given type.

        Args:
            value_type: The GGUF value type id.

        Raises:
            PackError: If the type id is unknown, or the file ends
                inside the value.
        """
        if value_type == _STRING_TYPE:
            self.read_string()
        elif value_type == _ARRAY_TYPE:
            element_type = int(self.read("<I"))
            count = int(self.read("<Q"))
            for _ in range(count):
                self.skip_value(element_type)
        elif value_type in _SCALAR_FORMATS:
            self.read(_SCALAR_FORMATS[value_type])
        else:
            raise PackError(f"unknown metadata value type {value_type}")


def _parse(parser: _Parser) -> GgufHeader:
    """Walk the header once.

    Args:
        parser: A parser at the file's first byte.

    Returns:
        The header.

    Raises:
        PackError: If the file is not a little-endian GGUF v2 or v3,
            ends early, or declares ``general.file_type`` at a type
            other than uint32.
    """
    if parser.handle.read(4) != _MAGIC:
        raise PackError("no GGUF magic")
    parser.offset = 4
    version = int(parser.read("<I"))
    if version not in _VERSIONS:
        raise PackError(f"GGUF version {version} is not a little-endian v2 or v3")
    n_tensors = int(parser.read("<Q"))
    n_kv = int(parser.read("<Q"))
    alignment = _DEFAULT_ALIGNMENT
    file_type_offset: int | None = None
    file_type = 0
    for _ in range(n_kv):
        key = parser.read_string()
        value_type = int(parser.read("<I"))
        if key == FILE_TYPE_KEY:
            if value_type != _UINT32_TYPE:
                raise PackError(f"{FILE_TYPE_KEY} holds type {value_type}, not uint32")
            file_type_offset = parser.offset
            file_type = int(parser.read("<I"))
        elif key == ALIGNMENT_KEY and value_type == _UINT32_TYPE:
            alignment = int(parser.read("<I"))
        else:
            parser.skip_value(value_type)
    infos_offset = parser.offset
    infos: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = parser.read_string()
        n_dims = int(parser.read("<I"))
        dims = tuple(int(parser.read("<Q")) for _ in range(n_dims))
        type_id = int(parser.read("<I"))
        infos.append(TensorInfo(name, dims, type_id, int(parser.read("<Q"))))
    data_start = -(-parser.offset // alignment) * alignment
    return GgufHeader(
        version=version,
        alignment=alignment,
        file_type_offset=file_type_offset,
        file_type=file_type,
        infos_offset=infos_offset,
        data_start=data_start,
        tensors=tuple(infos),
    )


def read_header(path: Path) -> GgufHeader:
    """Parse a GGUF's header.

    Args:
        path: The GGUF file.

    Returns:
        The header.

    Raises:
        PackError: If the file cannot be opened, is not a
            little-endian GGUF v2 or v3, or is truncated inside the
            header.

    Examples:
        Read the tensor names the file declares:

        ```python
        header = read_header(Path("model-f16.gguf"))
        assert header.tensors[0].name == "token_embd.weight"
        ```
    """
    try:
        with path.open("rb") as handle:
            return _parse(_Parser(handle))
    except OSError as exc:
        raise PackError(f"cannot read the GGUF {path}: {exc}") from exc
    except PackError as exc:
        raise PackError(f"cannot read the GGUF {path}: {exc}") from exc


def write_tensor_infos(handle: BinaryIO, tensors: tuple[TensorInfo, ...]) -> None:
    """Write tensor infos in the format `_parse` reads.

    Args:
        handle: The open file, positioned where the infos go.
        tensors: The infos to write, in order.
    """
    for info in tensors:
        name = info.name.encode("utf-8")
        handle.write(struct.pack("<Q", len(name)) + name)
        handle.write(struct.pack("<I", len(info.dims)))
        handle.writelines(struct.pack("<Q", dim) for dim in info.dims)
        handle.write(struct.pack("<I", info.type_id))
        handle.write(struct.pack("<Q", info.offset))
