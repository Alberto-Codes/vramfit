"""Declare the packed GGUF's file type as the modal tensor type by bytes.

``general.file_type`` is one enum for one uniform quantization. A
mixed-precision pack has no true value in it. ``llama-quantize``
stamps the positional base ftype, which is the recipe's floor
(ADR-0012 decision 3). On the published 30B pack that stamped
``Q2_K`` over a file holding no ``Q2_K`` tensor (#413). The
2026-09-04 amendment to decision 3 rules the value instead: the
type covering the most bytes in the packed file, written after the
quantizer runs (#414). On that pack the value is ``Q4_0``, at
74.3 % of the bytes.

The read is a header parse this module owns. gguf-py's reader
names every tensor type through its enum, and the PyPI release
can lag llama.cpp's type table — 0.19.0 carries no ``Q2_0``, which
every expert-stack pack at nominal 2 holds (ADR-0028). A reader
that refuses the file it must relabel is no reader. The parse walks
the metadata once, records where the ``general.file_type`` value
sits, and measures each tensor's bytes from the data-section
offsets. Measuring by offset needs no block-size table, so a type
the parser cannot name still counts its bytes. Naming it is the
one table this module keeps, and an unnamed type refuses rather
than stamping a label the project cannot state.

The write changes four bytes in place. The value's width and
position do not move, so the rest of the file is untouched.

Examples:
    Relabel a packed file after the quantizer wrote it:

    ```python
    from vramfit.adapters.outbound.gguf.file_type import stamp_modal_file_type

    declared = stamp_modal_file_type(Path("packed.gguf"))
    assert declared == "Q4_0"
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: The caller, right
      after the quantizer's zero exit.
    - [vramfit.domain.pack][]: `modal_type`, the pure rule.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import modal_type

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

# Tensor type ids to names (ggml/include/ggml.h, `ggml_type`). The
# rows this backend drives plus the plain block types beside them.
# ``Q2_0`` is 42 upstream since 2026-07-07, and PyPI gguf-py 0.19.0
# does not carry it.
TENSOR_TYPE_NAMES: Final[dict[int, str]] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    30: "BF16",
    41: "Q1_0",
    42: "Q2_0",
}

# Tensor type to the ftype that names it (include/llama.h,
# `llama_ftype`). A k-quant maps to its ``_S`` ftype, the same choice
# as `BASE_FTYPE_BY_BITS`: ``Q3_K`` and ``Q4_K`` are not ftypes.
# ``Q2_0`` is 41 upstream (``LLAMA_FTYPE_MOSTLY_Q2_0``).
FTYPE_BY_TENSOR_TYPE: Final[dict[str, tuple[str, int]]] = {
    "F32": ("F32", 0),
    "F16": ("F16", 1),
    "Q4_0": ("Q4_0", 2),
    "Q4_1": ("Q4_1", 3),
    "Q8_0": ("Q8_0", 7),
    "Q5_0": ("Q5_0", 8),
    "Q5_1": ("Q5_1", 9),
    "Q2_K": ("Q2_K", 10),
    "Q3_K": ("Q3_K_S", 11),
    "Q4_K": ("Q4_K_S", 14),
    "Q5_K": ("Q5_K_S", 16),
    "Q6_K": ("Q6_K", 18),
    "BF16": ("BF16", 32),
    "Q1_0": ("Q1_0", 40),
    "Q2_0": ("Q2_0", 41),
}


@dataclass(frozen=True, slots=True)
class PackedLayout:
    """What one header parse learned about a packed GGUF.

    Attributes:
        bytes_by_type (dict[str, int]): Data-section bytes each
            tensor type covers, keyed by type name. Alignment
            padding counts toward the tensor before it.
        file_type_offset (int): Absolute offset of the
            ``general.file_type`` uint32 value.
        file_type (int): The ftype value the file declares now.

    Examples:
        Read the composition the label must describe:

        ```python
        layout = read_layout(Path("packed.gguf"))
        share = layout.bytes_by_type["Q4_0"] / sum(layout.bytes_by_type.values())
        ```
    """

    bytes_by_type: dict[str, int]
    file_type_offset: int
    file_type: int


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


def _parse(parser: _Parser, file_size: int) -> PackedLayout:
    """Walk the header once and measure the data section by offsets.

    Args:
        parser: A parser at the file's first byte.
        file_size: The whole file's size in bytes.

    Returns:
        The layout the header describes.

    Raises:
        PackError: If the file is not a little-endian GGUF v2 or v3,
            ends early, declares no ``general.file_type`` or one that
            is not a uint32, or holds a tensor type the table cannot
            name.
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
    if file_type_offset is None:
        raise PackError(f"the file declares no {FILE_TYPE_KEY}")
    infos: list[tuple[int, int]] = []
    for _ in range(n_tensors):
        parser.read_string()
        n_dims = int(parser.read("<I"))
        for _ in range(n_dims):
            parser.read("<Q")
        type_id = int(parser.read("<I"))
        infos.append((int(parser.read("<Q")), type_id))
    data_start = -(-parser.offset // alignment) * alignment
    data_bytes = file_size - data_start
    if data_bytes < 0:
        raise PackError(f"the data section would start past the end, at {data_start}")
    return PackedLayout(_bytes_by_type(infos, data_bytes), file_type_offset, file_type)


def _bytes_by_type(infos: list[tuple[int, int]], data_bytes: int) -> dict[str, int]:
    """Sum each type's bytes from consecutive data offsets.

    Args:
        infos: ``(data offset, type id)`` per tensor, in any order.
        data_bytes: The data section's size, which bounds the last
            tensor.

    Returns:
        Bytes per type name. A file holding no tensors returns an
        empty table.

    Raises:
        PackError: If a tensor's offset lies past the data section,
            or its type id has no name in the table.
    """
    totals: dict[str, int] = {}
    ordered = sorted(infos)
    for index, (offset, type_id) in enumerate(ordered):
        end = ordered[index + 1][0] if index + 1 < len(ordered) else data_bytes
        if end < offset:
            raise PackError(f"tensor data at offset {offset} runs past the file")
        try:
            name = TENSOR_TYPE_NAMES[type_id]
        except KeyError:
            raise PackError(
                f"tensor type id {type_id} has no name in the file-type table"
            ) from None
        totals[name] = totals.get(name, 0) + (end - offset)
    return totals


def read_layout(path: Path) -> PackedLayout:
    """Parse a packed GGUF's header and measure its tensor bytes.

    Args:
        path: The packed GGUF.

    Returns:
        The layout: bytes per type, and where the file type sits.

    Raises:
        PackError: If the file cannot be opened, is not a
            little-endian GGUF v2 or v3, is truncated, declares no
            ``general.file_type``, or holds a tensor type the table
            cannot name.

    Examples:
        Check the file the quantizer wrote:

        ```python
        layout = read_layout(Path("packed.gguf"))
        assert "Q4_0" in layout.bytes_by_type
        ```
    """
    try:
        with path.open("rb") as handle:
            file_size = path.stat().st_size
            return _parse(_Parser(handle), file_size)
    except OSError as exc:
        raise PackError(f"cannot read the packed GGUF {path}: {exc}") from exc
    except PackError as exc:
        raise PackError(f"cannot read the packed GGUF {path}: {exc}") from exc


def write_file_type(path: Path, offset: int, value: int) -> None:
    """Overwrite the ``general.file_type`` value in place.

    Args:
        path: The packed GGUF.
        offset: Absolute offset of the uint32 value, from
            `read_layout`.
        value: The ftype value to write.

    Raises:
        PackError: If the file cannot be written.
    """
    try:
        with path.open("r+b") as handle:
            handle.seek(offset)
            handle.write(struct.pack("<I", value))
    except OSError as exc:
        raise PackError(f"cannot write {FILE_TYPE_KEY} into {path}: {exc}") from exc


def declared_file_type(bytes_by_type: dict[str, int]) -> tuple[str, int]:
    """Name the ftype a packed file declares: its modal type by bytes.

    Args:
        bytes_by_type: Data-section bytes per tensor type name.

    Returns:
        The ftype name and value, e.g. ``("Q4_0", 2)``.

    Raises:
        PackError: If the table is empty, or the modal type has no
            ftype.

    Examples:
        The published 30B pack's composition (#413):

        ```python
        table = {"Q4_0": 743, "Q8_0": 138, "Q2_0": 117, "F32": 2}
        assert declared_file_type(table) == ("Q4_0", 2)
        ```
    """
    try:
        modal = modal_type(bytes_by_type)
    except ValueError as exc:
        raise PackError(f"cannot pick a file type: {exc}") from exc
    try:
        return FTYPE_BY_TENSOR_TYPE[modal]
    except KeyError:
        raise PackError(
            f"the modal tensor type {modal} has no ftype in the file-type table"
        ) from None


def stamp_modal_file_type(path: Path) -> str:
    """Declare the packed file's modal type by bytes as its file type.

    ADR-0012 decision 3 as amended 2026-09-04: the quantizer stamps
    the base ftype, and this replaces it with the type covering the
    most bytes. The write happens after the quantizer exits 0.

    Args:
        path: The packed GGUF the quantizer wrote.

    Returns:
        The ftype name written, e.g. ``Q4_0``.

    Raises:
        PackError: If the file cannot be read or written, or its
            composition maps to no ftype.

    Examples:
        The pack step records the returned name in `PackResult`:

        ```python
        declared = stamp_modal_file_type(out_path)
        ```
    """
    layout = read_layout(path)
    name, value = declared_file_type(layout.bytes_by_type)
    if value != layout.file_type:
        write_file_type(path, layout.file_type_offset, value)
    return name
