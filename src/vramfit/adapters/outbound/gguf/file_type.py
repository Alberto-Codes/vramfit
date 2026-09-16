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

The read is the owned header parse in
[vramfit.adapters.outbound.gguf.header][]. gguf-py's reader
names every tensor type through its enum, and the PyPI release
can lag llama.cpp's type table — 0.19.0 carries no ``Q2_0``, which
every expert-stack pack at nominal 2 holds (ADR-0028). A reader
that refuses the file it must relabel is no reader. The parse walks
the metadata once, records where the ``general.file_type`` value
sits, and this module measures each tensor's bytes from the
data-section offsets. Measuring by offset needs no block-size
table, so a type the parser cannot name still counts its bytes.
Naming it is the one table this module keeps, and an unnamed type
refuses rather than stamping a label the project cannot state.

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
from typing import Final

from vramfit.adapters.outbound.gguf.header import (
    FILE_TYPE_KEY as FILE_TYPE_KEY,  # noqa: PLC0414 - re-export: the relabel's key reads from this module
)
from vramfit.adapters.outbound.gguf.header import GgufHeader, read_header
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import modal_type

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


def _layout(header: GgufHeader, file_size: int) -> PackedLayout:
    """Measure the data section by offsets.

    Args:
        header: The parsed header.
        file_size: The whole file's size in bytes.

    Returns:
        The layout the header describes.

    Raises:
        PackError: If the file declares no ``general.file_type``, a
            tensor's offset lies past the data section, or a tensor
            type id has no name in the table.
    """
    if header.file_type_offset is None:
        raise PackError(f"the file declares no {FILE_TYPE_KEY}")
    totals: dict[str, int] = {}
    spans = header.spans(file_size)
    for info in header.tensors:
        try:
            name = TENSOR_TYPE_NAMES[info.type_id]
        except KeyError:
            raise PackError(
                f"tensor type id {info.type_id} has no name in the file-type table"
            ) from None
        start, end = spans[info.name]
        totals[name] = totals.get(name, 0) + (end - start)
    return PackedLayout(totals, header.file_type_offset, header.file_type)


def read_layout(path: Path) -> PackedLayout:
    """Parse a packed GGUF's header and measure its tensor bytes.

    The parse is `header.read_header`, and this measures the data
    section it describes.

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
        return _layout(read_header(path), path.stat().st_size)
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
