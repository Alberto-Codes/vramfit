"""Torch-free ``block_q2_0`` decoding for the pack path (ADR-0032).

The pack path reads the ``Q2_0`` blocks the preprocessor wrote and
the blocks the packed file carries, and it imports no torch
(ADR-0005, ADR-0008). gguf-py 0.19.0 names no ``Q2_0`` in its type
table, so its dequantizer cannot serve either. This module owns the
block arithmetic instead: the stored layout is a little-endian fp16
scale followed by 16 bytes of two-bit codes, element ``j`` at bits
``2 * (j % 4)`` of byte ``j // 4``, and the level is ``code - 1``.
That is stock llama.cpp's ``dequantize_row_q2_0``.

Examples:
    Decode one row of 64 elements:

    ```python
    values = dequantize_q2_0(payload, elements=64)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.q2_0_assisted][]: The encoder
      whose blocks this decodes.
    - [vramfit.adapters.outbound.gguf.pre_encode][]: The pack-side
      verification that reads these blocks back.
"""

from __future__ import annotations

from typing import Any, Final

from vramfit.adapters.outbound.gguf.types import PackError

# ggml-common.h: QK2_0 groups 64 elements, and a block holds an fp16
# scale plus one byte per four elements.
QK2_0: Final[int] = 64
Q2_0_BLOCK_BYTES: Final[int] = 2 + QK2_0 // 4
# The ggml type id the GGUF header stores for Q2_0 (ggml.h, since
# 2026-07-07). The file-type table in `file_type` names it too.
Q2_0_TYPE_ID: Final[int] = 42
_CODES_PER_BYTE: Final[int] = 4
_CODE_MASK: Final[int] = 0x03


def q2_0_payload_bytes(elements: int) -> int:
    """Size the ``Q2_0`` payload for an element count.

    Args:
        elements: The tensor's element count.

    Returns:
        The stored byte count.

    Raises:
        PackError: If the count does not divide into blocks. A
            ``Q2_0`` row must hold whole blocks, and the quantizer
            refuses such a tensor by type fallback (ADR-0028).

    Examples:
        One block per 64 elements, 18 bytes each:

        ```python
        assert q2_0_payload_bytes(128) == 36
        ```
    """
    if elements <= 0 or elements % QK2_0:
        raise PackError(
            f"{elements} elements do not divide into {QK2_0}-element Q2_0 blocks"
        )
    return elements // QK2_0 * Q2_0_BLOCK_BYTES


def _load_numpy() -> Any:
    """Import numpy on first use, naming the gguf extra when absent.

    Returns:
        The imported ``numpy`` module.

    Raises:
        PackError: If numpy is not installed.
    """
    try:
        import numpy  # noqa: PLC0415 - lazy: base install has no numpy (ADR-0005)
    except ImportError as exc:
        raise PackError(
            "decoding Q2_0 blocks needs numpy — install the gguf extra: "
            "uv sync --extra gguf (ADR-0005)"
        ) from exc
    return numpy


def dequantize_q2_0(payload: bytes, elements: int) -> Any:
    r"""Decode stored ``Q2_0`` blocks to float32 values.

    Args:
        payload: The blocks, ``Q2_0_BLOCK_BYTES`` per 64 elements.
        elements: The element count the blocks hold.

    Returns:
        A float32 numpy array of ``elements`` values, in stored
        order.

    Raises:
        PackError: If the count does not divide into blocks, the
            payload's size disagrees with the count, or numpy is
            missing.

    Examples:
        A zero scale decodes to zeros whatever the codes:

        ```python
        values = dequantize_q2_0(b"\x00\x00" + b"\x55" * 16, 64)
        assert not values.any()
        ```
    """
    expected = q2_0_payload_bytes(elements)
    if len(payload) != expected:
        raise PackError(
            f"a Q2_0 payload for {elements} elements holds {expected} bytes, "
            f"got {len(payload)}"
        )
    np = _load_numpy()
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, Q2_0_BLOCK_BYTES)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    codes = blocks[:, 2:]
    shifts = np.arange(_CODES_PER_BYTE, dtype=np.uint8) * 2
    levels = ((codes[:, :, None] >> shifts) & _CODE_MASK).astype(np.float32) - 1.0
    return (levels.reshape(-1, QK2_0) * scales).reshape(-1)
