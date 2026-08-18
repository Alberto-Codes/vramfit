"""Block-quantizer quantize-dequantize (ADR-0018, `gguf-ref`).

Torch ports of llama.cpp's reference quantizers for ``Q2_0`` and
``Q4_0`` (``ggml-quants.c``, checkout 3653e6d6d, release b10326).
`Q8_0` reuses the ADR-0018 port in
[vramfit.adapters.outbound.scan.kquant][] — the two checkouts hold
byte-identical ``ggml-quants.c`` and ``ggml-common.h``, so one port
serves both. Like the other methods, the functions return
dequantized values.

The pack applies these types where no K-quant reaches. ADR-0028 maps
nominal 8, 4, and 2 onto ``Q8_0``, ``Q4_0``, and ``Q2_0`` on the
routed-expert stacks, whose rows of 2688 and 1856 refuse every
256-element super-block type. Nominal 3 has no type and this method
refuses it.

The two types round differently and the port keeps them apart.
``Q2_0`` calls ``roundf``, which rounds half away from zero.
``Q4_0`` truncates through ``(int8_t)(x*id + 8.5f)``, which rounds
half up. Both quantize against the pre-fp16 scale and dequantize
with the fp16-stored one.

``Q2_0`` reaches three levels, not four. The reference clamps
``round(w/amax)`` to ``[-1, 2]``, and ``|w| <= amax`` caps it at 1.

Examples:
    Simulate Q2_0 damage on one weight matrix:

    ```python
    perturbed = gguf_ref_quantize_dequantize(weight, bits=2)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.kquant][]: The K-quant port
      (ADR-0018) and the shared `Q8_0` round trip.
    - [vramfit.adapters.outbound.scan.quantize][]: The RTN v1 method.
"""

from __future__ import annotations

import torch

from vramfit.adapters.outbound.scan.kquant import (
    _fp16,
    _q8_0_round_trip,
    _sliced,
)

# ggml-common.h: QK2_0 groups 64 elements, QK4_0 groups 32.
QK2_0 = 64
QK4_0 = 32


def _round_half_away(v: torch.Tensor) -> torch.Tensor:
    """Round half away from zero, like C's ``roundf``.

    ``torch.round`` rounds half to even, so it cannot stand in here.

    Args:
        v: Float32 values.

    Returns:
        The rounded values, still float32.
    """
    return torch.trunc(v + 0.5 * torch.sign(v))


def _safe_inverse(d: torch.Tensor) -> torch.Tensor:
    """Build the reciprocal scale, zeroed where the C's would not apply.

    A subnormal scale overflows the reciprocal to infinity. The C
    then feeds infinity through an int cast, and the fp16-stored
    scale has already underflowed to zero, so every dequantized
    element reads zero whatever the cast produced. Zeroing the
    reciprocal reproduces that without relying on undefined behavior.

    Args:
        d: Per-block scales, shape ``(n, 1)``.

    Returns:
        The reciprocal where it is finite, zero elsewhere.
    """
    safe = torch.where(d != 0, d, torch.ones_like(d))
    raw = 1.0 / safe
    return torch.where((d != 0) & torch.isfinite(raw), raw, torch.zeros_like(d))


def _q2_0_round_trip(blocks: torch.Tensor) -> torch.Tensor:
    """Round-trip 64-element blocks through Q2_0.

    ``quantize_row_q2_0_ref`` sets the scale to the block absmax and
    stores ``roundf(w/d) + 1`` clamped to ``[0, 3]``.
    ``dequantize_row_q2_0`` returns ``(q - 1) * fp16(d)``, so the
    level lives in ``[-1, 2]``.

    Args:
        blocks: Shape ``(n, QK2_0)``, float32.

    Returns:
        Dequantized values, same shape.
    """
    d = blocks.abs().amax(dim=1, keepdim=True)
    levels = _round_half_away(blocks * _safe_inverse(d)).clamp(-1.0, 2.0)
    return levels * _fp16(d)


def _q4_0_round_trip(blocks: torch.Tensor) -> torch.Tensor:
    """Round-trip 32-element blocks through Q4_0.

    ``quantize_row_q4_0_ref`` takes the signed value at the first
    absmax position, sets ``d`` to ``max / -8``, and stores
    ``MIN(15, (int8_t)(x/d + 8.5f))``. The cast truncates toward
    zero. ``dequantize_row_q4_0`` returns ``(q - 8) * fp16(d)``.

    The C loop keeps the *first* strict maximum, so a block holding
    ``+a`` and ``-a`` takes the sign of whichever comes first. An
    index reduction reproduces that; a plain ``argmax`` does not
    promise it.

    Args:
        blocks: Shape ``(n, QK4_0)``, float32.

    Returns:
        Dequantized values, same shape.
    """
    magnitude = blocks.abs()
    amax = magnitude.amax(dim=1, keepdim=True)
    positions = torch.arange(blocks.shape[1], device=blocks.device)
    beyond = torch.full_like(positions, blocks.shape[1])
    first = torch.where(magnitude == amax, positions, beyond).amin(dim=1, keepdim=True)
    signed_max = blocks.gather(1, first)

    d = signed_max / -8.0
    # No lower clamp: the C applies MIN(15, .) alone, and x/d stays
    # inside [-8, 8] because |x| <= amax.
    levels = torch.trunc(blocks * _safe_inverse(d) + 8.5).clamp(max=15.0) - 8.0
    return levels * _fp16(d)


_ROUND_TRIPS = {
    2: (_q2_0_round_trip, QK2_0),
    4: (_q4_0_round_trip, QK4_0),
    8: (_q8_0_round_trip, QK4_0),
}
# The ported coverage, derived from the dispatch table so the two
# cannot drift. domain.scan.GGUF_REF_PRECISIONS must mirror this —
# the CLI validates against the domain copy before a model loads.
GGUF_REF_BITS = tuple(sorted(_ROUND_TRIPS))

# The GGUF type each precision maps onto (ADR-0028 decision 1), for
# error messages and for the block-size refusal in kquant.
GGUF_TYPE_NAMES = {2: "Q2_0", 4: "Q4_0", 8: "Q8_0"}


def gguf_ref_quantize_dequantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize a tensor through the GGUF block type for ``bits``.

    The tensor is flattened in row-major order — the order
    ``llama-quantize`` consumes rows — padded with zeros to the
    format's block multiple, and round-tripped through the ported
    reference quantizer. Every block size here divides the
    routed-expert rows of 2688 and 1856, so no block straddles two
    rows on this method's motivating target. The input is never
    modified. The computation runs on the input's device first and
    retries on the CPU when the float32 workspace does not fit the
    card. Large tensors fit in bounded slices of whole blocks.

    Args:
        weight: The tensor to perturb. Any shape, any float dtype.
        bits: Nominal precision — 2 (``Q2_0``), 4 (``Q4_0``), or
            8 (``Q8_0``).

    Returns:
        The dequantized tensor, same shape, dtype, and device as the
        input.

    Raises:
        ValueError: If ``bits`` has no port. ADR-0028 refuses
            nominal 3 at pack, and 5 and 6 wait for ports.

    Examples:
        Q2_0 keeps at most three levels per 64-element block:

        ```python
        import torch

        w = torch.randn(4, 128)
        q = gguf_ref_quantize_dequantize(w, 2)
        assert all(len(block.unique()) <= 3 for block in q.reshape(-1, 64))
        ```
    """
    if bits not in _ROUND_TRIPS:
        raise ValueError(
            f"gguf supports bits in {GGUF_REF_BITS}, got {bits} — ADR-0028 "
            "refuses nominal 3 at pack, and 5 and 6 have no port (ADR-0018)"
        )
    round_trip, block = _ROUND_TRIPS[bits]

    def prepare(device: torch.device | str) -> torch.Tensor:
        """Flatten to padded float32 blocks on ``device``.

        Args:
            device: Where the workspace copy lives.

        Returns:
            Blocks, shape ``(n, block)``.
        """
        flat = weight.detach().to(device=device, dtype=torch.float32).reshape(-1)
        pad = (-flat.numel()) % block
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        return flat.reshape(-1, block)

    try:
        result = _sliced(round_trip, prepare(weight.device))
    except torch.cuda.OutOfMemoryError:
        result = _sliced(round_trip, prepare("cpu"))
    result = result.reshape(-1)[: weight.numel()]
    return result.reshape(weight.shape).to(device=weight.device, dtype=weight.dtype)
