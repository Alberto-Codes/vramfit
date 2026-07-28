"""Round-to-nearest quantize-dequantize, the v1 within-group method.

Symmetric integer quantization with per-block scales (ADR-0006). The
scan never keeps integer weights — it measures the damage of the
round trip, so the function returns dequantized values in the input's
dtype, on the input's device, with a CPU fallback when the float32
workspace does not fit the card.

Examples:
    Simulate 4-bit damage on one weight matrix:

    ```python
    perturbed = rtn_quantize_dequantize(weight, bits=4)
    ```

See Also:
    - [quantfit.adapters.outbound.scan.meter][]: Applies this to whole
      layer groups.
"""

from __future__ import annotations

import torch

DEFAULT_BLOCK_SIZE = 32
MIN_BITS = 2


def _round_trip(flat: torch.Tensor, bits: int, block_size: int) -> torch.Tensor:
    """Quantize and dequantize a flat float32 tensor per block.

    Args:
        flat: Flattened float32 values, length a multiple of
            ``block_size``.
        bits: Target precision.
        block_size: Elements sharing one scale.

    Returns:
        The dequantized values, same shape as ``flat``.
    """
    qmax = 2 ** (bits - 1) - 1
    blocks = flat.reshape(-1, block_size)
    scale = blocks.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(blocks / scale).clamp(-qmax - 1, qmax) * scale
    return quantized.reshape(-1)


def rtn_quantize_dequantize(
    weight: torch.Tensor, bits: int, block_size: int = DEFAULT_BLOCK_SIZE
) -> torch.Tensor:
    """Quantize a tensor to ``bits`` and dequantize it back.

    The tensor is flattened, padded to a multiple of ``block_size``,
    and each block is scaled symmetrically to the signed integer range
    of ``bits``. Zero blocks pass through unchanged. The input is
    never modified.

    The round trip computes on the input's device first — a same-size
    GPU tensor round-trips in ~15 ms where the CPU takes ~1 s. The
    float32 temporaries cost ~4x the tensor's bytes, so on a CUDA
    out-of-memory the computation retries on the CPU (a 2 GiB
    embedding table needs >8 GiB of workspace, which a card packed
    with model shards does not have).

    Args:
        weight: The tensor to perturb. Any shape, any float dtype.
        bits: Target precision, at least 2.
        block_size: Elements sharing one scale.

    Returns:
        The dequantized tensor, same shape, dtype, and device as the
        input.

    Raises:
        ValueError: If ``bits`` is below 2 or ``block_size`` is not
            positive.

    Examples:
        8-bit round-trips are near-lossless:

        ```python
        import torch

        w = torch.randn(16, 16)
        assert torch.allclose(w, rtn_quantize_dequantize(w, 8), atol=0.05)
        ```
    """
    if bits < MIN_BITS:
        raise ValueError(f"bits must be at least {MIN_BITS}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    def prepare(device: torch.device | str) -> torch.Tensor:
        """Flatten to padded float32 on ``device`` without touching the input.

        Args:
            device: Where the workspace copy lives.

        Returns:
            The padded flat float32 copy.
        """
        flat = weight.detach().to(device=device, dtype=torch.float32).reshape(-1)
        pad = (-flat.numel()) % block_size
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        return flat

    try:
        result = _round_trip(prepare(weight.device), bits, block_size)
    except torch.cuda.OutOfMemoryError:
        result = _round_trip(prepare("cpu"), bits, block_size)
    result = result[: weight.numel()]
    return result.reshape(weight.shape).to(device=weight.device, dtype=weight.dtype)
