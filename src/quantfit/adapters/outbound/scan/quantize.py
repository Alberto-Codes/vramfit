"""Round-to-nearest quantize-dequantize, the v1 within-group method.

Symmetric integer quantization with per-block scales (ADR-0006). The
scan never keeps integer weights — it measures the damage of the
round trip, so the function returns dequantized values in the input's
dtype.

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


def rtn_quantize_dequantize(
    weight: torch.Tensor, bits: int, block_size: int = DEFAULT_BLOCK_SIZE
) -> torch.Tensor:
    """Quantize a tensor to ``bits`` and dequantize it back.

    The tensor is flattened, padded to a multiple of ``block_size``,
    and each block is scaled symmetrically to the signed integer range
    of ``bits``. Zero blocks pass through unchanged.

    Args:
        weight: The tensor to perturb. Any shape, any float dtype.
        bits: Target precision, at least 2.
        block_size: Elements sharing one scale.

    Returns:
        The dequantized tensor, same shape and dtype as the input.

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
    qmax = 2 ** (bits - 1) - 1
    flat = weight.detach().to(torch.float32).reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    scale = blocks.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(blocks / scale).clamp(-qmax - 1, qmax) * scale
    result = quantized.reshape(-1)
    if pad:
        result = result[: weight.numel()]
    return result.reshape(weight.shape).to(weight.dtype)
