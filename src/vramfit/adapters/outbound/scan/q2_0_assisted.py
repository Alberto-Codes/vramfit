"""The assisted ``Q2_0`` encoder: one fit for scan and pack (ADR-0032).

ADR-0032 decision 2 owns the numerical fit once. This module holds
it. The scan meter prices a covered nominal-2 cell through
`q2_0_assisted_quantize_dequantize`, and the preprocessor program
([vramfit.adapters.outbound.scan.q2_0_program][]) writes the blocks
`q2_0_encode_rows` returns into the temporary mixed GGUF. Both read
`q2_0_assisted_fit`, so the meter measures the dequantization of the
stored blocks, fp16 scales included.

The fit reimplements the reference specification recorded in
[issue #461](https://github.com/Alberto-Codes/vramfit/issues/461#issuecomment-5486773582).
It carries no third-party patch. Per 64-element block:

1. Seed the signed scale three ways: ``amax``, ``0.5 * amax``, and
   ``-0.5 * amax``.
2. Run four Lloyd iterations per seed. Each codes every element as
   ``clamp(roundf(x / d), -1, 2)``, then refits ``d`` as the weighted
   least-squares scale ``sum(w x q) / sum(w q q)``. An iteration
   keeps ``d`` when that denominator is zero, and the loop stops
   once ``d`` reaches zero.
3. Round ``d`` through fp16, recode the block against the stored
   scale, and score the weighted squared error inference sees.
4. Keep the first seed with the strictly smallest error.

The element weight is ``qw[j] * sqrt(sigma2 + x[j]^2)``, where
``qw`` is the imatrix column weight and ``sigma2`` is the row's mean
squared value, the formula stock llama.cpp applies to assisted
``Q4_0`` and the K-quants. Without weights every element weighs 1.
That unweighted search is the specification's weight-1.0 path. No
pack ships it: an uncovered or excluded ``Q2_0`` tensor packs
through stock ``llama-quantize``, which the reference round trip in
[vramfit.adapters.outbound.scan.q0_ref][] prices (decision 3).

``roundf`` rounds half away from zero, and ``torch.round`` rounds
half to even, so the coder uses `kquant.round_half_away`. The
reference sums in float32 sequentially and this port sums
vectorized, so a knife-edge candidate tie may resolve differently.
The fixtures in ``tests/data/q2_0_assisted/`` bound that drift.

Examples:
    Price one covered tensor the way the pack encodes it:

    ```python
    perturbed = q2_0_assisted_quantize_dequantize(weight, column_weights)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.q0_assisted][]: The assisted
      ``Q4_0`` port, which shares the row driver.
    - [vramfit.adapters.outbound.gguf.q2_0_blocks][]: The torch-free
      decoder for the blocks this module emits.
"""

from __future__ import annotations

from typing import Final

import torch

from vramfit.adapters.outbound.scan.kquant import _fp16, round_half_away
from vramfit.adapters.outbound.scan.q0_assisted import (
    check_q0_weights,
    run_assisted_rows,
)
from vramfit.adapters.outbound.scan.q0_ref import QK2_0

# Names the fit that produced a pre-encoded tensor. The packed
# result records it (ADR-0032 decision 3). Bump it when the fit's
# output changes for the same input.
Q2_0_ENCODER_REVISION: Final[str] = "vramfit-q2_0-assisted-1"

# One block stores an fp16 scale and 64 two-bit codes.
Q2_0_BLOCK_BYTES: Final[int] = 2 + QK2_0 // 4

# The three signed scale seeds, as multiples of the block absmax.
_SEEDS: Final[tuple[float, ...]] = (1.0, 0.5, -0.5)
_LLOYD_ITERATIONS: Final[int] = 4
_LEVEL_MIN: Final[float] = -1.0
_LEVEL_MAX: Final[float] = 2.0
# Codes per packed byte: four two-bit fields.
_CODES_PER_BYTE: Final[int] = 4


def _code(blocks: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """Code every element against a per-block reciprocal scale.

    Args:
        blocks: Shape ``(n, QK2_0)``, float32.
        inverse: Reciprocal scales, shape ``(n,)``. Zero codes a
            block to level 0 everywhere.

    Returns:
        Levels in ``[-1, 2]``, shape of ``blocks``, float32.
    """
    return round_half_away(blocks * inverse[:, None]).clamp(_LEVEL_MIN, _LEVEL_MAX)


def _safe_inverse(scale: torch.Tensor) -> torch.Tensor:
    """Build ``1 / scale`` where the scale is nonzero, else zero.

    Args:
        scale: Per-block scales, shape ``(n,)``.

    Returns:
        The reciprocal, zero where the scale is zero.
    """
    live = scale != 0
    return torch.where(
        live, 1.0 / torch.where(live, scale, torch.ones_like(scale)), 0.0
    )


def q2_0_fit_weights(rows: torch.Tensor, quant_weights: torch.Tensor) -> torch.Tensor:
    """Build the per-element fit weights for whole rows.

    The weight is ``qw[j] * sqrt(sigma2 + x[j]^2)`` with ``sigma2``
    the row's mean squared value, computed over the whole row.

    Args:
        rows: Shape ``(n, row)``, float32.
        quant_weights: Imatrix column weights, same shape.

    Returns:
        The weights, same shape.
    """
    sigma2 = (rows * rows).sum(dim=1, keepdim=True) / rows.shape[1]
    return quant_weights * torch.sqrt(sigma2 + rows * rows)


def q2_0_assisted_fit(
    rows: torch.Tensor, quant_weights: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the assisted ``Q2_0`` scale and codes for whole rows.

    This is the one fit ADR-0032 decision 2 names. Every row must
    divide into ``QK2_0`` blocks, because a block never straddles two
    rows and the row's ``sigma2`` shapes every weight in it.

    Args:
        rows: Shape ``(n, row)``, float32, with ``row`` a multiple of
            ``QK2_0``.
        quant_weights: Imatrix column weights, same shape, or None
            for the unweighted search.

    Returns:
        ``(scales, levels)``: the fp16-exact signed scale per block,
        shape ``(n * row / QK2_0,)``, and the levels in ``[-1, 2]``
        per element, shape ``(blocks, QK2_0)``, both float32.

    Raises:
        ValueError: If the row length does not divide into blocks.

    Examples:
        A block of ``-2`` and ones fits exactly with a negative scale:

        ```python
        import torch

        row = torch.full((1, 64), 1.0)
        row[0, 0] = -2.0
        scales, levels = q2_0_assisted_fit(row, None)
        assert float(scales[0]) == -1.0
        ```
    """
    row = int(rows.shape[1])
    if row % QK2_0:
        raise ValueError(
            f"rows of {row} do not divide into {QK2_0}-element Q2_0 blocks"
        )
    blocks = rows.reshape(-1, QK2_0)
    weights = (
        torch.ones_like(blocks)
        if quant_weights is None
        else q2_0_fit_weights(rows, quant_weights).reshape(-1, QK2_0)
    )
    amax = blocks.abs().amax(dim=1)
    ones = torch.ones_like(amax)
    best_error = torch.full_like(amax, float("inf"))
    best_scale = torch.zeros_like(amax)
    best_levels = torch.zeros_like(blocks)
    for seed in _SEEDS:
        scale = amax * seed
        for _ in range(_LLOYD_ITERATIONS):
            levels = _code(blocks, _safe_inverse(scale))
            sum_xq = (weights * blocks * levels).sum(dim=1)
            sum_q2 = (weights * levels * levels).sum(dim=1)
            refit = (scale != 0) & (sum_q2 > 0)
            scale = torch.where(refit, sum_xq / torch.where(refit, sum_q2, ones), scale)
        scale = _fp16(scale)
        levels = _code(blocks, _safe_inverse(scale))
        error = (weights * (blocks - scale[:, None] * levels) ** 2).sum(dim=1)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
        best_levels = torch.where(better[:, None], levels, best_levels)
    return best_scale, best_levels


def q2_0_dequantize_rows(
    rows: torch.Tensor, quant_weights: torch.Tensor | None
) -> torch.Tensor:
    """Round-trip whole rows through the assisted ``Q2_0`` fit.

    Args:
        rows: Shape ``(n, row)``, float32.
        quant_weights: Column weights per row, same shape, or None.

    Returns:
        The dequantized values, shape of ``rows``.
    """
    scales, levels = q2_0_assisted_fit(rows, quant_weights)
    return (levels * scales[:, None]).reshape(rows.shape)


def q2_0_encode_rows(rows: torch.Tensor, quant_weights: torch.Tensor | None) -> bytes:
    r"""Encode whole rows into stored ``block_q2_0`` bytes.

    The block layout is stock llama.cpp's: a little-endian fp16
    scale, then 16 bytes holding ``level + 1`` in two-bit fields,
    element ``j`` at bits ``2 * (j % 4)`` of byte ``j // 4``. Stock
    ``dequantize_row_q2_0`` reads these blocks unchanged.

    Args:
        rows: Shape ``(n, row)``, float32, on the CPU.
        quant_weights: Column weights per row, same shape, or None.

    Returns:
        The rows' blocks in row-major order, ``Q2_0_BLOCK_BYTES``
        per block.

    Examples:
        One zero row stores a zero scale and level 0 everywhere:

        ```python
        import torch

        payload = q2_0_encode_rows(torch.zeros(1, 64), None)
        assert payload == b"\x00\x00" + b"\x55" * 16
        ```
    """
    scales, levels = q2_0_assisted_fit(rows, quant_weights)
    codes = (levels + 1).to(torch.uint8).reshape(-1, QK2_0 // _CODES_PER_BYTE, 4)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)
    packed = (codes << shifts).sum(dim=2, dtype=torch.uint8)
    scale_bytes = scales.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
    return torch.cat([scale_bytes, packed], dim=1).numpy().tobytes()


def q2_0_assisted_quantize_dequantize(
    weight: torch.Tensor, quant_weights: torch.Tensor | None
) -> torch.Tensor:
    """Quantize a tensor through the assisted ``Q2_0`` encoder.

    Rows round-trip exactly as the preprocessor encodes them: every
    row of length ``weight.shape[-1]`` fits against its column
    weights. With 1-D weights every row shares them. With 2-D
    weights on a 3-D fused expert stack, expert ``i``'s rows fit
    against weight row ``i``, the mapping ``llama-quant.cpp`` uses
    per expert. The input is never modified.

    Args:
        weight: The tensor to perturb. Any shape, any float dtype.
        quant_weights: Imatrix column weights — 1-D of the row
            length, or 2-D ``(experts, row)`` against a 3-D fused
            expert stack — or None for the unweighted search.

    Returns:
        The dequantized tensor, same shape, dtype, and device as the
        input.

    Raises:
        ValueError: If the rows do not divide into ``QK2_0`` blocks,
            or the weights fit neither layout.

    Examples:
        Price a covered expert stack:

        ```python
        import torch

        stack = torch.randn(2, 4, 128)
        qw = torch.rand(2, 128)
        priced = q2_0_assisted_quantize_dequantize(stack, qw)
        ```
    """
    if quant_weights is not None:
        check_q0_weights(weight, quant_weights, block=QK2_0)
    elif int(weight.shape[-1]) % QK2_0:
        raise ValueError(
            f"rows of {int(weight.shape[-1])} do not divide into {QK2_0}-element "
            "Q2_0 blocks (ADR-0032)"
        )
    return run_assisted_rows(weight, quant_weights, q2_0_dequantize_rows, QK2_0)
