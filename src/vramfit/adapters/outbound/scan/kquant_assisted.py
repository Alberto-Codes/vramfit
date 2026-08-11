"""Imatrix-assisted K-quant quantize-dequantize (ADR-0020).

Torch ports of llama.cpp's imatrix-weighted quantizers for ``Q2_K``,
``Q3_K``, and ``Q4_K`` (``quantize_row_q2_K_impl`` and kin,
``ggml-quants.c``, checkout e9fa078). ``llama-quantize --imatrix``
routes covered tensors through these paths, so an assisted scan cell
prices the format the pack actually ships. The element fit weight is
``qw[i] * sqrt(sigma2 + x[i]^2)``, where ``qw`` is the imatrix column
weight and ``sigma2`` is a per-super-block variance term. ``Q8_0``
has no weighted path — ``quantize_q8_0`` discards the imatrix — so
8-bit cells reuse the unassisted port.

Like the unassisted module, the functions return dequantized values,
and knife-edge fitting ties may diverge from the C where vectorized
float sums order differently. The assisted golden fixtures bound
that drift.

The C's ``make_qkx3_quants`` reuses the ADR-0018 port of
``make_qkx2_quants``: the two C functions are identical when
``weights`` is non-NULL. They differ only in the NULL-weights
fallback (never taken here) and the degenerate guard —
``max <= min`` against ``max == min`` — which cannot diverge
because the ``min > 0 -> 0`` clamp keeps ``max >= min``.

Examples:
    Simulate assisted Q2_K damage on one weight matrix:

    ```python
    perturbed = kquant_assisted_quantize_dequantize(weight, 2, column_weights)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.kquant][]: The unassisted
      reference-path port (ADR-0018).
    - [vramfit.adapters.outbound.scan.imatrix][]: Loads the column
      weights from the imatrix artifact.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from vramfit.adapters.outbound.scan.kquant import (
    _CHUNK_ROWS,
    _GROUP_MAX_EPS,
    SUB_BLOCK,
    SUPER_BLOCK,
    _fp16,
    _make_qkx2_quants,
    kquant_quantize_dequantize,
)

# The precisions with a ported weighted path. 8 is assisted-valid but
# routes to the unassisted Q8_0 port, matching quantize_q8_0.
ASSISTED_BITS = (2, 3, 4, 8)


def _make_qp_quants(
    x: torch.Tensor, weights: torch.Tensor, nmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit non-negative super-block scale codes (``make_qp_quants``).

    A weighted absmax start, an 8-candidate grid search, then five
    sweeps of per-element coordinate refinement. The element loop
    runs in C order so the running sums update exactly like the
    reference.

    Args:
        x: Non-negative values to code, shape ``(n, m)``, float32.
        weights: Per-element fit weights, same shape.
        nmax: Largest code value.

    Returns:
        Per-row ``(scale, codes)`` — ``scale`` shape ``(n,)``,
        ``codes`` shape ``(n, m)`` in ``[0, nmax]``. The C stores
        codes through uint8 and would wrap a negative fit. Fitted
        sub-block scales and minima stay non-negative, so the clamp
        and the wrap agree on reachable data.
    """
    ones = torch.ones_like(x[:, 0])
    max_x = x.amax(dim=1)
    dead = max_x < _GROUP_MAX_EPS
    safe_max = torch.where(dead, ones, max_x)

    iscale = nmax / safe_max
    levels = torch.round(iscale[:, None] * x).clamp(0, nmax)
    best_mse = (weights * (x - (1.0 / iscale)[:, None] * levels) ** 2).sum(dim=1)
    for step in range(-4, 5):
        if step == 0:
            continue
        cand_iscale = (
            float(torch.tensor(0.1, dtype=torch.float32) * step) + nmax
        ) / safe_max
        cand = torch.round(cand_iscale[:, None] * x).clamp(0, nmax)
        mse = (weights * (x - (1.0 / cand_iscale)[:, None] * cand) ** 2).sum(dim=1)
        accept = mse < best_mse
        best_mse = torch.where(accept, mse, best_mse)
        iscale = torch.where(accept, cand_iscale, iscale)

    levels = torch.round(iscale[:, None] * x).clamp(0, nmax)
    sum_lx = (weights * x * levels).sum(dim=1)
    sum_l2 = (weights * levels * levels).sum(dim=1)
    for _ in range(5):
        changed_any = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for i in range(x.shape[1]):
            xi = x[:, i]
            wi = weights[:, i]
            li = levels[:, i]
            slx = sum_lx - wi * xi * li
            sl2 = sum_l2 - wi * li * li
            ok = (slx > 0) & (sl2 > 0)
            safe_slx = torch.where(ok, slx, ones)
            new_l = torch.round(xi * sl2 / safe_slx).clamp(0, nmax)
            new_slx = slx + wi * xi * new_l
            new_sl2 = sl2 + wi * new_l * new_l
            accept = (
                ok
                & (new_l != li)
                & (new_slx * new_slx * sum_l2 > sum_lx * sum_lx * new_sl2)
            )
            levels[:, i] = torch.where(accept, new_l, li)
            sum_lx = torch.where(accept, new_slx, sum_lx)
            sum_l2 = torch.where(accept, new_sl2, sum_l2)
            changed_any = changed_any | accept
        if not bool(changed_any.any()):
            break

    pos = sum_l2 > 0
    scale = torch.where(pos, sum_lx / torch.where(pos, sum_l2, ones), 0.0)
    scale = torch.where(dead, torch.zeros_like(scale), scale)
    levels = torch.where(dead[:, None], torch.zeros_like(levels), levels)
    return scale, levels


def _make_qx_quants(
    x: torch.Tensor, weights: torch.Tensor, nmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one symmetric scale per row (``make_qx_quants``, rmse type 1).

    A signed absmax start, then an 18-candidate grid search that
    keeps the weighted least-squares winner.

    Args:
        x: Values to code, shape ``(n, m)``, float32.
        weights: Per-element fit weights, same shape.
        nmax: Level magnitude bound — levels live in
            ``[-nmax, nmax - 1]``.

    Returns:
        Per-row ``(scale, levels)`` — ``scale`` shape ``(n,)``,
        signed ``levels`` shape ``(n, m)``.
    """
    ones = torch.ones_like(x[:, 0])
    amax, argmax = x.abs().max(dim=1)
    signed_max = x.gather(1, argmax[:, None]).squeeze(1)
    dead = amax < _GROUP_MAX_EPS
    safe_max = torch.where(dead, ones, signed_max)

    iscale = -nmax / safe_max
    levels = torch.round(iscale[:, None] * x).clamp(-nmax, nmax - 1)
    sum_lx = (weights * x * levels).sum(dim=1)
    sum_l2 = (weights * levels * levels).sum(dim=1)
    pos = sum_l2 > 0
    scale = torch.where(pos, sum_lx / torch.where(pos, sum_l2, ones), 0.0)
    best = scale * sum_lx
    for step in range(-9, 10):
        if step == 0:
            continue
        cand_iscale = -(nmax + float(torch.tensor(0.1, dtype=torch.float32) * step)) / (
            safe_max
        )
        cand = torch.round(cand_iscale[:, None] * x).clamp(-nmax, nmax - 1)
        cand_lx = (weights * x * cand).sum(dim=1)
        cand_l2 = (weights * cand * cand).sum(dim=1)
        accept = (cand_l2 > 0) & (cand_lx * cand_lx > best * cand_l2)
        levels = torch.where(accept[:, None], cand, levels)
        scale = torch.where(accept, cand_lx / torch.where(accept, cand_l2, ones), scale)
        best = torch.where(accept, scale * cand_lx, best)

    scale = torch.where(dead, torch.zeros_like(scale), scale)
    levels = torch.where(dead[:, None], torch.zeros_like(levels), levels)
    return scale, levels


def _fit_weights(
    blocks: torch.Tensor, qw: torch.Tensor, sub: int, sigma_factor: float
) -> torch.Tensor:
    """Build the per-element fit weights for one sub-block width.

    The C forms ``qw[l] * sqrt(sigma2 + x[l]^2)`` with ``sigma2``
    computed once per super-block.

    Args:
        blocks: Super-blocks, shape ``(n, SUPER_BLOCK)``, float32.
        qw: Imatrix column weights aligned to ``blocks``, same shape.
        sub: Sub-block width — 16 for Q2_K/Q3_K, 32 for Q4_K.
        sigma_factor: 1.0 for Q2_K, 2.0 for Q3_K/Q4_K.

    Returns:
        Fit weights as sub-blocks, shape ``(n * SUPER_BLOCK/sub, sub)``.
    """
    sigma2 = sigma_factor * (blocks * blocks).sum(dim=1) / SUPER_BLOCK
    subs = blocks.reshape(-1, sub)
    expanded = sigma2.repeat_interleave(SUPER_BLOCK // sub)[:, None]
    return qw.reshape(-1, sub) * torch.sqrt(expanded + subs * subs)


def _q2k_assisted(blocks: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
    """Round-trip super-blocks through assisted Q2_K.

    Args:
        blocks: Shape ``(n, SUPER_BLOCK)``, float32.
        qw: Imatrix column weights aligned to ``blocks``, same shape.

    Returns:
        Dequantized values, same shape.
    """
    n = blocks.shape[0]
    subs = blocks.reshape(-1, SUB_BLOCK)
    weights = _fit_weights(blocks, qw, SUB_BLOCK, sigma_factor=1.0)
    scale, the_min = _make_qkx2_quants(
        subs, weights, nmax=3, rmin=-0.9, rdelta=0.05, nstep=36, use_mad=False
    )
    scales = scale.reshape(n, -1)
    mins = the_min.reshape(n, -1)
    sw = weights.sum(dim=1).reshape(n, -1)

    dm, sc_q = _make_qp_quants(scales, sw, 15)
    mm, m_q = _make_qp_quants(mins, sw, 15)
    d = _fp16(dm)
    dmin = _fp16(mm)

    dl = (d[:, None] * sc_q).repeat_interleave(SUB_BLOCK, dim=1)
    ml = (dmin[:, None] * m_q).repeat_interleave(SUB_BLOCK, dim=1)
    live = dl != 0
    safe_dl = torch.where(live, dl, torch.ones_like(dl))
    levels = torch.round((blocks + ml) / safe_dl).clamp(0, 3)
    return torch.where(live, dl * levels - ml, -ml)


def _q3k_assisted(blocks: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
    """Round-trip super-blocks through assisted Q3_K.

    Args:
        blocks: Shape ``(n, SUPER_BLOCK)``, float32.
        qw: Imatrix column weights aligned to ``blocks``, same shape.

    Returns:
        Dequantized values, same shape.
    """
    n = blocks.shape[0]
    subs = blocks.reshape(-1, SUB_BLOCK)
    weights = _fit_weights(blocks, qw, SUB_BLOCK, sigma_factor=2.0)
    scale, _ = _make_qx_quants(subs, weights, nmax=4)
    scales = scale.reshape(n, -1)
    sw = weights.sum(dim=1).reshape(n, -1)

    d_block, sc_q = _make_qx_quants(scales, sw, nmax=32)
    d = _fp16(d_block)

    dl = (d[:, None] * sc_q).repeat_interleave(SUB_BLOCK, dim=1)
    live = dl != 0
    safe_dl = torch.where(live, dl, torch.ones_like(dl))
    levels = torch.round(blocks / safe_dl).clamp(-4, 3)
    return torch.where(live, dl * levels, torch.zeros_like(blocks))


def _q4k_assisted(blocks: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
    """Round-trip super-blocks through assisted Q4_K.

    Args:
        blocks: Shape ``(n, SUPER_BLOCK)``, float32.
        qw: Imatrix column weights aligned to ``blocks``, same shape.

    Returns:
        Dequantized values, same shape.
    """
    n = blocks.shape[0]
    sub = 32
    subs = blocks.reshape(-1, sub)
    weights = _fit_weights(blocks, qw, sub, sigma_factor=2.0)
    scale, the_min = _make_qkx2_quants(
        subs, weights, nmax=15, rmin=-0.9, rdelta=0.05, nstep=36, use_mad=False
    )
    scales = scale.reshape(n, -1)
    mins = the_min.reshape(n, -1)
    sw = weights.sum(dim=1).reshape(n, -1)

    d_block, sc_q = _make_qp_quants(scales, sw, 63)
    m_block, m_q = _make_qp_quants(mins, sw, 63)
    d = _fp16(d_block)
    dmin = _fp16(m_block)

    dl = (d[:, None] * sc_q).repeat_interleave(sub, dim=1)
    ml = (dmin[:, None] * m_q).repeat_interleave(sub, dim=1)
    live = dl != 0
    safe_dl = torch.where(live, dl, torch.ones_like(dl))
    levels = torch.round((blocks + ml) / safe_dl).clamp(0, 15)
    return torch.where(live, dl * levels - ml, -ml)


_ASSISTED_ROUND_TRIPS: dict[
    int, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    2: _q2k_assisted,
    3: _q3k_assisted,
    4: _q4k_assisted,
}


def _sliced_assisted(
    round_trip: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    blocks: torch.Tensor,
    row_qw: torch.Tensor,
) -> torch.Tensor:
    """Run the assisted round trip over bounded slices of whole blocks.

    Every row shares one imatrix weight vector, so the per-block
    weights repeat with period ``row_qw.shape[0]``. Slices index into
    that one small tensor instead of materializing a full-size copy.

    Args:
        round_trip: The per-block assisted round-trip function.
        blocks: Shape ``(n, SUPER_BLOCK)``, float32.
        row_qw: One row's weights as blocks,
            shape ``(row/SUPER_BLOCK, SUPER_BLOCK)``.

    Returns:
        Dequantized values, shape of ``blocks``.
    """
    period = row_qw.shape[0]
    if blocks.shape[0] <= _CHUNK_ROWS:
        idx = torch.arange(blocks.shape[0], device=blocks.device) % period
        return round_trip(blocks, row_qw[idx])
    out = torch.empty_like(blocks)
    for start in range(0, blocks.shape[0], _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, blocks.shape[0])
        idx = torch.arange(start, stop, device=blocks.device) % period
        out[start:stop] = round_trip(blocks[start:stop], row_qw[idx])
    return out


def kquant_assisted_quantize_dequantize(
    weight: torch.Tensor, bits: int, quant_weights: torch.Tensor
) -> torch.Tensor:
    """Quantize a tensor through the assisted K-quant format for ``bits``.

    Rows are round-tripped exactly as ``llama-quantize --imatrix``
    consumes them: every row of length ``weight.shape[-1]`` fits
    against the same imatrix column weights. 8-bit routes to the
    unassisted ``Q8_0`` port — ``quantize_q8_0`` discards the
    imatrix. The input is never modified. The computation runs on
    the input's device first and retries on the CPU when the float32
    workspace does not fit the card.

    Args:
        weight: The tensor to perturb. Any shape, any float dtype.
        bits: Nominal precision — 2, 3, 4, or 8.
        quant_weights: Imatrix column weights, 1-D, length
            ``weight.shape[-1]``, finite and non-negative.

    Returns:
        The dequantized tensor, same shape, dtype, and device as the
        input.

    Raises:
        ValueError: If ``bits`` has no assisted port, the row length
            does not divide into super-blocks (the C asserts
            ``n_per_row % QK_K == 0``), ``quant_weights`` does not
            match the row length, or a weight is negative or
            non-finite — garbage weights would corrupt every damage
            downstream. Every check runs for 8-bit too, before the
            Q8_0 route discards the weights — validity must not
            depend on which branch consumes the argument.
    """
    row = int(weight.shape[-1])
    if row % SUPER_BLOCK:
        raise ValueError(
            f"assisted kquant needs rows divisible by {SUPER_BLOCK}, "
            f"got row length {row}"
        )
    if quant_weights.dim() != 1 or quant_weights.numel() != row:
        raise ValueError(
            f"quant_weights must be 1-D with {row} entries, got shape "
            f"{tuple(quant_weights.shape)}"
        )
    if not bool(torch.isfinite(quant_weights).all()) or bool((quant_weights < 0).any()):
        raise ValueError("quant_weights must be finite and non-negative")
    if bits == 8:  # noqa: PLR2004 - the Q8_0 nominal precision
        return kquant_quantize_dequantize(weight, 8)
    if bits not in _ASSISTED_ROUND_TRIPS:
        raise ValueError(
            f"assisted kquant supports bits in {ASSISTED_BITS}, got {bits} (ADR-0020)"
        )
    round_trip = _ASSISTED_ROUND_TRIPS[bits]

    def prepare(device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build float32 blocks and one row's weight blocks on ``device``.

        Args:
            device: Where the workspace copies live.

        Returns:
            ``(blocks, row_qw)`` shaped ``(n, SUPER_BLOCK)`` and
            ``(row/SUPER_BLOCK, SUPER_BLOCK)``.
        """
        blocks = (
            weight.detach()
            .to(device=device, dtype=torch.float32)
            .reshape(-1, SUPER_BLOCK)
        )
        row_qw = (
            quant_weights.detach()
            .to(device=device, dtype=torch.float32)
            .reshape(-1, SUPER_BLOCK)
        )
        return blocks, row_qw

    try:
        result = _sliced_assisted(round_trip, *prepare(weight.device))
    except torch.cuda.OutOfMemoryError:
        result = _sliced_assisted(round_trip, *prepare("cpu"))
    return result.reshape(weight.shape).to(device=weight.device, dtype=weight.dtype)
