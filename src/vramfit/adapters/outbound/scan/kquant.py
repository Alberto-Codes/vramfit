"""K-quant-faithful quantize-dequantize (ADR-0018).

Torch ports of llama.cpp's reference quantizers for ``Q2_K``,
``Q3_K``, ``Q4_K``, and ``Q8_0`` (``ggml-quants.c``, checkout
e9fa078). The ports reproduce the full round trip: sub-block
scale/min fitting, super-block scale re-quantization, fp16 storage
rounding, and the final level recomputation. Like the RTN module,
the functions return dequantized values — the scan measures
round-trip damage, never keeps integers. Large tensors fit in
bounded slices of whole blocks, which caps the fp32 workspace.

The C reference rounds with ``nearest_int``, a round-half-to-even
magic-number trick. ``torch.round`` rounds half to even, so the
rounding modes match. Vectorized reductions can still order float
sums differently from the C loops, so knife-edge fitting choices may
diverge on isolated sub-blocks. The golden fixtures bound that
drift.

Examples:
    Simulate Q2_K damage on one weight matrix:

    ```python
    perturbed = kquant_quantize_dequantize(weight, bits=2)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.quantize][]: The RTN v1 method.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

SUPER_BLOCK = 256
SUB_BLOCK = 16
Q8_BLOCK = 32
# llama.cpp's all-zero guard for a sub-block (GROUP_MAX_EPS).
_GROUP_MAX_EPS = 1e-15


def _fp16(t: torch.Tensor) -> torch.Tensor:
    """Round through fp16 storage, like GGML_FP32_TO_FP16.

    Args:
        t: Float32 values.

    Returns:
        The values after an fp16 round trip, as float32.
    """
    return t.half().float()


def _make_qkx2_quants(
    x: torch.Tensor,
    weights: torch.Tensor,
    nmax: int,
    rmin: float,
    rdelta: float,
    nstep: int,
    use_mad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one scale and one minimum per sub-block (Q2_K/Q4_K path).

    Vectorized port of ``make_qkx2_quants``. The candidate loop
    evaluates every scale candidate on every sub-block and keeps the
    first strict improvement, matching the C loop's accept order.

    Args:
        x: Sub-blocks, shape ``(n, sub)``, float32.
        weights: Per-element fit weights, same shape as ``x``.
        nmax: Largest quantized level.
        rmin: First candidate offset.
        rdelta: Candidate step.
        nstep: Number of candidate steps.
        use_mad: Score candidates by weighted absolute deviation
            instead of weighted squared error.

    Returns:
        Per-sub-block ``(scale, the_min)``, each shape ``(n,)``.
        ``the_min`` is the negated fitted minimum, non-negative.
    """

    def score(diff: torch.Tensor) -> torch.Tensor:
        """Turn reconstruction differences into the fit's error terms.

        Args:
            diff: Reconstruction minus input, any shape.

        Returns:
            Absolute or squared deviations, per ``use_mad``.
        """
        return diff.abs() if use_mad else diff * diff

    w = weights
    sum_w = w.sum(dim=1)
    sum_x = (w * x).sum(dim=1)
    lo = x.amin(dim=1).clamp(max=0.0)
    hi = x.amax(dim=1)
    degenerate = hi == lo
    span = torch.where(degenerate, torch.ones_like(hi), hi - lo)

    iscale = nmax / span
    scale = 1.0 / iscale
    fmin = lo.clone()
    levels = torch.round(iscale[:, None] * (x - lo[:, None])).clamp(0, nmax)
    best_error = (w * score(scale[:, None] * levels + lo[:, None] - x)).sum(dim=1)

    # The C loop reads the *current* fitted minimum when it builds each
    # candidate grid — an accepted candidate shifts every later grid.
    # The candidate numerator is fp32 scalar arithmetic in C.
    numerators = (
        torch.tensor(rmin, dtype=torch.float32)
        + torch.tensor(rdelta, dtype=torch.float32)
        * torch.arange(nstep + 1, dtype=torch.float32)
        + nmax
    )
    for step in range(nstep + 1):
        span_now = torch.where(degenerate, torch.ones_like(hi), hi - fmin)
        cand_iscale = float(numerators[step]) / span_now
        cand_levels = torch.round(cand_iscale[:, None] * (x - fmin[:, None])).clamp(
            0, nmax
        )
        sum_l = (w * cand_levels).sum(dim=1)
        sum_l2 = (w * cand_levels * cand_levels).sum(dim=1)
        sum_xl = (w * cand_levels * x).sum(dim=1)
        det = sum_w * sum_l2 - sum_l * sum_l
        safe_det = torch.where(det > 0, det, torch.ones_like(det))
        cand_scale = (sum_w * sum_xl - sum_x * sum_l) / safe_det
        cand_min = (sum_l2 * sum_x - sum_l * sum_xl) / safe_det
        safe_l2 = torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2))
        clip = cand_min > 0
        cand_scale = torch.where(clip, sum_xl / safe_l2, cand_scale)
        cand_min = torch.where(clip, torch.zeros_like(cand_min), cand_min)
        cand_error = (
            w * score(cand_scale[:, None] * cand_levels + cand_min[:, None] - x)
        ).sum(dim=1)
        accept = (det > 0) & (cand_error < best_error)
        scale = torch.where(accept, cand_scale, scale)
        fmin = torch.where(accept, cand_min, fmin)
        best_error = torch.where(accept, cand_error, best_error)

    scale = torch.where(degenerate, torch.zeros_like(scale), scale)
    fmin = torch.where(degenerate, lo, fmin)
    return scale, -fmin


def _make_q3_quants(x: torch.Tensor, nmax: int) -> torch.Tensor:
    """Fit one symmetric scale per sub-block (Q3_K path).

    Vectorized port of ``make_q3_quants`` with ``do_rmse=True``: an
    absmax start, then five sweeps of per-element coordinate
    refinement under weights ``x**2``. The element loop runs in C
    order so the running sums update exactly like the reference.

    Args:
        x: Sub-blocks, shape ``(n, SUB_BLOCK)``, float32.
        nmax: Level magnitude bound — levels live in
            ``[-nmax, nmax - 1]``.

    Returns:
        Per-sub-block scale, shape ``(n,)``.
    """
    amax, argmax = x.abs().max(dim=1)
    signed_max = x.gather(1, argmax[:, None]).squeeze(1)
    dead = amax < _GROUP_MAX_EPS
    safe_max = torch.where(dead, torch.ones_like(signed_max), signed_max)

    iscale = -nmax / safe_max
    levels = torch.round(iscale[:, None] * x).clamp(-nmax, nmax - 1)
    w = x * x
    sum_lx = (w * x * levels).sum(dim=1)
    sum_l2 = (w * levels * levels).sum(dim=1)

    for _ in range(5):
        changed_any = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for i in range(x.shape[1]):
            xi = x[:, i]
            wi = w[:, i]
            li = levels[:, i]
            slx = sum_lx - wi * xi * li
            positive = slx > 0
            sl2 = sum_l2 - wi * li * li
            safe_slx = torch.where(positive, slx, torch.ones_like(slx))
            new_l = torch.round(xi * sl2 / safe_slx).clamp(-nmax, nmax - 1)
            new_slx = slx + wi * xi * new_l
            new_sl2 = sl2 + wi * new_l * new_l
            accept = (
                positive
                & (new_l != li)
                & (new_sl2 > 0)
                & (new_slx * new_slx * sum_l2 > sum_lx * sum_lx * new_sl2)
            )
            levels[:, i] = torch.where(accept, new_l, li)
            sum_lx = torch.where(accept, new_slx, sum_lx)
            sum_l2 = torch.where(accept, new_sl2, sum_l2)
            changed_any = changed_any | accept
        if not bool(changed_any.any()):
            break

    safe_l2 = torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2))
    scale = torch.where(sum_l2 > 0, sum_lx / safe_l2, torch.zeros_like(sum_lx))
    return torch.where(dead, torch.zeros_like(scale), scale)


def _q2k_round_trip(blocks: torch.Tensor) -> torch.Tensor:
    """Round-trip super-blocks through Q2_K.

    Fits with absolute deviation under ``|x|`` weights, the
    reference's Q2_K arguments.

    Args:
        blocks: Shape ``(n, SUPER_BLOCK)``, float32.

    Returns:
        Dequantized values, same shape.
    """
    n = blocks.shape[0]
    subs = blocks.reshape(n * (SUPER_BLOCK // SUB_BLOCK), SUB_BLOCK)
    scale, the_min = _make_qkx2_quants(
        subs, subs.abs(), nmax=3, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True
    )
    scale = scale.reshape(n, -1)
    the_min = the_min.reshape(n, -1)

    q4scale = 15.0
    max_scale = scale.amax(dim=1)
    max_min = the_min.amax(dim=1)
    pos_scale = max_scale > 0
    safe_ms = torch.where(pos_scale, max_scale, torch.ones_like(max_scale))
    sc_q = torch.round(q4scale * scale / safe_ms[:, None]).clamp(0, 15)
    sc_q = torch.where(pos_scale[:, None], sc_q, torch.zeros_like(sc_q))
    d = torch.where(pos_scale, _fp16(max_scale / q4scale), torch.zeros_like(max_scale))
    pos_min = max_min > 0
    safe_mm = torch.where(pos_min, max_min, torch.ones_like(max_min))
    m_q = torch.round(q4scale * the_min / safe_mm[:, None]).clamp(0, 15)
    m_q = torch.where(pos_min[:, None], m_q, torch.zeros_like(m_q))
    dmin = torch.where(pos_min, _fp16(max_min / q4scale), torch.zeros_like(max_min))

    dl = (d[:, None] * sc_q).repeat_interleave(SUB_BLOCK, dim=1)
    ml = (dmin[:, None] * m_q).repeat_interleave(SUB_BLOCK, dim=1)
    live = dl != 0
    safe_dl = torch.where(live, dl, torch.ones_like(dl))
    levels = torch.round((blocks + ml) / safe_dl).clamp(0, 3)
    return torch.where(live, dl * levels - ml, -ml)


def _q3k_round_trip(blocks: torch.Tensor) -> torch.Tensor:
    """Round-trip super-blocks through Q3_K.

    Args:
        blocks: Shape ``(n, SUPER_BLOCK)``, float32.

    Returns:
        Dequantized values, same shape.
    """
    n = blocks.shape[0]
    subs = blocks.reshape(n * (SUPER_BLOCK // SUB_BLOCK), SUB_BLOCK)
    scale = _make_q3_quants(subs, nmax=4).reshape(n, -1)

    _, arg = scale.abs().max(dim=1)
    max_scale = scale.gather(1, arg[:, None]).squeeze(1)
    live_block = max_scale != 0
    safe_ms = torch.where(live_block, max_scale, torch.ones_like(max_scale))
    iscale = -32.0 / safe_ms
    sc_q = torch.round(iscale[:, None] * scale).clamp(-32, 31)
    sc_q = torch.where(live_block[:, None], sc_q, torch.zeros_like(sc_q))
    d = torch.where(live_block, _fp16(1.0 / iscale), torch.zeros_like(max_scale))

    dl = (d[:, None] * sc_q).repeat_interleave(SUB_BLOCK, dim=1)
    live = dl != 0
    safe_dl = torch.where(live, dl, torch.ones_like(dl))
    levels = torch.round(blocks / safe_dl).clamp(-4, 3)
    return torch.where(live, dl * levels, torch.zeros_like(blocks))


def _q4k_round_trip(blocks: torch.Tensor) -> torch.Tensor:
    """Round-trip super-blocks through Q4_K.

    Fits with squared error under ``av_x + |x|`` weights over
    32-element sub-blocks, then re-quantizes scales to 6 bits.

    Args:
        blocks: Shape ``(n, SUPER_BLOCK)``, float32.

    Returns:
        Dequantized values, same shape.
    """
    n = blocks.shape[0]
    sub = 32
    subs = blocks.reshape(n * (SUPER_BLOCK // sub), sub)
    av_x = (subs * subs).mean(dim=1, keepdim=True).sqrt()
    scale, the_min = _make_qkx2_quants(
        subs, av_x + subs.abs(), nmax=15, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False
    )
    scale = scale.reshape(n, -1)
    the_min = the_min.reshape(n, -1)

    q6scale = 63.0
    max_scale = scale.amax(dim=1)
    max_min = the_min.amax(dim=1)
    pos_scale = max_scale > 0
    safe_ms = torch.where(pos_scale, max_scale, torch.ones_like(max_scale))
    # C stores through uint8 with MIN(63, ls) and would wrap a
    # negative fit. Fitted scales stay positive (the minimum is
    # deducted), so the clamp and the wrap agree on reachable data.
    sc_q = torch.round(q6scale * scale / safe_ms[:, None]).clamp(max=63)
    sc_q = torch.where(pos_scale[:, None], sc_q, torch.zeros_like(sc_q))
    d = torch.where(pos_scale, _fp16(max_scale / q6scale), torch.zeros_like(max_scale))
    pos_min = max_min > 0
    safe_mm = torch.where(pos_min, max_min, torch.ones_like(max_min))
    m_q = torch.round(q6scale * the_min / safe_mm[:, None]).clamp(max=63)
    m_q = torch.where(pos_min[:, None], m_q, torch.zeros_like(m_q))
    dmin = torch.where(pos_min, _fp16(max_min / q6scale), torch.zeros_like(max_min))

    dl = (d[:, None] * sc_q).repeat_interleave(sub, dim=1)
    ml = (dmin[:, None] * m_q).repeat_interleave(sub, dim=1)
    live = dl != 0
    safe_dl = torch.where(live, dl, torch.ones_like(dl))
    levels = torch.round((blocks + ml) / safe_dl).clamp(0, 15)
    return torch.where(live, dl * levels - ml, -ml)


def _q8_0_round_trip(blocks: torch.Tensor) -> torch.Tensor:
    """Round-trip 32-element blocks through Q8_0.

    ``quantize_row_q8_0_ref`` rounds with ``roundf`` — half away from
    zero — and quantizes against the pre-fp16 scale while dequantizing
    with the fp16-stored one. A subnormal scale dequantizes to zero,
    matching the C path, whose fp16 scale underflows.

    Args:
        blocks: Shape ``(n, Q8_BLOCK)``, float32.

    Returns:
        Dequantized values, same shape.
    """
    amax = blocks.abs().amax(dim=1, keepdim=True)
    d = amax / 127.0
    safe_d = torch.where(d != 0, d, torch.ones_like(d))
    raw = 1.0 / safe_d
    # A subnormal scale overflows the reciprocal to inf, and
    # inf * fp16(d) is NaN where the C path dequantizes to zero —
    # its fp16 scale underflows first. Zero the levels instead.
    inverse = torch.where((d != 0) & torch.isfinite(raw), raw, torch.zeros_like(d))
    v = blocks * inverse
    levels = torch.trunc(v + 0.5 * torch.sign(v))
    return levels * _fp16(d)


_ROUND_TRIPS = {
    2: (_q2k_round_trip, SUPER_BLOCK),
    3: (_q3k_round_trip, SUPER_BLOCK),
    4: (_q4k_round_trip, SUPER_BLOCK),
    8: (_q8_0_round_trip, Q8_BLOCK),
}
# The ported coverage, derived from the dispatch table so the two
# cannot drift. domain.scan.KQUANT_PRECISIONS must mirror this —
# the CLI validates against the domain copy before a model loads.
KQUANT_BITS = tuple(sorted(_ROUND_TRIPS))

# Rows per fitting slice: 134M elements at super-block width, so
# the candidate loops peak near 4 GiB of temporaries however large
# the tensor. The 49B head fits 1.05B parameters in one call.
# A slice holds whole blocks, so it cannot change a fitted value.
_CHUNK_ROWS = 1 << 19


def _sliced(
    round_trip: Callable[[torch.Tensor], torch.Tensor], blocks: torch.Tensor
) -> torch.Tensor:
    """Run the round trip over bounded slices of whole blocks.

    Every fit is local to one block, so a slice boundary is
    bit-invisible — this only caps the workspace. Slices write into
    one preallocated output, so no second full-size copy exists.

    Args:
        round_trip: The per-block round-trip function.
        blocks: Shape ``(n, block)``, float32.

    Returns:
        Dequantized values, same shape.
    """
    if blocks.shape[0] <= _CHUNK_ROWS:
        return round_trip(blocks)
    # Preallocate the output: collecting slices for a cat would hold
    # a second full-size copy (~4.2 GiB on the 49B head).
    out = torch.empty_like(blocks)
    for start in range(0, blocks.shape[0], _CHUNK_ROWS):
        out[start : start + _CHUNK_ROWS] = round_trip(
            blocks[start : start + _CHUNK_ROWS]
        )
    return out


def kquant_quantize_dequantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize a tensor through the K-quant format for ``bits``.

    The tensor is flattened in row-major order — the order
    ``llama-quantize`` consumes rows — padded with zeros to the
    format's block multiple, and round-tripped through the ported
    reference quantizer. Flat blocks match the per-row C layout only
    when row length divides by the block size. Rows that do not
    divide would straddle blocks (every 49B weight row divides).
    The input is never modified. Like the RTN
    round trip, the computation runs on the input's device first and
    retries on the CPU when the float32 workspace does not fit the
    card. Large tensors fit in bounded slices of whole blocks. The
    49B output head would otherwise need ~38 GiB of fp32
    temporaries.

    Args:
        weight: The tensor to perturb. Any shape, any float dtype.
        bits: Nominal precision — 2 (``Q2_K``), 3 (``Q3_K``),
            4 (``Q4_K``), or 8 (``Q8_0``).

    Returns:
        The dequantized tensor, same shape, dtype, and device as the
        input.

    Raises:
        ValueError: If ``bits`` has no K-quant port (ADR-0018 leaves
            5 and 6 open).

    Examples:
        Q2_K keeps at most four levels per 16-element sub-block:

        ```python
        import torch

        w = torch.randn(4, 256)
        q = kquant_quantize_dequantize(w, 2)
        assert all(len(sub.unique()) <= 4 for sub in q.reshape(-1, 16))
        ```
    """
    if bits not in _ROUND_TRIPS:
        raise ValueError(
            f"kquant supports bits in {KQUANT_BITS}, got {bits} (ADR-0018)"
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
