"""Imatrix-assisted `q0` quantize-dequantize (ADR-0018, `q0-imx`).

Torch port of llama.cpp's ``quantize_row_q4_0_impl``
(``ggml-quants.c``, checkout 4801e3c56, release b10362 — the
campaign arms' build). ``llama-quantize --imatrix`` routes covered
tensors through this path, so an assisted cell prices the format the
pack actually ships. The element fit weight is
``qw[j] * sqrt(sigma2 + x[j]^2)``, where ``qw`` is the imatrix
column weight and ``sigma2`` is the row's mean squared value. Each
32-element block then runs `kquant_assisted._make_qx_quants`'
candidate-scale search at ``nmax`` 8, the same C function the
assisted ``Q3_K`` port already reuses.

Only nominal 4 fits with weights. ``quantize_q2_0`` and
``quantize_q8_0`` discard the matrix, so nominal 2 and 8 route to
the unassisted [vramfit.adapters.outbound.scan.q0_ref][] port — the
assisted method keeps the pack's frame at every width it covers
(ADR-0018, 2026-08-21 amendment).

Unlike the K-quant paths, one parameter here can carry one weight
row per expert: a fused expert stack fits expert ``i``'s rows
against imatrix row ``i``, exactly as ``llama-quant.cpp`` slices the
matrix per expert. A 2-D ``quant_weights`` states that layout. Like
the other methods, the function returns dequantized values, and
knife-edge fitting ties may diverge from the C where vectorized
float sums order differently — the assisted golden fixtures bound
that drift.

Examples:
    Simulate assisted Q4_0 damage on one weight matrix:

    ```python
    perturbed = q0_assisted_quantize_dequantize(weight, 4, column_weights)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.q0_ref][]: The unassisted
      reference-path port.
    - [vramfit.adapters.outbound.scan.imatrix_q0][]: Resolves the
      per-expert weight rows from the imatrix artifact.
"""

from __future__ import annotations

import torch

from vramfit.adapters.outbound.scan.kquant import _CHUNK_ROWS, _fp16
from vramfit.adapters.outbound.scan.kquant_assisted import _make_qx_quants
from vramfit.adapters.outbound.scan.q0_ref import QK4_0, q0_ref_quantize_dequantize

# The nominal precision whose C path consumes the matrix. Every
# other covered width routes to the unassisted port, matching
# quantize_q2_0 and quantize_q8_0.
_WEIGHTED_BITS = 4


def check_q0_weights(weight: torch.Tensor, quant_weights: torch.Tensor) -> None:
    """Refuse weights the assisted Q4_0 fit cannot consume.

    Args:
        weight: The tensor the weights would assist.
        quant_weights: Imatrix column weights — 1-D of the row
            length, or 2-D ``(experts, row)`` against a 3-D fused
            expert stack.

    Raises:
        ValueError: If the shape fits neither layout, the row length
            does not divide into ``QK4_0`` blocks (the C asserts the
            multiple, and padding would misalign every column
            weight), or a weight is negative or non-finite — garbage
            weights would corrupt every damage downstream.
    """
    row = int(weight.shape[-1])
    if quant_weights.dim() == 2 and (  # noqa: PLR2004 - the per-expert layout
        weight.dim() != 3 or quant_weights.shape[0] != weight.shape[0]  # noqa: PLR2004
    ):
        raise ValueError(
            f"2-D quant_weights pair expert rows with a 3-D expert stack — "
            f"got weights {tuple(quant_weights.shape)} against a parameter "
            f"of shape {tuple(weight.shape)}"
        )
    if quant_weights.dim() not in (1, 2) or quant_weights.shape[-1] != row:
        raise ValueError(
            f"quant_weights must be 1-D with {row} entries, or 2-D with "
            f"{row} columns, got shape {tuple(quant_weights.shape)}"
        )
    if row % QK4_0:
        raise ValueError(
            f"rows of {row} do not divide into {QK4_0}-element Q4_0 "
            "blocks — the assisted fit cannot align its column weights "
            "(ADR-0018)"
        )
    if not bool(torch.isfinite(quant_weights).all()) or bool((quant_weights < 0).any()):
        raise ValueError("quant_weights must be finite and non-negative")


def _assisted_rows(rows: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
    """Round-trip whole rows through the weighted Q4_0 fit.

    Args:
        rows: Shape ``(n, row)``, float32.
        qw: Column weights per row, same shape.

    Returns:
        Dequantized values, same shape.
    """
    n, row = rows.shape
    # The C computes sigma2 once per row, over the whole row.
    sigma2 = (rows * rows).sum(dim=1, keepdim=True) / row
    weights = qw * torch.sqrt(sigma2 + rows * rows)
    scale, levels = _make_qx_quants(
        rows.reshape(-1, QK4_0), weights.reshape(-1, QK4_0), nmax=8
    )
    return (levels * _fp16(scale)[:, None]).reshape(n, row)


def q0_assisted_quantize_dequantize(
    weight: torch.Tensor, bits: int, quant_weights: torch.Tensor
) -> torch.Tensor:
    """Quantize a tensor through the assisted q0 format for ``bits``.

    Rows are round-tripped exactly as ``llama-quantize --imatrix``
    consumes them: every row of length ``weight.shape[-1]`` fits
    against its imatrix column weights. With 1-D weights every row
    shares them. With 2-D weights on a 3-D fused expert stack,
    expert ``i``'s rows fit against weight row ``i``. Nominal 2 and
    8 route to the unassisted port — ``quantize_q2_0`` and
    ``quantize_q8_0`` discard the imatrix. The input is never
    modified. The computation runs on the input's device first and
    retries on the CPU when the float32 workspace does not fit the
    card.

    Args:
        weight: The tensor to perturb. Any shape, any float dtype.
        bits: Nominal precision — 4 (weighted ``Q4_0``), or 2 or 8
            (the reference path).
        quant_weights: Imatrix column weights — 1-D of the row
            length, or 2-D ``(experts, row)`` against a 3-D fused
            expert stack. Finite and non-negative.

    Returns:
        The dequantized tensor, same shape, dtype, and device as the
        input.

    Raises:
        ValueError: If ``bits`` has no q0 port (the refusal names
            the covered set), or `check_q0_weights` refuses the
            weights. Every weight check runs for 2- and 8-bit too,
            before those routes discard the weights — validity must
            not depend on which branch consumes the argument.

    Examples:
        The assisted fit differs from the reference at nominal 4:

        ```python
        import torch

        w = torch.randn(4, 512)
        qw = torch.rand(512)
        assisted = q0_assisted_quantize_dequantize(w, 4, qw)
        ```
    """
    check_q0_weights(weight, quant_weights)
    if bits != _WEIGHTED_BITS:
        return q0_ref_quantize_dequantize(weight, bits)
    row = int(weight.shape[-1])

    def prepare(device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build float32 rows and the per-matrix weights on ``device``.

        Args:
            device: Where the workspace copies live.

        Returns:
            ``(rows, qw)`` shaped ``(n, row)`` and
            ``(matrices, row)``.
        """
        rows = weight.detach().to(device=device, dtype=torch.float32).reshape(-1, row)
        qw = quant_weights.detach().to(device=device, dtype=torch.float32)
        return rows, qw.reshape(1, row) if qw.dim() == 1 else qw

    def run(rows: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
        """Fit bounded slices of whole rows against their weights.

        Every slice holds whole rows, so a boundary never splits a
        row's ``sigma2``. One shared weight row broadcasts instead
        of gathering a slice-sized copy.

        Args:
            rows: Shape ``(n, row)``, float32.
            qw: Shape ``(matrices, row)`` — each matrix covers
                ``n / matrices`` consecutive rows.

        Returns:
            Dequantized values, shape of ``rows``.
        """
        per_matrix = rows.shape[0] // qw.shape[0]

        def weight_rows(start: int, stop: int) -> torch.Tensor:
            """Select the weight row for each row in ``[start, stop)``.

            Args:
                start: First row index of the slice.
                stop: Past-the-end row index of the slice.

            Returns:
                Weights that broadcast against the slice — the one
                shared row stays ``(1, row)`` rather than gathering
                a slice-sized copy.
            """
            if qw.shape[0] == 1:
                return qw
            idx = torch.arange(start, stop, device=rows.device) // per_matrix
            return qw[idx]

        chunk = max(1, (_CHUNK_ROWS * QK4_0) // row)
        if rows.shape[0] <= chunk:
            return _assisted_rows(rows, weight_rows(0, rows.shape[0]))
        out = torch.empty_like(rows)
        for start in range(0, rows.shape[0], chunk):
            stop = min(start + chunk, rows.shape[0])
            out[start:stop] = _assisted_rows(rows[start:stop], weight_rows(start, stop))
        return out

    try:
        result = run(*prepare(weight.device))
    except torch.cuda.OutOfMemoryError:
        result = run(*prepare("cpu"))
    return result.reshape(weight.shape).to(device=weight.device, dtype=weight.dtype)
