"""Within-group method dispatch (ADR-0018, ADR-0020).

One place decides which quantizer perturbs a cell, and which imatrix
reader family serves a method. The meter holds
the choice as a method name and this module turns it into a round
trip, a weight resolution, and a construction-time weight gate.
Keeping the dispatch here means a new method touches one
module rather than the meter's measurement loop.

The methods:

- ``rtn`` — the ADR-0006 v1 round-to-nearest, at 32-element scale
  blocks.
- ``kquant`` — the ported K-quant reference quantizers, assisted by
  an imatrix wherever the parameter carries column weights
  (ADR-0020).
- ``q0`` — the ported block quantizers ``Q2_0``, ``Q4_0``, and
  ``Q8_0``, which reach the rows no K-quant tiles. With imatrix
  weights, nominal 4 fits through the assisted ``Q4_0`` port
  (ADR-0018, 2026-08-21 amendment).

Examples:
    Perturb one tensor under the q0 method:

    ```python
    name = "model.layers.0.mixer.experts.up_proj"
    perturbed = perturb(weight, 2, name, "q0", 32, None)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.meter][]: The caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from vramfit.adapters.outbound.scan.imatrix import (
    ImatrixEntry,
    check_imatrix_weights,
    resolve_assisted_weights,
)
from vramfit.adapters.outbound.scan.imatrix_q0 import (
    check_q0_imatrix_weights,
    resolve_q0_assisted_weights,
)
from vramfit.adapters.outbound.scan.kquant import kquant_quantize_dequantize
from vramfit.adapters.outbound.scan.kquant_assisted import (
    kquant_assisted_quantize_dequantize,
)
from vramfit.adapters.outbound.scan.q0_assisted import q0_assisted_quantize_dequantize
from vramfit.adapters.outbound.scan.q0_ref import q0_ref_quantize_dequantize
from vramfit.adapters.outbound.scan.quantize import rtn_quantize_dequantize

WithinGroupMethod = Literal["rtn", "kquant", "q0"]
# The method names the meter and the CLI accept. An unknown value
# must refuse rather than fall back — a silent RTN fallback would
# record every damage under the wrong token.
METHODS: tuple[WithinGroupMethod, ...] = ("rtn", "kquant", "q0")


def resolve_method_weights(
    method: WithinGroupMethod,
    by_gguf_name: Mapping[str, ImatrixEntry],
    shapes: Mapping[str, Sequence[int]],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Resolve imatrix weights through the method's reader family.

    One reader serves one method family (ADR-0018, 2026-08-21
    amendment, decision 2): the ``q0`` reader accepts fused expert
    stacks, and the ``kquant`` reader keeps its fused-stack refusal
    and its super-block gate, unchanged.

    Args:
        method: The within-group method name.
        by_gguf_name: Entries keyed by GGUF tensor name, from
            ``load_imatrix``.
        shapes: Parameter shapes keyed by the names the loaded
            model reports.

    Returns:
        ``(covered, uncovered)`` — weights keyed by parameter name,
        and the names the imatrix does not cover, in input order.

    Raises:
        ValueError: If the family's resolver refuses — a shape or
            matrix-count mismatch, two parameters claiming one row,
            or zero coverage.
    """
    if method == "q0":
        return resolve_q0_assisted_weights(by_gguf_name, shapes)
    return resolve_assisted_weights(
        by_gguf_name, {name: int(shape[-1]) for name, shape in shapes.items()}
    )


def check_method_weights(
    method: WithinGroupMethod,
    weights: Mapping[str, torch.Tensor],
    shapes: Mapping[str, Sequence[int]],
) -> None:
    """Gate imatrix weights through the method's reader family.

    The meter runs this at construction over any weight source —
    resolved from a file or passed directly.

    Args:
        method: The within-group method name.
        weights: Weights keyed by HF parameter name.
        shapes: Parameter shapes keyed by discovered parameter name.

    Raises:
        ValueError: If the family's gate refuses — an unknown name,
            a layout or length mismatch, rows the family's blocks
            cannot align, or a negative or non-finite weight.
    """
    if method == "q0":
        check_q0_imatrix_weights(weights, shapes)
    else:
        check_imatrix_weights(
            weights, {name: int(shape[-1]) for name, shape in shapes.items()}
        )


def perturb(
    param: torch.Tensor,
    bits: int,
    name: str,
    method: WithinGroupMethod,
    block_size: int,
    column_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Round-trip one tensor through the selected method.

    Args:
        param: The tensor to perturb.
        bits: Candidate precision.
        name: The parameter's dotted name, for the refusal message.
        method: The within-group method name.
        block_size: Scale-block width for the ``rtn`` method.
        column_weights: Imatrix weights for this parameter, or None
            to price unassisted (ADR-0020). ``kquant`` reads a 1-D
            column vector. ``q0`` also reads a 2-D per-expert
            tensor on a fused expert stack (ADR-0018, 2026-08-21
            amendment). ``rtn`` never reads them.

    Returns:
        The dequantized tensor, same shape, dtype, and device.

    Raises:
        ValueError: If the method has no port for ``bits`` —
            ``kquant`` covers 8, 4, 3, and 2, and ``q0`` covers 8,
            4, and 2 — or the mapped type's block size does not
            divide the tensor's row length. The message names the
            parameter.
    """
    try:
        if method == "q0":
            if column_weights is not None:
                return q0_assisted_quantize_dequantize(param, bits, column_weights)
            return q0_ref_quantize_dequantize(param, bits)
        if method == "kquant":
            if column_weights is not None:
                return kquant_assisted_quantize_dequantize(param, bits, column_weights)
            return kquant_quantize_dequantize(param, bits)
        return rtn_quantize_dequantize(param, bits, block_size)
    except ValueError as exc:
        # A method refusal names the type and the row length. A scan
        # runs hours over hundreds of cells, so it must also name
        # the tensor that stopped it.
        raise ValueError(f"{name}: {exc}") from exc
