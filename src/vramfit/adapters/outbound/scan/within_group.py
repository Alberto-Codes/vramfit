"""Within-group method dispatch (ADR-0018, ADR-0020).

One place decides which quantizer perturbs a cell. The meter holds
the choice as a method name and this module turns it into a round
trip. Keeping the dispatch here means a new method touches one
function rather than the meter's measurement loop.

The methods:

- ``rtn`` — the ADR-0006 v1 round-to-nearest, at 32-element scale
  blocks.
- ``kquant`` — the ported K-quant reference quantizers, assisted by
  an imatrix wherever the parameter carries column weights
  (ADR-0020).
- ``gguf`` — the ported block quantizers ``Q2_0``, ``Q4_0``, and
  ``Q8_0``, which reach the rows no K-quant tiles.

Examples:
    Perturb one tensor under the gguf method:

    ```python
    perturbed = perturb(weight, 2, "blk.0.ffn_up_exps", "gguf", 32, None)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.meter][]: The caller.
"""

from __future__ import annotations

from typing import Literal

import torch

from vramfit.adapters.outbound.scan.gguf_ref import gguf_ref_quantize_dequantize
from vramfit.adapters.outbound.scan.kquant import kquant_quantize_dequantize
from vramfit.adapters.outbound.scan.kquant_assisted import (
    kquant_assisted_quantize_dequantize,
)
from vramfit.adapters.outbound.scan.quantize import rtn_quantize_dequantize

WithinGroupMethod = Literal["rtn", "kquant", "gguf"]
# The method names the meter and the CLI accept. An unknown value
# must refuse rather than fall back — a silent RTN fallback would
# record every damage under the wrong token.
METHODS: tuple[WithinGroupMethod, ...] = ("rtn", "kquant", "gguf")


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
        column_weights: Imatrix column weights for this parameter,
            or None to price unassisted (ADR-0020). Only ``kquant``
            reads them.

    Returns:
        The dequantized tensor, same shape, dtype, and device.

    Raises:
        ValueError: If the method has no port for ``bits`` —
            ``kquant`` covers 8, 4, 3, and 2, and ``gguf`` covers 8,
            4, and 2 — or the mapped type's block size does not
            divide the tensor's row length. The message names the
            parameter.
    """
    try:
        if method == "gguf":
            return gguf_ref_quantize_dequantize(param, bits)
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
