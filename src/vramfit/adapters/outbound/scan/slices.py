"""Slice cell validation for the slice perturbation path.

ADR-0026's 2026-08-13 #200 amendment rules the mechanism: on a fused
expert layout a per-expert cell is a dim-0 slice of the parameter,
not a name. The meter quantizes the slice in place and measures
damage as usual ([vramfit.adapters.outbound.scan.meter][],
`measure_slices`). This module vouches a slice cell against the
loaded parameters before any weight changes, so a malformed cell
refuses cleanly instead of perturbing the wrong weights.

Examples:
    Vouch one single-expert cell:

    ```python
    check_slice_cell({"model.layers.7.mixer.experts.up_proj": (5, 6)}, params)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.meter][]: The `DamageMeter`
      adapter that applies validated slice cells.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

_STACK_NDIM = 3


def check_slice_cell(
    slices: Mapping[str, tuple[int, int]],
    params: Mapping[str, torch.Tensor],
) -> None:
    """Vouch one slice cell against the loaded parameters.

    Args:
        slices: Half-open expert index range per fused expert stack,
            keyed by the loaded parameter name.
        params: Every quantizable parameter the meter discovered,
            keyed by the loaded parameter name.

    Raises:
        ValueError: If ``slices`` is empty, names a parameter the
            meter did not discover, names a parameter that is not a
            fused expert stack, or a range is empty or out of
            bounds.
    """
    if not slices:
        raise ValueError("slices must not be empty")
    for name, (low, high) in slices.items():
        param = params.get(name)
        if param is None:
            raise ValueError(f'unknown parameter "{name}"')
        if param.ndim != _STACK_NDIM:
            raise ValueError(
                f'"{name}" is {param.ndim}D — a slice cell needs a fused '
                "expert stack, 3D and expert-indexed on dim 0 (ADR-0026)"
            )
        if not 0 <= low < high <= param.shape[0]:
            raise ValueError(
                f"slice [{low}, {high}) is not a valid expert range for "
                f'"{name}" with {param.shape[0]} experts'
            )
