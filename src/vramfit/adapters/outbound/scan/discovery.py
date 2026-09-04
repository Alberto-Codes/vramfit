"""Construction helpers for the torch meter: groups, memory, counts.

Split out of [vramfit.adapters.outbound.scan.meter][] to keep that
module inside the size cap. Discovery walks the loaded model's
parameters and filters them. The naming rule itself lives in
[vramfit.domain.scan][] (`group_key`), and so does the class rule
that skips a parameter no quantizer touches
([vramfit.domain.runtime][], `unquantizable_class`, #204), so the
fast suite covers every granularity without torch.

Examples:
    Group a loaded model the way the meter does:

    ```python
    groups = discover_groups(model, "layer")
    print(list(groups))
    ```

See Also:
    - [vramfit.adapters.outbound.scan.meter][]: The `DamageMeter`
      adapter these helpers assemble.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch

from vramfit.adapters.outbound.scan.imatrix import expert_stack_count_vectors
from vramfit.domain.model import ImatrixCountSummary
from vramfit.domain.runtime import unquantizable_class
from vramfit.domain.scan import (
    group_key,
    matches_a_layer,
    summarize_imatrix_counts,
)


def discover_groups(
    model: torch.nn.Module, group_by: Literal["layer", "tensor", "stack"]
) -> dict[str, list[str]]:
    """Group the model's quantizable parameters.

    Args:
        model: The loaded model.
        group_by: Grouping granularity, passed through to `group_key`.

    Returns:
        Ordered mapping of group name to member parameter names. Only
        floating-point tensors with 2+ dimensions are included, and a
        tensor of a class the quantizer refuses stays out: the 30B
        target's 23 ``mixer.conv1d`` weights are 3-D, and the pack
        holds them at the convert dtype whatever the map says (#204).
        The class is read off the tensor-granularity group name, so
        the skip applies under every granularity.

    Raises:
        ValueError: If no quantizable parameters are found, or
            ``layer`` grouping finds no per-layer structure — silently
            degrading to per-tensor groups would misrepresent the map.
    """
    groups: dict[str, list[str]] = {}
    layer_matches = 0
    for name, param in model.named_parameters():
        if param.ndim < 2 or not param.is_floating_point():  # noqa: PLR2004
            continue
        if unquantizable_class(group_key(name, "tensor")) is not None:
            continue
        layer_matches += group_by == "layer" and matches_a_layer(name)
        groups.setdefault(group_key(name, group_by), []).append(name)
    if not groups:
        raise ValueError(f"no quantizable parameters found in {model.__class__}")
    if group_by == "layer" and layer_matches == 0:
        raise ValueError(
            "no per-layer structure found in this model's parameter names — "
            "pass --group-by tensor"
        )
    return groups


def max_memory_map(
    device: str, max_gpu_memory: int | None
) -> dict[int | str, int] | None:
    """Build the accelerate ``max_memory`` map for a GPU shard cap.

    The cap applies to GPU 0 only — the reference box has one card.
    The integer device key is required: accelerate rejects ``"0"``.

    Args:
        device: The ``device_map`` value.
        max_gpu_memory: Byte cap on GPU 0 shards, or None for no cap.

    Returns:
        The map for ``auto`` sharding with a cap, otherwise None.
    """
    if max_gpu_memory is None or device != "auto":
        return None
    return {0: max_gpu_memory, "cpu": 999 * 2**30}


def group_count_summaries(
    counts: Mapping[str, int | tuple[int, ...]],
    groups: Mapping[str, list[str]],
) -> dict[str, ImatrixCountSummary]:
    """Pool each group's resolved expert-stack vectors into a summary.

    The meter's half of ADR-0026 decision 4: select each group's
    vectors through `expert_stack_count_vectors` (all or nothing per
    group, the #201 amendment) and reduce through
    `summarize_imatrix_counts`. A group that selects nothing records
    no entry, so its map field stays absent.

    Args:
        counts: Resolved counts per parameter name, from
            `resolve_imatrix_counts`.
        groups: Member parameter names per group name.

    Returns:
        One summary per group that resolved every expert-stack
        member. Empty when nothing resolved.
    """
    return {
        group: summarize_imatrix_counts(vectors)
        for group, members in groups.items()
        if (vectors := expert_stack_count_vectors(counts, members)) is not None
    }
