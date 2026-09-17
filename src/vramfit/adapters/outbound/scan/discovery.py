"""Construction helpers for the torch meter: groups, memory, counts.

Split out of [vramfit.adapters.outbound.scan.meter][] to keep that
module inside the size cap. Discovery walks the loaded model's
parameters and filters them, then assembles the map-ready specs:
`group_specs` carries each group's reference bytes, per-tensor
sizes, count summary, and the row width `group_row_widths` measured
(issue #558). The naming rule itself lives in
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

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import torch

from vramfit.adapters.outbound.scan.imatrix import expert_stack_count_vectors
from vramfit.domain.model import ImatrixCountSummary
from vramfit.domain.runtime import routes_by_row_width, unquantizable_class
from vramfit.domain.scan import (
    GroupSpec,
    group_key,
    matches_a_layer,
    summarize_imatrix_counts,
)
from vramfit.domain.sizes import fold_row_width


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


def group_row_widths(
    groups: Mapping[str, Sequence[str]],
    parameter: Callable[[str], torch.Tensor],
) -> dict[str, int]:
    """Measure the row width of every group the map should record.

    The scan already reads each row width to refuse a cell whose
    mapped type cannot tile the rows (ADR-0018). Issue #558 keeps
    that number: a map that records it routes the 256 super-block
    decision with no checkpoint beside it.

    The gate mirrors `vramfit.domain.sizes.discovered_group_rows`, so
    the map and a checkpoint read cover the same groups and their
    widths compare directly. A whole-layer group holds classes of
    several widths, so it records none and keeps the ADR-0012
    k-quant table.

    Args:
        groups: Group name to its member parameter names, as
            `discover_groups` built it.
        parameter: Resolves a member's name to its loaded tensor.

    Returns:
        Elements per row per group name. Groups the decision does not
        reach are absent.

    Raises:
        SizeSourceError: If two members of one group state different
            row widths. One group packs under one type, so two widths
            have no single answer.

    Examples:
        ```python
        widths = group_row_widths(groups, model.get_parameter)
        ```
    """
    rows: dict[str, int] = {}
    for group, members in groups.items():
        if not routes_by_row_width(group):
            continue
        for tensor in members:
            fold_row_width(rows, group, tensor, int(parameter(tensor).shape[-1]))
    return rows


def group_specs(
    groups: Mapping[str, Sequence[str]],
    parameter: Callable[[str], torch.Tensor],
    summaries: Mapping[str, ImatrixCountSummary],
) -> tuple[GroupSpec, ...]:
    """Assemble the meter's discovered groups into map-ready specs.

    One pass carries everything the map records about a group that no
    measurement supplies: its reference bytes, its per-tensor sizes
    for the protection pricing (ADR-0022), its pooled expert-stack
    count summary (ADR-0026 decision 4), and its measured row width
    (issue #558).

    Args:
        groups: Group name to its member parameter names, as
            `discover_groups` built it.
        parameter: Resolves a member's name to its loaded tensor.
        summaries: Pooled count summary per group, empty for an
            unassisted meter.

    Returns:
        One spec per group, in module order.

    Raises:
        SizeSourceError: If two members of one group state different
            row widths (issue #515).

    Examples:
        ```python
        specs = group_specs(groups, model.get_parameter, {})
        ```
    """
    rows = group_row_widths(groups, parameter)
    return tuple(
        GroupSpec(
            name=name,
            tensors=tuple(tensors),
            bytes_fp16=sum(parameter(t).numel() * 2 for t in tensors),
            tensor_bytes={t: parameter(t).numel() * 2 for t in tensors},
            imatrix_counts=summaries.get(name),
            row_width=rows.get(name),
        )
        for name, tensors in groups.items()
    )
