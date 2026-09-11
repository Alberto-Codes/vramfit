"""Merged projections: the second name gap between a map and a checkpoint.

`plan --checkpoint` matches the map's group names against the
checkpoint's by exact string. ADR-0029 decision 7 reconciles the
naming *root* before that match. This module reconciles the *leaf*,
which is the gap issue #576 measured.

`vramfit scan` names a group from the loaded `transformers` model.
`vramfit plan --checkpoint` names it from the safetensors headers.
The two disagree whenever the installed `transformers` loads several
of the checkpoint's projections as one parameter. Qwen3-MoE on
`transformers` 5.16.1 loads gate and up as one 3-D
`mlp.experts.gate_up_proj`, while the checkpoint keeps
`mlp.experts.gate_proj` and `mlp.experts.up_proj` apart. A finished
384-cell scan of Qwen3-Coder-30B-A3B matched 21 of its 43 expert
names against 144 checkpoint names, so 96 checkpoint groups held at
reference precision and the solver refused a budget it could not
reach. The refusal advised a scan those names can never produce.

The merge is a `transformers`-version property and not a model
property, so a version bound would hide the gap and an upgrade would
return it. `split_merged_projections` therefore reconciles the names
and reads no version.

**A split group inherits its damage, and the map says so.** The scan
perturbed gate and up together, so one measurement covers both
halves. Each split group carries that curve verbatim — the split
invents no damage number, and dividing one would invent one. The
reconciled map records the split under `SensitivityMap.derived`,
naming every group it split, so a reader tells an inherited curve
from a measured one. `plan` echoes the same note.

Two consequences follow, and both are stated rather than hidden. The
solver may assign the halves different precisions, because it sees
two groups whose curves happen to agree. And `predicted_damage`
counts the merged measurement once per half, which over-states it —
the direction ADR-0006 calls safe.

Attributes:
    MERGED_PROJECTIONS (Mapping[str, tuple[str, ...]]): Loaded
        parameter leaf to the checkpoint leaves it holds. The table
        is explicit and closed, like `NAME_TABLE_ROOTS` and
        `CHECKPOINT_ROOTS`. A prefix wildcard would map one
        projection onto another family's (#177), so each new merge
        costs one entry.
    DERIVED_NOTE (str): The sentence the reconciled map records
        under `SensitivityMap.derived`, before the split group names.

Examples:
    Split a merged scan group against a checkpoint that keeps the
    halves apart:

    ```python
    from vramfit.domain.projections import merged_parts

    assert merged_parts("model.layers.0.mlp.experts.gate_up_proj") == (
        "model.layers.0.mlp.experts.gate_proj",
        "model.layers.0.mlp.experts.up_proj",
    )
    ```

See Also:
    - [vramfit.domain.sizes][]: `reconcile_root`, the root half of
      the same reconciliation.
    - [vramfit.domain.solver][]: Prices the reconciled map.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from vramfit.domain.model import LayerGroup, SensitivityMap

# Transformers loads Qwen3-MoE gate and up as one 3-D parameter, and
# the checkpoint keeps them apart (#576). The table is closed on
# purpose: the split only fires on a leaf named here, so no wildcard
# maps one family's projection onto another's (#177).
MERGED_PROJECTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"gate_up_proj": ("gate_proj", "up_proj")}
)

DERIVED_NOTE: Final[str] = (
    "Derived by vramfit plan --checkpoint: split merged projections the "
    "loaded model fused and the checkpoint keeps apart (#576). Each split "
    "group inherits its merged group's damage curve verbatim and takes its "
    "own bytes from the checkpoint. Not a scan artifact. Split groups: "
)


def merged_parts(group: str) -> tuple[str, ...]:
    """Name the checkpoint groups one merged group covers.

    The match reads the group name's last dot-separated segment
    against `MERGED_PROJECTIONS` and rewrites only that segment. The
    prefix passes through untouched, so a vision tower's projection
    can never name a decoder group.

    Args:
        group: A group name, as `vramfit.domain.scan.group_key`
            produces it.

    Returns:
        One name per checkpoint projection the merged parameter
        holds, in table order. Empty when the name carries no merged
        leaf, and empty for a bare leaf with no prefix.

    Examples:
        ```python
        from vramfit.domain.projections import merged_parts

        assert merged_parts("model.layers.3.mlp.experts.gate_up_proj") == (
            "model.layers.3.mlp.experts.gate_proj",
            "model.layers.3.mlp.experts.up_proj",
        )
        assert merged_parts("model.layers.3.mlp.experts.down_proj") == ()
        ```
    """
    prefix, dot, leaf = group.rpartition(".")
    parts = MERGED_PROJECTIONS.get(leaf)
    if not dot or parts is None:
        return ()
    return tuple(f"{prefix}.{part}" for part in parts)


def _splittable(
    group: LayerGroup, discovered_bytes: Mapping[str, int], covered: frozenset[str]
) -> tuple[str, ...]:
    """Decide whether one map group splits against this checkpoint.

    Four conditions hold together. The name carries a merged leaf.
    The checkpoint does not carry the merged name itself, because a
    checkpoint that stores the parameter merged already agrees with
    the map. The checkpoint carries every half, so the split never
    invents a name no checkpoint states. And the map carries none of
    the halves, because a map holding both spellings measures them
    and the split would collide with a measured group.

    Args:
        group: One map group.
        discovered_bytes: Bytes at reference precision per group the
            checkpoint holds.
        covered: Every group name the map carries.

    Returns:
        The checkpoint group names to split into, or empty when this
        group stays as it is.
    """
    parts = merged_parts(group.name)
    if not parts or group.name in discovered_bytes:
        return ()
    if any(part not in discovered_bytes or part in covered for part in parts):
        return ()
    return parts


def split_merged_projections(
    sensitivity_map: SensitivityMap,
    discovered_bytes: Mapping[str, int] | None,
) -> tuple[SensitivityMap, tuple[tuple[str, tuple[str, ...]], ...]]:
    """Reconcile a map's merged projections against a checkpoint.

    Each merged group becomes one group per checkpoint projection it
    covers. A split group inherits the merged group's damage curve
    verbatim and takes its bytes from the checkpoint, so the plan
    prices what the checkpoint holds and the solver reaches it. Every
    other group passes through unchanged, and the group order holds.

    The reconciled map records the split under
    `SensitivityMap.derived`, appending to a note the map already
    carried. A map with nothing to split returns unchanged, `derived`
    included.

    Args:
        sensitivity_map: The map `plan` loaded.
        discovered_bytes: Bytes at reference precision per group the
            checkpoint holds, from
            `vramfit.domain.sizes.discovered_group_bytes`, or None
            when the caller read no checkpoint. None splits nothing:
            without a checkpoint no name is authoritative.

    Returns:
        The reconciled map, and one ``(merged group, split names)``
        pair per group split, in map order. The pairs are empty when
        the map and the checkpoint already agree.

    Examples:
        ```python
        from vramfit.domain.model import LayerGroup, ScanMeta, SensitivityMap
        from vramfit.domain.projections import split_merged_projections

        merged = "model.layers.0.mlp.experts.gate_up_proj"
        map_ = SensitivityMap(
            model_id="m",
            scan=ScanMeta("kl_divergence", "wikitext", 1, (8,), "stack", "t"),
            groups=(
                LayerGroup(
                    name=merged, tensors=(merged,), bytes_fp16=8, sensitivity={8: 0.5}
                ),
            ),
        )
        checkpoint = {
            "model.layers.0.mlp.experts.gate_proj": 4,
            "model.layers.0.mlp.experts.up_proj": 4,
        }
        reconciled, split = split_merged_projections(map_, checkpoint)
        assert [g.name for g in reconciled.groups] == sorted(checkpoint)
        assert reconciled.groups[0].sensitivity[8] == 0.5
        assert split == ((merged, tuple(sorted(checkpoint))),)
        ```
    """
    if discovered_bytes is None:
        return sensitivity_map, ()
    covered = frozenset(group.name for group in sensitivity_map.groups)
    groups: list[LayerGroup] = []
    split: list[tuple[str, tuple[str, ...]]] = []
    for group in sensitivity_map.groups:
        parts = _splittable(group, discovered_bytes, covered)
        if not parts:
            groups.append(group)
            continue
        groups.extend(_split_group(group, parts, discovered_bytes))
        split.append((group.name, parts))
    if not split:
        return sensitivity_map, ()
    names = ", ".join(name for name, _ in split)
    note = f"{DERIVED_NOTE}{names}."
    if sensitivity_map.derived is not None:
        note = f"{sensitivity_map.derived} {note}"
    reconciled = SensitivityMap(
        model_id=sensitivity_map.model_id,
        scan=sensitivity_map.scan,
        groups=tuple(groups),
        derived=note,
    )
    return reconciled, tuple(split)


def _split_group(
    group: LayerGroup, parts: tuple[str, ...], discovered_bytes: Mapping[str, int]
) -> tuple[LayerGroup, ...]:
    """Build the checkpoint-named groups one merged group becomes.

    Each part names itself as its own tensor, which mirrors what the
    merged group carried: the meter records the loaded parameter's
    name, and that name is the group name. The checkpoint states each
    part's bytes, so the split reads a measurement rather than
    halving the merged total.

    Args:
        group: The merged map group.
        parts: The checkpoint group names it covers.
        discovered_bytes: Bytes at reference precision per group the
            checkpoint holds.

    Returns:
        One group per part, in table order.
    """
    return tuple(
        LayerGroup(
            name=part,
            tensors=(part,),
            bytes_fp16=discovered_bytes[part],
            sensitivity=group.sensitivity,
            tensor_bytes={part: discovered_bytes[part]} if group.tensor_bytes else {},
            imatrix_counts=group.imatrix_counts,
        )
        for part in parts
    )
