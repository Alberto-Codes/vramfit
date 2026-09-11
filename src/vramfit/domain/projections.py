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
return it. `MERGED_PROJECTIONS` therefore records the merge itself,
and every surface reads that one table.

**One measurement stays one group.** The scan perturbed gate and up
together, so one damage curve covers both halves.
`reconcile_merged_projections` folds the checkpoint's halves onto the
merged name the map carries, so the solver prices one group, counts
that measurement once, and ranks the pair as a unit for
damage-per-byte. `split_assignments` then names the checkpoint's
projections in the recipe, because `pack` addresses `ffn_gate_exps`
and `ffn_up_exps` separately (#159). The split rows share the pair's
precision. The first row carries the measured damage and the rest
carry 0.0, the value this codebase records for a row no measurement
prices (ADR-0029 decision 3). Nothing divides the curve, which would
invent a number the scan never measured.

`merged_assignments` runs the same table the other way. `vramfit
validate` discovers `gate_up_proj` from the loaded model, so it folds
the recipe's split rows back onto that name before it measures.

Attributes:
    MERGED_PROJECTIONS (Mapping[str, tuple[str, ...]]): Loaded
        parameter leaf to the checkpoint leaves it holds. The table
        is explicit and closed, like `NAME_TABLE_ROOTS` and
        `CHECKPOINT_ROOTS`. A prefix wildcard would map one
        projection onto another family's (#177), so each new merge
        costs one entry.
    DERIVED_NOTE (str): The sentence the reconciled map records
        under `SensitivityMap.derived`, before the merged group
        names. The note reaches the `plan` echo and no artifact —
        the recipe schema carries no provenance field for it.

Examples:
    Name the checkpoint projections one merged parameter holds:

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
    - [vramfit.domain.solver][]: Prices the reconciled groups.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from vramfit.domain.model import Assignment, Recipe, SensitivityMap

# Transformers loads Qwen3-MoE gate and up as one 3-D parameter, and
# the checkpoint keeps them apart (#576). The table is closed on
# purpose: the fold only fires on a leaf named here, so no wildcard
# maps one family's projection onto another's (#177).
MERGED_PROJECTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"gate_up_proj": ("gate_proj", "up_proj")}
)

DERIVED_NOTE: Final[str] = (
    "Derived by vramfit plan --checkpoint: reconciled merged projections the "
    "loaded model fused and the checkpoint keeps apart (#576). The plan "
    "prices each merged projection as one group, so its one measurement "
    "counts once, and the recipe names the checkpoint's projections. Not a "
    "scan artifact. Merged projections: "
)


@dataclass(frozen=True, slots=True)
class MergedReconciliation:
    """A map and a checkpoint, reconciled on their merged projections.

    Attributes:
        sensitivity_map (SensitivityMap): The map, carrying
            `DERIVED_NOTE` when anything was reconciled. Its groups
            are untouched: the merged name is the one the solver
            prices.
        bytes (Mapping[str, int]): Bytes at reference precision per
            checkpoint group, with each merged projection's halves
            summed under the merged name.
        rows (Mapping[str, int]): Elements per row per checkpoint
            group, keyed the same way.
        splits (Mapping[str, Mapping[str, int]]): Merged group name
            to the checkpoint projections it holds and their
            reference bytes, in table order. Empty when the two name
            sets already agree.

    Examples:
        ```python
        reconciled = reconcile_merged_projections(map_, bytes_, rows)
        recipe = split_assignments(solve(reconciled.sensitivity_map), ...)
        ```
    """

    sensitivity_map: SensitivityMap
    bytes: Mapping[str, int]
    rows: Mapping[str, int]
    splits: Mapping[str, Mapping[str, int]]


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


def _foldable(
    group: str,
    discovered_bytes: Mapping[str, int],
    row_widths: Mapping[str, int],
    covered: frozenset[str],
) -> tuple[str, ...]:
    """Decide whether one map group folds this checkpoint's halves.

    Five conditions hold together. The name carries a merged leaf.
    The checkpoint does not carry the merged name itself, because a
    checkpoint that stores the parameter merged already agrees with
    the map. The checkpoint carries every half, so the fold never
    reads a name no checkpoint states. The map carries none of the
    halves, because a map holding both spellings measures them and
    the fold would discard a measurement. And the halves state one
    row width or none, because one group packs under one type
    (ADR-0028, issue #515).

    Args:
        group: One map group's name.
        discovered_bytes: Bytes at reference precision per group the
            checkpoint holds.
        row_widths: Elements per row per group the checkpoint holds.
        covered: Every group name the map carries.

    Returns:
        The checkpoint group names to fold, or empty when this group
        stays as it is.
    """
    parts = merged_parts(group)
    if not parts or group in discovered_bytes:
        return ()
    if any(part not in discovered_bytes or part in covered for part in parts):
        return ()
    measured = [row_widths[part] for part in parts if part in row_widths]
    if measured and (len(measured) != len(parts) or len(set(measured)) != 1):
        return ()
    return parts


def reconcile_merged_projections(
    sensitivity_map: SensitivityMap,
    discovered_bytes: Mapping[str, int],
    row_widths: Mapping[str, int],
) -> MergedReconciliation:
    """Fold a checkpoint's split projections onto the map's merged names.

    The scan measured the merged parameter once, so the plan prices
    it once. Each merged projection keeps its map group, and the
    checkpoint's halves sum into that group's checkpoint entry. The
    coverage match then finds the group measured, the solver ranks
    the pair as one unit of damage-per-byte, and
    `plan.predicted_damage` counts the measurement once.

    The map records the reconciliation under
    `SensitivityMap.derived`, appending to a note the map already
    carried. A map with nothing to reconcile returns unchanged,
    `derived` included.

    Args:
        sensitivity_map: The map `plan` loaded.
        discovered_bytes: Bytes at reference precision per group the
            checkpoint holds, from
            `vramfit.domain.sizes.discovered_group_bytes`.
        row_widths: Elements per row per group the checkpoint holds,
            from `vramfit.domain.sizes.discovered_group_rows`.

    Returns:
        The reconciled map, the folded checkpoint sizes and widths,
        and the split record `split_assignments` reads.

    Examples:
        ```python
        from vramfit.domain.projections import reconcile_merged_projections

        merged = "model.layers.0.mlp.experts.gate_up_proj"
        checkpoint = {
            "model.layers.0.mlp.experts.gate_proj": 4,
            "model.layers.0.mlp.experts.up_proj": 4,
        }
        reconciled = reconcile_merged_projections(map_, checkpoint, {})
        assert reconciled.bytes == {merged: 8}
        ```
    """
    covered = frozenset(group.name for group in sensitivity_map.groups)
    sizes = dict(discovered_bytes)
    rows = dict(row_widths)
    splits: dict[str, Mapping[str, int]] = {}
    for group in sensitivity_map.groups:
        parts = _foldable(group.name, discovered_bytes, row_widths, covered)
        if not parts:
            continue
        splits[group.name] = {part: discovered_bytes[part] for part in parts}
        sizes[group.name] = sum(splits[group.name].values())
        for part in parts:
            del sizes[part]
            if part in rows:
                rows[group.name] = rows.pop(part)
    if not splits:
        return MergedReconciliation(
            sensitivity_map, discovered_bytes, row_widths, splits
        )
    note = f"{DERIVED_NOTE}{', '.join(splits)}."
    if sensitivity_map.derived is not None:
        note = f"{sensitivity_map.derived} {note}"
    reconciled = SensitivityMap(
        model_id=sensitivity_map.model_id,
        scan=sensitivity_map.scan,
        groups=sensitivity_map.groups,
        derived=note,
    )
    return MergedReconciliation(reconciled, sizes, rows, splits)


def _split_row(
    assignment: Assignment, parts: Mapping[str, int]
) -> tuple[Assignment, ...]:
    """Name one merged assignment's checkpoint projections.

    The rows share the merged group's precision, because one
    measurement priced them together. Their bytes divide in
    proportion to the reference bytes the checkpoint states, and the
    last row takes the remainder, so the rows sum to the merged
    prediction exactly. The first row carries the measured damage and
    the rest carry 0.0.

    Args:
        assignment: The merged group's assignment.
        parts: Checkpoint projection name to reference bytes, in
            table order.

    Returns:
        One assignment per checkpoint projection.
    """
    reference = sum(parts.values())
    spent = 0
    rows: list[Assignment] = []
    for index, (name, part_bytes) in enumerate(parts.items()):
        last = index == len(parts) - 1
        size = (
            assignment.bytes - spent
            if last
            else assignment.bytes * part_bytes // reference
        )
        spent += size
        rows.append(
            Assignment(
                group=name,
                bits=assignment.bits,
                bytes=size,
                damage=assignment.damage if index == 0 else 0.0,
            )
        )
    return tuple(rows)


def split_assignments(
    recipe: Recipe, splits: Mapping[str, Mapping[str, int]]
) -> Recipe:
    """Name the checkpoint's projections in a recipe's assignments.

    `pack` addresses `ffn_gate_exps` and `ffn_up_exps` separately
    (#159), so a recipe naming the merged parameter reaches neither.
    Every other row passes through unchanged, and the row order
    holds. The budget accounting does not move: the split rows sum to
    the merged row's bytes and carry its damage once.

    Args:
        recipe: The recipe the solver built over the reconciled map.
        splits: Merged group name to its checkpoint projections and
            their reference bytes, from `reconcile_merged_projections`.

    Returns:
        The recipe under the checkpoint's names. The same recipe when
        ``splits`` is empty.

    Examples:
        ```python
        from vramfit.domain.projections import split_assignments

        packable = split_assignments(recipe, reconciled.splits)
        ```
    """
    if not splits:
        return recipe
    rows: list[Assignment] = []
    for assignment in recipe.assignments:
        parts = splits.get(assignment.group)
        if parts is None:
            rows.append(assignment)
            continue
        rows.extend(_split_row(assignment, parts))
    return Recipe(
        model_id=recipe.model_id,
        plan=recipe.plan,
        assignments=tuple(rows),
        runtime=recipe.runtime,
        within_group=recipe.within_group,
        imatrix=recipe.imatrix,
        protected_tensors=recipe.protected_tensors,
    )


def merged_assignments(
    assignments: Mapping[str, int], discovered: Collection[str]
) -> dict[str, int]:
    """Fold a recipe's split rows back onto the names a model reports.

    `vramfit validate` loads the same `transformers` the scan did, so
    it discovers `gate_up_proj` where the recipe names `gate_proj`
    and `up_proj` (#576). Without this fold the pass refuses the
    recipe and advises a scan no version can produce. The rows fold
    only where the model reports the merged name, the recipe names
    every half, and the halves share one precision. A pair at two
    precisions is unmeasurable on this version, so it stays split and
    the group check refuses it.

    Args:
        assignments: Precision per recipe group name.
        discovered: Group names the loaded model reports.

    Returns:
        Precision per group name under the model's own names.

    Examples:
        ```python
        from vramfit.domain.projections import merged_assignments

        merged = "model.layers.0.mlp.experts.gate_up_proj"
        rows = {
            "model.layers.0.mlp.experts.gate_proj": 4,
            "model.layers.0.mlp.experts.up_proj": 4,
        }
        assert merged_assignments(rows, [merged]) == {merged: 4}
        ```
    """
    folded = dict(assignments)
    for group in discovered:
        parts = merged_parts(group)
        if not parts or group in folded:
            continue
        shared = {folded.get(part) for part in parts}
        if len(shared) != 1:
            continue
        bits = shared.pop()
        if bits is None:
            continue
        for part in parts:
            del folded[part]
        folded[group] = bits
    return folded
