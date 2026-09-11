"""The guards that keep the merged-projection split from inventing names.

[vramfit.domain.projections][] reconciles the leaf half of the map-to-
checkpoint name gap (issue #576). The split fires only where the
checkpoint states every half, so these pin each condition that holds
it back, and the provenance the reconciled map records.
"""

from __future__ import annotations

import pytest

from vramfit.domain.model import LayerGroup, ScanMeta, SensitivityMap
from vramfit.domain.projections import (
    MERGED_PROJECTIONS,
    merged_parts,
    split_merged_projections,
)

pytestmark = pytest.mark.unit

MERGED = "model.layers.0.mlp.experts.gate_up_proj"
GATE = "model.layers.0.mlp.experts.gate_proj"
UP = "model.layers.0.mlp.experts.up_proj"
CURVE = {8: 0.001, 2: 0.500}
SPLIT_CHECKPOINT = {GATE: 400, UP: 600}


def make_map(*groups: LayerGroup, derived: str | None = None) -> SensitivityMap:
    """Build a stack-granularity map over the given groups."""
    return SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="wikitext",
            calibration_tokens=1024,
            precisions=(8, 2),
            group_by="stack",
            started_at="2026-09-11T00:00:00Z",
        ),
        groups=groups,
        derived=derived,
    )


def merged_group(name: str = MERGED, **kwargs) -> LayerGroup:
    """One merged group, sized as gate and up together."""
    return LayerGroup(
        name=name,
        tensors=kwargs.pop("tensors", (name,)),
        bytes_fp16=kwargs.pop("bytes_fp16", 1000),
        sensitivity=CURVE,
        **kwargs,
    )


class TestMergedParts:
    """Which names the table rewrites, and which it leaves alone."""

    def test_a_merged_leaf_names_both_checkpoint_projections(self) -> None:
        assert merged_parts(MERGED) == (GATE, UP)

    def test_an_unmerged_leaf_names_nothing(self) -> None:
        assert merged_parts("model.layers.0.mlp.experts.down_proj") == ()

    def test_only_the_leaf_is_rewritten_so_a_prefix_cannot_cross_towers(self) -> None:
        # #177 measured what a prefix wildcard costs. The rewrite
        # touches the last segment only, so a vision tower's merged
        # projection can only ever name that tower's own halves.
        tower = "model.vision_tower.layers.0.mlp.gate_up_proj"
        assert merged_parts(tower) == (
            "model.vision_tower.layers.0.mlp.gate_proj",
            "model.vision_tower.layers.0.mlp.up_proj",
        )

    def test_a_bare_leaf_with_no_prefix_names_nothing(self) -> None:
        assert merged_parts("gate_up_proj") == ()

    def test_the_table_is_a_closed_list_of_known_merges(self) -> None:
        assert dict(MERGED_PROJECTIONS) == {"gate_up_proj": ("gate_proj", "up_proj")}


class TestTheSplit:
    """What the reconciled map carries after a split."""

    def test_each_half_takes_its_own_bytes_from_the_checkpoint(self) -> None:
        reconciled, _ = split_merged_projections(
            make_map(merged_group()), SPLIT_CHECKPOINT
        )

        assert {g.name: g.bytes_fp16 for g in reconciled.groups} == SPLIT_CHECKPOINT

    def test_each_half_inherits_the_merged_damage_curve_verbatim(self) -> None:
        # Dividing the measured damage would invent a number the scan
        # never measured. The curve carries over whole instead.
        reconciled, _ = split_merged_projections(
            make_map(merged_group()), SPLIT_CHECKPOINT
        )

        for group in reconciled.groups:
            assert dict(group.sensitivity) == CURVE

    def test_the_reconciled_map_records_the_split_under_derived(self) -> None:
        reconciled, split = split_merged_projections(
            make_map(merged_group()), SPLIT_CHECKPOINT
        )

        assert split == ((MERGED, (GATE, UP)),)
        assert reconciled.derived is not None
        assert MERGED in reconciled.derived
        assert "inherits its merged group's damage curve" in reconciled.derived

    def test_a_map_that_was_already_derived_keeps_its_own_note(self) -> None:
        reconciled, _ = split_merged_projections(
            make_map(merged_group(), derived="Hand-edited for the 2-bit probe."),
            SPLIT_CHECKPOINT,
        )

        assert reconciled.derived is not None
        assert reconciled.derived.startswith("Hand-edited for the 2-bit probe.")

    def test_a_split_half_records_its_own_tensor_size_when_the_map_carries_them(
        self,
    ) -> None:
        group = merged_group(bytes_fp16=1000, tensor_bytes={MERGED: 1000})
        reconciled, _ = split_merged_projections(make_map(group), SPLIT_CHECKPOINT)

        assert {g.name: dict(g.tensor_bytes) for g in reconciled.groups} == {
            GATE: {GATE: 400},
            UP: {UP: 600},
        }

    def test_groups_the_table_does_not_reach_pass_through_in_order(self) -> None:
        down = LayerGroup(
            name="model.layers.0.mlp.experts.down_proj",
            tensors=("model.layers.0.mlp.experts.down_proj",),
            bytes_fp16=500,
            sensitivity=CURVE,
        )
        reconciled, _ = split_merged_projections(
            make_map(merged_group(), down), SPLIT_CHECKPOINT | {down.name: 500}
        )

        assert [g.name for g in reconciled.groups] == [GATE, UP, down.name]


class TestTheSplitHoldsBack:
    """Every condition that leaves the map exactly as it was."""

    def test_no_checkpoint_splits_nothing(self) -> None:
        # Without a size source no name is authoritative, so the map
        # keeps the names the loaded model reported.
        map_ = make_map(merged_group())

        reconciled, split = split_merged_projections(map_, None)

        assert reconciled is map_
        assert split == ()

    def test_a_checkpoint_that_stores_the_parameter_merged_splits_nothing(self) -> None:
        # The two surfaces already agree. Splitting would name groups
        # the checkpoint does not hold.
        map_ = make_map(merged_group())

        reconciled, split = split_merged_projections(
            map_, {MERGED: 1000} | SPLIT_CHECKPOINT
        )

        assert reconciled is map_
        assert split == ()

    def test_a_checkpoint_missing_one_half_splits_nothing(self) -> None:
        map_ = make_map(merged_group())

        reconciled, split = split_merged_projections(map_, {GATE: 400})

        assert reconciled is map_
        assert split == ()

    def test_a_map_already_measuring_a_half_splits_nothing(self) -> None:
        # The split would collide with a measured group, and a
        # measurement outranks an inherited curve.
        gate = LayerGroup(name=GATE, tensors=(GATE,), bytes_fp16=400, sensitivity=CURVE)
        map_ = make_map(merged_group(), gate)

        reconciled, split = split_merged_projections(map_, SPLIT_CHECKPOINT)

        assert reconciled is map_
        assert split == ()
