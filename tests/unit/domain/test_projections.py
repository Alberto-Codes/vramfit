"""The guards that keep the merged-projection reconciliation honest.

[vramfit.domain.projections][] reconciles the leaf half of the map-to-
checkpoint name gap (issue #576). The fold fires only where the
checkpoint states every half, so these pin each condition that holds
it back, the provenance the reconciled map records, and the two
directions the same table drives: the checkpoint's names into the
recipe, and the recipe's names back for `vramfit validate`.
"""

from __future__ import annotations

import pytest

from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    ScanMeta,
    SensitivityMap,
)
from vramfit.domain.projections import (
    MERGED_PROJECTIONS,
    merged_assignments,
    merged_parts,
    reconcile_merged_projections,
    split_assignments,
)

pytestmark = pytest.mark.unit

MERGED = "model.layers.0.mlp.experts.gate_up_proj"
GATE = "model.layers.0.mlp.experts.gate_proj"
UP = "model.layers.0.mlp.experts.up_proj"
DOWN = "model.layers.0.mlp.experts.down_proj"
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


def make_recipe(*assignments: Assignment) -> Recipe:
    """Wrap assignments in the smallest recipe that validates."""
    return Recipe(
        model_id="test/model",
        plan=PlanMeta(
            vram_budget_bytes=100,
            kv_headroom_bytes=10,
            weight_budget_bytes=90,
            predicted_total_bytes=sum(a.bytes for a in assignments),
            predicted_damage=sum(a.damage for a in assignments),
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=assignments,
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


class TestMergedParts:
    """Which names the table rewrites, and which it leaves alone."""

    def test_a_merged_leaf_names_both_checkpoint_projections(self) -> None:
        assert merged_parts(MERGED) == (GATE, UP)

    def test_an_unmerged_leaf_names_nothing(self) -> None:
        assert merged_parts(DOWN) == ()

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


class TestTheFold:
    """What the reconciliation carries after it folds a checkpoint."""

    def test_the_halves_sum_under_the_name_the_scan_measured(self) -> None:
        # One measurement stays one group, so the solver ranks the
        # pair as one unit of damage-per-byte and counts its curve
        # once.
        reconciled = reconcile_merged_projections(
            make_map(merged_group()), SPLIT_CHECKPOINT, {}
        )

        assert dict(reconciled.bytes) == {MERGED: 1000}
        assert reconciled.splits == {MERGED: SPLIT_CHECKPOINT}

    def test_the_map_keeps_the_groups_and_curves_the_scan_wrote(self) -> None:
        map_ = make_map(merged_group())

        reconciled = reconcile_merged_projections(map_, SPLIT_CHECKPOINT, {})

        assert [g.name for g in reconciled.sensitivity_map.groups] == [MERGED]
        assert dict(reconciled.sensitivity_map.groups[0].sensitivity) == CURVE

    def test_the_halves_row_width_carries_to_the_merged_name(self) -> None:
        # `refuse_unmeasured_rows` reads the width under the name the
        # solver prices, so the fold moves it there (issue #515).
        reconciled = reconcile_merged_projections(
            make_map(merged_group()), SPLIT_CHECKPOINT, {GATE: 2688, UP: 2688}
        )

        assert dict(reconciled.rows) == {MERGED: 2688}

    def test_the_reconciled_map_records_the_merge_under_derived(self) -> None:
        reconciled = reconcile_merged_projections(
            make_map(merged_group()), SPLIT_CHECKPOINT, {}
        )

        derived = reconciled.sensitivity_map.derived
        assert derived is not None
        assert MERGED in derived
        assert "counts once" in derived

    def test_a_map_that_was_already_derived_keeps_its_own_note(self) -> None:
        reconciled = reconcile_merged_projections(
            make_map(merged_group(), derived="Hand-edited for the 2-bit probe."),
            SPLIT_CHECKPOINT,
            {},
        )

        derived = reconciled.sensitivity_map.derived
        assert derived is not None
        assert derived.startswith("Hand-edited for the 2-bit probe.")

    def test_groups_the_table_does_not_reach_pass_through(self) -> None:
        down = LayerGroup(name=DOWN, tensors=(DOWN,), bytes_fp16=500, sensitivity=CURVE)
        reconciled = reconcile_merged_projections(
            make_map(merged_group(), down), SPLIT_CHECKPOINT | {DOWN: 500}, {}
        )

        assert dict(reconciled.bytes) == {MERGED: 1000, DOWN: 500}


class TestTheFoldHoldsBack:
    """Every condition that leaves both name sets exactly as they were."""

    def test_a_checkpoint_that_stores_the_parameter_merged_folds_nothing(self) -> None:
        # The two surfaces already agree. Folding would discard the
        # checkpoint's own merged entry.
        checkpoint = {MERGED: 1000} | SPLIT_CHECKPOINT

        reconciled = reconcile_merged_projections(
            make_map(merged_group()), checkpoint, {}
        )

        assert dict(reconciled.bytes) == checkpoint
        assert reconciled.splits == {}

    def test_a_checkpoint_missing_one_half_folds_nothing(self) -> None:
        reconciled = reconcile_merged_projections(
            make_map(merged_group()), {GATE: 400}, {}
        )

        assert dict(reconciled.bytes) == {GATE: 400}
        assert reconciled.splits == {}

    def test_a_map_already_measuring_a_half_folds_nothing(self) -> None:
        # Folding would discard a measurement, and a measurement
        # outranks an inherited curve.
        gate = LayerGroup(name=GATE, tensors=(GATE,), bytes_fp16=400, sensitivity=CURVE)
        map_ = make_map(merged_group(), gate)

        reconciled = reconcile_merged_projections(map_, SPLIT_CHECKPOINT, {})

        assert reconciled.sensitivity_map is map_
        assert reconciled.splits == {}

    def test_halves_of_two_row_widths_fold_nothing(self) -> None:
        # One group packs under one type (ADR-0028, issue #515), so
        # two widths have no single answer and the fold stands back.
        reconciled = reconcile_merged_projections(
            make_map(merged_group()), SPLIT_CHECKPOINT, {GATE: 2688, UP: 2048}
        )

        assert reconciled.splits == {}
        assert dict(reconciled.bytes) == SPLIT_CHECKPOINT


class TestSplitAssignments:
    """The recipe rows `pack` addresses, built from one merged row."""

    def test_the_merged_row_becomes_one_row_per_checkpoint_projection(self) -> None:
        # `pack` addresses `ffn_gate_exps` and `ffn_up_exps` by name
        # (#159), so a merged row reaches neither.
        recipe = make_recipe(Assignment(group=MERGED, bits=4, bytes=250, damage=0.01))

        split = split_assignments(recipe, {MERGED: SPLIT_CHECKPOINT})

        assert [a.group for a in split.assignments] == [GATE, UP]
        assert [a.bits for a in split.assignments] == [4, 4]

    def test_the_rows_sum_to_the_merged_rows_bytes(self) -> None:
        recipe = make_recipe(Assignment(group=MERGED, bits=4, bytes=251, damage=0.01))

        split = split_assignments(recipe, {MERGED: SPLIT_CHECKPOINT})

        assert sum(a.bytes for a in split.assignments) == 251
        assert split.plan.predicted_total_bytes == 251

    def test_the_measurement_lands_on_one_row_and_the_sum_counts_it_once(self) -> None:
        # Halving the curve would invent a number the scan never
        # measured, and charging it twice would over-state
        # `predicted_damage`.
        recipe = make_recipe(Assignment(group=MERGED, bits=4, bytes=250, damage=0.01))

        split = split_assignments(recipe, {MERGED: SPLIT_CHECKPOINT})

        assert [a.damage for a in split.assignments] == [0.01, 0.0]
        assert sum(a.damage for a in split.assignments) == 0.01

    def test_rows_outside_the_split_record_pass_through_in_order(self) -> None:
        down = Assignment(group=DOWN, bits=8, bytes=500, damage=0.02)
        recipe = make_recipe(
            Assignment(group=MERGED, bits=4, bytes=250, damage=0.01), down
        )

        split = split_assignments(recipe, {MERGED: SPLIT_CHECKPOINT})

        assert [a.group for a in split.assignments] == [GATE, UP, DOWN]

    def test_an_empty_split_record_returns_the_same_recipe(self) -> None:
        recipe = make_recipe(Assignment(group=DOWN, bits=8, bytes=500, damage=0.02))

        assert split_assignments(recipe, {}) is recipe


class TestMergedAssignments:
    """The fold `vramfit validate` runs before it measures a recipe."""

    def test_the_split_rows_fold_onto_the_name_the_model_reports(self) -> None:
        # The meter discovers `gate_up_proj` on the same
        # `transformers` the scan used, so a recipe naming the halves
        # would otherwise refuse with advice no scan can satisfy.
        assert merged_assignments({GATE: 4, UP: 4, DOWN: 8}, [MERGED, DOWN]) == {
            MERGED: 4,
            DOWN: 8,
        }

    def test_a_model_that_reports_the_halves_folds_nothing(self) -> None:
        rows = {GATE: 4, UP: 4}

        assert merged_assignments(rows, [GATE, UP]) == rows

    def test_a_pair_at_two_precisions_stays_split_for_the_group_check(self) -> None:
        # The model holds one parameter, so two precisions have no
        # single answer. The group check then refuses the recipe.
        rows = {GATE: 4, UP: 2}

        assert merged_assignments(rows, [MERGED]) == rows

    def test_a_recipe_missing_one_half_stays_as_it_is(self) -> None:
        assert merged_assignments({GATE: 4}, [MERGED]) == {GATE: 4}

    def test_a_recipe_that_already_names_the_merged_group_is_untouched(self) -> None:
        assert merged_assignments({MERGED: 4}, [MERGED]) == {MERGED: 4}
