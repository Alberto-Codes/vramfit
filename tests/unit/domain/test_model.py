from __future__ import annotations

import math

import pytest

from vramfit.domain.model import (
    Assignment,
    ImatrixCountSummary,
    LayerGroup,
    PlanMeta,
    ProtectedTensor,
    Recipe,
)

pytestmark = pytest.mark.unit


def make_plan() -> PlanMeta:
    return PlanMeta(
        vram_budget_bytes=4_000,
        kv_headroom_bytes=1_000,
        weight_budget_bytes=3_000,
        predicted_total_bytes=2_500,
        predicted_damage=0.05,
        solver="greedy-damage-per-byte",
        pins={},
        protections={},
        format_overhead=0.05,
        trace=(),
    )


class TestRecipeInvariants:
    def test_unique_groups_construct_succeeds(self) -> None:
        recipe = Recipe(
            model_id="test/model",
            plan=make_plan(),
            assignments=(
                Assignment(group="model.layers.0", bits=8, bytes=1_500, damage=0.01),
                Assignment(group="model.layers.1", bits=4, bytes=1_000, damage=0.02),
            ),
            runtime=None,
            within_group=None,
            imatrix=None,
            protected_tensors=(),
        )

        assert len(recipe.assignments) == 2

    def test_duplicate_assignment_groups_raise_value_error(self) -> None:
        duplicated = Assignment(
            group="model.layers.0", bits=4, bytes=1_000, damage=0.02
        )

        with pytest.raises(ValueError, match="unique"):
            Recipe(
                model_id="test/model",
                plan=make_plan(),
                assignments=(
                    Assignment(
                        group="model.layers.0", bits=8, bytes=1_500, damage=0.01
                    ),
                    duplicated,
                ),
                runtime=None,
                within_group=None,
                imatrix=None,
                protected_tensors=(),
            )

    def test_empty_within_group_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="within_group"):
            Recipe(
                model_id="test/model",
                plan=make_plan(),
                assignments=(
                    Assignment(group="model.layers.0", bits=8, bytes=1, damage=0.0),
                ),
                runtime=None,
                within_group="",
                imatrix=None,
                protected_tensors=(),
            )

    def test_empty_assignments_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Recipe(
                model_id="test/model",
                plan=make_plan(),
                assignments=(),
                runtime=None,
                within_group=None,
                imatrix=None,
                protected_tensors=(),
            )


@pytest.mark.unit
class TestLayerGroupTensorBytes:
    def make_group(self, tensor_bytes: dict[str, int]) -> LayerGroup:
        return LayerGroup(
            name="model.layers.0",
            tensors=("a", "b"),
            bytes_fp16=1_000,
            sensitivity={8: 0.0, 4: 0.1},
            tensor_bytes=tensor_bytes,
        )

    def test_empty_tensor_bytes_means_unknown(self) -> None:
        assert dict(self.make_group({}).tensor_bytes) == {}

    def test_full_coverage_constructs(self) -> None:
        group = self.make_group({"a": 400, "b": 600})

        assert group.tensor_bytes["a"] == 400

    def test_partial_coverage_rejected(self) -> None:
        with pytest.raises(ValueError, match="tensor_bytes keys"):
            self.make_group({"a": 400})

    def test_extra_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="tensor_bytes keys"):
            self.make_group({"a": 400, "b": 300, "c": 300})

    def test_non_positive_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            self.make_group({"a": 0, "b": 1_000})

    def test_tensor_bytes_not_summing_to_group_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to bytes_fp16"):
            self.make_group({"a": 400, "b": 700})

    def test_tensor_bytes_cannot_alias_the_callers_dict(self) -> None:
        sizes = {"a": 400, "b": 600}
        group = self.make_group(sizes)
        sizes["a"] = 1

        assert group.tensor_bytes["a"] == 400


class TestImatrixCountSummary:
    def test_ordered_values_construct(self) -> None:
        summary = ImatrixCountSummary(
            minimum=426, median=18_114.0, maximum=192_191, covered=128
        )

        assert summary.maximum == 192_191

    def test_negative_minimum_rejected(self) -> None:
        # A count is a chunk tally, so it is never negative.
        with pytest.raises(ValueError, match="minimum must not be negative"):
            ImatrixCountSummary(minimum=-1, median=0.0, maximum=1, covered=1)

    def test_median_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            ImatrixCountSummary(minimum=10, median=5.0, maximum=20, covered=3)

    def test_median_above_maximum_rejected(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            ImatrixCountSummary(minimum=1, median=30.0, maximum=20, covered=3)

    def test_non_finite_median_rejected(self) -> None:
        with pytest.raises(ValueError, match="median must be finite"):
            ImatrixCountSummary(minimum=0, median=math.nan, maximum=1, covered=2)

    def test_non_positive_covered_rejected(self) -> None:
        # A summary that reduces no member is not a distribution.
        with pytest.raises(ValueError, match="covered must be positive"):
            ImatrixCountSummary(minimum=0, median=1.0, maximum=2, covered=0)

    def test_from_counts_reduces_to_three_numbers(self) -> None:
        summary = ImatrixCountSummary.from_counts([192_191, 426, 18_114])

        assert (summary.minimum, summary.median, summary.maximum) == (
            426,
            18_114,
            192_191,
        )

    def test_from_counts_sizes_the_summary_by_its_members(self) -> None:
        # 3 counts of a 128-expert stack must not read like a small
        # stack covered whole.
        assert ImatrixCountSummary.from_counts([1, 2, 3]).covered == 3

    def test_from_counts_averages_the_two_middle_values(self) -> None:
        # An even member count has no single middle expert.
        assert ImatrixCountSummary.from_counts([4, 2]).median == 3.0

    def test_from_counts_on_one_count_collapses_to_it(self) -> None:
        summary = ImatrixCountSummary.from_counts([421_370])

        assert summary.minimum == summary.maximum == 421_370

    def test_from_counts_keeps_a_zero_count(self) -> None:
        # Zero is a real count the pack path reports (ADR-0026
        # decision 5), never a missing one.
        assert ImatrixCountSummary.from_counts([0, 10]).minimum == 0

    def test_from_counts_on_no_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="counts must not be empty"):
            ImatrixCountSummary.from_counts([])


class TestLayerGroupImatrixCounts:
    def make_group(self, counts: ImatrixCountSummary | None) -> LayerGroup:
        return LayerGroup(
            name="backbone.layers.1.mixer.experts.up_proj",
            tensors=("a",),
            bytes_fp16=1_000,
            sensitivity={8: 0.0, 4: 0.1},
            imatrix_counts=counts,
        )

    def test_absent_summary_means_unknown(self) -> None:
        assert self.make_group(None).imatrix_counts is None

    def test_present_summary_rides_on_the_group(self) -> None:
        summary = ImatrixCountSummary(
            minimum=426, median=18_114.0, maximum=192_191, covered=128
        )

        assert self.make_group(summary).imatrix_counts == summary


@pytest.mark.unit
class TestProtectedTensor:
    def test_valid_pair_constructs(self) -> None:
        pair = ProtectedTensor(tensor="model.layers.4.self_attn.v_proj.weight", bits=5)

        assert pair.bits == 5

    def test_empty_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="tensor"):
            ProtectedTensor(tensor="", bits=5)

    def test_non_positive_bits_rejected(self) -> None:
        with pytest.raises(ValueError, match="bits"):
            ProtectedTensor(tensor="t", bits=0)


@pytest.mark.unit
class TestRecipeProtectionInvariants:
    def make_protected_recipe(
        self,
        protections: dict[str, int],
        protected_tensors: tuple[ProtectedTensor, ...],
        imatrix_exclusions: tuple[str, ...] = (),
    ) -> Recipe:
        return Recipe(
            model_id="test/model",
            plan=PlanMeta(
                vram_budget_bytes=100,
                kv_headroom_bytes=10,
                weight_budget_bytes=90,
                predicted_total_bytes=80,
                predicted_damage=0.5,
                solver="greedy-damage-per-byte",
                pins={},
                protections=protections,
                format_overhead=0.05,
                trace=(),
                imatrix_exclusions=imatrix_exclusions,
            ),
            assignments=(Assignment(group="g0", bits=4, bytes=40, damage=0.1),),
            runtime=None,
            within_group=None,
            imatrix=None,
            protected_tensors=protected_tensors,
        )

    def test_protections_with_resolved_pairs_construct(self) -> None:
        recipe = self.make_protected_recipe(
            {"*.v_proj.weight": 5}, (ProtectedTensor("t", 5),)
        )

        assert recipe.protected_tensors[0].bits == 5

    def test_duplicate_protected_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            self.make_protected_recipe(
                {"*.v_proj.weight": 5},
                (ProtectedTensor("t", 5), ProtectedTensor("t", 6)),
            )

    def test_protections_without_resolved_pairs_construct(self) -> None:
        # Legal since issue #59: a rule whose every floor is a
        # per-tensor no-op resolves to zero pairs.
        recipe = self.make_protected_recipe({"*.v_proj.weight": 5}, ())

        assert recipe.protected_tensors == ()

    def test_resolved_pairs_without_protections_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"requires plan\.protections"):
            self.make_protected_recipe({}, (ProtectedTensor("t", 5),))

    def test_exclusion_patterns_with_marked_pair_construct(self) -> None:
        recipe = self.make_protected_recipe(
            {"*.v_proj.weight": 5},
            (ProtectedTensor("t", 5, exclude_imatrix=True),),
            imatrix_exclusions=("t",),
        )

        assert recipe.protected_tensors[0].exclude_imatrix is True

    def test_exclusion_patterns_without_marks_rejected(self) -> None:
        # The solver refuses an exclusion whose every pair dropped
        # (issue #59), so a record without marks is a broken artifact.
        with pytest.raises(ValueError, match="both"):
            self.make_protected_recipe(
                {"*.v_proj.weight": 5},
                (ProtectedTensor("t", 5),),
                imatrix_exclusions=("t",),
            )

    def test_marks_without_exclusion_patterns_rejected(self) -> None:
        with pytest.raises(ValueError, match="exclusion"):
            self.make_protected_recipe(
                {"*.v_proj.weight": 5},
                (ProtectedTensor("t", 5, exclude_imatrix=True),),
            )

    def test_exclude_imatrix_defaults_to_false(self) -> None:
        assert ProtectedTensor("t", 5).exclude_imatrix is False
