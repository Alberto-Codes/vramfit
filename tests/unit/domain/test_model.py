from __future__ import annotations

import pytest

from quantfit.domain.model import (
    Assignment,
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

    def test_protections_without_resolved_pairs_rejected(self) -> None:
        with pytest.raises(ValueError, match="both"):
            self.make_protected_recipe({"*.v_proj.weight": 5}, ())

    def test_resolved_pairs_without_protections_rejected(self) -> None:
        with pytest.raises(ValueError, match="both"):
            self.make_protected_recipe({}, (ProtectedTensor("t", 5),))

    def test_exclusion_patterns_with_marked_pair_construct(self) -> None:
        recipe = self.make_protected_recipe(
            {"*.v_proj.weight": 5},
            (ProtectedTensor("t", 5, exclude_imatrix=True),),
            imatrix_exclusions=("t",),
        )

        assert recipe.protected_tensors[0].exclude_imatrix is True

    def test_exclusion_patterns_without_marks_rejected(self) -> None:
        with pytest.raises(ValueError, match="exclusion"):
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
