from __future__ import annotations

import pytest

from vramfit.domain.model import Assignment, PlanMeta, ProtectedTensor, Recipe
from vramfit.domain.pack import (
    PackResult,
    TypeOverride,
    ZeroCountExpert,
    collapsed_tensors,
    smoke_passed,
    weight_budget_margin,
    without_protections,
)

pytestmark = pytest.mark.unit


def make_recipe(weight_budget_bytes: int) -> Recipe:
    return Recipe(
        model_id="test/model",
        plan=PlanMeta(
            vram_budget_bytes=weight_budget_bytes + 1_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=weight_budget_bytes,
            predicted_total_bytes=weight_budget_bytes - 500,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


class TestPackResult:
    def test_valid_result_constructs(self) -> None:
        result = PackResult(
            packed_bytes=100,
            base_type="Q4_K_S",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(TypeOverride(pattern=r"blk\.0\.", quant_type="q4_k"),),
        )

        assert result.packed_bytes == 100

    def test_empty_token_embedding_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="token_embedding_type"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type="",
                output_tensor_type=None,
                overrides=(),
            )

    def test_duplicate_override_patterns_raise_value_error(self) -> None:
        duplicate = TypeOverride(pattern=r"blk\.0\.", quant_type="q4_k")
        shadowed = TypeOverride(pattern=r"blk\.0\.", quant_type="q8_0")

        with pytest.raises(ValueError, match="unique"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(duplicate, shadowed),
            )

    @pytest.mark.parametrize("bad_bytes", [0, -1], ids=["zero", "negative"])
    def test_non_positive_packed_bytes_raises_value_error(self, bad_bytes) -> None:
        with pytest.raises(ValueError, match="packed_bytes"):
            PackResult(
                packed_bytes=bad_bytes,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
            )

    def test_empty_base_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="base_type"):
            PackResult(
                packed_bytes=1,
                base_type="",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
            )

    def test_empty_output_tensor_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="output_tensor_type"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type="",
                overrides=(),
            )

    @pytest.mark.parametrize(
        ("pattern", "quant_type"),
        [("", "q4_k"), (r"blk\.0\.", "")],
        ids=["empty-pattern", "empty-type"],
    )
    def test_empty_override_half_raises_value_error(self, pattern, quant_type) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            TypeOverride(pattern=pattern, quant_type=quant_type)

    def test_empty_imatrix_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="imatrix_path"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
                imatrix_path="",
            )

    def test_excluded_tensors_without_imatrix_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="imatrix_excluded"):
            PackResult(
                packed_bytes=500,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
                imatrix_path=None,
                imatrix_excluded=("blk.0.attn_v.weight",),
            )

    def test_uncovered_tensors_without_imatrix_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="imatrix_uncovered"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
                imatrix_uncovered=("token_embd.weight",),
            )

    def test_imatrix_path_defaults_to_none(self) -> None:
        result = PackResult(
            packed_bytes=1,
            base_type="Q4_K_S",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(),
        )

        assert result.imatrix_path is None

    def test_zero_count_experts_without_imatrix_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="imatrix_zero_count_experts"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
                imatrix_zero_count_experts=(
                    ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=57),
                ),
            )

    def test_repeated_zero_count_expert_raises_value_error(self) -> None:
        starved = ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=57)
        with pytest.raises(ValueError, match="zero-count experts must be unique"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
                imatrix_path="model.imatrix.gguf",
                imatrix_zero_count_experts=(starved, starved),
            )

    def test_zero_count_experts_default_to_empty(self) -> None:
        result = PackResult(
            packed_bytes=1,
            base_type="Q4_K_S",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(),
        )

        assert result.imatrix_zero_count_experts == ()

    def test_zero_count_experts_are_separate_from_uncovered_tensors(self) -> None:
        # ADR-0023 fenced imatrix_uncovered to unintentional gaps
        # over whole tensors. An expert inside a covered stack is a
        # third case, so the two records never mix (ADR-0026).
        result = PackResult(
            packed_bytes=1,
            base_type="Q4_K_S",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(),
            imatrix_path="model.imatrix.gguf",
            imatrix_uncovered=("token_embd.weight",),
            imatrix_zero_count_experts=(
                ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=57),
            ),
        )

        assert result.imatrix_uncovered == ("token_embd.weight",)
        assert result.imatrix_zero_count_experts[0].expert == 57


class TestZeroCountExpert:
    def test_valid_report_constructs(self) -> None:
        starved = ZeroCountExpert(stack="blk.20.ffn_up_exps.weight", expert=0)

        assert starved.expert == 0

    def test_empty_stack_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="stack must not be empty"):
            ZeroCountExpert(stack="", expert=1)

    def test_negative_expert_index_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="expert must not be negative"):
            ZeroCountExpert(stack="blk.20.ffn_up_exps.weight", expert=-1)


class TestWeightBudgetMargin:
    def test_under_budget_margin_is_positive(self) -> None:
        assert weight_budget_margin(make_recipe(3_000), 2_900) == 100

    def test_over_budget_margin_is_negative(self) -> None:
        assert weight_budget_margin(make_recipe(3_000), 3_001) == -1

    def test_exact_fit_margin_is_zero(self) -> None:
        assert weight_budget_margin(make_recipe(3_000), 3_000) == 0

    def test_non_positive_packed_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="packed_bytes"):
            weight_budget_margin(make_recipe(3_000), 0)


class TestSmokePassed:
    def test_working_perplexity_under_ceiling_passes(self) -> None:
        assert smoke_passed(9.92, threshold=1000.0) is True

    def test_destroyed_perplexity_over_ceiling_fails(self) -> None:
        assert smoke_passed(1_020_627.87, threshold=1000.0) is False

    def test_perplexity_at_the_ceiling_fails(self) -> None:
        assert smoke_passed(1000.0, threshold=1000.0) is False

    @pytest.mark.parametrize(
        "perplexity",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_non_finite_perplexity_fails(self, perplexity) -> None:
        assert smoke_passed(perplexity, threshold=1000.0) is False

    def test_non_positive_threshold_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            smoke_passed(9.5, threshold=0.0)

    def test_infinite_threshold_raises_value_error(self) -> None:
        # An infinite ceiling would silently disable the gate.
        with pytest.raises(ValueError, match="threshold"):
            smoke_passed(1_020_627.87, threshold=float("inf"))

    def test_perplexity_below_one_fails(self) -> None:
        # Perplexity is mathematically at least 1 — lower values
        # signal a broken tool, not a good artifact.
        assert smoke_passed(0.0, threshold=1000.0) is False


class TestWithoutProtections:
    def test_strips_protections_and_resolved_pairs(self) -> None:

        recipe = make_recipe(2_000)
        protected = Recipe(
            model_id=recipe.model_id,
            plan=PlanMeta(
                vram_budget_bytes=3_000,
                kv_headroom_bytes=1_000,
                weight_budget_bytes=2_000,
                predicted_total_bytes=1_500,
                predicted_damage=0.05,
                solver="greedy-damage-per-byte",
                pins={},
                protections={"*.v_proj.weight": 5},
                format_overhead=0.05,
                trace=(),
                imatrix_exclusions=("model.layers.0.*",),
            ),
            assignments=recipe.assignments,
            runtime=None,
            within_group=None,
            imatrix=None,
            protected_tensors=(
                ProtectedTensor(
                    "model.layers.0.self_attn.v_proj.weight", 5, exclude_imatrix=True
                ),
            ),
        )

        stripped = without_protections(protected)

        assert stripped.protected_tensors == ()
        assert dict(stripped.plan.protections) == {}
        assert stripped.plan.imatrix_exclusions == ()
        assert stripped.assignments == protected.assignments
        assert stripped.model_id == protected.model_id

    def test_unprotected_recipe_round_trips_unchanged(self) -> None:

        recipe = make_recipe(2_000)

        assert without_protections(recipe) == recipe


class TestCollapsedTensors:
    def test_protected_closer_to_f16_passes(self) -> None:
        assert collapsed_tensors({"t": 0.001}, {"t": 0.004}) == ()

    def test_g1_collapse_signature_is_named(self) -> None:
        assert collapsed_tensors({"t": 0.0241}, {"t": 0.0048}) == ("t",)

    def test_equal_error_counts_as_collapsed(self) -> None:
        assert collapsed_tensors({"t": 0.004}, {"t": 0.004}) == ("t",)

    def test_nan_measurement_counts_as_collapsed(self) -> None:
        assert collapsed_tensors({"t": float("nan")}, {"t": 0.004}) == ("t",)

    def test_collapsed_names_sort_deterministically(self) -> None:

        collapsed = collapsed_tensors(
            {"b": 0.9, "a": 0.9, "c": 0.001}, {"b": 0.1, "a": 0.1, "c": 0.1}
        )

        assert collapsed == ("a", "b")

    def test_mismatched_tensor_sets_raise_value_error(self) -> None:

        with pytest.raises(ValueError, match="same tensors"):
            collapsed_tensors({"a": 0.1}, {"b": 0.1})
