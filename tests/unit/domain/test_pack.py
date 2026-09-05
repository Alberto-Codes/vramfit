from __future__ import annotations

import pytest

from vramfit.domain.model import Assignment, PlanMeta, ProtectedTensor, Recipe
from vramfit.domain.pack import (
    PackResult,
    TypeOverride,
    collapsed_tensors,
    modal_type,
    predicted_bytes_delta,
    predicted_bytes_within_tolerance,
    smoke_passed,
    weight_budget_margin,
    without_protections,
    zero_count_experts,
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

    def test_floored_layers_need_no_imatrix(self) -> None:
        # Unlike the three imatrix reports, this one comes from the
        # base GGUF's own tensor names, so a matrix-less pack carries
        # it (#307).
        result = PackResult(
            packed_bytes=1,
            base_type="Q4_K_S",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(),
            floored_layers=("blk.52.",),
        )

        assert result.floored_layers == ("blk.52.",)

    def test_empty_floored_layer_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="floored layer"):
            PackResult(
                packed_bytes=1,
                base_type="Q4_K_S",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
                floored_layers=("",),
            )

    def test_floored_layers_default_to_empty(self) -> None:
        result = PackResult(
            packed_bytes=1,
            base_type="Q4_K_S",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(),
        )

        assert result.floored_layers == ()

    def test_file_type_defaults_to_none(self) -> None:
        result = PackResult(
            packed_bytes=1,
            base_type="Q2_K",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(),
        )

        assert result.file_type is None

    def test_empty_file_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="file_type"):
            PackResult(
                packed_bytes=1,
                base_type="Q2_K",
                token_embedding_type=None,
                output_tensor_type=None,
                overrides=(),
                file_type="",
            )


class TestModalType:
    def test_published_30b_composition_names_q4_0(self) -> None:
        # The tensor table #413 read from the published file, in tenths
        # of a percent of bytes: Q4_0 covers 74.3 %, and the quantizer
        # had stamped Q2_K over no Q2_K tensor.
        table = {"F32": 2, "Q8_0": 138, "Q4_0": 743, "Q2_0": 117}

        assert modal_type(table) == "Q4_0"

    def test_single_type_names_itself(self) -> None:
        assert modal_type({"Q8_0": 1}) == "Q8_0"

    def test_tensor_count_does_not_decide(self) -> None:
        # Many small tensors lose to one large one: bytes, not counts.
        assert modal_type({"F32": 237, "Q4_0": 1_000}) == "Q4_0"

    def test_tie_goes_to_the_name_that_sorts_first(self) -> None:
        assert modal_type({"Q8_0": 5, "Q4_0": 5}) == "Q4_0"
        assert modal_type({"Q4_0": 5, "Q8_0": 5}) == "Q4_0"

    def test_empty_table_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            modal_type({})

    @pytest.mark.parametrize("count", [0, -1])
    def test_non_positive_count_raises_value_error(self, count: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            modal_type({"Q4_0": count})

    def test_empty_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="name"):
            modal_type({"": 5})


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


class TestPredictedBytesDelta:
    def test_packed_over_prediction_is_positive(self) -> None:
        assert predicted_bytes_delta(2_000, 2_100) == 100

    def test_packed_under_prediction_is_negative(self) -> None:
        assert predicted_bytes_delta(2_000, 1_900) == -100

    def test_exact_prediction_is_zero(self) -> None:
        assert predicted_bytes_delta(2_000, 2_000) == 0

    @pytest.mark.parametrize("predicted", [0, -1], ids=["zero", "negative"])
    def test_non_positive_prediction_raises_value_error(self, predicted) -> None:
        with pytest.raises(ValueError, match="predicted_total_bytes"):
            predicted_bytes_delta(predicted, 2_000)

    def test_non_positive_packed_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="packed_bytes"):
            predicted_bytes_delta(2_000, 0)


class TestPredictedBytesWithinTolerance:
    def test_delta_at_the_tolerance_is_within(self) -> None:
        assert predicted_bytes_within_tolerance(10_000, 10_100) is True

    def test_delta_past_the_tolerance_is_outside(self) -> None:
        assert predicted_bytes_within_tolerance(10_000, 10_101) is False

    def test_undershoot_past_the_tolerance_is_outside(self) -> None:
        assert predicted_bytes_within_tolerance(10_000, 9_899) is False

    def test_explicit_tolerance_replaces_the_default(self) -> None:
        assert predicted_bytes_within_tolerance(10_000, 10_500, 0.05) is True

    @pytest.mark.parametrize(
        "tolerance",
        [-0.01, float("inf"), float("nan")],
        ids=["negative", "inf", "nan"],
    )
    def test_bad_tolerance_raises_value_error(self, tolerance) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            predicted_bytes_within_tolerance(10_000, 10_000, tolerance)


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


class TestZeroCountExperts:
    def test_zero_counts_name_their_stack_and_expert_index(self) -> None:
        pairs = zero_count_experts({"blk.3.ffn_up_exps.weight": (7, 0, 12, 0)})

        assert pairs == (
            ("blk.3.ffn_up_exps.weight", 1),
            ("blk.3.ffn_up_exps.weight", 3),
        )

    def test_healthy_stacks_report_nothing(self) -> None:
        assert zero_count_experts({"blk.0.ffn_up_exps.weight": (1, 2)}) == ()

    def test_pairs_sort_by_stack_then_expert(self) -> None:
        pairs = zero_count_experts(
            {
                "blk.10.ffn_up_exps.weight": (0,),
                "blk.2.ffn_down_exps.weight": (0, 0),
            }
        )

        assert pairs == (
            ("blk.10.ffn_up_exps.weight", 0),
            ("blk.2.ffn_down_exps.weight", 0),
            ("blk.2.ffn_down_exps.weight", 1),
        )

    def test_empty_counts_report_nothing(self) -> None:
        assert zero_count_experts({}) == ()


class TestZeroCountResultInvariants:
    def make_result(self, **overrides) -> PackResult:
        fields: dict = {
            "packed_bytes": 500,
            "base_type": "q4_0",
            "token_embedding_type": None,
            "output_tensor_type": None,
            "overrides": (),
            "imatrix_path": "model.imatrix.gguf",
            "imatrix_zero_count_experts": (("blk.0.ffn_up_exps.weight", 1),),
        }
        fields.update(overrides)
        return PackResult(**fields)

    def test_zero_count_report_requires_an_imatrix_path(self) -> None:
        with pytest.raises(ValueError, match="imatrix_path"):
            self.make_result(imatrix_path=None)

    def test_zero_count_report_with_its_imatrix_constructs(self) -> None:
        result = self.make_result()

        assert result.imatrix_zero_count_experts == (("blk.0.ffn_up_exps.weight", 1),)

    def test_negative_expert_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            self.make_result(
                imatrix_zero_count_experts=(("blk.0.ffn_up_exps.weight", -1),)
            )

    def test_empty_stack_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="stack"):
            self.make_result(imatrix_zero_count_experts=(("", 0),))
