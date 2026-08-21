from __future__ import annotations

import json

import pytest

from tests.unit.conftest import make_recipe_dict
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.recipe_json import (
    load_recipe,
    recipe_from_dict,
    recipe_to_dict,
    save_recipe,
)
from vramfit.domain.model import Assignment, PlanMeta, TraceStep


@pytest.mark.unit
class TestRecipe:
    def test_round_trip_preserves_data(self) -> None:
        raw = make_recipe_dict()

        recipe = recipe_from_dict(raw)
        again = recipe_from_dict(recipe_to_dict(recipe))

        assert again == recipe
        assert recipe.plan.trace == (
            TraceStep(
                step=1,
                group="g1",
                from_bits=8,
                to_bits=4,
                damage_delta=0.1,
                bytes_freed=20,
                ratio=0.005,
            ),
        )
        assert recipe.assignments[0] == Assignment(
            group="g0", bits=8, bytes=40, damage=0.0
        )

    def test_file_round_trip_equals_original(self, tmp_path) -> None:
        recipe = recipe_from_dict(make_recipe_dict())
        path = tmp_path / "recipe.json"

        save_recipe(recipe, path)

        assert load_recipe(path) == recipe

    def test_duplicate_key_raises_artifact_error(self, tmp_path) -> None:
        # A repeated key needs raw text — `json.dumps` cannot write one.
        # A duplicate that changes the recipe is the case that matters,
        # so this one re-states the budget the solver planned against.
        text = json.dumps(make_recipe_dict())
        assert text.count('"vram_budget_bytes": 100') == 1
        path = tmp_path / "recipe.json"
        path.write_text(
            text.replace(
                '"vram_budget_bytes": 100',
                '"vram_budget_bytes": 100, "vram_budget_bytes": 200',
                1,
            ),
            encoding="utf-8",
        )

        with pytest.raises(ArtifactError, match='duplicate key "vram_budget_bytes"'):
            load_recipe(path)

    def test_absent_within_group_defaults_to_none(self) -> None:
        raw = make_recipe_dict()
        assert "within_group" not in raw

        recipe = recipe_from_dict(raw)

        assert recipe.within_group is None
        # The writer still records the absence explicitly.
        assert recipe_to_dict(recipe)["within_group"] is None

    def test_within_group_round_trips(self) -> None:
        raw = make_recipe_dict()
        raw["within_group"] = "kquant-ref"

        recipe = recipe_from_dict(raw)
        again = recipe_from_dict(recipe_to_dict(recipe))

        assert recipe.within_group == "kquant-ref"
        assert again == recipe

    def test_assisted_provenance_round_trips(self) -> None:
        raw = make_recipe_dict()
        raw["within_group"] = "kquant-imx"
        raw["imatrix"] = "/runs/model.imatrix.gguf"

        recipe = recipe_from_dict(raw)
        again = recipe_from_dict(recipe_to_dict(recipe))

        assert recipe.within_group == "kquant-imx"
        assert recipe.imatrix == "/runs/model.imatrix.gguf"
        assert again == recipe

    def test_assisted_token_without_imatrix_rejected(self) -> None:
        # A recipe claiming assisted pricing without naming its
        # imatrix is corrupted provenance (ADR-0020).
        raw = make_recipe_dict()
        raw["within_group"] = "kquant-imx"

        with pytest.raises(ArtifactError, match="imatrix"):
            recipe_from_dict(raw)

    def test_imatrix_without_the_assisted_token_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["within_group"] = "kquant-ref"
        raw["imatrix"] = "/runs/model.imatrix.gguf"

        with pytest.raises(ArtifactError, match="kquant-imx"):
            recipe_from_dict(raw)

    def test_q0_assisted_provenance_round_trips(self) -> None:
        # The q0-imx token pairs with the imatrix field exactly like
        # kquant-imx (ADR-0018, 2026-08-21 amendment, decision 6).
        raw = make_recipe_dict()
        raw["within_group"] = "q0-imx"
        raw["imatrix"] = "/runs/model.imatrix.gguf"

        recipe = recipe_from_dict(raw)
        again = recipe_from_dict(recipe_to_dict(recipe))

        assert recipe.within_group == "q0-imx"
        assert recipe.imatrix == "/runs/model.imatrix.gguf"
        assert again == recipe

    def test_q0_imx_token_without_imatrix_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["within_group"] = "q0-imx"

        with pytest.raises(ArtifactError, match="imatrix"):
            recipe_from_dict(raw)

    def test_imatrix_on_a_q0_ref_recipe_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["within_group"] = "q0-ref"
        raw["imatrix"] = "/runs/model.imatrix.gguf"

        with pytest.raises(ArtifactError, match="q0-imx"):
            recipe_from_dict(raw)

    def test_empty_within_group_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["within_group"] = ""

        with pytest.raises(ArtifactError, match="within_group"):
            recipe_from_dict(raw)

    def test_runtime_round_trips(self) -> None:
        recipe = recipe_from_dict(make_recipe_dict())

        assert recipe.runtime == "llama.cpp"
        assert recipe_to_dict(recipe)["runtime"] == "llama.cpp"

    def test_null_runtime_loads_as_none(self) -> None:
        raw = make_recipe_dict()
        raw["runtime"] = None

        recipe = recipe_from_dict(raw)

        assert recipe.runtime is None
        assert recipe_to_dict(recipe)["runtime"] is None

    def test_missing_runtime_rejected(self) -> None:
        raw = make_recipe_dict()
        del raw["runtime"]

        with pytest.raises(ArtifactError, match="runtime"):
            recipe_from_dict(raw)

    def test_empty_runtime_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["runtime"] = ""

        with pytest.raises(ArtifactError, match="must not be empty"):
            recipe_from_dict(raw)

    def test_known_runtime_with_unservable_bits_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["runtime"] = "vllm"
        raw["assignments"][1]["bits"] = 3

        with pytest.raises(ArtifactError, match='not servable by runtime "vllm"'):
            recipe_from_dict(raw)

    def test_unknown_runtime_loads_untouched(self) -> None:
        raw = make_recipe_dict()
        raw["runtime"] = "some-future-runtime"

        recipe = recipe_from_dict(raw)

        assert recipe.runtime == "some-future-runtime"

    def test_version_one_recipe_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["vramfit_schema"] = 1

        with pytest.raises(ArtifactError, match="unsupported schema version 1"):
            recipe_from_dict(raw)

    def test_pre_rename_envelope_key_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["quantfit_schema"] = raw.pop("vramfit_schema")

        with pytest.raises(ArtifactError, match='renamed to "vramfit_schema"'):
            recipe_from_dict(raw)

    def test_pre_rename_envelope_key_current_version_omits_migration_advice(
        self,
    ) -> None:
        raw = make_recipe_dict()
        raw["quantfit_schema"] = raw.pop("vramfit_schema")

        with pytest.raises(ArtifactError) as excinfo:
            recipe_from_dict(raw)

        assert "A key rename alone" not in excinfo.value.message

    def test_missing_assignment_field_rejected(self) -> None:
        raw = make_recipe_dict()
        del raw["assignments"][0]["bits"]

        with pytest.raises(ArtifactError) as excinfo:
            recipe_from_dict(raw)

        assert excinfo.value.json_path == "$.assignments[0]"

    def test_duplicate_assignment_groups_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["assignments"][1]["group"] = "g0"

        with pytest.raises(ArtifactError, match="duplicate group"):
            recipe_from_dict(raw)

    def test_nonpositive_bits_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["assignments"][0]["bits"] = 0

        with pytest.raises(ArtifactError) as excinfo:
            recipe_from_dict(raw)

        assert excinfo.value.json_path == "$.assignments[0].bits"

    def test_nonpositive_bytes_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["assignments"][1]["bytes"] = -5

        with pytest.raises(ArtifactError) as excinfo:
            recipe_from_dict(raw)

        assert excinfo.value.json_path == "$.assignments[1].bytes"

    def test_empty_assignments_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["assignments"] = []

        with pytest.raises(ArtifactError, match="must not be empty"):
            recipe_from_dict(raw)

    @pytest.mark.parametrize("missing", ["format_overhead", "trace", "pins"])
    def test_missing_plan_field_rejected(self, missing: str) -> None:
        raw = make_recipe_dict()
        del raw["plan"][missing]

        with pytest.raises(ArtifactError, match=missing):
            recipe_from_dict(raw)

    def test_infinite_predicted_damage_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["plan"]["predicted_damage"] = float("inf")

        with pytest.raises(ArtifactError, match="finite"):
            recipe_from_dict(raw)

    def test_plan_meta_requires_explicit_provenance(self) -> None:
        with pytest.raises(TypeError):
            PlanMeta(  # type: ignore
                vram_budget_bytes=1,
                kv_headroom_bytes=1,
                weight_budget_bytes=1,
                predicted_total_bytes=1,
                predicted_damage=0.0,
                solver="s",
                pins={},
                protections={},
            )


@pytest.mark.unit
class TestProtections:
    def make_protected_dict(self) -> dict:
        raw = make_recipe_dict()
        raw["plan"]["protections"] = {"*.self_attn.v_proj.weight": 5}
        raw["protected_tensors"] = [
            {
                "tensor": "model.layers.0.self_attn.v_proj.weight",
                "bits": 5,
                "exclude_imatrix": False,
            }
        ]
        return raw

    def test_protections_round_trip(self) -> None:
        raw = self.make_protected_dict()

        recipe = recipe_from_dict(raw)
        again = recipe_from_dict(recipe_to_dict(recipe))

        assert again == recipe
        assert dict(recipe.plan.protections) == {"*.self_attn.v_proj.weight": 5}
        assert recipe.protected_tensors[0].bits == 5

    def test_old_schema_version_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["vramfit_schema"] = 3

        with pytest.raises(ArtifactError, match="version 6"):
            recipe_from_dict(raw)

    def test_missing_protections_field_rejected(self) -> None:
        raw = make_recipe_dict()
        del raw["plan"]["protections"]

        with pytest.raises(ArtifactError, match="protections"):
            recipe_from_dict(raw)

    def test_missing_protected_tensors_field_rejected(self) -> None:
        raw = make_recipe_dict()
        del raw["protected_tensors"]

        with pytest.raises(ArtifactError, match="protected_tensors"):
            recipe_from_dict(raw)

    def test_patterns_without_resolved_pairs_load(self) -> None:
        # Legal since issue #59: a rule whose every floor is a
        # per-tensor no-op resolves to zero pairs.
        raw = make_recipe_dict()
        raw["plan"]["protections"] = {"*.v_proj.weight": 5}

        recipe = recipe_from_dict(raw)

        assert recipe.protected_tensors == ()

    def test_duplicate_protected_tensor_rejected(self) -> None:
        raw = self.make_protected_dict()
        raw["protected_tensors"].append(
            {
                "tensor": "model.layers.0.self_attn.v_proj.weight",
                "bits": 6,
                "exclude_imatrix": False,
            }
        )

        with pytest.raises(ArtifactError, match="duplicate"):
            recipe_from_dict(raw)

    def test_unservable_protected_bits_rejected_for_known_runtime(self) -> None:
        raw = self.make_protected_dict()
        raw["protected_tensors"][0]["bits"] = 7

        with pytest.raises(ArtifactError, match="not servable"):
            recipe_from_dict(raw)

    def test_non_positive_protected_bits_rejected(self) -> None:
        raw = self.make_protected_dict()
        raw["protected_tensors"][0]["bits"] = 0

        with pytest.raises(ArtifactError, match="positive"):
            recipe_from_dict(raw)


@pytest.mark.unit
class TestImatrixExclusions:
    def make_excluded_dict(self) -> dict:
        raw = make_recipe_dict()
        raw["plan"]["protections"] = {"*.self_attn.v_proj.weight": 5}
        raw["plan"]["imatrix_exclusions"] = ["model.layers.0.*"]
        raw["protected_tensors"] = [
            {
                "tensor": "model.layers.0.self_attn.v_proj.weight",
                "bits": 5,
                "exclude_imatrix": True,
            }
        ]
        return raw

    def test_exclusions_round_trip(self) -> None:
        raw = self.make_excluded_dict()

        recipe = recipe_from_dict(raw)
        again = recipe_from_dict(recipe_to_dict(recipe))

        assert again == recipe
        assert recipe.plan.imatrix_exclusions == ("model.layers.0.*",)
        assert recipe.protected_tensors[0].exclude_imatrix is True

    def test_missing_exclusions_field_rejected(self) -> None:
        raw = make_recipe_dict()
        del raw["plan"]["imatrix_exclusions"]

        with pytest.raises(ArtifactError, match="imatrix_exclusions"):
            recipe_from_dict(raw)

    def test_missing_exclude_imatrix_mark_rejected(self) -> None:
        raw = self.make_excluded_dict()
        del raw["protected_tensors"][0]["exclude_imatrix"]

        with pytest.raises(ArtifactError, match="exclude_imatrix"):
            recipe_from_dict(raw)

    def test_non_boolean_exclude_imatrix_rejected(self) -> None:
        raw = self.make_excluded_dict()
        raw["protected_tensors"][0]["exclude_imatrix"] = 1

        with pytest.raises(ArtifactError, match="boolean"):
            recipe_from_dict(raw)

    def test_patterns_without_marks_rejected(self) -> None:
        # The solver refuses an exclusion whose every pair dropped
        # (issue #59), so a record without marks is a broken artifact.
        raw = self.make_excluded_dict()
        raw["protected_tensors"][0]["exclude_imatrix"] = False

        with pytest.raises(ArtifactError, match="both"):
            recipe_from_dict(raw)

    def test_marks_without_patterns_rejected(self) -> None:
        raw = self.make_excluded_dict()
        raw["plan"]["imatrix_exclusions"] = []

        with pytest.raises(ArtifactError, match="both"):
            recipe_from_dict(raw)

    def test_empty_exclusion_pattern_rejected(self) -> None:
        raw = self.make_excluded_dict()
        raw["plan"]["imatrix_exclusions"] = [""]

        with pytest.raises(ArtifactError, match="empty"):
            recipe_from_dict(raw)
