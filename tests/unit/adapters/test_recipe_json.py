from __future__ import annotations

import pytest

from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.recipe_json import (
    load_recipe,
    recipe_from_dict,
    recipe_to_dict,
    save_recipe,
)
from quantfit.domain.model import Assignment, PlanMeta, TraceStep


def make_recipe_dict() -> dict:
    return {
        "quantfit_schema": 1,
        "model_id": "test/model",
        "plan": {
            "vram_budget_bytes": 100,
            "kv_headroom_bytes": 10,
            "weight_budget_bytes": 90,
            "predicted_total_bytes": 80,
            "predicted_damage": 0.5,
            "solver": "greedy-damage-per-byte",
            "pins": {"g*": 8},
            "format_overhead": 0.05,
            "trace": [
                {
                    "step": 1,
                    "group": "g1",
                    "from_bits": 8,
                    "to_bits": 4,
                    "damage_delta": 0.1,
                    "bytes_freed": 20,
                    "ratio": 0.005,
                }
            ],
        },
        "assignments": [
            {"group": "g0", "bits": 8, "bytes": 40, "damage": 0.0},
            {"group": "g1", "bits": 4, "bytes": 40, "damage": 0.5},
        ],
    }


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

    def test_empty_assignments_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["assignments"] = []

        with pytest.raises(ArtifactError, match="must not be empty"):
            recipe_from_dict(raw)

    def test_missing_format_overhead_defaults(self) -> None:
        raw = make_recipe_dict()
        del raw["plan"]["format_overhead"]
        del raw["plan"]["trace"]

        recipe = recipe_from_dict(raw)

        assert recipe.plan.format_overhead == 0.05
        assert recipe.plan.trace == ()

    def test_plan_meta_defaults_match_schema(self) -> None:
        meta = PlanMeta(
            vram_budget_bytes=1,
            kv_headroom_bytes=1,
            weight_budget_bytes=1,
            predicted_total_bytes=1,
            predicted_damage=0.0,
            solver="s",
            pins={},
        )

        assert meta.format_overhead == 0.05
        assert meta.trace == ()
