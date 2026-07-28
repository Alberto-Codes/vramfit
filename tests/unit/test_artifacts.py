from __future__ import annotations

import pytest

from quantfit.artifacts import (
    ArtifactError,
    Assignment,
    PlanMeta,
    Recipe,
    SensitivityMap,
    TraceStep,
)
from tests.unit.conftest import make_map


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
class TestSensitivityMap:
    def test_round_trip_preserves_data(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])

        map_ = SensitivityMap.from_dict(raw)
        again = SensitivityMap.from_dict(map_.to_dict())

        assert again == map_
        assert map_.groups[0].sensitivity == {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3}
        assert map_.scan.precisions == (8, 4, 3, 2)

    def test_load_file_round_trip_equals_original(self, tmp_path) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        map_ = SensitivityMap.from_dict(raw)
        path = tmp_path / "map.json"

        map_.save(path)

        assert SensitivityMap.load(path) == map_
        assert path.read_text().endswith("\n")

    def test_missing_field_raises_error_with_json_path(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        del raw["scan"]["metric"]

        with pytest.raises(ArtifactError) as excinfo:
            SensitivityMap.from_dict(raw)

        assert excinfo.value.json_path == "$.scan"
        assert "metric" in excinfo.value.message

    def test_wrong_schema_version_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["quantfit_schema"] = 2

        with pytest.raises(ArtifactError, match="unsupported schema version 2"):
            SensitivityMap.from_dict(raw)

    def test_non_integer_precision_key_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["groups"][0]["sensitivity"]["4x"] = 0.5

        with pytest.raises(ArtifactError, match="not an integer precision"):
            SensitivityMap.from_dict(raw)

    def test_bool_damage_value_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["groups"][0]["sensitivity"]["4"] = True

        with pytest.raises(ArtifactError, match="boolean"):
            SensitivityMap.from_dict(raw)

    def test_bool_calibration_tokens_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["scan"]["calibration_tokens"] = True

        with pytest.raises(ArtifactError, match="boolean"):
            SensitivityMap.from_dict(raw)

    def test_duplicate_group_names_rejected(self) -> None:
        curve = {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3}
        raw = make_map([("g0", 1000, curve), ("g0", 2000, curve)])

        with pytest.raises(ArtifactError, match="duplicate group name"):
            SensitivityMap.from_dict(raw)

    def test_group_missing_scanned_precision_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2})])

        with pytest.raises(ArtifactError, match="must equal scan.precisions"):
            SensitivityMap.from_dict(raw)

    def test_nonpositive_bytes_fp16_rejected(self) -> None:
        raw = make_map([("g0", 0, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])

        with pytest.raises(ArtifactError, match="must be positive"):
            SensitivityMap.from_dict(raw)

    def test_empty_groups_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["groups"] = []

        with pytest.raises(ArtifactError, match="must not be empty"):
            SensitivityMap.from_dict(raw)

    def test_duplicate_precisions_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["scan"]["precisions"] = [8, 8, 4]

        with pytest.raises(ArtifactError, match="duplicates"):
            SensitivityMap.from_dict(raw)

    def test_unsorted_precisions_rejected(self) -> None:
        raw = make_map(
            [("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})],
            precisions=(4, 8, 2, 3),
        )

        with pytest.raises(ArtifactError, match="strictly descending"):
            SensitivityMap.from_dict(raw)

    def test_malformed_json_file_raises_artifact_error(self, tmp_path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json")

        with pytest.raises(ArtifactError, match="invalid JSON"):
            SensitivityMap.load(path)

    def test_non_object_top_level_rejected(self) -> None:
        with pytest.raises(ArtifactError, match="expected a JSON object"):
            SensitivityMap.from_dict([1, 2, 3])


@pytest.mark.unit
class TestRecipe:
    def test_round_trip_preserves_data(self) -> None:
        raw = make_recipe_dict()

        recipe = Recipe.from_dict(raw)
        again = Recipe.from_dict(recipe.to_dict())

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
        recipe = Recipe.from_dict(make_recipe_dict())
        path = tmp_path / "recipe.json"

        recipe.save(path)

        assert Recipe.load(path) == recipe

    def test_missing_assignment_field_rejected(self) -> None:
        raw = make_recipe_dict()
        del raw["assignments"][0]["bits"]

        with pytest.raises(ArtifactError) as excinfo:
            Recipe.from_dict(raw)

        assert excinfo.value.json_path == "$.assignments[0]"

    def test_duplicate_assignment_groups_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["assignments"][1]["group"] = "g0"

        with pytest.raises(ArtifactError, match="duplicate group"):
            Recipe.from_dict(raw)

    def test_empty_assignments_rejected(self) -> None:
        raw = make_recipe_dict()
        raw["assignments"] = []

        with pytest.raises(ArtifactError, match="must not be empty"):
            Recipe.from_dict(raw)

    def test_missing_format_overhead_defaults(self) -> None:
        raw = make_recipe_dict()
        del raw["plan"]["format_overhead"]
        del raw["plan"]["trace"]

        recipe = Recipe.from_dict(raw)

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
