from __future__ import annotations

import json

import pytest

from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.sensitivity_map_json import (
    load_sensitivity_map,
    map_from_dict,
    map_to_dict,
    save_sensitivity_map,
)
from tests.unit.conftest import make_map


@pytest.mark.unit
class TestSensitivityMap:
    def test_round_trip_preserves_data(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])

        map_ = map_from_dict(raw)
        again = map_from_dict(map_to_dict(map_))

        assert again == map_
        assert map_.groups[0].sensitivity == {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3}
        assert map_.scan.precisions == (8, 4, 3, 2)

    def test_load_file_round_trip_equals_original(self, tmp_path) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        map_ = map_from_dict(raw)
        path = tmp_path / "map.json"

        save_sensitivity_map(map_, path)

        assert load_sensitivity_map(path) == map_
        assert path.read_text().endswith("\n")

    def test_absent_within_group_defaults_to_rtn(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        assert "within_group" not in raw["scan"]

        map_ = map_from_dict(raw)

        assert map_.scan.within_group == "rtn-block32"

    def test_within_group_round_trips(self) -> None:
        raw = make_map([("g0", 1000, {3: 0.2, 2: 0.3})], precisions=(3, 2))
        raw["scan"]["within_group"] = "kquant-ref"

        map_ = map_from_dict(raw)
        again = map_from_dict(map_to_dict(map_))

        assert map_.scan.within_group == "kquant-ref"
        assert again == map_

    def test_empty_within_group_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["scan"]["within_group"] = ""

        with pytest.raises(ArtifactError, match="within_group"):
            map_from_dict(raw)

    def test_absent_imatrix_defaults_to_none(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        assert "imatrix" not in raw["scan"]

        map_ = map_from_dict(raw)

        assert map_.scan.imatrix is None

    def test_null_imatrix_loads_as_none(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["scan"]["imatrix"] = None

        map_ = map_from_dict(raw)

        assert map_.scan.imatrix is None

    def test_assisted_map_round_trips_the_imatrix(self) -> None:
        raw = make_map([("g0", 1000, {3: 0.2, 2: 0.3})], precisions=(3, 2))
        raw["scan"]["within_group"] = "kquant-imx"
        raw["scan"]["imatrix"] = "/runs/model.imatrix.gguf"

        map_ = map_from_dict(raw)
        again = map_from_dict(map_to_dict(map_))

        assert map_.scan.within_group == "kquant-imx"
        assert map_.scan.imatrix == "/runs/model.imatrix.gguf"
        assert again == map_

    def test_assisted_token_without_imatrix_rejected(self) -> None:
        # A map claiming assistance without naming its imatrix is
        # corrupted provenance (ADR-0020).
        raw = make_map([("g0", 1000, {3: 0.2, 2: 0.3})], precisions=(3, 2))
        raw["scan"]["within_group"] = "kquant-imx"

        with pytest.raises(ArtifactError, match="imatrix"):
            map_from_dict(raw)

    def test_imatrix_without_the_assisted_token_rejected(self) -> None:
        raw = make_map([("g0", 1000, {3: 0.2, 2: 0.3})], precisions=(3, 2))
        raw["scan"]["within_group"] = "kquant-ref"
        raw["scan"]["imatrix"] = "/runs/model.imatrix.gguf"

        with pytest.raises(ArtifactError, match="kquant-imx"):
            map_from_dict(raw)

    def test_empty_imatrix_rejected(self) -> None:
        raw = make_map([("g0", 1000, {3: 0.2, 2: 0.3})], precisions=(3, 2))
        raw["scan"]["within_group"] = "kquant-imx"
        raw["scan"]["imatrix"] = ""

        with pytest.raises(ArtifactError, match="imatrix"):
            map_from_dict(raw)

    def test_missing_field_raises_error_with_json_path(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        del raw["scan"]["metric"]

        with pytest.raises(ArtifactError) as excinfo:
            map_from_dict(raw)

        assert excinfo.value.json_path == "$.scan"
        assert "metric" in excinfo.value.message

    def test_wrong_schema_version_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["quantfit_schema"] = 2

        with pytest.raises(ArtifactError, match="unsupported schema version 2"):
            map_from_dict(raw)

    def test_non_integer_precision_key_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["groups"][0]["sensitivity"]["4x"] = 0.5

        with pytest.raises(ArtifactError, match="not an integer precision"):
            map_from_dict(raw)

    def test_bool_damage_value_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["groups"][0]["sensitivity"]["4"] = True

        with pytest.raises(ArtifactError, match="boolean"):
            map_from_dict(raw)

    def test_bool_calibration_tokens_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["scan"]["calibration_tokens"] = True

        with pytest.raises(ArtifactError, match="boolean"):
            map_from_dict(raw)

    def test_duplicate_group_names_rejected(self) -> None:
        curve = {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3}
        raw = make_map([("g0", 1000, curve), ("g0", 2000, curve)])

        with pytest.raises(ArtifactError, match="duplicate group name"):
            map_from_dict(raw)

    def test_group_missing_scanned_precision_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2})])

        with pytest.raises(ArtifactError, match="must equal scan"):
            map_from_dict(raw)

    def test_nonpositive_bytes_fp16_rejected(self) -> None:
        raw = make_map([("g0", 0, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])

        with pytest.raises(ArtifactError, match="must be positive"):
            map_from_dict(raw)

    def test_empty_groups_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["groups"] = []

        with pytest.raises(ArtifactError, match="must not be empty"):
            map_from_dict(raw)

    def test_duplicate_precisions_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["scan"]["precisions"] = [8, 8, 4]

        with pytest.raises(ArtifactError, match="duplicates"):
            map_from_dict(raw)

    def test_unsorted_precisions_rejected(self) -> None:
        raw = make_map(
            [("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})],
            precisions=(4, 8, 2, 3),
        )

        with pytest.raises(ArtifactError, match="strictly descending"):
            map_from_dict(raw)

    def test_nan_damage_in_file_rejected(self, tmp_path) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        path = tmp_path / "map.json"
        path.write_text(json.dumps(raw).replace("0.1", "NaN"))

        with pytest.raises(ArtifactError, match="finite"):
            load_sensitivity_map(path)

    def test_unknown_group_by_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["scan"]["group_by"] = "block"

        with pytest.raises(ArtifactError, match="layer"):
            map_from_dict(raw)

    def test_non_utf8_file_rejected(self, tmp_path) -> None:
        path = tmp_path / "map.json"
        path.write_bytes(b'{"quantfit_schema": 1, "model_id": "\xff\xfe"}')

        with pytest.raises(ArtifactError, match="UTF-8"):
            load_sensitivity_map(path)

    def test_missing_file_raises_artifact_error(self, tmp_path) -> None:
        with pytest.raises(ArtifactError, match="cannot read"):
            load_sensitivity_map(tmp_path / "absent.json")

    def test_malformed_json_file_raises_artifact_error(self, tmp_path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json")

        with pytest.raises(ArtifactError, match="invalid JSON"):
            load_sensitivity_map(path)

    def test_non_object_top_level_rejected(self) -> None:
        with pytest.raises(ArtifactError, match="expected a JSON object"):
            map_from_dict([1, 2, 3])
