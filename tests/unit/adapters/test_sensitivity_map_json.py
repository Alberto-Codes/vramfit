from __future__ import annotations

import json

import pytest

from tests.unit.conftest import make_map
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.sensitivity_map_json import (
    load_sensitivity_map,
    map_from_dict,
    map_to_dict,
    save_sensitivity_map,
)


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
        raw["vramfit_schema"] = 4

        with pytest.raises(ArtifactError, match="unsupported schema version 4"):
            map_from_dict(raw)

    def test_schema_version_two_map_still_reads(self) -> None:
        # Version 3 only widened group_by with "stack" (#161), so every
        # version-2 map is already a valid version-3 document. The
        # published maps dataset ships version 2.
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["vramfit_schema"] = 2

        assert map_from_dict(raw).scan.group_by == "layer"

    def test_schema_version_three_map_reads(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["vramfit_schema"] = 3

        assert map_from_dict(raw).scan.group_by == "layer"

    def test_writer_emits_schema_version_three(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])

        assert map_to_dict(map_from_dict(raw))["vramfit_schema"] == 3

    def test_pre_rename_envelope_key_rejected(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw["quantfit_schema"] = raw.pop("vramfit_schema")

        with pytest.raises(ArtifactError, match='renamed to "vramfit_schema"'):
            map_from_dict(raw)

    def test_pre_rename_map_at_a_readable_version_is_not_told_to_bump(self) -> None:
        # #154 tells a reader to bump when a key rename alone will not
        # load the document. Version 2 now reads (#161), so a
        # pre-rename schema-2 map needs the rename and nothing else.
        # Naming both blockers here would send the reader on a
        # pointless edit.
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw.pop("vramfit_schema")
        raw = {"quantfit_schema": 2, **raw}

        with pytest.raises(ArtifactError) as excinfo:
            map_from_dict(raw)

        message = excinfo.value.message
        assert "new key at version 2 or 3" in message
        assert "does not make it load" not in message

    def test_pre_rename_map_at_an_unreadable_version_names_both_blockers(self) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3})])
        raw.pop("vramfit_schema")
        raw = {"quantfit_schema": 1, **raw}

        with pytest.raises(ArtifactError) as excinfo:
            map_from_dict(raw)

        message = excinfo.value.message
        assert "declares version 1" in message
        assert "does not make it load" in message

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

    def test_stack_group_by_round_trips(self) -> None:
        # The pack-addressable key (#161): one group per fused expert
        # stack, which llama.cpp addresses as one tensor (#159).
        raw = make_map(
            [
                (
                    "model.layers.0.mlp.experts.up_proj",
                    1000,
                    {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3},
                )
            ]
        )
        raw["scan"]["group_by"] = "stack"

        map_ = map_from_dict(raw)

        assert map_.scan.group_by == "stack"
        assert map_from_dict(map_to_dict(map_)) == map_

    def test_non_utf8_file_rejected(self, tmp_path) -> None:
        path = tmp_path / "map.json"
        path.write_bytes(b'{"vramfit_schema": 2, "model_id": "\xff\xfe"}')

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

    def test_duplicate_key_in_a_list_element_raises_artifact_error(
        self, tmp_path
    ) -> None:
        # The hook covers every object in the document, including one
        # inside a list. A repeated key needs raw text. `json.dumps`
        # cannot write one, so this test builds the map, then edits it.
        raw = make_map([("model.layers.0", 1_000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.4})])
        text = json.dumps(raw)
        assert text.count('"name": "model.layers.0"') == 1
        path = tmp_path / "map.json"
        path.write_text(
            text.replace(
                '"name": "model.layers.0"',
                '"name": "model.layers.1", "name": "model.layers.0"',
                1,
            ),
            encoding="utf-8",
        )

        with pytest.raises(ArtifactError, match='duplicate key "name"') as caught:
            load_sensitivity_map(path)

        # The hook has no ancestry, so the path is the artifact root
        # rather than the `$.groups[0].name` an extractor would report.
        assert caught.value.json_path == "$"

    def test_non_object_top_level_rejected(self) -> None:
        with pytest.raises(ArtifactError, match="expected a JSON object"):
            map_from_dict([1, 2, 3])


@pytest.mark.unit
class TestTensorBytes:
    def make_sized_dict(self) -> dict:
        raw = make_map([("model.layers.0", 1_000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.4})])
        raw["groups"][0]["tensors"] = ["a.weight", "b.weight"]
        raw["groups"][0]["tensor_bytes"] = {"a.weight": 400, "b.weight": 600}
        return raw

    def test_tensor_bytes_round_trip(self) -> None:
        raw = self.make_sized_dict()

        map_ = map_from_dict(raw)
        again = map_from_dict(map_to_dict(map_))

        assert again == map_
        assert dict(map_.groups[0].tensor_bytes) == {
            "a.weight": 400,
            "b.weight": 600,
        }

    def test_absent_tensor_bytes_loads_empty(self) -> None:
        raw = self.make_sized_dict()
        del raw["groups"][0]["tensor_bytes"]

        map_ = map_from_dict(raw)

        assert dict(map_.groups[0].tensor_bytes) == {}
        # The writer omits the unknown field instead of writing null.
        assert "tensor_bytes" not in map_to_dict(map_)["groups"][0]

    def test_partial_tensor_bytes_rejected(self) -> None:
        raw = self.make_sized_dict()
        del raw["groups"][0]["tensor_bytes"]["b.weight"]

        with pytest.raises(ArtifactError, match="tensor_bytes"):
            map_from_dict(raw)

    def test_non_positive_tensor_bytes_rejected(self) -> None:
        raw = self.make_sized_dict()
        raw["groups"][0]["tensor_bytes"]["a.weight"] = 0

        with pytest.raises(ArtifactError, match="positive"):
            map_from_dict(raw)

    def test_null_tensor_bytes_rejected(self) -> None:
        # The writer omits the field when unknown and never writes
        # null — an explicit null is a hand-edit (ADR-0022).
        raw = self.make_sized_dict()
        raw["groups"][0]["tensor_bytes"] = None

        with pytest.raises(ArtifactError, match="tensor_bytes"):
            map_from_dict(raw)


@pytest.mark.unit
class TestImatrixCounts:
    def make_summarized_dict(self) -> dict:
        raw = make_map([("model.layers.0", 1_000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.4})])
        raw["groups"][0]["imatrix_counts"] = {
            "min": 426,
            "median": 18114.0,
            "max": 192191,
        }
        return raw

    def test_imatrix_counts_round_trip(self) -> None:
        raw = self.make_summarized_dict()

        map_ = map_from_dict(raw)
        again = map_from_dict(map_to_dict(map_))

        assert again == map_
        summary = map_.groups[0].imatrix_counts
        assert summary is not None
        assert (summary.min, summary.median, summary.max) == (426, 18114.0, 192191)

    def test_median_writes_a_float_for_an_integer_value(self) -> None:
        # statistics.median returns an int for odd-length integer
        # input, and one field must not write two JSON types. The
        # fixture stores the median as an int to force the coercion.
        raw = self.make_summarized_dict()
        raw["groups"][0]["imatrix_counts"]["median"] = 18114

        written = map_to_dict(map_from_dict(raw))["groups"][0]["imatrix_counts"]

        assert isinstance(written["median"], float)

    def test_absent_imatrix_counts_loads_none(self) -> None:
        raw = self.make_summarized_dict()
        del raw["groups"][0]["imatrix_counts"]

        map_ = map_from_dict(raw)

        assert map_.groups[0].imatrix_counts is None
        # The writer omits the absent field instead of writing null.
        assert "imatrix_counts" not in map_to_dict(map_)["groups"][0]

    def test_null_imatrix_counts_rejected(self) -> None:
        # The writer omits the field when absent and never writes
        # null — an explicit null is a hand-edit (ADR-0026).
        raw = self.make_summarized_dict()
        raw["groups"][0]["imatrix_counts"] = None

        with pytest.raises(ArtifactError, match="imatrix_counts"):
            map_from_dict(raw)

    def test_extra_summary_key_rejected(self) -> None:
        # PR #195's proposed fourth field stays out (the #201
        # amendment): decision 4 is three numbers.
        raw = self.make_summarized_dict()
        raw["groups"][0]["imatrix_counts"]["covered"] = 128

        with pytest.raises(ArtifactError, match="exactly"):
            map_from_dict(raw)

    def test_missing_summary_key_rejected(self) -> None:
        raw = self.make_summarized_dict()
        del raw["groups"][0]["imatrix_counts"]["median"]

        with pytest.raises(ArtifactError, match="exactly"):
            map_from_dict(raw)

    def test_unordered_summary_rejected(self) -> None:
        raw = self.make_summarized_dict()
        raw["groups"][0]["imatrix_counts"]["min"] = 200_000

        with pytest.raises(ArtifactError, match="ordered"):
            map_from_dict(raw)

    def test_negative_min_rejected(self) -> None:
        raw = self.make_summarized_dict()
        raw["groups"][0]["imatrix_counts"]["min"] = -1

        with pytest.raises(ArtifactError, match="negative"):
            map_from_dict(raw)

    def test_non_integer_min_rejected(self) -> None:
        raw = self.make_summarized_dict()
        raw["groups"][0]["imatrix_counts"]["min"] = 426.5

        with pytest.raises(ArtifactError, match="min"):
            map_from_dict(raw)


@pytest.mark.unit
class TestDerived:
    # The published no-2 maps carry this note, and a load-then-save
    # deleted it (#136).
    NOTE = (
        "Derived from sensitivity-64k-kquant-imx.json by removing the "
        "2-bit column. Not a scan artifact."
    )

    def make_derived_dict(self) -> dict:
        raw = make_map(
            [("model.layers.0", 1_000, {8: 0.0, 4: 0.1, 3: 0.2})], precisions=(8, 4, 3)
        )
        raw["derived"] = self.NOTE
        return raw

    def test_derived_round_trip(self) -> None:
        raw = self.make_derived_dict()

        map_ = map_from_dict(raw)
        again = map_from_dict(map_to_dict(map_))

        assert map_.derived == self.NOTE
        assert again == map_

    def test_schema_two_map_keeps_the_note(self) -> None:
        # The published dataset ships version 2, and both carriers of
        # the note sit at that version.
        raw = self.make_derived_dict()
        raw["vramfit_schema"] = 2

        assert map_from_dict(raw).derived == self.NOTE

    def test_derived_writes_last_to_match_the_published_maps(self) -> None:
        # Key order is deliberate: a republished map should differ from
        # its source in content, never in field order.
        map_ = map_from_dict(self.make_derived_dict())

        assert list(map_to_dict(map_))[-1] == "derived"

    def test_save_after_load_keeps_the_note(self, tmp_path) -> None:
        path = tmp_path / "map.json"
        out = tmp_path / "resaved.json"
        path.write_text(json.dumps(self.make_derived_dict()))

        save_sensitivity_map(load_sensitivity_map(path), out)

        assert json.loads(out.read_text())["derived"] == self.NOTE

    def test_absent_derived_loads_none(self) -> None:
        raw = self.make_derived_dict()
        del raw["derived"]

        map_ = map_from_dict(raw)

        assert map_.derived is None
        # The writer omits the absent note instead of writing null.
        assert "derived" not in map_to_dict(map_)

    def test_null_derived_rejected(self) -> None:
        # The writer omits the note when absent and never writes null
        # — an explicit null is a hand-edit (#136).
        raw = self.make_derived_dict()
        raw["derived"] = None

        with pytest.raises(ArtifactError, match="derived"):
            map_from_dict(raw)

    def test_empty_derived_rejected(self) -> None:
        raw = self.make_derived_dict()
        raw["derived"] = ""

        with pytest.raises(ArtifactError, match="derived"):
            map_from_dict(raw)

    def test_non_string_derived_rejected(self) -> None:
        raw = self.make_derived_dict()
        raw["derived"] = {"note": self.NOTE}

        with pytest.raises(ArtifactError, match="derived"):
            map_from_dict(raw)
