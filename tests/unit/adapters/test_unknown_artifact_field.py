"""The unknown-field rule across all four artifact readers (#261).

ADR-0013's 2026-08-16 amendment sets one level for every reader: a
field the reader does not know reports and loads. The document still
parses, and a save still drops the field — the report is what changed.

The published-artifact test is the load-bearing one. It pins the fact
the ruling rested on: no shipped artifact carries a field the readers
do not know, so no report fires on the dataset today.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unit.conftest import make_map, make_recipe_dict
from vramfit.adapters.outbound.evals_sidecar_json import (
    load_evals_sidecar,
    sidecar_from_dict,
    sidecar_to_dict,
)
from vramfit.adapters.outbound.json_common import (
    ArtifactError,
    UnknownArtifactFieldWarning,
    reset_unknown_field_reporter,
    set_unknown_field_reporter,
)
from vramfit.adapters.outbound.recipe_json import recipe_from_dict, recipe_to_dict
from vramfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from vramfit.adapters.outbound.sensitivity_map_json import map_from_dict, map_to_dict
from vramfit.domain.model import (
    ImatrixCountSummary,
    LayerGroup,
    ScanMeta,
    SensitivityMap,
)
from vramfit.domain.scan import Measurement

pytestmark = pytest.mark.unit

PUBLISHED = Path(__file__).parents[2] / "data" / "published-evals"
FINGERPRINT = "model|calib|64000|8,4,3,2|layer|rtn-block32|none"


def report_for(path: str) -> str:
    """Build the report text a reader raises for one unknown field.

    `TestEveryReaderReports` pins the literal wording once. Every other
    assertion compares paths through this helper.

    Args:
        path: JSON path of the field, e.g. ``$.tier1.notes``.

    Returns:
        The full report text.
    """
    return f"{path}: vramfit does not know this field. A save drops it."


@pytest.fixture
def reports() -> Iterator[list[str]]:
    """Collect every unknown-field report raised inside the test.

    Yields:
        The list the installed reporter appends to.
    """
    collected: list[str] = []
    token = set_unknown_field_reporter(collected.append)
    yield collected
    reset_unknown_field_reporter(token)


def a_map() -> dict:
    """Build a valid two-group sensitivity map.

    Returns:
        The map as parsed JSON.
    """
    return make_map(
        [
            ("g0", 1000, {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.3}),
            ("g1", 1000, {8: 0.0, 4: 0.2, 3: 0.3, 2: 0.4}),
        ]
    )


def a_sidecar() -> dict:
    """Read a published sidecar that carries every tier.

    Returns:
        The sidecar as parsed JSON.
    """
    return json.loads(
        (PUBLISHED / "fit24gib.gguf.evals.json").read_text(encoding="utf-8")
    )


def a_checkpoint() -> dict:
    """Build a valid one-measurement scan checkpoint.

    Returns:
        The checkpoint as parsed JSON.
    """
    return {
        "vramfit_schema": 2,
        "fingerprint": FINGERPRINT,
        "measurements": [{"group": "g0", "bits": 4, "damage": 0.25}],
    }


class TestEveryReaderReports:
    """One reader, one object, one report — the uniform level."""

    def test_map_reports_at_every_object_the_schema_fixes(self, reports) -> None:
        raw = a_map()
        raw["notes"] = "hand note"
        raw["scan"]["pod"] = "runpod-abc"
        raw["groups"][1]["comment"] = "watch this one"

        map_from_dict(raw)

        assert reports == [
            "$.notes: vramfit does not know this field. A save drops it.",
            "$.scan.pod: vramfit does not know this field. A save drops it.",
            "$.groups[1].comment: vramfit does not know this field. A save drops it.",
        ]
        assert reports[0] == report_for("$.notes")

    def test_recipe_reports_at_every_object_the_schema_fixes(self, reports) -> None:
        raw = make_recipe_dict()
        raw["notes"] = "hand note"
        raw["plan"]["commit"] = "abc1234"
        raw["plan"]["trace"][0]["why"] = "cheapest"
        raw["assignments"][0]["note"] = "pinned"
        raw["protected_tensors"] = [
            {"tensor": "t0", "bits": 8, "exclude_imatrix": False, "note": "keep"}
        ]
        raw["plan"]["protections"] = {"t0": 8}

        recipe_from_dict(raw)

        assert reports == [
            report_for("$.notes"),
            report_for("$.plan.commit"),
            report_for("$.plan.trace[0].why"),
            report_for("$.assignments[0].note"),
            report_for("$.protected_tensors[0].note"),
        ]

    def test_sidecar_reports_at_every_object_the_schema_fixes(self, reports) -> None:
        raw = a_sidecar()
        raw["notes"] = "hand note"
        raw["artifact"]["pod"] = "runpod-abc"
        raw["toolchain"]["cuda"] = "12.4"
        raw["tier1"]["notes"] = "re-measured"
        raw["tier2"]["notes"] = "paired"
        raw["tier2"]["windows"][0]["host"] = "box"
        raw["tier3"]["notes"] = "five tasks"
        raw["tier3"]["tasks"][0]["seed"] = "1234"

        sidecar_from_dict(raw)

        assert reports == [
            report_for("$.notes"),
            report_for("$.artifact.pod"),
            report_for("$.toolchain.cuda"),
            report_for("$.tier1.notes"),
            report_for("$.tier2.notes"),
            report_for("$.tier2.windows[0].host"),
            report_for("$.tier3.notes"),
            report_for("$.tier3.tasks[0].seed"),
        ]

    def test_checkpoint_reports_at_every_object(self, reports, tmp_path) -> None:
        raw = a_checkpoint()
        raw["notes"] = "resumed twice"
        raw["measurements"][0]["seconds"] = 42.0
        path = tmp_path / "scan.checkpoint.json"
        path.write_text(json.dumps(raw), encoding="utf-8")

        JsonScanCheckpointFile(path).load(FINGERPRINT)

        assert reports == [
            report_for("$.notes"),
            report_for("$.measurements[0].seconds"),
        ]


class TestTheDocumentStillLoads:
    """Warn, not refuse — the field reports and the load continues."""

    def test_map_loads_equal_to_the_same_map_without_the_field(self, reports) -> None:
        raw = a_map()
        clean = map_from_dict(json.loads(json.dumps(raw)))
        raw["notes"] = "hand note"

        assert map_from_dict(raw) == clean
        assert len(reports) == 1

    def test_sidecar_loads_and_the_save_drops_the_field(self, reports) -> None:
        raw = a_sidecar()
        raw["tier1"]["notes"] = "re-measured"

        again = sidecar_to_dict(sidecar_from_dict(raw))

        # The report exists precisely because the round trip is lossy.
        assert "notes" not in again["tier1"]
        assert len(reports) == 1

    def test_recipe_loads_and_the_save_drops_the_field(self, reports) -> None:
        raw = make_recipe_dict()
        raw["plan"]["commit"] = "abc1234"

        again = recipe_to_dict(recipe_from_dict(raw))

        assert "commit" not in again["plan"]
        assert len(reports) == 1


class TestObjectsTheSchemaDoesNotFix:
    """An open-keyed object never reports — its own rule checks it."""

    def test_sensitivity_and_tensor_bytes_keys_never_report(self, reports) -> None:
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1})], precisions=(8, 4))
        raw["groups"][0]["tensor_bytes"] = {"g0.weight": 1000}
        raw["groups"][0]["tensors"] = ["g0.weight"]

        map_from_dict(raw)

        assert reports == []

    def test_pins_and_protections_patterns_never_report(self, reports) -> None:
        raw = make_recipe_dict()
        raw["plan"]["pins"] = {"blk.*.attn": 8, "output": 6}

        recipe_from_dict(raw)

        assert reports == []

    def test_imatrix_counts_still_refuses_its_own_extra_key(self, reports) -> None:
        # ADR-0026 fixes those three keys exactly, so that object
        # refuses where every other object reports. The rule leaves it
        # alone.
        raw = make_map([("g0", 1000, {8: 0.0, 4: 0.1})], precisions=(8, 4))
        raw["groups"][0]["imatrix_counts"] = {
            "min": 1,
            "median": 2.0,
            "max": 3,
            "mean": 2.0,
        }

        with pytest.raises(ArtifactError, match="must hold exactly"):
            map_from_dict(raw)

        assert reports == []


class TestPublishedArtifactsStaySilent:
    """The measured fact the ruling rested on, pinned as a test."""

    @pytest.mark.parametrize(
        "name",
        [
            "baseline-iq3-xs.gguf.evals.json",
            "baseline-iq3-xxs.gguf.evals.json",
            "baseline-q3-k-s.gguf.evals.json",
            "baseline-ud-iq3-xxs.gguf.evals.json",
            "fit24gib.gguf.evals.json",
        ],
    )
    def test_no_published_sidecar_reports_an_unknown_field(self, reports, name) -> None:
        load_evals_sidecar(PUBLISHED / name)

        assert reports == []


class TestNoWriterReportsAgainstItself:
    """A freshly written artifact must never report a field.

    This is the regression that bites when someone adds a field to a
    writer and forgets its known-key set. The map case exercises every
    optional writer key at once, because no published map carries
    `imatrix_counts` and the published set alone would miss it.
    """

    def test_map_writer_output_reports_nothing(self, reports) -> None:
        map_ = SensitivityMap(
            model_id="test/model",
            derived="Derived by removing the 2-bit column.",
            scan=ScanMeta(
                metric="kl_divergence",
                calibration="calib.txt",
                calibration_tokens=64_000,
                precisions=(8, 4),
                group_by="stack",
                started_at="2026-08-16T00:00:00Z",
                within_group="kquant-imx",
                imatrix="/runs/model.imatrix.gguf",
            ),
            groups=(
                LayerGroup(
                    name="blk.0.ffn_up_exps",
                    tensors=("t0",),
                    bytes_fp16=1000,
                    sensitivity={8: 0.0, 4: 0.1},
                    tensor_bytes={"t0": 1000},
                    imatrix_counts=ImatrixCountSummary(min=1, median=2.0, max=3),
                ),
            ),
        )

        map_from_dict(map_to_dict(map_))

        assert reports == []

    def test_recipe_writer_output_reports_nothing(self, reports) -> None:
        recipe_from_dict(recipe_to_dict(recipe_from_dict(make_recipe_dict())))

        assert reports == []

    def test_sidecar_writer_output_reports_nothing(self, reports) -> None:
        sidecar_from_dict(sidecar_to_dict(sidecar_from_dict(a_sidecar())))

        assert reports == []

    def test_checkpoint_writer_output_reports_nothing(self, reports, tmp_path) -> None:
        store = JsonScanCheckpointFile(tmp_path / "scan.checkpoint.json")
        store.append(FINGERPRINT, Measurement(group="g0", bits=4, damage=0.25))

        store.load(FINGERPRINT)

        assert reports == []


class TestTheReporterSeam:
    """The default reports through `warnings`, and a caller may reroute."""

    def test_default_reporter_raises_the_warning_category(self) -> None:
        raw = a_map()
        raw["notes"] = "hand note"

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            map_from_dict(raw)

        assert len(seen) == 1
        assert seen[0].category is UnknownArtifactFieldWarning
        assert "$.notes" in str(seen[0].message)

    def test_every_document_reports_under_stdlib_default_filters(self) -> None:
        # One call site, three documents — a tool walking a directory.
        # `warnings.warn` reported only the first: its default filter
        # keys on the message, the category, and the raising module and
        # line, and the loop never moves that line. Two rules keep this
        # test honest. The loads must share a source line, and the test
        # must not call `simplefilter("always")` — that filter is what
        # hid the defect from the suite.
        documents = []
        for _ in range(3):
            raw = a_map()
            raw["notes"] = "hand note"
            documents.append(raw)
        # `record=True` hooks the collector and sets the "always"
        # filter. `resetwarnings` then drops every filter, so the
        # default action applies and the collector still records.
        with warnings.catch_warnings(record=True) as seen:
            warnings.resetwarnings()
            for document in documents:
                map_from_dict(document)

        assert [str(entry.message) for entry in seen] == [report_for("$.notes")] * 3

    def test_a_token_restores_the_reporter_it_replaced(self) -> None:
        outer: list[str] = []
        inner: list[str] = []
        outer_token = set_unknown_field_reporter(outer.append)
        inner_token = set_unknown_field_reporter(inner.append)
        raw = a_map()
        raw["notes"] = "hand note"

        map_from_dict(raw)
        reset_unknown_field_reporter(inner_token)
        map_from_dict(raw)
        reset_unknown_field_reporter(outer_token)

        assert len(inner) == 1
        assert len(outer) == 1

    def test_an_installed_reporter_replaces_the_warning(self, reports) -> None:
        raw = a_map()
        raw["notes"] = "hand note"

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            map_from_dict(raw)

        assert len(reports) == 1
        assert seen == []
