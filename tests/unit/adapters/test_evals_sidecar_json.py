"""Serialization of the evals-sidecar JSON adapter (ADR-0025).

The published fixtures in ``tests/data/published-evals`` are the
shipped sidecars, byte for byte. #137 added the reader precisely
because nothing could execute against them before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vramfit.adapters.outbound.evals_sidecar_json import (
    EVALS_SIDECAR_SCHEMA_VERSION,
    JsonEvalsSidecarFile,
    load_evals_sidecar,
    save_evals_sidecar,
    sidecar_from_dict,
    sidecar_to_dict,
)
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.domain.evals import (
    EvalsSidecar,
    EvalToolchain,
    EvaluatedArtifact,
    Tier1Result,
    Tier2Result,
    Tier2Window,
    Tier3Result,
    Tier3Task,
)

pytestmark = pytest.mark.unit


def full_sidecar() -> EvalsSidecar:
    return EvalsSidecar(
        artifact=EvaluatedArtifact("model.gguf", "ab" * 32, 21860214272),
        toolchain=EvalToolchain(
            llama_cpp_build="b10172-bc71c24c9",
            lm_eval="0.4.12",
            llama_cpp_python="0.3.34",
            lane="in-process llama-cpp-python",
        ),
        tier1=Tier1Result("2026-08-09", "wikitext-2-test", 564, 8.5168, 0.06308),
        tier2=Tier2Result(
            "f16",
            "wikitext-2-test",
            (Tier2Window("2026-08-09", 564, 0.28727, 0.003219, 82.917, 0.099),),
        ),
        tier3=Tier3Result(
            (
                Tier3Task(
                    "2026-08-09",
                    "gsm8k",
                    "3.0",
                    5,
                    1319,
                    "exact_match,strict-match",
                    0.93177,
                    0.00695,
                    4847.2,
                ),
            ),
        ),
    )


def tier1_only_sidecar() -> EvalsSidecar:
    return EvalsSidecar(
        artifact=EvaluatedArtifact("baseline.gguf", "cd" * 32, 19519022592),
        toolchain=EvalToolchain(llama_cpp_build="b10172-bc71c24c9"),
        tier1=Tier1Result("2026-08-10", "wikitext-2-test", 564, 8.723, 0.065),
    )


class TestSidecarToDict:
    def test_full_sidecar_serializes_every_block(self) -> None:
        data = sidecar_to_dict(full_sidecar())

        assert data["vramfit_schema"] == EVALS_SIDECAR_SCHEMA_VERSION
        assert data["artifact"]["sha256"] == "ab" * 32
        assert data["tier1"]["ppl"] == 8.5168
        assert data["tier2"]["windows"][0]["same_top_pct"] == 82.917
        assert data["tier3"]["tasks"][0]["metric"] == "exact_match,strict-match"

    def test_absent_tiers_serialize_as_null(self) -> None:
        data = sidecar_to_dict(tier1_only_sidecar())

        assert data["tier2"] is None
        assert data["tier3"] is None

    def test_absent_harness_toolchain_serializes_as_null(self) -> None:
        data = sidecar_to_dict(tier1_only_sidecar())

        assert data["toolchain"]["lm_eval"] is None
        assert data["toolchain"]["llama_cpp_python"] is None
        assert data["toolchain"]["lane"] is None

    def test_key_set_matches_between_full_and_tier1_only(self) -> None:
        full = sidecar_to_dict(full_sidecar())
        partial = sidecar_to_dict(tier1_only_sidecar())

        assert full.keys() == partial.keys()
        assert full["toolchain"].keys() == partial["toolchain"].keys()


class TestSaveEvalsSidecar:
    def test_save_writes_parseable_json_with_trailing_newline(self, tmp_path) -> None:
        path = tmp_path / "model.gguf.evals.json"

        save_evals_sidecar(full_sidecar(), path)

        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text) == sidecar_to_dict(full_sidecar())

    def test_save_leaves_no_temp_file_behind(self, tmp_path) -> None:
        path = tmp_path / "model.gguf.evals.json"

        save_evals_sidecar(full_sidecar(), path)

        assert [p.name for p in tmp_path.iterdir()] == ["model.gguf.evals.json"]


PUBLISHED = Path(__file__).parents[2] / "data" / "published-evals"
PUBLISHED_SIDECARS = [
    "fit24gib.gguf.evals.json",
    "baseline-iq3-xs.gguf.evals.json",
]


class TestSidecarFromDict:
    def test_written_sidecar_reads_back_to_an_equal_value(self) -> None:
        sidecar = full_sidecar()

        assert sidecar_from_dict(sidecar_to_dict(sidecar)) == sidecar

    def test_tier1_only_sidecar_reads_back_to_an_equal_value(self) -> None:
        sidecar = tier1_only_sidecar()

        assert sidecar_from_dict(sidecar_to_dict(sidecar)) == sidecar

    def test_pre_rename_envelope_key_names_the_new_key(self) -> None:
        data = sidecar_to_dict(full_sidecar())
        data["quantfit_schema"] = data.pop("vramfit_schema")

        with pytest.raises(ArtifactError, match="vramfit_schema"):
            sidecar_from_dict(data)

    def test_unsupported_schema_version_is_refused(self) -> None:
        data = sidecar_to_dict(full_sidecar())
        data["vramfit_schema"] = EVALS_SIDECAR_SCHEMA_VERSION + 99

        with pytest.raises(ArtifactError, match="unsupported schema version"):
            sidecar_from_dict(data)

    def test_missing_tier_key_is_refused(self) -> None:
        data = sidecar_to_dict(full_sidecar())
        del data["tier2"]

        with pytest.raises(ArtifactError, match='missing required field "tier2"'):
            sidecar_from_dict(data)

    def test_domain_invariant_failure_names_the_json_path(self) -> None:
        # `same_top_pct` is a percentage. The domain rejects 120, and
        # the reader must say where in the document that value sits.
        data = sidecar_to_dict(full_sidecar())
        data["tier2"]["windows"][0]["same_top_pct"] = 120.0

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.tier2.windows[0]"
        assert "same_top_pct" in caught.value.message

    def test_non_finite_number_is_refused(self) -> None:
        data = sidecar_to_dict(full_sidecar())
        data["tier1"]["ppl"] = float("nan")

        with pytest.raises(ArtifactError, match="finite"):
            sidecar_from_dict(data)


class TestLoadEvalsSidecar:
    @pytest.mark.parametrize("name", PUBLISHED_SIDECARS, ids=["certified", "baseline"])
    def test_published_sidecar_round_trips_byte_for_byte(
        self, name: str, tmp_path
    ) -> None:
        # The #121 re-upload verified every other artifact by loading
        # it through the merged reader. The sidecars could not be
        # checked that way, because no reader existed (#137).
        source = PUBLISHED / name
        loaded = load_evals_sidecar(source)

        out = tmp_path / name
        save_evals_sidecar(loaded, out)

        assert out.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", PUBLISHED_SIDECARS, ids=["certified", "baseline"])
    def test_published_sidecar_names_its_artifact_and_build(self, name: str) -> None:
        sidecar = load_evals_sidecar(PUBLISHED / name)

        assert sidecar.artifact.file.endswith(".gguf")
        assert len(sidecar.artifact.sha256) == 64
        assert sidecar.toolchain.llama_cpp_build

    def test_absent_file_raises_artifact_error(self, tmp_path) -> None:
        with pytest.raises(ArtifactError, match="cannot read file"):
            load_evals_sidecar(tmp_path / "missing.evals.json")

    def test_malformed_json_raises_artifact_error(self, tmp_path) -> None:
        path = tmp_path / "model.gguf.evals.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ArtifactError, match="invalid JSON"):
            load_evals_sidecar(path)

    def test_adapter_load_matches_the_module_function(self, tmp_path) -> None:
        path = tmp_path / "model.gguf.evals.json"
        save_evals_sidecar(full_sidecar(), path)

        assert JsonEvalsSidecarFile(path).load() == load_evals_sidecar(path)
