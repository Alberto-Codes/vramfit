"""Serialization of the evals-sidecar JSON adapter (ADR-0025).

The fixtures in ``tests/data/published-evals`` are all five shipped
sidecars, byte for byte. #137 added the reader precisely because
nothing could execute against them before.
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

# All five sidecars publication #1 ships: the certified artifact and
# the four baselines. #137 named their unverifiability as the cost of
# having no reader, so the fixture set is the whole set, not a sample.
PUBLISHED_SIDECARS = [
    "fit24gib.gguf.evals.json",
    "baseline-iq3-xs.gguf.evals.json",
    "baseline-iq3-xxs.gguf.evals.json",
    "baseline-q3-k-s.gguf.evals.json",
    "baseline-ud-iq3-xxs.gguf.evals.json",
]
PUBLISHED_IDS = ["certified", "iq3-xs", "iq3-xxs", "q3-k-s", "ud-iq3-xxs"]


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

    def test_number_too_large_for_a_float_names_its_path(self) -> None:
        # `float()` on such an int raises OverflowError, an
        # ArithmeticError that escapes the VramfitError root (#260).
        data = sidecar_to_dict(full_sidecar())
        data["tier1"]["ppl"] = 10**400

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.tier1.ppl"
        assert "too large for a float" in caught.value.message

    @pytest.mark.parametrize("block", ["artifact", "toolchain"])
    def test_missing_required_block_reports_it_as_missing(self, block: str) -> None:
        # `dict.get` would turn the absent key into None and report
        # "expected a JSON object", sending the reader after a type
        # error in a block that is not there.
        data = sidecar_to_dict(full_sidecar())
        del data[block]

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$"
        assert f'missing required field "{block}"' in caught.value.message

    @pytest.mark.parametrize("block", ["artifact", "toolchain"])
    def test_null_required_block_reports_the_wrong_type(self, block: str) -> None:
        data = sidecar_to_dict(full_sidecar())
        data[block] = None

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == f"$.{block}"
        assert "expected a JSON object" in caught.value.message

    def test_every_tier_null_is_refused_at_the_root(self) -> None:
        # `EvalsSidecar.__post_init__` owns this rule, and the root
        # `_built` is the only thing that restates it by path.
        data = sidecar_to_dict(tier1_only_sidecar())
        data["tier1"] = None

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$"
        assert "at least one tier must be present" in caught.value.message

    def test_tier3_without_the_harness_toolchain_is_refused_at_the_root(self) -> None:
        data = sidecar_to_dict(full_sidecar())
        data["toolchain"]["lm_eval"] = None

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$"
        assert "tier3 requires the toolchain's lm_eval" in caught.value.message


class TestLoadEvalsSidecar:
    @pytest.mark.parametrize("name", PUBLISHED_SIDECARS, ids=PUBLISHED_IDS)
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

    @pytest.mark.parametrize("name", PUBLISHED_SIDECARS, ids=PUBLISHED_IDS)
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

    def test_huge_number_literal_raises_artifact_error(self, tmp_path) -> None:
        # `json.loads` parses a 400-digit literal to a Python int, which
        # passes the number check. Only a file proves that — a dict
        # fixture cannot show what the parser produces (#260).
        source = PUBLISHED / "baseline-iq3-xs.gguf.evals.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        data["tier1"]["ppl"] = 10**400
        path = tmp_path / "baseline-iq3-xs.gguf.evals.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ArtifactError, match="too large for a float"):
            load_evals_sidecar(path)

    def test_number_literal_past_the_digit_limit_raises_artifact_error(
        self, tmp_path
    ) -> None:
        # `sys.get_int_max_str_digits` caps integer-string conversion at
        # 4300 digits. `json.loads` raises a plain ValueError, not a
        # JSONDecodeError, so the load step must catch it too (#260).
        source = PUBLISHED / "baseline-iq3-xs.gguf.evals.json"
        text = source.read_text(encoding="utf-8")
        # A literal past the limit cannot survive `json.dumps`, so this
        # test edits the text. Assert the target first: a replacement
        # that silently missed would fail as a bare "DID NOT RAISE".
        assert text.count("8.5543") == 1
        path = tmp_path / "baseline-iq3-xs.gguf.evals.json"
        path.write_text(text.replace("8.5543", "1" + "0" * 5000, 1), encoding="utf-8")

        with pytest.raises(ArtifactError, match="cannot parse JSON"):
            load_evals_sidecar(path)

    def test_adapter_load_matches_the_module_function(self, tmp_path) -> None:
        path = tmp_path / "model.gguf.evals.json"
        save_evals_sidecar(full_sidecar(), path)

        assert JsonEvalsSidecarFile(path).load() == load_evals_sidecar(path)
