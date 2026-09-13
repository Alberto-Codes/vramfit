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
    EVALS_SIDECAR_SCHEMA_ALSO_READS,
    EVALS_SIDECAR_SCHEMA_VERSION,
    JsonEvalsSidecarFile,
    load_evals_sidecar,
    save_evals_sidecar,
    sidecar_from_dict,
    sidecar_to_dict,
)
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.domain.evals import (
    CorpusReference,
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


# One entry, named by both tier 1 and tier 2. The digest is the
# fixture's own, not any published corpus's.
WIKITEXT = CorpusReference(
    source="Salesforce/wikitext",
    revision="b08601e",
    file="wikitext-2-raw/wiki.test.raw",
    sha256="ef" * 32,
    size_bytes=1_288_556,
    provenance="re_derived",
)


def pinned_sidecar() -> EvalsSidecar:
    """A schema-3 sidecar whose tiers resolve to one corpus entry."""
    base = full_sidecar()
    return EvalsSidecar(
        artifact=base.artifact,
        toolchain=base.toolchain,
        tier1=base.tier1,
        tier2=base.tier2,
        tier3=base.tier3,
        corpora={"wikitext-2-test": WIKITEXT},
    )


def revision_only_sidecar() -> EvalsSidecar:
    """A schema-3 sidecar whose entry records no content identity."""
    base = tier1_only_sidecar()
    return EvalsSidecar(
        artifact=base.artifact,
        toolchain=base.toolchain,
        tier1=base.tier1,
        corpora={
            "wikitext-2-test": CorpusReference(
                source="Salesforce/wikitext", revision="b08601e"
            )
        },
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

    def test_absent_corpora_serializes_as_null(self) -> None:
        assert sidecar_to_dict(full_sidecar())["corpora"] is None

    def test_corpus_entry_serializes_every_field(self) -> None:
        entry = sidecar_to_dict(pinned_sidecar())["corpora"]["wikitext-2-test"]

        assert entry == {
            "source": "Salesforce/wikitext",
            "revision": "b08601e",
            "file": "wikitext-2-raw/wiki.test.raw",
            "sha256": "ef" * 32,
            "size_bytes": 1_288_556,
            "provenance": "re_derived",
        }

    def test_unrecorded_corpus_fields_serialize_as_null(self) -> None:
        entry = sidecar_to_dict(revision_only_sidecar())["corpora"]["wikitext-2-test"]

        assert entry["source"] == "Salesforce/wikitext"
        assert entry["sha256"] is None
        assert entry["size_bytes"] is None
        assert entry["provenance"] is None

    def test_both_tiers_name_the_same_corpus_key(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())

        assert data["tier1"]["dataset"] == data["tier2"]["dataset"]
        assert data["tier1"]["dataset"] in data["corpora"]


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

    def test_unbounded_size_bytes_is_refused(self) -> None:
        # A sidecar is published evidence and the artifact class most
        # likely to be hand-edited. Before #260 it could claim a file
        # size no file can have, and the reader took it as provenance.
        data = sidecar_to_dict(full_sidecar())
        data["artifact"]["size_bytes"] = 10**400

        with pytest.raises(ArtifactError, match="signed 64-bit range"):
            sidecar_from_dict(data)

    def test_unbounded_size_bytes_names_its_json_path(self) -> None:
        data = sidecar_to_dict(full_sidecar())
        data["artifact"]["size_bytes"] = 10**400

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)
        assert "$.artifact.size_bytes" in str(caught.value)

    def test_real_artifact_size_still_reads(self) -> None:
        data = sidecar_to_dict(full_sidecar())
        data["artifact"]["size_bytes"] = 20_908_008_960

        assert sidecar_from_dict(data).artifact.size_bytes == 20_908_008_960

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


class TestCorporaFromDict:
    def test_pinned_sidecar_reads_back_to_an_equal_value(self) -> None:
        sidecar = pinned_sidecar()

        assert sidecar_from_dict(sidecar_to_dict(sidecar)) == sidecar

    def test_absent_corpora_key_reads_as_no_corpus_identity(self) -> None:
        # How every schema-2 document loads: the key predates
        # version 3, so absent reads the same as null.
        data = sidecar_to_dict(full_sidecar())
        del data["corpora"]
        data["vramfit_schema"] = EVALS_SIDECAR_SCHEMA_ALSO_READS[0]

        assert sidecar_from_dict(data).corpora is None

    def test_tier_naming_an_unresolvable_key_is_refused_at_the_root(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())
        data["tier2"]["dataset"] = "wikitext-103-test"

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$"
        assert "tier2.dataset" in caught.value.message

    def test_digest_without_provenance_names_its_entry(self) -> None:
        # The refusal the mark exists for. An unlabelled digest must
        # not reach a reader as though a run had measured it.
        data = sidecar_to_dict(pinned_sidecar())
        data["corpora"]["wikitext-2-test"]["provenance"] = None

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.corpora.wikitext-2-test"
        assert "sha256 and provenance must pair" in caught.value.message

    def test_mark_without_its_referent_names_its_entry(self) -> None:
        # The fixture is re_derived, so dropping the revision leaves
        # a mark asserting a derivation the record cannot name.
        data = sidecar_to_dict(pinned_sidecar())
        data["corpora"]["wikitext-2-test"]["revision"] = None

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.corpora.wikitext-2-test"
        assert 'provenance "re_derived" requires revision' in caught.value.message

    def test_undeclared_provenance_names_its_entry(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())
        data["corpora"]["wikitext-2-test"]["provenance"] = "downloaded"

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.corpora.wikitext-2-test"
        assert "provenance must be one of" in caught.value.message

    def test_malformed_digest_names_its_entry(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())
        data["corpora"]["wikitext-2-test"]["sha256"] = "EF" * 32

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.corpora.wikitext-2-test"
        assert "64 lowercase hex digits" in caught.value.message

    def test_missing_corpus_field_reports_it_as_missing(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())
        del data["corpora"]["wikitext-2-test"]["revision"]

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.corpora.wikitext-2-test"
        assert 'missing required field "revision"' in caught.value.message

    def test_non_object_corpus_entry_reports_the_wrong_type(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())
        data["corpora"]["wikitext-2-test"] = "wikitext-2-test"

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$.corpora.wikitext-2-test"
        assert "expected a JSON object" in caught.value.message

    def test_empty_corpora_map_is_refused_at_the_root(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())
        data["corpora"] = {}

        with pytest.raises(ArtifactError) as caught:
            sidecar_from_dict(data)

        assert caught.value.json_path == "$"
        assert "corpora must not be empty" in caught.value.message

    def test_unknown_field_in_a_corpus_entry_warns_and_loads(self) -> None:
        data = sidecar_to_dict(pinned_sidecar())
        data["corpora"]["wikitext-2-test"]["license"] = "CC BY-SA 3.0"

        with pytest.warns(UserWarning, match="license"):
            sidecar = sidecar_from_dict(data)

        assert sidecar.corpora is not None
        assert sidecar.corpora["wikitext-2-test"].source == "Salesforce/wikitext"

    def test_reader_never_fills_an_unrecorded_digest(self) -> None:
        # Nothing in vramfit computes a corpus digest. A null stays
        # null through a load and a save.
        data = sidecar_to_dict(revision_only_sidecar())

        loaded = sidecar_from_dict(data)

        assert loaded.corpora is not None
        assert loaded.corpora["wikitext-2-test"].sha256 is None
        entry = sidecar_to_dict(loaded)["corpora"]["wikitext-2-test"]
        assert entry["sha256"] is None
        assert entry["size_bytes"] is None


class TestLoadEvalsSidecar:
    @pytest.mark.parametrize("name", PUBLISHED_SIDECARS, ids=PUBLISHED_IDS)
    def test_published_sidecar_still_loads(self, name: str) -> None:
        # The five shipped files sit at schema 2 and stay untouched.
        # They are the evidence that version 3 only added.
        sidecar = load_evals_sidecar(PUBLISHED / name)

        assert sidecar.corpora is None

    @pytest.mark.parametrize("name", PUBLISHED_SIDECARS, ids=PUBLISHED_IDS)
    def test_published_sidecar_rewrites_with_only_the_version_3_additions(
        self, name: str, tmp_path
    ) -> None:
        # This test read byte for byte before version 3. The writer
        # now emits 3, so a schema-2 file cannot re-serialize to its
        # own bytes. Assert the exact delta instead: the envelope and
        # the null corpora map. Every measured number must still come
        # back untouched.
        source = PUBLISHED / name
        out = tmp_path / name
        save_evals_sidecar(load_evals_sidecar(source), out)

        before = json.loads(source.read_text(encoding="utf-8"))
        assert before["vramfit_schema"] == EVALS_SIDECAR_SCHEMA_ALSO_READS[0]
        expected = dict(before)
        expected["vramfit_schema"] = EVALS_SIDECAR_SCHEMA_VERSION
        expected["corpora"] = None

        assert json.loads(out.read_text(encoding="utf-8")) == expected

    def test_pinned_sidecar_round_trips_byte_for_byte(self, tmp_path) -> None:
        # A schema-3 document is what the writer emits, so it must
        # survive a load and a save unchanged.
        first = tmp_path / "model.gguf.evals.json"
        save_evals_sidecar(pinned_sidecar(), first)
        second = tmp_path / "again.evals.json"

        save_evals_sidecar(load_evals_sidecar(first), second)

        assert second.read_text(encoding="utf-8") == first.read_text(encoding="utf-8")

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

    def test_duplicate_key_raises_artifact_error(self, tmp_path) -> None:
        # The realistic path is a re-measured tier pasted in beside the
        # old line. `json.dumps` cannot write a repeated key, so this
        # test edits the text. Assert the target first: a replacement
        # that silently missed would fail as a bare "DID NOT RAISE".
        source = PUBLISHED / "baseline-iq3-xs.gguf.evals.json"
        text = source.read_text(encoding="utf-8")
        assert text.count('"ppl": 8.5543,') == 1
        path = tmp_path / "baseline-iq3-xs.gguf.evals.json"
        path.write_text(
            text.replace('"ppl": 8.5543,', '"ppl": 999.0,\n    "ppl": 8.5543,', 1),
            encoding="utf-8",
        )

        with pytest.raises(ArtifactError, match='duplicate key "ppl"') as caught:
            load_evals_sidecar(path)

        assert "cannot parse JSON" not in str(caught.value)

    def test_adapter_load_matches_the_module_function(self, tmp_path) -> None:
        path = tmp_path / "model.gguf.evals.json"
        save_evals_sidecar(full_sidecar(), path)

        assert JsonEvalsSidecarFile(path).load() == load_evals_sidecar(path)
