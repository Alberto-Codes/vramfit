"""Invariants of the evals-sidecar domain types (ADR-0025)."""

from __future__ import annotations

import pytest

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

SHA = "ab" * 32
ARTIFACT = EvaluatedArtifact("model.gguf", SHA, 1024)
TIER3_TOOLCHAIN = EvalToolchain(
    llama_cpp_build="b10172", lm_eval="0.4.12", llama_cpp_python="0.3.34", lane="lane"
)
TIER1 = Tier1Result("2026-08-09", "wikitext-2-test", 564, 8.5168, 0.06308)
WINDOW = Tier2Window("2026-08-09", 564, 0.28727, 0.003219, 82.917, 0.099)
TASK = Tier3Task("2026-08-09", "mmlu", "2", 5, 14042, "acc", 0.7829, 0.00332, 3831.2)
CORPUS = CorpusReference(
    source="Salesforce/wikitext",
    revision="b08601e",
    sha256="cd" * 32,
    size_bytes=1_288_556,
    provenance="measured",
)
CORPORA = {"wikitext-2-test": CORPUS}


class TestCorpusReference:
    def test_source_and_revision_without_a_digest_is_valid(self) -> None:
        # A tier-3 task's corpus is fetched at run time, so it has no
        # local file to hash. Identity and revision still record.
        corpus = CorpusReference(source="hails/mmlu_no_train", revision="99a5c73")

        assert corpus.sha256 is None
        assert corpus.provenance is None

    @pytest.mark.parametrize(
        "sha", ["ab" * 31, "AB" * 32, "zz" * 32], ids=["short", "uppercase", "non-hex"]
    )
    def test_malformed_sha256_raises_value_error(self, sha) -> None:
        with pytest.raises(ValueError, match="sha256"):
            CorpusReference(sha256=sha, provenance="measured")

    def test_non_positive_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="size_bytes"):
            CorpusReference(size_bytes=0)

    def test_digest_without_provenance_raises_value_error(self) -> None:
        # The whole point of the field: a digest nobody labelled can
        # be read as a measurement it is not.
        with pytest.raises(ValueError, match="sha256 and provenance must pair"):
            CorpusReference(sha256="cd" * 32)

    def test_provenance_without_a_digest_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sha256 and provenance must pair"):
            CorpusReference(source="Salesforce/wikitext", provenance="measured")

    @pytest.mark.parametrize("mark", ["measured", "recovered", "re_derived"])
    def test_every_declared_provenance_is_accepted(self, mark) -> None:
        assert CorpusReference(sha256="cd" * 32, provenance=mark).provenance == mark

    @pytest.mark.parametrize(
        "mark",
        ["rederived", "re-derived", "Measured"],
        ids=["run-on", "hyphen", "case"],
    )
    def test_undeclared_provenance_raises_value_error(self, mark) -> None:
        with pytest.raises(ValueError, match="provenance must be one of"):
            CorpusReference(sha256="cd" * 32, provenance=mark)

    def test_empty_string_field_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="use None"):
            CorpusReference(source="")

    def test_reference_recording_nothing_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one of"):
            CorpusReference()


class TestEvaluatedArtifact:
    @pytest.mark.parametrize(
        "sha",
        ["", "ab" * 31, "AB" * 32, "zz" * 32],
        ids=["empty", "short", "uppercase", "non-hex"],
    )
    def test_malformed_sha256_raises_value_error(self, sha) -> None:
        with pytest.raises(ValueError, match="sha256"):
            EvaluatedArtifact("model.gguf", sha, 1024)

    def test_non_positive_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="size_bytes"):
            EvaluatedArtifact("model.gguf", SHA, 0)


class TestEvalToolchain:
    def test_empty_build_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="llama_cpp_build"):
            EvalToolchain(llama_cpp_build="")

    def test_empty_optional_field_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="use None"):
            EvalToolchain(llama_cpp_build="b10172", lm_eval="")


class TestTier1Result:
    def test_non_positive_ppl_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="ppl"):
            Tier1Result("2026-08-09", "wikitext-2-test", 564, 0.0, 0.06)

    def test_nan_ppl_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Tier1Result("2026-08-09", "wikitext-2-test", 564, float("nan"), 0.06)

    def test_negative_stderr_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            Tier1Result("2026-08-09", "wikitext-2-test", 564, 8.5, -0.01)


class TestTier2:
    def test_negative_mean_kld_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="mean_kld"):
            Tier2Window("2026-08-09", 564, -0.1, 0.003, 82.9, 0.1)

    def test_same_top_over_100_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="same_top_pct"):
            Tier2Window("2026-08-09", 564, 0.28, 0.003, 100.5, 0.1)

    def test_empty_windows_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="windows"):
            Tier2Result("f16", "wikitext-2-test", ())

    def test_duplicate_window_chunks_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            Tier2Result("f16", "wikitext-2-test", (WINDOW, WINDOW))


class TestTier3:
    def test_negative_few_shot_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="few_shot"):
            Tier3Task("2026-08-09", "mmlu", "2", -1, 14042, "acc", 0.78, 0.003, 3831.2)

    def test_non_positive_wall_clock_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="wall_clock_seconds"):
            Tier3Task("2026-08-09", "mmlu", "2", 5, 14042, "acc", 0.78, 0.003, 0.0)

    def test_empty_tasks_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tasks"):
            Tier3Result(())

    def test_duplicate_task_names_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            Tier3Result((TASK, TASK))


class TestEvalsSidecar:
    def test_all_tiers_absent_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one tier"):
            EvalsSidecar(artifact=ARTIFACT, toolchain=TIER3_TOOLCHAIN)

    def test_tier3_without_harness_toolchain_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tier3 requires"):
            EvalsSidecar(
                artifact=ARTIFACT,
                toolchain=EvalToolchain(llama_cpp_build="b10172"),
                tier3=Tier3Result((TASK,)),
            )

    def test_harness_toolchain_without_tier3_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="pair"):
            EvalsSidecar(
                artifact=ARTIFACT,
                toolchain=EvalToolchain(llama_cpp_build="b10172", lm_eval="0.4.12"),
                tier1=TIER1,
            )

    def test_tiers_1_and_2_without_harness_toolchain_is_valid(self) -> None:
        sidecar = EvalsSidecar(
            artifact=ARTIFACT,
            toolchain=EvalToolchain(llama_cpp_build="b10172"),
            tier1=TIER1,
            tier2=Tier2Result("f16", "wikitext-2-test", (WINDOW,)),
        )

        assert sidecar.tier3 is None


class TestEvalsSidecarCorpora:
    def _tiers_1_and_2(self, corpora, dataset: str = "wikitext-2-test"):
        return EvalsSidecar(
            artifact=ARTIFACT,
            toolchain=EvalToolchain(llama_cpp_build="b10172"),
            tier1=TIER1,
            tier2=Tier2Result("f16", dataset, (WINDOW,)),
            corpora=corpora,
        )

    def test_absent_corpora_records_no_corpus_identity(self) -> None:
        assert self._tiers_1_and_2(None).corpora is None

    def test_both_tiers_naming_one_key_resolve_to_one_entry(self) -> None:
        # The structural defect this map closes: two equal strings are
        # not proof that two tiers ran over the same bytes. One entry
        # is.
        sidecar = self._tiers_1_and_2(CORPORA)

        assert sidecar.corpora is not None
        assert sidecar.tier1 is not None
        assert sidecar.tier2 is not None
        assert sidecar.corpora[sidecar.tier1.dataset] is CORPUS
        assert sidecar.corpora[sidecar.tier2.dataset] is CORPUS

    def test_tier2_naming_an_absent_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"tier2\.dataset"):
            self._tiers_1_and_2(CORPORA, dataset="wikitext-103-test")

    def test_tier1_naming_an_absent_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"tier1\.dataset"):
            EvalsSidecar(
                artifact=ARTIFACT,
                toolchain=EvalToolchain(llama_cpp_build="b10172"),
                tier1=Tier1Result("2026-08-09", "ptb", 564, 8.5168, 0.06308),
                corpora=CORPORA,
            )

    def test_tier3_task_naming_an_absent_key_raises_value_error(self) -> None:
        task = Tier3Task(
            "2026-08-09", "mmlu", "2", 5, 14042, "acc", 0.78, 0.003, 3831.2, "mmlu-test"
        )
        with pytest.raises(ValueError, match=r"tier3.tasks\[0\].corpus"):
            EvalsSidecar(
                artifact=ARTIFACT,
                toolchain=TIER3_TOOLCHAIN,
                tier1=TIER1,
                tier3=Tier3Result((task,)),
                corpora=CORPORA,
            )

    def test_tier3_task_naming_no_corpus_needs_no_entry(self) -> None:
        sidecar = EvalsSidecar(
            artifact=ARTIFACT,
            toolchain=TIER3_TOOLCHAIN,
            tier1=TIER1,
            tier3=Tier3Result((TASK,)),
            corpora=CORPORA,
        )

        assert sidecar.tier3 is not None
        assert sidecar.tier3.tasks[0].corpus is None

    def test_empty_corpora_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="corpora must not be empty"):
            self._tiers_1_and_2({})

    def test_corpora_is_read_only_after_construction(self) -> None:
        sidecar = self._tiers_1_and_2(dict(CORPORA))

        with pytest.raises(TypeError):
            sidecar.corpora["wikitext-2-test"] = CORPUS

    def test_empty_task_corpus_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="corpus must not be empty"):
            Tier3Task(
                "2026-08-09", "mmlu", "2", 5, 14042, "acc", 0.78, 0.003, 3831.2, ""
            )
