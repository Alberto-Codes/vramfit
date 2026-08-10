"""Invariants of the evals-sidecar domain types (ADR-0025)."""

from __future__ import annotations

import pytest

from quantfit.domain.evals import (
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

    def test_tiers_1_and_2_without_harness_toolchain_is_valid(self) -> None:
        sidecar = EvalsSidecar(
            artifact=ARTIFACT,
            toolchain=EvalToolchain(llama_cpp_build="b10172"),
            tier1=TIER1,
            tier2=Tier2Result("f16", "wikitext-2-test", (WINDOW,)),
        )

        assert sidecar.tier3 is None
