"""Serialization of the evals-sidecar JSON adapter (ADR-0025)."""

from __future__ import annotations

import json

import pytest

from vramfit.adapters.outbound.evals_sidecar_json import (
    EVALS_SIDECAR_SCHEMA_VERSION,
    save_evals_sidecar,
    sidecar_to_dict,
)
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
