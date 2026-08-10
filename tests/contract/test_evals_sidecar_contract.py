"""Verified-fake contract suite for `EvalsSidecarSink` (ADR-0009).

The port is writer-only (ADR-0025, issue #65), so readback goes
through the serialized dict: the real adapter's file parses back to
`sidecar_to_dict` of what was saved, and the fake's captured value
serializes to the same dict.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from quantfit.adapters.outbound.evals_sidecar_json import (
    EVALS_SIDECAR_SCHEMA_VERSION,
    JsonEvalsSidecarFile,
    sidecar_to_dict,
)
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
from quantfit.ports.outbound import EvalsSidecarSink
from tests.fakes import MemoryEvalsSidecarSink


def sample_sidecar() -> EvalsSidecar:
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
            (
                Tier2Window("2026-08-09", 100, 0.153821, 0.002304, 83.490, 0.233),
                Tier2Window("2026-08-09", 564, 0.287270, 0.003219, 82.917, 0.099),
            ),
        ),
        tier3=Tier3Result(
            (
                Tier3Task(
                    "2026-08-09",
                    "winogrande",
                    "1.0",
                    5,
                    1267,
                    "acc",
                    0.7845,
                    0.011555,
                    351.6,
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


def _real_sink(
    tmp_path: Path,
) -> tuple[EvalsSidecarSink, Callable[[], dict[str, Any]]]:
    path = tmp_path / "model.gguf.evals.json"
    sink = JsonEvalsSidecarFile(path)
    return sink, lambda: json.loads(path.read_text(encoding="utf-8"))


def _fake_sink(
    tmp_path: Path,
) -> tuple[EvalsSidecarSink, Callable[[], dict[str, Any]]]:
    sink = MemoryEvalsSidecarSink()
    return sink, lambda: sidecar_to_dict(sink.last)


@pytest.mark.contract
@pytest.mark.parametrize(
    "build", [_real_sink, _fake_sink], ids=["real-json", "fake-memory"]
)
class TestEvalsSidecarSinkContract:
    def test_saved_sidecar_reads_back_equal(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)
        sidecar = sample_sidecar()

        sink.save(sidecar)

        assert readback() == sidecar_to_dict(sidecar)

    def test_saved_sidecar_carries_schema_envelope(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)

        sink.save(sample_sidecar())

        assert readback()["quantfit_schema"] == EVALS_SIDECAR_SCHEMA_VERSION

    def test_absent_tiers_read_back_null(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)

        sink.save(tier1_only_sidecar())

        data = readback()
        assert data["tier1"] is not None
        assert data["tier2"] is None
        assert data["tier3"] is None

    def test_second_save_wins(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)

        sink.save(sample_sidecar())
        sink.save(tier1_only_sidecar())

        assert readback() == sidecar_to_dict(tier1_only_sidecar())
