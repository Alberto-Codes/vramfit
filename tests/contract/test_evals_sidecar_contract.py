"""Verified-fake contract suites for the evals sidecar ports (ADR-0009).

Two ports, two suites. `EvalsSidecarSink` readback goes through the
serialized dict: the real adapter's file parses back to
`sidecar_to_dict` of what was saved, and the fake's captured value
serializes to the same dict. `EvalsSidecarSource` (#137) round-trips
the domain value itself, and both implementations refuse an
unreadable source with the same error type.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import MemoryEvalsSidecarStore
from vramfit.adapters.outbound.evals_sidecar_json import (
    EVALS_SIDECAR_SCHEMA_VERSION,
    JsonEvalsSidecarFile,
    save_evals_sidecar,
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
from vramfit.ports.outbound import EvalsSidecarSink, EvalsSidecarSource


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
    sink = MemoryEvalsSidecarStore()
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

        assert readback()["vramfit_schema"] == EVALS_SIDECAR_SCHEMA_VERSION

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


# --- EvalsSidecarSource ---------------------------------------------------- #
#
# Each side is seeded through its own natural mechanism and the tests
# then call `load` only. `save` is not on this Protocol, so seeding
# through it would test a capability the port does not promise.


def _real_sidecar_source(tmp_path: Path, sidecar: EvalsSidecar | None):
    path = tmp_path / "model.gguf.evals.json"
    if sidecar is None:
        path.write_text("{}", encoding="utf-8")
    else:
        save_evals_sidecar(sidecar, path)
    return JsonEvalsSidecarFile(path)


def _fake_sidecar_source(tmp_path: Path, sidecar: EvalsSidecar | None):
    store = MemoryEvalsSidecarStore()
    if sidecar is not None:
        store.save(sidecar)
    return store


@pytest.mark.contract
@pytest.mark.parametrize(
    "build",
    [_real_sidecar_source, _fake_sidecar_source],
    ids=["real-json", "fake-memory"],
)
class TestEvalsSidecarSourceContract:
    def test_load_returns_the_configured_sidecar(self, build, tmp_path) -> None:
        expected = sample_sidecar()
        source: EvalsSidecarSource = build(tmp_path, expected)

        assert source.load() == expected

    def test_load_returns_absent_tiers_as_none(self, build, tmp_path) -> None:
        source: EvalsSidecarSource = build(tmp_path, tier1_only_sidecar())

        loaded = source.load()

        assert loaded.tier1 is not None
        assert loaded.tier2 is None
        assert loaded.tier3 is None

    def test_load_without_valid_sidecar_raises_artifact_error(
        self, build, tmp_path
    ) -> None:
        source: EvalsSidecarSource = build(tmp_path, None)

        with pytest.raises(ArtifactError):
            source.load()

    def test_load_is_repeatable(self, build, tmp_path) -> None:
        # The real adapter re-reads the file each time. The fake
        # returns a held object. Both must answer the same twice.
        source: EvalsSidecarSource = build(tmp_path, sample_sidecar())

        assert source.load() == source.load()
