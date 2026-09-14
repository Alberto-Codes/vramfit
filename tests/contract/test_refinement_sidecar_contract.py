"""Verified-fake contract suite for `RefinementSidecarSink` (ADR-0009).

One port, one suite. The port has no reader, so readback goes through
the serialized dict the way the evals sink's suite does: the real
adapter's file parses back to `sidecar_to_dict` of what was saved, and
the fake's captured value serializes to the same dict.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import MemoryRefinementSidecarStore
from vramfit.adapters.outbound.refinement_sidecar_json import (
    REFINEMENT_SIDECAR_SCHEMA_VERSION,
    JsonRefinementSidecarFile,
    sidecar_to_dict,
)
from vramfit.domain.evals import CorpusReference
from vramfit.domain.refinement_record import (
    CONTROL_ARM,
    ArmRecord,
    MeasurementFrame,
    RefinementSidecar,
)
from vramfit.ports.outbound import RefinementSidecarSink


def _frame() -> MeasurementFrame:
    return MeasurementFrame(
        runtime_build="b10362",
        hardware="H100 SXM",
        corpus=CorpusReference(
            file="wiki.test.raw",
            sha256="ef" * 32,
            size_bytes=1_288_556,
            provenance="measured",
        ),
        reference="f16 base logits",
    )


def won_sidecar() -> RefinementSidecar:
    """A pass that measured two arms and kept the one that won."""
    control = ArmRecord(
        arm=CONTROL_ARM,
        promoted=None,
        demoted=None,
        from_bits=None,
        to_bits=None,
        mean=0.204223,
        delta=0.0,
        sigma=0.0,
        better_chunks=0,
        chunks=594,
        packed_bytes=21_860_214_272,
    )
    arms = (
        ArmRecord(
            arm="arm01",
            promoted="g1",
            demoted="g2",
            from_bits=2,
            to_bits=4,
            mean=0.191855,
            delta=-0.012368,
            sigma=-14.4,
            better_chunks=470,
            chunks=594,
            predicted_delta=0.00312,
            packed_bytes=21_860_214_272,
        ),
        ArmRecord(
            arm="arm02",
            promoted="g0",
            demoted="g3",
            from_bits=2,
            to_bits=4,
            mean=0.211004,
            delta=0.006781,
            sigma=6.2,
            better_chunks=201,
            chunks=594,
            predicted_delta=-0.00104,
            packed_bytes=21_860_214_272,
        ),
    )
    return RefinementSidecar(
        model_id="test/model",
        frame=_frame(),
        bar=7.8,
        control=control,
        arms=arms,
        winner="arm01",
        declined=None,
        neighbourhood_moves=385,
    )


def declined_sidecar() -> RefinementSidecar:
    """A pass that found no legal swap and measured nothing."""
    return RefinementSidecar(
        model_id="test/49b",
        frame=_frame(),
        bar=7.8,
        control=None,
        arms=(),
        winner=None,
        declined="every group sits at 3 bits, so no swap moves precision",
        neighbourhood_moves=0,
    )


def _real_sink(
    tmp_path: Path,
) -> tuple[RefinementSidecarSink, Callable[[], dict[str, Any]]]:
    path = tmp_path / "recipe.refinement.json"
    sink = JsonRefinementSidecarFile(path)
    return sink, lambda: json.loads(path.read_text(encoding="utf-8"))


def _fake_sink(
    tmp_path: Path,
) -> tuple[RefinementSidecarSink, Callable[[], dict[str, Any]]]:
    sink = MemoryRefinementSidecarStore()
    return sink, lambda: sidecar_to_dict(sink.last)


@pytest.mark.contract
@pytest.mark.parametrize(
    "build", [_real_sink, _fake_sink], ids=["real-json", "fake-memory"]
)
class TestRefinementSidecarSinkContract:
    def test_saved_sidecar_reads_back_equal(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)
        sidecar = won_sidecar()

        sink.save(sidecar)

        assert readback() == sidecar_to_dict(sidecar)

    def test_saved_sidecar_carries_schema_envelope(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)

        sink.save(won_sidecar())

        assert readback()["vramfit_schema"] == REFINEMENT_SIDECAR_SCHEMA_VERSION

    def test_saved_sidecar_names_the_neighbourhood_the_arms_came_from(
        self, build, tmp_path
    ) -> None:
        sink, readback = build(tmp_path)

        sink.save(won_sidecar())

        data = readback()
        assert len(data["arms"]) == 2
        assert data["neighbourhood_moves"] == 385

    def test_declined_pass_reads_back_no_neighbourhood_move(
        self, build, tmp_path
    ) -> None:
        sink, readback = build(tmp_path)

        sink.save(declined_sidecar())

        assert readback()["neighbourhood_moves"] == 0

    def test_saved_sidecar_keeps_every_arm_not_only_the_winner(
        self, build, tmp_path
    ) -> None:
        sink, readback = build(tmp_path)

        sink.save(won_sidecar())

        data = readback()
        assert [a["arm"] for a in data["arms"]] == ["arm01", "arm02"]
        assert data["winner"] == "arm01"

    def test_saved_frame_keeps_the_corpus_content_identity(
        self, build, tmp_path
    ) -> None:
        sink, readback = build(tmp_path)

        sink.save(won_sidecar())

        corpus = readback()["frame"]["corpus"]
        assert corpus["sha256"] == "ef" * 32
        assert corpus["size_bytes"] == 1_288_556
        assert corpus["provenance"] == "measured"

    def test_declined_pass_reads_back_with_no_measurement(
        self, build, tmp_path
    ) -> None:
        sink, readback = build(tmp_path)

        sink.save(declined_sidecar())

        data = readback()
        assert data["declined"]
        assert data["control"] is None
        assert data["arms"] == []
        assert data["winner"] is None

    def test_second_save_wins(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)

        sink.save(won_sidecar())
        sink.save(declined_sidecar())

        assert readback() == sidecar_to_dict(declined_sidecar())
