"""Unit tests for the refinement sidecar's invariants."""

from __future__ import annotations

import pytest

from vramfit.domain.evals import CorpusReference
from vramfit.domain.refinement_record import (
    CONTROL_ARM,
    ArmRecord,
    MeasurementFrame,
    RefinementRecordError,
    RefinementSidecar,
)


def _frame() -> MeasurementFrame:
    return MeasurementFrame(
        runtime_build="b10362",
        hardware="H100 SXM",
        corpus=CorpusReference(file="wiki.test.raw"),
        reference="f16 base logits",
    )


def _control(chunks: int = 594) -> ArmRecord:
    return ArmRecord(
        arm=CONTROL_ARM,
        promoted=None,
        demoted=None,
        from_bits=None,
        to_bits=None,
        mean=0.204223,
        delta=0.0,
        sigma=0.0,
        better_chunks=0,
        chunks=chunks,
    )


def _arm(name: str = "arm11", sigma: float = -14.4, chunks: int = 594) -> ArmRecord:
    return ArmRecord(
        arm=name,
        promoted="g1",
        demoted="g2",
        from_bits=2,
        to_bits=4,
        mean=0.191855,
        delta=-0.012368,
        sigma=sigma,
        better_chunks=436,
        chunks=chunks,
        predicted_delta=0.004,
        packed_bytes=17_000_000_000,
    )


def _sidecar(
    *,
    model_id: str = "test/model",
    bar: float = 7.8,
    control: ArmRecord | None = None,
    arms: tuple[ArmRecord, ...] | None = None,
    winner: str | None = "arm11",
    declined: str | None = None,
) -> RefinementSidecar:
    return RefinementSidecar(
        model_id=model_id,
        frame=_frame(),
        bar=bar,
        control=_control() if control is None and declined is None else control,
        arms=(_arm(),) if arms is None else arms,
        winner=winner,
        declined=declined,
    )


def test_a_pass_records_its_winner() -> None:
    won = _sidecar().winning_arm()
    assert won is not None
    assert won.mean == pytest.approx(0.191855)


def test_a_pass_that_kept_nothing_has_no_winning_arm() -> None:
    assert _sidecar(winner=None).winning_arm() is None


def test_a_winner_that_missed_the_bar_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="does not clear the stated"):
        _sidecar(arms=(_arm(sigma=-2.0),))


def test_a_winner_that_measured_worse_is_refused() -> None:
    worse = ArmRecord(
        arm="arm11",
        promoted="g1",
        demoted="g2",
        from_bits=2,
        to_bits=4,
        mean=0.21,
        delta=0.01,
        sigma=14.4,
        better_chunks=100,
        chunks=594,
    )
    with pytest.raises(RefinementRecordError, match="does not clear the stated"):
        _sidecar(arms=(worse,))


def test_a_winner_naming_no_measured_arm_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="names no measured arm"):
        _sidecar(winner="ghost")


def test_arms_measuring_different_chunk_counts_are_refused() -> None:
    with pytest.raises(RefinementRecordError, match="pairing is broken"):
        _sidecar(arms=(_arm(chunks=512),), winner=None)


def test_repeated_arm_names_are_refused() -> None:
    with pytest.raises(RefinementRecordError, match="must be unique"):
        _sidecar(arms=(_arm(), _arm()), winner=None)


def test_a_negative_bar_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="must not be negative"):
        _sidecar(bar=-1.0)


def test_a_declined_pass_measures_nothing() -> None:
    sidecar = _sidecar(arms=(), winner=None, declined="no byte-neutral neighbour")
    assert sidecar.control is None
    assert sidecar.arms == ()


def test_a_declined_pass_carrying_a_measurement_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="measures nothing"):
        _sidecar(winner=None, declined="no byte-neutral neighbour")


def test_a_pass_that_ran_without_a_control_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="needs its control"):
        RefinementSidecar(
            model_id="test/model",
            frame=_frame(),
            bar=7.8,
            control=None,
            arms=(_arm(),),
            winner=None,
            declined=None,
        )


def test_the_control_must_carry_the_reserved_name() -> None:
    # A swap-describing arm cannot stand in as the control, or a
    # reader cannot tell which measurement the deltas are against.
    with pytest.raises(RefinementRecordError, match="must be named"):
        _sidecar(control=_arm(name="arm01"), winner=None)


def test_an_arm_other_than_the_control_must_describe_a_swap() -> None:
    with pytest.raises(RefinementRecordError, match="half-describes"):
        ArmRecord(
            arm="baseline",
            promoted=None,
            demoted=None,
            from_bits=None,
            to_bits=None,
            mean=0.2,
            delta=0.0,
            sigma=0.0,
            better_chunks=0,
            chunks=594,
        )


def test_the_control_arm_names_no_swap() -> None:
    with pytest.raises(RefinementRecordError, match="names no swap"):
        ArmRecord(
            arm=CONTROL_ARM,
            promoted="g1",
            demoted="g2",
            from_bits=2,
            to_bits=4,
            mean=0.2,
            delta=0.0,
            sigma=0.0,
            better_chunks=0,
            chunks=594,
        )


def test_a_half_described_swap_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="half-describes"):
        ArmRecord(
            arm="arm01",
            promoted="g1",
            demoted=None,
            from_bits=2,
            to_bits=4,
            mean=0.2,
            delta=0.0,
            sigma=0.0,
            better_chunks=0,
            chunks=594,
        )


def test_an_arm_counting_more_better_chunks_than_it_measured_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="better chunks"):
        ArmRecord(
            arm="arm01",
            promoted="g1",
            demoted="g2",
            from_bits=2,
            to_bits=4,
            mean=0.2,
            delta=0.0,
            sigma=0.0,
            better_chunks=600,
            chunks=594,
        )


def test_a_frame_that_names_no_instrument_is_refused() -> None:
    with pytest.raises(RefinementRecordError, match="runtime_build"):
        MeasurementFrame(
            runtime_build="",
            hardware="H100 SXM",
            corpus=CorpusReference(file="wiki.test.raw"),
            reference="f16 base logits",
        )
