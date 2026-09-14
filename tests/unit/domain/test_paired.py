"""Unit tests for the paired comparison of refinement arms."""

from __future__ import annotations

import pytest

from vramfit.domain.paired import (
    PairedError,
    PairedResult,
    compare,
    per_chunk,
    select,
)


def _result(delta: float, sigma: float, mean: float) -> PairedResult:
    """Build a standing with the fields selection reads."""
    return PairedResult(
        mean=mean,
        control_mean=0.2,
        delta=delta,
        sigma=sigma,
        better_chunks=500,
        chunks=594,
    )


def test_per_chunk_recovers_a_constant_series() -> None:
    assert per_chunk([2.0, 2.0, 2.0]) == pytest.approx((2.0, 2.0, 2.0))


def test_per_chunk_recovers_a_rising_series() -> None:
    assert per_chunk([1.0, 2.0]) == pytest.approx((1.0, 3.0))


def test_per_chunk_refuses_an_empty_series() -> None:
    with pytest.raises(PairedError, match="no chunks"):
        per_chunk([])


def test_compare_of_an_arm_against_itself_measures_no_difference() -> None:
    chunks = [0.2, 0.3, 0.25, 0.4]
    result = compare(chunks, chunks)
    assert result.delta == pytest.approx(0.0)
    assert result.sigma == pytest.approx(0.0)
    assert result.better_chunks == 0


def test_compare_reports_a_negative_delta_for_an_improvement() -> None:
    control = [0.20, 0.30, 0.25, 0.40]
    candidate = [0.19, 0.29, 0.24, 0.39]
    result = compare(candidate, control)
    assert result.delta == pytest.approx(-0.01)
    assert result.sigma < 0
    assert result.better_chunks == 4


def test_compare_reverses_sign_when_the_arms_swap() -> None:
    a = [0.20, 0.31, 0.24, 0.44]
    b = [0.19, 0.29, 0.26, 0.39]
    assert compare(a, b).delta == pytest.approx(-compare(b, a).delta)


def test_compare_refuses_arms_of_different_length() -> None:
    with pytest.raises(PairedError, match="4 and 3 chunks"):
        compare([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3])


def test_compare_refuses_a_single_chunk() -> None:
    with pytest.raises(PairedError, match="at least 2 chunks"):
        compare([0.1], [0.2])


def test_improved_accepts_an_arm_past_the_bar() -> None:
    assert _result(delta=-0.01, sigma=-14.4, mean=0.19).improved(7.8)


def test_improved_rejects_an_arm_short_of_the_bar() -> None:
    assert not _result(delta=-0.01, sigma=-3.0, mean=0.19).improved(7.8)


def test_improved_rejects_an_arm_that_measured_worse() -> None:
    assert not _result(delta=0.01, sigma=14.4, mean=0.21).improved(7.8)


def test_improved_refuses_a_negative_bar() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        _result(delta=0.01, sigma=1.0, mean=0.21).improved(-1.0)


def test_select_returns_the_lowest_mean_among_arms_past_the_bar() -> None:
    results = {
        "arm01": _result(delta=-0.006, sigma=-9.0, mean=0.197942),
        "arm11": _result(delta=-0.012, sigma=-14.4, mean=0.191855),
        "arm07": _result(delta=-0.001, sigma=-2.0, mean=0.203),
    }
    winner, refusal = select(results, bar=7.8)
    assert winner == "arm11"
    assert refusal is None


def test_select_ignores_a_lower_mean_that_missed_the_bar() -> None:
    results = {
        "loud": _result(delta=-0.006, sigma=-9.0, mean=0.197),
        "quiet": _result(delta=-0.012, sigma=-1.0, mean=0.191),
    }
    winner, _ = select(results, bar=7.8)
    assert winner == "loud"


def test_select_declines_when_no_arm_clears_the_bar() -> None:
    results = {"arm01": _result(delta=-0.001, sigma=-2.0, mean=0.203)}
    winner, refusal = select(results, bar=7.8)
    assert winner is None
    assert refusal is not None
    assert "-2.0" in refusal


def test_select_declines_when_nothing_was_measured() -> None:
    winner, refusal = select({}, bar=7.8)
    assert winner is None
    assert refusal == "no arm was measured"


def test_select_refuses_a_negative_bar() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        select({}, bar=-1.0)


def test_compare_reports_unbounded_sigma_for_a_shift_every_chunk_agrees_on() -> None:
    control = [0.20, 0.30, 0.25, 0.40]
    candidate = [c - 0.01 for c in control]
    result = compare(candidate, control)
    assert result.sigma == -float("inf")
    assert result.improved(7.8)


def test_compare_reports_unbounded_sigma_for_a_uniform_regression() -> None:
    control = [0.20, 0.30, 0.25, 0.40]
    candidate = [c + 0.01 for c in control]
    result = compare(candidate, control)
    assert result.sigma == float("inf")
    assert not result.improved(7.8)
