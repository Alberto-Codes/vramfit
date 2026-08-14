"""Slice cell validation (ADR-0026, the 2026-08-13 #200 amendment).

`check_slice_cell` vouches a slice cell against the loaded
parameters before the meter changes any weight. The meter's own
suite covers the perturb-measure-restore path on a real fused
checkpoint. These checks pin the refusal matrix. They skip cleanly
where the scan extra is absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.slices import check_slice_cell

pytestmark = pytest.mark.unit

STACK = "model.layers.0.mixer.experts.up_proj"
DOWN = "model.layers.0.mixer.experts.down_proj"
DENSE = "model.layers.0.mixer.out_proj.weight"

PARAMS = {
    STACK: torch.zeros(8, 4, 64),
    DOWN: torch.zeros(8, 64, 4),
    DENSE: torch.zeros(4, 64),
}


class TestCheckSliceCell:
    def test_valid_single_expert_range_passes(self) -> None:
        check_slice_cell({STACK: (0, 1)}, PARAMS)

    def test_valid_full_range_passes(self) -> None:
        check_slice_cell({STACK: (0, 8)}, PARAMS)

    def test_empty_slices_refuse(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            check_slice_cell({}, PARAMS)

    def test_unknown_parameter_refuses(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            check_slice_cell({"model.layers.9.missing": (0, 1)}, PARAMS)

    def test_dense_2d_parameter_refuses(self) -> None:
        with pytest.raises(ValueError, match="fused expert stack"):
            check_slice_cell({DENSE: (0, 1)}, PARAMS)

    @pytest.mark.parametrize(
        ("low", "high"),
        [(-1, 1), (3, 3), (5, 2), (0, 9), (8, 9)],
        ids=["negative", "empty", "inverted", "past-end", "at-end"],
    )
    def test_bad_expert_range_refuses(self, low: int, high: int) -> None:
        with pytest.raises(ValueError, match="not a valid expert range"):
            check_slice_cell({STACK: (low, high)}, PARAMS)

    def test_one_bad_entry_refuses_the_whole_cell(self) -> None:
        with pytest.raises(ValueError, match="not a valid expert range"):
            check_slice_cell({STACK: (0, 1), DOWN: (0, 9)}, PARAMS)
