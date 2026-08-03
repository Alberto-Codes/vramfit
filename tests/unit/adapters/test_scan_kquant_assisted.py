"""Checks of the assisted K-quant port against libggml goldens.

The fixtures in ``tests/data/kquant/golden-assisted.npz`` hold
inputs, imatrix column weights, and the dequantized outputs of
llama.cpp's weighted quantizers (``ggml_quantize_chunk`` with a
non-NULL imatrix, checkout e9fa078) — regenerate with
``scripts/gen_kquant_assisted_goldens.py``. Clean-data cases
reproduce the C bit-exactly. Outlier and spike cases admit
representation ties (two fits with equal weighted error) that the
super-block scale coding amplifies, so their gate is reconstruction
error parity plus a floor on exact elements — the ADR-0018 gate
shape, applied to the ADR-0020 paths.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")

from quantfit.adapters.outbound.scan.kquant import kquant_quantize_dequantize
from quantfit.adapters.outbound.scan.kquant_assisted import (
    ASSISTED_BITS,
    kquant_assisted_quantize_dequantize,
)

pytestmark = pytest.mark.unit

GOLDEN = Path(__file__).parent.parent.parent / "data" / "kquant" / "golden-assisted.npz"
CASES = (
    "gauss_act",
    "gauss_ones",
    "gauss_spike",
    "gauss_holes",
    "outliers_act",
    "positive_act",
    "constant_act",
    "zeros_act",
    "tiny_act",
)
# Fixtures whose representation ties survive amplification — the C's
# own compiler-ordered float sums pick a different tie winner there.
# Q2_K's 4-level grid ties most often, Q4_K's 16-level grid least.
# Measured exact fractions (2026-08-03): worst 0.641 (q3 outliers) —
# the floors sit below with margin for reduction-order drift.
TIE_CASES = {
    2: frozenset({"outliers_act", "gauss_spike", "positive_act"}),
    3: frozenset({"outliers_act", "gauss_spike"}),
    4: frozenset({"outliers_act"}),
}


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN) as data:
        return dict(data)


def _parity(
    golden: dict[str, np.ndarray], case: str, bits: int, key: str, exact_floor: float
) -> None:
    """Assert error parity and an exact-element floor against the C.

    Two-sided: a systematically better fit is also a mis-port — the
    pack ships the C behavior, not an improvement on it.
    """
    x = golden[f"x_{case}"]
    qw = torch.from_numpy(golden[f"qw_{case}"])

    ours = kquant_assisted_quantize_dequantize(torch.from_numpy(x), bits, qw).numpy()

    reference = golden[f"{key}_{case}"]
    if case not in TIE_CASES[bits]:
        assert np.array_equal(ours, reference), case
        return
    c_mse = float(np.mean((x - reference) ** 2))
    our_mse = float(np.mean((x - ours) ** 2))
    assert c_mse / 1.10 - 1e-12 <= our_mse <= c_mse * 1.10 + 1e-12
    assert float((ours == reference).mean()) > exact_floor


class TestAgainstReference:
    @pytest.mark.parametrize("case", CASES)
    def test_assisted_q2k_matches_the_c_within_ties(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        _parity(golden, case, 2, "q2", exact_floor=0.7)

    @pytest.mark.parametrize("case", CASES)
    def test_assisted_q3k_matches_the_c_within_ties(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        _parity(golden, case, 3, "q3", exact_floor=0.55)

    @pytest.mark.parametrize("case", CASES)
    def test_assisted_q4k_matches_the_c_within_ties(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        _parity(golden, case, 4, "q4", exact_floor=0.85)

    def test_8bit_routes_to_the_unassisted_q8_0_port(self) -> None:
        # quantize_q8_0 discards the imatrix — assisted 8-bit must
        # price identically to unassisted 8-bit.
        torch.manual_seed(0)
        w = torch.randn(4, 512)
        qw = torch.rand(512) + 0.1

        assisted = kquant_assisted_quantize_dequantize(w, 8, qw)

        assert torch.equal(assisted, kquant_quantize_dequantize(w, 8))


class TestRoundTripProperties:
    def test_weights_change_the_fit(self) -> None:
        # A spiked column weight must move the fit — a port that
        # ignores its weights would price unassisted under the
        # assisted label.
        torch.manual_seed(0)
        w = torch.randn(4, 512)
        flat = torch.ones(512)
        spiked = flat.clone()
        spiked[13] = 1e4

        flat_fit = kquant_assisted_quantize_dequantize(w, 2, flat)
        spiked_fit = kquant_assisted_quantize_dequantize(w, 2, spiked)

        assert not torch.equal(flat_fit, spiked_fit)

    def test_shape_dtype_and_device_survive(self) -> None:
        w = torch.randn(6, 512, dtype=torch.bfloat16)
        qw = torch.rand(512) + 0.1

        result = kquant_assisted_quantize_dequantize(w, 3, qw)

        assert result.shape == w.shape
        assert result.dtype == w.dtype
        assert not torch.equal(result, w)

    def test_input_is_not_modified(self) -> None:
        w = torch.randn(2, 256)
        original = w.clone()

        kquant_assisted_quantize_dequantize(w, 2, torch.ones(256))

        assert torch.equal(w, original)

    @pytest.mark.parametrize("chunk_rows", [7, 8])
    def test_slicing_keeps_weight_tiling_aligned(
        self, monkeypatch, chunk_rows: int
    ) -> None:
        # Slice starts index the tiled weights globally — a slice
        # that restarted the tiling period would fit every later
        # block against the wrong columns. Chunk 7 puts slice
        # boundaries mid-row (period 2), chunk 8 divides evenly.
        from quantfit.adapters.outbound.scan import kquant_assisted

        torch.manual_seed(0)
        w = torch.randn(8, 512)
        qw = torch.rand(512) + 0.1

        whole = {b: kquant_assisted_quantize_dequantize(w, b, qw) for b in (2, 3, 4)}
        monkeypatch.setattr(kquant_assisted, "_CHUNK_ROWS", chunk_rows)
        sliced = {b: kquant_assisted_quantize_dequantize(w, b, qw) for b in (2, 3, 4)}

        for bits in (2, 3, 4):
            assert torch.equal(whole[bits], sliced[bits]), bits


class TestRefusals:
    def test_uncovered_bits_are_refused(self) -> None:
        with pytest.raises(ValueError, match="ADR-0020"):
            kquant_assisted_quantize_dequantize(torch.randn(2, 256), 6, torch.ones(256))
        assert 6 not in ASSISTED_BITS

    def test_rows_that_straddle_super_blocks_are_refused(self) -> None:
        # The C asserts n_per_row % QK_K == 0 — padding would
        # misalign every column weight after the first row.
        with pytest.raises(ValueError, match="divisible"):
            kquant_assisted_quantize_dequantize(torch.randn(4, 100), 2, torch.ones(100))

    def test_wrong_length_weights_are_refused(self) -> None:
        with pytest.raises(ValueError, match="quant_weights"):
            kquant_assisted_quantize_dequantize(torch.randn(2, 512), 2, torch.ones(256))

    def test_negative_weights_are_refused(self) -> None:
        qw = torch.ones(256)
        qw[7] = -1.0

        with pytest.raises(ValueError, match="non-negative"):
            kquant_assisted_quantize_dequantize(torch.randn(2, 256), 2, qw)

    def test_non_finite_weights_are_refused(self) -> None:
        qw = torch.ones(256)
        qw[7] = float("nan")

        with pytest.raises(ValueError, match="finite"):
            kquant_assisted_quantize_dequantize(torch.randn(2, 256), 2, qw)

    def test_8bit_still_validates_the_weights_it_discards(self) -> None:
        # Validity must not depend on which branch consumes the
        # argument — a caller smoke-testing 8-bit cells would gain
        # false confidence in garbage weights otherwise.
        qw = torch.ones(256)
        qw[7] = float("nan")

        with pytest.raises(ValueError, match="finite"):
            kquant_assisted_quantize_dequantize(torch.randn(2, 256), 8, qw)
