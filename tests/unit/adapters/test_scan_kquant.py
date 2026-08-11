"""Checks of the K-quant port against libggml goldens.

The fixtures in ``tests/data/kquant/golden.npz`` hold inputs and the
dequantized outputs of llama.cpp's reference quantizers
(``ggml_quantize_chunk``, checkout e9fa078, no imatrix). Q3_K and
Q8_0 reproduce the C output bit-exactly. Q2_K and Q4_K admit
representation ties — sub-blocks where two (level, scale) encodings
reconstruct identically and float summation order picks the winner —
so their gate is reconstruction-error parity plus a floor on exact
elements (ADR-0018). The suite sits in the unit tier so the port is
guarded on every commit: it needs no model, no card, and no network,
and it skips cleanly where the scan extra is absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")

from vramfit.adapters.outbound.scan import kquant
from vramfit.adapters.outbound.scan.kquant import (
    _ROUND_TRIPS,
    KQUANT_BITS,
    kquant_quantize_dequantize,
)
from vramfit.adapters.outbound.scan.quantize import rtn_quantize_dequantize
from vramfit.domain.scan import KQUANT_PRECISIONS

pytestmark = pytest.mark.unit

GOLDEN = Path(__file__).parent.parent.parent / "data" / "kquant" / "golden.npz"
CASES = ("gauss", "outliers", "positive", "constant", "zeros", "tiny")


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN) as data:
        return dict(data)


class TestAgainstReference:
    @pytest.mark.parametrize("case", CASES)
    def test_q3k_matches_the_c_reference_exactly(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        x = torch.from_numpy(golden[f"x_{case}"])

        ours = kquant_quantize_dequantize(x, 3).numpy()

        assert np.array_equal(ours, golden[f"q3_{case}"])

    @pytest.mark.parametrize("case", CASES)
    def test_q8_0_matches_the_c_reference_exactly(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        x = torch.from_numpy(golden[f"x_{case}"])

        ours = kquant_quantize_dequantize(x, 8).numpy()

        assert np.array_equal(ours, golden[f"q8_{case}"])

    @pytest.mark.parametrize("case", CASES)
    def test_q4k_matches_the_c_reference_error_parity(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        # Q4_K fits with squared error, so ties are rare — the exact
        # floor sits far higher than Q2_K's.
        x = golden[f"x_{case}"]

        ours = kquant_quantize_dequantize(torch.from_numpy(x), 4).numpy()

        c_mse = float(np.mean((x - golden[f"q4_{case}"]) ** 2))
        our_mse = float(np.mean((x - ours) ** 2))
        assert c_mse / 1.01 - 1e-12 <= our_mse <= c_mse * 1.01 + 1e-12
        assert float((ours == golden[f"q4_{case}"]).mean()) > 0.9

    @pytest.mark.parametrize("case", CASES)
    def test_q2k_matches_the_c_reference_error_parity(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        # Q2_K fitting admits representation ties (levels-to-2 at 1.5x
        # scale vs levels-to-3) whose sub-block errors agree to float
        # noise. The super-block scale re-quantization amplifies the
        # pick, so outlier-heavy data drifts furthest from the C
        # binary's own compiler-ordered arithmetic. The bound guards
        # against a mis-port, not against tie noise.
        x = golden[f"x_{case}"]

        ours = kquant_quantize_dequantize(torch.from_numpy(x), 2).numpy()

        c_mse = float(np.mean((x - golden[f"q2_{case}"]) ** 2))
        our_mse = float(np.mean((x - ours) ** 2))
        factor = 1.10 if case == "outliers" else 1.01
        # Two-sided: a systematically better fit is also a mis-port —
        # the pack ships the C behavior, not an improvement on it.
        assert c_mse / factor - 1e-12 <= our_mse <= c_mse * factor + 1e-12
        assert float((ours == golden[f"q2_{case}"]).mean()) > 0.5


class TestRoundTripProperties:
    def test_q2k_uses_at_most_four_levels_per_sub_block(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(4, 256)

        q = kquant_quantize_dequantize(w, 2)

        for sub in q.reshape(-1, 16):
            assert len(sub.unique()) <= 4

    def test_q3k_uses_at_most_eight_levels_per_sub_block(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(4, 256)

        q = kquant_quantize_dequantize(w, 3)

        for sub in q.reshape(-1, 16):
            assert len(sub.unique()) <= 8

    def test_2bit_damage_exceeds_3bit_damage(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(16, 512)

        mse2 = (w - kquant_quantize_dequantize(w, 2)).pow(2).mean()
        mse3 = (w - kquant_quantize_dequantize(w, 3)).pow(2).mean()

        assert mse3 < mse2

    def test_shape_dtype_and_device_survive_including_padding(self) -> None:
        w = torch.randn(17, 5, dtype=torch.bfloat16)

        result = kquant_quantize_dequantize(w, 2)

        assert result.shape == w.shape
        assert result.dtype == w.dtype
        assert result.device == w.device

    def test_zero_tensor_passes_through_unchanged(self) -> None:
        w = torch.zeros(2, 256)

        assert torch.equal(kquant_quantize_dequantize(w, 3), w)

    def test_input_tensor_is_never_modified(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(3, 256)
        before = w.clone()

        kquant_quantize_dequantize(w, 2)

        assert torch.equal(w, before)

    def test_uncovered_bits_raise(self) -> None:
        for bits in (5, 6):
            with pytest.raises(ValueError, match="kquant"):
                kquant_quantize_dequantize(torch.randn(1, 256), bits)

    def test_domain_coverage_mirrors_the_dispatch_table(self) -> None:
        # The CLI pre-flight validates against the domain copy; the
        # adapter gates on the dispatch table. Drift between them
        # fails an hour into a model load.
        assert set(KQUANT_PRECISIONS) == set(_ROUND_TRIPS)
        assert set(KQUANT_BITS) == set(_ROUND_TRIPS)

    @pytest.mark.parametrize("chunk_rows", [7, 8], ids=["remainder", "exact"])
    def test_slicing_is_bit_invisible(self, monkeypatch, chunk_rows: int) -> None:
        # Every fit is local to one block, so slice boundaries must
        # not change a single value — slicing only caps workspace.
        # Chunk 7 leaves a short final slice, chunk 8 divides evenly.
        torch.manual_seed(0)
        w = torch.randn(64, 256)

        whole = {b: kquant_quantize_dequantize(w, b) for b in KQUANT_BITS}
        monkeypatch.setattr(kquant, "_CHUNK_ROWS", chunk_rows)
        sliced = {b: kquant_quantize_dequantize(w, b) for b in KQUANT_BITS}

        for bits in KQUANT_BITS:
            assert torch.equal(whole[bits], sliced[bits]), bits

    def test_subnormal_scale_blocks_stay_finite(self) -> None:
        # 0 < amax < 127/FLT_MAX overflows a reciprocal to inf — the
        # C path dequantizes such blocks to zero, never NaN.
        w = torch.full((2, 256), 1e-38)
        w[1] = 0.0

        for bits in KQUANT_BITS:
            q = kquant_quantize_dequantize(w, bits)
            assert torch.isfinite(q).all(), bits

    def test_damage_shrinks_as_bits_rise(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(16, 512)

        mse = {
            bits: (w - kquant_quantize_dequantize(w, bits)).pow(2).mean().item()
            for bits in (8, 4, 3, 2)
        }

        assert mse[8] < mse[4] < mse[3] < mse[2]

    def test_kquant_and_rtn_produce_different_round_trips(self) -> None:
        # The two methods must not collapse to the same grid — the
        # flag exists because the structures differ (ADR-0018). Which
        # one damages the model more is the probe's question, not a
        # test invariant.
        torch.manual_seed(0)
        w = torch.randn(4, 512) * 0.02

        for bits in (2, 3):
            kq = kquant_quantize_dequantize(w, bits)
            rtn = rtn_quantize_dequantize(w, bits)
            assert not torch.equal(kq, rtn)
