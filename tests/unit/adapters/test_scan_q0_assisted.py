"""Checks of the assisted Q4_0 port against libggml goldens.

The fixtures in ``tests/data/q0_ref/golden-assisted.npz`` hold
inputs, imatrix column weights, and the dequantized outputs of
llama.cpp's ``quantize_row_q4_0_impl`` (``ggml_quantize_chunk`` with
a non-NULL imatrix, checkout 4801e3c56) — regenerate with
``scripts/gen_q0_assisted_goldens.py``. The bar is ADR-0018's
decision 4 shape, applied by the 2026-08-21 amendment's decision 3:
clean cases reproduce the C bit-exactly, and cases where
``make_qx_quants``' candidate-scale search ties admit error parity
plus a floor on exact elements.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.q0_assisted import (
    ASSISTED_BITS,
    q0_assisted_quantize_dequantize,
)
from vramfit.adapters.outbound.scan.q0_ref import q0_ref_quantize_dequantize

pytestmark = pytest.mark.unit

GOLDEN = Path(__file__).parent.parent.parent / "data" / "q0_ref" / "golden-assisted.npz"
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
# Fixtures where the 18-candidate scale search ties: two fits with
# equal weighted error, resolved differently by the C's sequential
# sums and torch's vectorized ones. Measured exact fractions
# (2026-08-21): constant_act 0.625 at exact error parity,
# gauss_holes 0.998 at parity within 0.1 %.
TIE_CASES = frozenset({"constant_act", "gauss_holes"})


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN) as data:
        return dict(data)


class TestAgainstReference:
    @pytest.mark.parametrize("case", CASES)
    def test_assisted_q4_0_matches_the_c_within_ties(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        x = golden[f"x_{case}"]
        qw = torch.from_numpy(golden[f"qw_{case}"])

        ours = q0_assisted_quantize_dequantize(torch.from_numpy(x), 4, qw).numpy()

        reference = golden[f"q4_0_{case}"]
        if case not in TIE_CASES:
            assert np.array_equal(ours, reference), case
            return
        # Two-sided parity: a systematically better fit is also a
        # mis-port — the pack ships the C behavior.
        c_mse = float(np.mean((x - reference) ** 2))
        our_mse = float(np.mean((x - ours) ** 2))
        assert c_mse / 1.01 - 1e-12 <= our_mse <= c_mse * 1.01 + 1e-12
        assert float((ours == reference).mean()) > 0.6

    def test_the_assisted_fit_differs_from_the_reference_path(
        self, golden: dict[str, np.ndarray]
    ) -> None:
        # A silently dropped weight argument would price q0-imx cells
        # under the unassisted arithmetic.
        x = torch.from_numpy(golden["x_gauss_act"])
        qw = torch.from_numpy(golden["qw_gauss_act"])

        assisted = q0_assisted_quantize_dequantize(x, 4, qw)

        assert not torch.equal(assisted, q0_ref_quantize_dequantize(x, 4))


class TestRouting:
    def test_2bit_routes_to_the_reference_path(self) -> None:
        # quantize_q2_0 discards the imatrix — assisted 2-bit must
        # price identically to unassisted 2-bit.
        torch.manual_seed(0)
        w = torch.randn(4, 128)
        qw = torch.rand(128)

        assisted = q0_assisted_quantize_dequantize(w, 2, qw)

        assert torch.equal(assisted, q0_ref_quantize_dequantize(w, 2))

    def test_8bit_routes_to_the_reference_path(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(4, 128)
        qw = torch.rand(128)

        assisted = q0_assisted_quantize_dequantize(w, 8, qw)

        assert torch.equal(assisted, q0_ref_quantize_dequantize(w, 8))

    def test_assisted_bits_names_the_covered_set(self) -> None:
        assert sorted(ASSISTED_BITS) == [2, 4, 8]

    def test_3bit_refuses_with_the_q0_message(self) -> None:
        with pytest.raises(ValueError, match="q0 covers bits"):
            q0_assisted_quantize_dequantize(torch.randn(2, 64), 3, torch.rand(64))


class TestFusedExpertStacks:
    def test_each_expert_fits_against_its_own_weight_row(self) -> None:
        # 2-D weights on a 3-D stack must reproduce a per-expert
        # loop exactly — llama-quant.cpp applies imatrix + i03 * ne0.
        torch.manual_seed(1)
        stack = torch.randn(3, 5, 64)
        qw = torch.rand(3, 64) + 0.05

        fused = q0_assisted_quantize_dequantize(stack, 4, qw)

        per_expert = torch.stack(
            [q0_assisted_quantize_dequantize(stack[i], 4, qw[i]) for i in range(3)]
        )
        assert torch.equal(fused, per_expert)

    def test_distinct_weight_rows_change_the_fit(self) -> None:
        torch.manual_seed(2)
        stack = torch.randn(2, 4, 64)
        same = torch.rand(64) + 0.05
        flat = q0_assisted_quantize_dequantize(stack, 4, same)
        skewed = torch.stack([same, same * torch.rand(64) * 10 + 0.01])

        fused = q0_assisted_quantize_dequantize(stack, 4, skewed)

        assert torch.equal(fused[0], flat[0])
        assert not torch.equal(fused[1], flat[1])


class TestValidation:
    def test_2d_weights_on_a_2d_parameter_refuse(self) -> None:
        with pytest.raises(ValueError, match="3-D expert stack"):
            q0_assisted_quantize_dequantize(torch.randn(4, 64), 4, torch.rand(4, 64))

    def test_2d_weights_with_a_wrong_expert_count_refuse(self) -> None:
        with pytest.raises(ValueError, match="3-D expert stack"):
            q0_assisted_quantize_dequantize(torch.randn(3, 4, 64), 4, torch.rand(2, 64))

    def test_wrong_length_weights_refuse(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D with 64 entries"):
            q0_assisted_quantize_dequantize(torch.randn(4, 64), 4, torch.rand(32))

    def test_misaligned_rows_refuse(self) -> None:
        with pytest.raises(ValueError, match="32-element Q4_0"):
            q0_assisted_quantize_dequantize(torch.randn(4, 48), 4, torch.rand(48))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_garbage_weights_refuse(self, bad: float) -> None:
        qw = torch.rand(64)
        qw[7] = bad
        with pytest.raises(ValueError, match="finite and non-negative"):
            q0_assisted_quantize_dequantize(torch.randn(4, 64), 4, qw)

    def test_checks_run_before_the_2bit_route_discards_the_weights(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D with 64 entries"):
            q0_assisted_quantize_dequantize(torch.randn(4, 64), 2, torch.rand(32))


class TestShapePreservation:
    def test_output_keeps_shape_dtype_and_values_change(self) -> None:
        torch.manual_seed(3)
        w = torch.randn(6, 96, dtype=torch.bfloat16)
        qw = torch.rand(96) + 0.05

        out = q0_assisted_quantize_dequantize(w, 4, qw)

        assert out.shape == w.shape
        assert out.dtype == w.dtype
        assert not torch.equal(out, w)

    def test_the_input_is_never_modified(self) -> None:
        torch.manual_seed(4)
        w = torch.randn(4, 64)
        before = w.clone()

        q0_assisted_quantize_dequantize(w, 4, torch.rand(64))

        assert torch.equal(w, before)
