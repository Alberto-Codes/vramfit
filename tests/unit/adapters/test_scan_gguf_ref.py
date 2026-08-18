"""Checks of the `gguf-ref` port against libggml goldens.

The fixtures in ``tests/data/gguf_ref/golden.npz`` hold inputs and
the dequantized outputs of llama.cpp's reference quantizers for
``Q2_0``, ``Q4_0``, and ``Q8_0`` (``ggml_quantize_chunk``, no
imatrix). ADR-0018's 2026-08-17 amendment sets the bar at
**bit-exact** for all three: neither type fits a candidate grid, so
none carries ``Q2_K``'s representation ties. The suite sits in the
unit tier so the port is guarded on every commit — it needs no
model, no card, and no network, and it skips cleanly where the scan
extra is absent (ADR-0009).

Three cases carry the rules a plain port gets wrong. ``halfway``
lands 459 elements exactly on ``.5``, where ``Q2_0``'s ``roundf``
disagrees with a half-to-even round. ``ties`` gives all 64 ``Q4_0``
blocks a signed tie at the absmax, where the C keeps the *first*
maximum and decides the block's sign by it. ``subnormal`` overflows
every block's reciprocal scale to infinity.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")

from vramfit.adapters.outbound.scan import kquant
from vramfit.adapters.outbound.scan.gguf_ref import (
    _ROUND_TRIPS,
    GGUF_REF_BITS,
    QK2_0,
    QK4_0,
    gguf_ref_quantize_dequantize,
)
from vramfit.domain.scan import GGUF_REF_PRECISIONS

pytestmark = pytest.mark.unit

GOLDEN = Path(__file__).parent.parent.parent / "data" / "gguf_ref" / "golden.npz"
CASES = (
    "gauss",
    "outliers",
    "constant",
    "zeros",
    "tiny",
    "subnormal",
    "ties",
    "halfway",
    "near_half",
)
# The routed-expert row lengths on the 30B target (#159). Every
# block size here divides both, which is the point of the method.
EXPERT_ROWS = (2688, 1856)


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN) as data:
        return dict(data)


class TestAgainstReference:
    @pytest.mark.parametrize("case", CASES)
    def test_q2_0_matches_the_c_reference_exactly(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        x = torch.from_numpy(golden[f"x_{case}"])

        ours = gguf_ref_quantize_dequantize(x, 2).numpy()

        assert np.array_equal(ours, golden[f"q2_0_{case}"])

    @pytest.mark.parametrize("case", CASES)
    def test_q4_0_matches_the_c_reference_exactly(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        x = torch.from_numpy(golden[f"x_{case}"])

        ours = gguf_ref_quantize_dequantize(x, 4).numpy()

        assert np.array_equal(ours, golden[f"q4_0_{case}"])

    @pytest.mark.parametrize("case", CASES)
    def test_q8_0_matches_the_c_reference_exactly(
        self, golden: dict[str, np.ndarray], case: str
    ) -> None:
        # Nominal 8 reuses the ADR-0018 port rather than porting
        # Q8_0 twice. These fixtures prove the reuse is sound.
        x = torch.from_numpy(golden[f"x_{case}"])

        ours = gguf_ref_quantize_dequantize(x, 8).numpy()

        assert np.array_equal(ours, golden[f"q8_0_{case}"])

    def test_a_half_to_even_round_would_fail_the_q2_0_tie_case(
        self, golden: dict[str, np.ndarray]
    ) -> None:
        # Guards the guard: if `halfway` stopped holding exact ties,
        # the bit-exact test above would pass vacuously.
        x = torch.from_numpy(golden["x_halfway"]).reshape(-1, QK2_0)
        scale = x.abs().amax(dim=1, keepdim=True)

        half_to_even = torch.round(x / scale).clamp(-1, 2) * scale.half().float()

        expected = torch.from_numpy(golden["q2_0_halfway"]).reshape(-1, QK2_0)
        assert not torch.equal(half_to_even, expected)

    def test_a_shift_and_truncate_round_would_fail_the_near_half_case(
        self, golden: dict[str, np.ndarray]
    ) -> None:
        # `trunc(v + 0.5 * sign(v))` is the obvious way to write
        # round-half-away, and the add is inexact: 0.49999997 + 0.5
        # reaches the exact midpoint under 1.0 and snaps up, where
        # `roundf` returns 0. This case pins the difference.
        x = torch.from_numpy(golden["x_near_half"]).reshape(-1, QK2_0)
        scale = x.abs().amax(dim=1, keepdim=True)
        v = x / scale

        shifted = torch.trunc(v + 0.5 * torch.sign(v)).clamp(-1, 2)
        shifted = shifted * scale.half().float()

        expected = torch.from_numpy(golden["q2_0_near_half"]).reshape(-1, QK2_0)
        assert not torch.equal(shifted, expected)

    def test_a_last_maximum_pick_would_fail_the_q4_0_tie_case(
        self, golden: dict[str, np.ndarray]
    ) -> None:
        # The C keeps the first strict maximum. A block holding +a
        # and -a flips sign under a last-maximum pick, so this case
        # is what pins the index reduction.
        blocks = torch.from_numpy(golden["x_ties"]).reshape(-1, QK4_0)
        magnitude = blocks.abs()
        amax = magnitude.amax(dim=1, keepdim=True)

        signed_first = blocks.gather(1, (magnitude == amax).float().argmax(1)[:, None])
        signed_last = blocks.gather(
            1, (QK4_0 - 1 - (magnitude == amax).flip(1).float().argmax(1))[:, None]
        )

        assert not torch.equal(signed_first, signed_last)


class TestRoundTripProperties:
    def test_q2_0_reaches_three_levels_per_block(self) -> None:
        # The reference clamps round(w/amax) to [-1, 2] and
        # |w| <= amax caps it at 1, so the fourth code is dead.
        torch.manual_seed(0)
        w = torch.randn(4, QK2_0 * 8)

        q = gguf_ref_quantize_dequantize(w, 2)

        for block in q.reshape(-1, QK2_0):
            assert len(block.unique()) <= 3

    def test_q4_0_reaches_sixteen_levels_per_block(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(4, QK4_0 * 8)

        q = gguf_ref_quantize_dequantize(w, 4)

        for block in q.reshape(-1, QK4_0):
            assert len(block.unique()) <= 16

    @pytest.mark.parametrize("bits", GGUF_REF_PRECISIONS)
    @pytest.mark.parametrize("row", EXPERT_ROWS)
    def test_the_routed_expert_rows_round_trip(self, bits: int, row: int) -> None:
        # 2688 and 1856 refuse every 256-element super-block type.
        # Reaching them is this method's whole reason to exist.
        torch.manual_seed(0)
        w = torch.randn(2, row)

        q = gguf_ref_quantize_dequantize(w, bits)

        assert q.shape == w.shape
        assert torch.isfinite(q).all()

    def test_the_input_is_never_modified(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(2, 256)
        before = w.clone()

        gguf_ref_quantize_dequantize(w, 2)

        assert torch.equal(w, before)

    def test_shape_dtype_and_device_survive(self) -> None:
        w = torch.randn(3, 5, 64, dtype=torch.bfloat16)

        q = gguf_ref_quantize_dequantize(w, 4)

        assert q.shape == w.shape
        assert q.dtype == w.dtype
        assert q.device == w.device

    @pytest.mark.parametrize("chunk_rows", [7, 8], ids=["remainder", "exact"])
    def test_slicing_is_bit_invisible(self, monkeypatch, chunk_rows: int) -> None:
        # Every fit is local to one block, so a slice boundary must
        # not change a value. Slicing only caps the fp32 workspace.
        # `_sliced` is shared with kquant and reads the constant from
        # that module, so the patch lands there.
        torch.manual_seed(0)
        w = torch.randn(64, QK2_0)

        whole = {bits: gguf_ref_quantize_dequantize(w, bits) for bits in GGUF_REF_BITS}
        monkeypatch.setattr(kquant, "_CHUNK_ROWS", chunk_rows)
        sliced = {bits: gguf_ref_quantize_dequantize(w, bits) for bits in GGUF_REF_BITS}

        for bits in GGUF_REF_BITS:
            assert torch.equal(whole[bits], sliced[bits]), bits


class TestRefusals:
    @pytest.mark.parametrize("bits", (3, 5, 6))
    def test_uncovered_bits_are_refused(self, bits: int) -> None:
        # ADR-0028 refuses nominal 3 at pack, and 5 and 6 have no
        # port. A silent fallback would price an unreachable frame.
        with pytest.raises(ValueError, match="gguf covers bits"):
            gguf_ref_quantize_dequantize(torch.randn(2, 64), bits)

        assert bits not in GGUF_REF_BITS

    def test_the_domain_copy_mirrors_the_dispatch_table(self) -> None:
        # The CLI validates precisions against the domain copy
        # before a model loads, so the two must not drift.
        assert sorted(GGUF_REF_PRECISIONS) == sorted(GGUF_REF_BITS)
        assert sorted(_ROUND_TRIPS) == sorted(GGUF_REF_PRECISIONS)
