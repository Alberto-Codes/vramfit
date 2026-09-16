"""Checks of the assisted ``Q2_0`` encoder against its reference fixtures.

The fixtures in ``tests/data/q2_0_assisted/fixtures.json`` come from
``scripts/gen_q2_0_assisted_fixtures.py``, a scalar float32
transliteration of the reference specification (ADR-0032 decision
2). Four crafted cases carry hand-checked expectations, which the
tests assert directly as well: a signed scale, a rounding tie, a
zero block, and a weighted candidate selection. Two random cases
cover bulk agreement.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")

from vramfit.adapters.outbound.gguf.q2_0_blocks import dequantize_q2_0
from vramfit.adapters.outbound.scan.q2_0_assisted import (
    Q2_0_BLOCK_BYTES,
    Q2_0_ENCODER_REVISION,
    q2_0_assisted_fit,
    q2_0_assisted_quantize_dequantize,
    q2_0_encode_rows,
)
from vramfit.adapters.outbound.scan.within_group import perturb

pytestmark = pytest.mark.unit

FIXTURES = (
    Path(__file__).parent.parent.parent / "data" / "q2_0_assisted" / "fixtures.json"
)
QK = 64


@pytest.fixture(scope="module")
def fixtures() -> dict[str, dict]:
    return json.loads(FIXTURES.read_text())


def _case(
    fixtures: dict[str, dict], name: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    case = fixtures[name]
    rows = torch.tensor(case["rows"], dtype=torch.float32)
    qw = None if case["qw"] is None else torch.tensor(case["qw"], dtype=torch.float32)
    return rows, qw


def _expected(
    fixtures: dict[str, dict], name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = [block for row in fixtures[name]["blocks"] for block in row]
    scales = torch.tensor([block["scale"] for block in blocks], dtype=torch.float32)
    levels = torch.tensor([block["levels"] for block in blocks], dtype=torch.float32)
    return scales, levels


class TestAgainstReferenceFixtures:
    @pytest.mark.parametrize(
        "name",
        [
            "signed_scale",
            "rounding_tie",
            "zero_block",
            "weighted_selection",
            "random_weighted",
            "random_unweighted",
        ],
    )
    def test_fit_matches_the_scalar_reference_exactly(
        self, fixtures: dict[str, dict], name: str
    ) -> None:
        rows, qw = _case(fixtures, name)
        scales, levels = q2_0_assisted_fit(rows, qw)
        expected_scales, expected_levels = _expected(fixtures, name)
        assert torch.equal(scales, expected_scales), name
        assert torch.equal(levels, expected_levels), name

    def test_signed_scale_case_fits_exactly_with_a_negative_scale(
        self, fixtures: dict[str, dict]
    ) -> None:
        rows, qw = _case(fixtures, "signed_scale")
        scales, levels = q2_0_assisted_fit(rows, qw)
        assert float(scales[0]) == -1.0
        assert levels[0, 0] == 2.0
        assert torch.all(levels[0, 1:] == -1.0)
        assert torch.equal(levels * scales[:, None], rows)

    def test_rounding_tie_case_rounds_half_away_from_zero(
        self, fixtures: dict[str, dict]
    ) -> None:
        rows, qw = _case(fixtures, "rounding_tie")
        scales, levels = q2_0_assisted_fit(rows, qw)
        # Half to even would keep d = 2 and code the 0.5 elements 0.
        assert float(scales[0]) == float(np.float16(np.float32(35.5) / np.float32(67)))
        assert levels[0, 0] == 2.0
        assert torch.all(levels[0, 1:] == 1.0)

    def test_zero_block_stores_zero_scale_and_level_zero_bytes(
        self, fixtures: dict[str, dict]
    ) -> None:
        rows, qw = _case(fixtures, "zero_block")
        assert q2_0_encode_rows(rows, qw) == b"\x00\x00" + b"\x55" * 16

    def test_weights_change_the_candidate_the_block_keeps(
        self, fixtures: dict[str, dict]
    ) -> None:
        rows, qw = _case(fixtures, "weighted_selection")
        weighted, _ = q2_0_assisted_fit(rows, qw)
        unweighted, _ = q2_0_assisted_fit(rows, None)
        assert float(weighted[0]) == 0.94921875
        assert float(unweighted[0]) != float(weighted[0])


class TestBlockLayout:
    def test_encoded_rows_decode_to_the_fit_through_the_torch_free_decoder(
        self, fixtures: dict[str, dict]
    ) -> None:
        rows, qw = _case(fixtures, "random_weighted")
        scales, levels = q2_0_assisted_fit(rows, qw)
        payload = q2_0_encode_rows(rows, qw)
        assert len(payload) == rows.numel() // QK * Q2_0_BLOCK_BYTES
        decoded = dequantize_q2_0(payload, rows.numel())
        assert np.array_equal(decoded, (levels * scales[:, None]).reshape(-1).numpy())

    def test_signed_scale_block_stores_the_fp16_scale_first(
        self, fixtures: dict[str, dict]
    ) -> None:
        rows, qw = _case(fixtures, "signed_scale")
        payload = q2_0_encode_rows(rows, qw)
        assert payload[:2] == np.float16(-1.0).tobytes()
        # Element 0 codes level 2 as 3, the rest code level -1 as 0.
        assert payload[2] == 0b00000011
        assert payload[3:] == bytes(15)

    def test_row_not_dividing_into_blocks_refuses(self) -> None:
        with pytest.raises(ValueError, match="do not divide into 64-element"):
            q2_0_assisted_fit(torch.zeros(1, 96), None)


class TestQuantizeDequantize:
    def test_expert_stack_fits_each_expert_against_its_own_weight_row(self) -> None:
        stack = torch.randn(2, 4, 128)
        qw = torch.rand(2, 128)
        priced = q2_0_assisted_quantize_dequantize(stack, qw)
        for expert in range(2):
            rows = stack[expert]
            scales, levels = q2_0_assisted_fit(rows, qw[expert].expand(4, 128))
            assert torch.equal(
                priced[expert], (levels * scales[:, None]).reshape(4, 128)
            )

    def test_shared_weight_row_broadcasts_across_a_matrix(self) -> None:
        weight = torch.randn(6, 64, dtype=torch.float16)
        qw = torch.rand(64)
        priced = q2_0_assisted_quantize_dequantize(weight, qw)
        assert priced.shape == weight.shape
        assert priced.dtype == weight.dtype
        scales, levels = q2_0_assisted_fit(weight.float(), qw.expand(6, 64))
        assert torch.equal(
            priced.float(), (levels * scales[:, None]).reshape(6, 64).half().float()
        )

    def test_unweighted_search_is_the_specification_weight_one_path(self) -> None:
        weight = torch.randn(3, 128)
        priced = q2_0_assisted_quantize_dequantize(weight, None)
        scales, levels = q2_0_assisted_fit(weight, None)
        assert torch.equal(priced, (levels * scales[:, None]).reshape(3, 128))

    def test_misaligned_rows_refuse_before_fitting(self) -> None:
        with pytest.raises(ValueError, match="Q2_0"):
            q2_0_assisted_quantize_dequantize(torch.randn(2, 96), torch.rand(96))
        with pytest.raises(ValueError, match="Q2_0"):
            q2_0_assisted_quantize_dequantize(torch.randn(2, 96), None)

    def test_encoder_revision_is_a_stable_token(self) -> None:
        assert Q2_0_ENCODER_REVISION == "vramfit-q2_0-assisted-1"


class TestSuccessorMethodDispatch:
    def test_weighted_nominal_two_routes_to_the_encoder(self) -> None:
        weight = torch.randn(4, 128)
        qw = torch.rand(128)
        priced = perturb(weight, 2, "p", "q0-successor", 32, qw)
        assert torch.equal(priced, q2_0_assisted_quantize_dequantize(weight, qw))

    def test_weighted_nominal_two_differs_from_the_stock_q0_imx_route(self) -> None:
        weight = torch.randn(4, 128)
        qw = torch.rand(128)
        successor = perturb(weight, 2, "p", "q0-successor", 32, qw)
        stock = perturb(weight, 2, "p", "q0", 32, qw)
        assert not torch.equal(successor, stock)

    @pytest.mark.parametrize("bits", [4, 8])
    def test_other_precisions_keep_the_q0_imx_route(self, bits: int) -> None:
        weight = torch.randn(4, 128)
        qw = torch.rand(128)
        assert torch.equal(
            perturb(weight, bits, "p", "q0-successor", 32, qw),
            perturb(weight, bits, "p", "q0", 32, qw),
        )

    def test_uncovered_nominal_two_keeps_the_reference_arithmetic(self) -> None:
        weight = torch.randn(4, 128)
        assert torch.equal(
            perturb(weight, 2, "p", "q0-successor", 32, None),
            perturb(weight, 2, "p", "q0", 32, None),
        )
