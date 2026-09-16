"""Checks of the pre-encoding fields on the pack record (ADR-0032)."""

from __future__ import annotations

import pytest

from vramfit.domain.pack import PackResult, PreEncodeCost, TypeOverride

pytestmark = pytest.mark.unit


def result(
    *,
    imatrix_path: str | None = "m.gguf",
    pre_encoded: tuple[str, ...] = (),
    q2_0_encoder: str | None = None,
    pre_encode_cost: PreEncodeCost | None = None,
) -> PackResult:
    return PackResult(
        packed_bytes=100,
        base_type="Q2_K",
        token_embedding_type=None,
        output_tensor_type=None,
        overrides=(TypeOverride(r"blk\.0\.", "q2_0"),),
        imatrix_path=imatrix_path,
        pre_encoded=pre_encoded,
        q2_0_encoder=q2_0_encoder,
        pre_encode_cost=pre_encode_cost,
    )


COST = PreEncodeCost(peak_rss_bytes=1, mixed_gguf_bytes=2, payload_bytes=3)


class TestPreEncodeCost:
    def test_records_the_three_measurements(self) -> None:
        assert (COST.peak_rss_bytes, COST.mixed_gguf_bytes, COST.payload_bytes) == (
            1,
            2,
            3,
        )

    def test_negative_measurement_refuses(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            PreEncodeCost(peak_rss_bytes=-1, mixed_gguf_bytes=0, payload_bytes=0)


class TestPackResultPreEncoding:
    def test_defaults_record_no_stage(self) -> None:
        r = result()
        assert r.pre_encoded == ()
        assert r.q2_0_encoder is None
        assert r.pre_encode_cost is None

    def test_full_record_holds(self) -> None:
        r = result(
            pre_encoded=("blk.0.ffn_down_exps.weight",),
            q2_0_encoder="rev",
            pre_encode_cost=COST,
        )
        assert r.pre_encoded == ("blk.0.ffn_down_exps.weight",)

    def test_pre_encoded_requires_an_imatrix(self) -> None:
        with pytest.raises(ValueError, match="requires an imatrix_path"):
            result(
                imatrix_path=None,
                pre_encoded=("x",),
                q2_0_encoder="rev",
                pre_encode_cost=COST,
            )

    def test_empty_name_refuses(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            result(pre_encoded=("",), q2_0_encoder="rev", pre_encode_cost=COST)

    @pytest.mark.parametrize(
        ("pre_encoded", "encoder", "cost"),
        [
            (("x",), None, None),
            ((), "rev", None),
            ((), None, COST),
            (("x",), "rev", None),
            ((), "rev", COST),
        ],
        ids=["tensors-only", "encoder-only", "cost-only", "no-cost", "no-tensors"],
    )
    def test_half_stated_stage_refuses(
        self,
        pre_encoded: tuple[str, ...],
        encoder: str | None,
        cost: PreEncodeCost | None,
    ) -> None:
        with pytest.raises(ValueError, match="set all three or none"):
            result(pre_encoded=pre_encoded, q2_0_encoder=encoder, pre_encode_cost=cost)

    def test_empty_encoder_refuses(self) -> None:
        with pytest.raises(ValueError, match="q2_0_encoder must not be empty"):
            result(pre_encoded=("x",), q2_0_encoder="", pre_encode_cost=COST)
