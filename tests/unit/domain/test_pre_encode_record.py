"""Checks of the pre-encoding fields on the pack record (ADR-0032)."""

from __future__ import annotations

import pytest

from vramfit.domain.pack import PackResult, TypeOverride

pytestmark = pytest.mark.unit


def result(
    *,
    imatrix_path: str | None = "m.gguf",
    pre_encoded: tuple[str, ...] = (),
    q2_0_encoder: str | None = None,
    pre_encode_assisted: bool = False,
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
        pre_encode_assisted=pre_encode_assisted,
    )


class TestPackResultPreEncoding:
    def test_defaults_record_no_stage(self) -> None:
        r = result()
        assert r.pre_encoded == ()
        assert r.q2_0_encoder is None
        assert not r.pre_encode_assisted

    def test_full_record_holds(self) -> None:
        r = result(
            pre_encoded=("blk.0.ffn_down_exps.weight",),
            q2_0_encoder="rev",
            pre_encode_assisted=True,
        )
        assert r.pre_encoded == ("blk.0.ffn_down_exps.weight",)
        assert r.q2_0_encoder == "rev"
        assert r.pre_encode_assisted

    def test_assisted_pre_encoding_requires_an_imatrix(self) -> None:
        with pytest.raises(ValueError, match="requires an imatrix_path"):
            result(
                imatrix_path=None,
                pre_encoded=("x",),
                q2_0_encoder="rev",
                pre_encode_assisted=True,
            )

    def test_matrix_free_pre_encoding_holds_without_an_imatrix(self) -> None:
        # The matrix-free encoder reads no matrix, so the record
        # names none (ADR-0018, 2026-09-17 amendment).
        r = result(imatrix_path=None, pre_encoded=("x",), q2_0_encoder="rev")
        assert r.pre_encoded == ("x",)
        assert not r.pre_encode_assisted

    def test_assisted_claim_without_a_stage_refuses(self) -> None:
        with pytest.raises(ValueError, match="requires pre_encoded tensors"):
            result(pre_encode_assisted=True)

    def test_empty_name_refuses(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            result(pre_encoded=("",), q2_0_encoder="rev")

    @pytest.mark.parametrize(
        ("pre_encoded", "encoder"),
        [(("x",), None), ((), "rev")],
        ids=["tensors-only", "encoder-only"],
    )
    def test_half_stated_stage_refuses(
        self, pre_encoded: tuple[str, ...], encoder: str | None
    ) -> None:
        with pytest.raises(ValueError, match="set both or neither"):
            result(pre_encoded=pre_encoded, q2_0_encoder=encoder)

    def test_empty_encoder_refuses(self) -> None:
        with pytest.raises(ValueError, match="q2_0_encoder must not be empty"):
            result(pre_encoded=("x",), q2_0_encoder="")
