"""Checks of the imatrix adapter: name mapping and GGUF loading.

The loader tests write miniature imatrix GGUF files with
``gguf.GGUFWriter`` — the same library the adapter reads with — so
they stay hermetic. They skip cleanly where the scan extra is
absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
gguf = pytest.importorskip("gguf", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")

from quantfit.adapters.outbound.scan.imatrix import (
    assisted_weights_for_params,
    gguf_tensor_name,
    load_imatrix,
)

pytestmark = pytest.mark.unit


def _write_imatrix(
    path: Path,
    tensors: dict[str, np.ndarray],
    general_type: str | None = "imatrix",
) -> Path:
    writer = gguf.GGUFWriter(str(path), "imatrix")
    if general_type is not None:
        writer.add_type(general_type)
    for name, data in tensors.items():
        writer.add_tensor(name, data.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TestGgufTensorName:
    @pytest.mark.parametrize(
        ("param", "expected"),
        [
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.41.self_attn.k_proj.weight", "blk.41.attn_k.weight"),
            ("model.layers.7.self_attn.v_proj.weight", "blk.7.attn_v.weight"),
            ("model.layers.7.self_attn.o_proj.weight", "blk.7.attn_output.weight"),
            ("model.layers.79.mlp.gate_proj.weight", "blk.79.ffn_gate.weight"),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            ("model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"),
            ("lm_head.weight", "output.weight"),
            ("model.embed_tokens.weight", "token_embd.weight"),
        ],
    )
    def test_llama_family_names_map(self, param: str, expected: str) -> None:
        assert gguf_tensor_name(param) == expected

    @pytest.mark.parametrize(
        "param",
        [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.unknown_proj.weight",
            "transformer.h.0.attn.c_attn.weight",
        ],
    )
    def test_unmapped_names_return_none(self, param: str) -> None:
        assert gguf_tensor_name(param) is None


class TestLoadImatrix:
    def test_weights_are_sums_over_counts(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.arange(8.0),
                "blk.0.attn_q.weight.counts": np.array([4.0]),
            },
        )

        weights = load_imatrix(path)

        expected = torch.arange(8.0) / 4.0
        assert torch.equal(weights["blk.0.attn_q.weight"], expected)

    def test_zero_count_columns_weigh_one(self, tmp_path) -> None:
        # The llama-quantize load formula: a column with no chunks
        # falls back to weight 1, not 0 — zero would erase it from
        # the fit.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([3.0, 5.0]),
                "t.weight.counts": np.array([0.0]),
            },
        )

        weights = load_imatrix(path)

        assert torch.equal(weights["t.weight"], torch.ones(2))

    def test_per_expert_counts_normalize_row_wise(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([2.0, 4.0, 30.0, 90.0]),
                "t.weight.counts": np.array([2.0, 3.0]),
            },
        )

        weights = load_imatrix(path)

        assert torch.equal(weights["t.weight"], torch.tensor([1.0, 2.0, 10.0, 30.0]))

    def test_sum_without_counts_twin_raises(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {"t.weight.in_sum2": np.array([1.0])},
        )

        with pytest.raises(ValueError, match="counts twin"):
            load_imatrix(path)

    def test_non_imatrix_file_is_refused(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "not-im.gguf",
            {"t.weight.in_sum2": np.array([1.0]), "t.weight.counts": np.array([1.0])},
            general_type="model",
        )

        with pytest.raises(ValueError, match="not an imatrix"):
            load_imatrix(path)


class TestAssistedWeightsForParams:
    def test_covered_and_uncovered_split(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones(8),
                "blk.0.attn_q.weight.counts": np.array([1.0]),
            },
        )
        shapes = {
            "model.layers.0.self_attn.q_proj.weight": (8, 8),
            "model.layers.1.self_attn.q_proj.weight": (8, 8),
            "model.embed_tokens.weight": (16, 8),
        }

        covered, uncovered = assisted_weights_for_params(path, shapes)

        assert set(covered) == {"model.layers.0.self_attn.q_proj.weight"}
        assert uncovered == (
            "model.layers.1.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
        )

    def test_row_length_mismatch_raises(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones(8),
                "blk.0.attn_q.weight.counts": np.array([1.0]),
            },
        )

        with pytest.raises(ValueError, match="rows have 16"):
            assisted_weights_for_params(
                path, {"model.layers.0.self_attn.q_proj.weight": (8, 16)}
            )
