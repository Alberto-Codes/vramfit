from __future__ import annotations

import json

import pytest

from quantfit.budget import (
    Budget,
    ModelShape,
    format_size,
    kv_bytes_per_token,
    kv_cache_bytes,
    parse_size,
)

NEMOTRON_SHAPE = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)


@pytest.mark.unit
class TestParseSize:
    def test_gib_returns_binary_bytes(self) -> None:
        assert parse_size("24GiB") == 24 * 2**30

    def test_gb_returns_decimal_bytes(self) -> None:
        assert parse_size("24GB") == 24 * 10**9

    def test_bare_integer_is_bytes(self) -> None:
        assert parse_size("1073741824") == 2**30

    def test_fractional_size_with_unit(self) -> None:
        assert parse_size("1.5GiB") == int(1.5 * 2**30)

    def test_whitespace_and_case_tolerated(self) -> None:
        assert parse_size(" 4 gib ") == 4 * 2**30

    @pytest.mark.parametrize("bad", ["", "GiB", "12XB", "1.2.3GiB", "-4GiB"])
    def test_malformed_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError, match="not a recognizable size"):
            parse_size(bad)


@pytest.mark.unit
class TestFormatSize:
    def test_renders_gib_with_two_decimals(self) -> None:
        assert format_size(2 * 2**30) == "2.00 GiB"

    def test_renders_small_values_as_bytes(self) -> None:
        assert format_size(512) == "512 B"

    def test_round_trips_with_parse_size(self) -> None:
        assert parse_size(format_size(4 * 2**20)) == 4 * 2**20


@pytest.mark.unit
class TestKvMath:
    def test_bytes_per_token_nemotron_shape_matches_hand_calc(self) -> None:
        # 2 (K+V) x 49 layers x 8 kv_heads x 128 head_dim x 2 bytes
        assert kv_bytes_per_token(NEMOTRON_SHAPE) == 200_704

    def test_bytes_per_token_fp8_is_half_of_fp16(self) -> None:
        fp16 = kv_bytes_per_token(NEMOTRON_SHAPE, "fp16")

        assert kv_bytes_per_token(NEMOTRON_SHAPE, "fp8") == fp16 // 2

    def test_unknown_dtype_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            kv_bytes_per_token(NEMOTRON_SHAPE, "int4")

    def test_cache_scales_linearly_with_context_and_sequences(self) -> None:
        base = kv_cache_bytes(NEMOTRON_SHAPE, context=1024)

        assert kv_cache_bytes(NEMOTRON_SHAPE, context=2048) == 2 * base
        assert kv_cache_bytes(NEMOTRON_SHAPE, context=1024, sequences=3) == 3 * base

    def test_heterogeneous_shape_sums_per_layer_heads(self) -> None:
        shape = ModelShape(kv_heads_per_layer=(8, 4), head_dim=128)

        assert kv_bytes_per_token(shape) == 2 * 128 * (8 + 4) * 2


@pytest.mark.unit
class TestModelShapeFromConfig:
    def _decilm_config(self) -> dict:
        def block() -> dict:
            return {
                "attention": {"n_heads_in_group": 8, "no_op": False},
                "ffn": {"ffn_mult": 5.25},
            }

        no_op_block = {
            "attention": {"n_heads_in_group": None, "no_op": True},
            "ffn": {"ffn_mult": 1.0},
        }
        return {
            "num_attention_heads": 64,
            "hidden_size": 8192,
            "block_configs": [block(), block(), no_op_block, block()],
        }

    def test_decilm_config_skips_no_op_blocks(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(self._decilm_config()))

        shape = ModelShape.from_config_json(path)

        assert shape.kv_heads_per_layer == (8, 8, 8)
        assert shape.head_dim == 128

    def test_decilm_config_handles_varying_gqa_group_size(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][1]["attention"]["n_heads_in_group"] = 16
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = ModelShape.from_config_json(path)

        assert shape.kv_heads_per_layer == (8, 4, 8)

    def test_decilm_config_missing_group_size_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][0]["attention"]["n_heads_in_group"] = None
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="n_heads_in_group"):
            ModelShape.from_config_json(path)

    def test_llama_config_uniform_layers(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "num_attention_heads": 32,
                    "hidden_size": 4096,
                }
            )
        )

        shape = ModelShape.from_config_json(path)

        assert shape.kv_heads_per_layer == (8,) * 32
        assert shape.head_dim == 128

    def test_llama_config_explicit_head_dim_wins(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 16,
                    "hidden_size": 2048,
                    "head_dim": 64,
                }
            )
        )

        shape = ModelShape.from_config_json(path)

        assert shape.head_dim == 64

    def test_missing_fields_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"num_hidden_layers": 2}))

        with pytest.raises(ValueError, match="num_key_value_heads"):
            ModelShape.from_config_json(path)

    def test_invalid_json_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{oops")

        with pytest.raises(ValueError, match="invalid JSON"):
            ModelShape.from_config_json(path)


@pytest.mark.unit
class TestBudget:
    def test_weight_budget_subtracts_kv_and_overhead(self) -> None:
        ledger = Budget(
            vram_total_bytes=parse_size("24GiB"),
            kv_cache_bytes=parse_size("3GiB"),
            runtime_overhead_bytes=parse_size("2GiB"),
        )

        assert ledger.weight_budget_bytes == parse_size("19GiB")

    def test_weight_budget_negative_when_overcommitted(self) -> None:
        ledger = Budget(
            vram_total_bytes=parse_size("8GiB"),
            kv_cache_bytes=parse_size("6GiB"),
            runtime_overhead_bytes=parse_size("4GiB"),
        )

        assert ledger.weight_budget_bytes < 0
