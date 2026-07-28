from __future__ import annotations

import json

import pytest

from quantfit.adapters.outbound.hf_config import shape_from_config_json


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

        shape = shape_from_config_json(path)

        assert shape.kv_heads_per_layer == (8, 8, 8)
        assert shape.head_dim == 128

    def test_decilm_config_handles_varying_gqa_group_size(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][1]["attention"]["n_heads_in_group"] = 16
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert shape.kv_heads_per_layer == (8, 4, 8)

    def test_decilm_config_missing_group_size_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][0]["attention"]["n_heads_in_group"] = None
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="n_heads_in_group"):
            shape_from_config_json(path)

    def test_decilm_config_non_divisible_group_size_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][0]["attention"]["n_heads_in_group"] = 6
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="does not divide"):
            shape_from_config_json(path)

    def test_decilm_config_group_size_above_heads_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][0]["attention"]["n_heads_in_group"] = 128
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="does not divide"):
            shape_from_config_json(path)

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

        shape = shape_from_config_json(path)

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

        shape = shape_from_config_json(path)

        assert shape.head_dim == 64

    def test_missing_fields_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"num_hidden_layers": 2}))

        with pytest.raises(ValueError, match="num_key_value_heads"):
            shape_from_config_json(path)

    def test_invalid_json_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{oops")

        with pytest.raises(ValueError, match="invalid JSON"):
            shape_from_config_json(path)
