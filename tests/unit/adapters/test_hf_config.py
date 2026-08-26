from __future__ import annotations

import json

import pytest

from vramfit.adapters.outbound.hf_config import shape_from_config_json


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

    def test_decilm_config_boolean_group_size_raises(self, tmp_path) -> None:
        # `bool` subclasses `int`, so `true` read as one head group and
        # returned a shape with no report (#348).
        config = self._decilm_config()
        config["block_configs"][0]["attention"]["n_heads_in_group"] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a positive integer"):
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

    def test_decilm_config_skips_replace_with_linear_blocks(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][0]["attention"] = {
            "replace_with_linear": True,
            "no_op": False,
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert shape.kv_heads_per_layer == (8, 8)

    def test_decilm_config_non_bool_skip_flag_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][0]["attention"]["no_op"] = "false"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a boolean"):
            shape_from_config_json(path)

    def test_decilm_config_all_blocks_skipped_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        for block in config["block_configs"]:
            block["attention"]["no_op"] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="no block has real attention"):
            shape_from_config_json(path)

    def test_decilm_config_non_list_block_configs_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"] = {"0": "not-a-list"}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a list"):
            shape_from_config_json(path)

    def test_decilm_config_block_without_attention_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][1] = {"ffn": {}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="has no attention object"):
            shape_from_config_json(path)

    def test_non_divisible_hidden_size_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 64,
                    "hidden_size": 8190,
                }
            )
        )

        with pytest.raises(ValueError, match="not divisible"):
            shape_from_config_json(path)

    def test_present_but_invalid_head_dim_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 16,
                    "hidden_size": 2048,
                    "head_dim": 64.0,
                }
            )
        )

        with pytest.raises(ValueError, match="head_dim"):
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

    def test_duplicate_key_raises(self, tmp_path) -> None:
        # The publisher owns this file. Taking the last value would give
        # a 32-layer budget for a config that also says 64 (#283).
        path = tmp_path / "config.json"
        path.write_text(
            '{"num_hidden_layers": 64, "num_hidden_layers": 32, '
            '"num_key_value_heads": 8, "num_attention_heads": 32, '
            '"hidden_size": 4096}'
        )

        with pytest.raises(ValueError, match='duplicate key "num_hidden_layers"'):
            shape_from_config_json(path)

    def test_duplicate_key_in_a_nested_object_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        raw = json.dumps(config).replace(
            '"n_heads_in_group": 8', '"n_heads_in_group": 8, "n_heads_in_group": 16', 1
        )
        path = tmp_path / "config.json"
        path.write_text(raw)

        with pytest.raises(ValueError, match='duplicate key "n_heads_in_group"'):
            shape_from_config_json(path)

    def test_layer_count_above_the_64_bit_range_raises(self, tmp_path) -> None:
        # `ModelShape.uniform` repeats a tuple by this count, which
        # raised `OverflowError` past the CLI's clause and printed a
        # traceback where ADR-0011 decision 5 wants an `error:` line
        # (#314).
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 10**30,
                    "num_key_value_heads": 8,
                    "num_attention_heads": 32,
                    "hidden_size": 4096,
                }
            )
        )

        with pytest.raises(ValueError, match="the largest integer this format"):
            shape_from_config_json(path)

    def test_kv_heads_one_past_the_64_bit_range_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2**63,
                    "num_attention_heads": 32,
                    "hidden_size": 4096,
                }
            )
        )

        with pytest.raises(ValueError, match='"num_key_value_heads" exceeds'):
            shape_from_config_json(path)

    def test_head_dim_at_the_largest_representable_integer_parses(
        self, tmp_path
    ) -> None:
        # The bound refuses what the format cannot carry and nothing
        # else. The domain rules whether the value means anything
        # (ADR-0008, 2026-08-16).
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 8,
                    "num_attention_heads": 32,
                    "head_dim": 2**63 - 1,
                }
            )
        )

        shape = shape_from_config_json(path)

        assert shape.head_dim == 2**63 - 1

    def test_head_dim_above_the_64_bit_range_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 8,
                    "num_attention_heads": 32,
                    "head_dim": 10**25,
                }
            )
        )

        with pytest.raises(ValueError, match='"head_dim" exceeds'):
            shape_from_config_json(path)

    def test_decilm_group_size_above_the_64_bit_range_raises(self, tmp_path) -> None:
        # The divisibility refusal below this bound renders the value
        # into its message, so an unbounded literal would reach the
        # operator's terminal in full.
        config = self._decilm_config()
        config["block_configs"][0]["attention"]["n_heads_in_group"] = 10**40
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match=r"n_heads_in_group exceeds \d+"):
            shape_from_config_json(path)

    def _text_config(self) -> dict:
        # The container shape of the official Gemma 4 family (#420):
        # decoder geometry nested under `text_config`, no decoder
        # fields at the top level. The defaults here are the supported
        # subset; each unrepresentable marker gets its own test.
        return {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "vision_config": {"model_type": "siglip_vision_model"},
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "num_attention_heads": 8,
                "hidden_size": 1024,
                "layer_types": ["full_attention"] * 4,
                "attention_k_eq_v": False,
                "num_kv_shared_layers": 0,
            },
        }

    def test_text_config_with_uniform_full_attention_parses(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(self._text_config()))

        shape = shape_from_config_json(path)

        assert shape.kv_heads_per_layer == (2, 2, 2, 2)
        assert shape.head_dim == 128

    def test_text_config_not_an_object_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"] = "gemma4_text"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match='"text_config" must be a JSON object'):
            shape_from_config_json(path)

    def test_text_config_beside_top_level_layer_count_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["num_hidden_layers"] = 4
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="decoder config is ambiguous"):
            shape_from_config_json(path)

    def test_text_config_beside_block_configs_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["block_configs"] = self._decilm_config()["block_configs"]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="decoder config is ambiguous"):
            shape_from_config_json(path)

    def test_text_config_mixed_layer_types_raises(self, tmp_path) -> None:
        # The Gemma 4 31B pattern: 50 sliding layers, 10 global, 5:1.
        # Flattening to uniform would price a wrong KV cache (#421).
        config = self._text_config()
        config["text_config"]["num_hidden_layers"] = 60
        config["text_config"]["layer_types"] = (
            ["sliding_attention"] * 5 + ["full_attention"]
        ) * 10

        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="cannot represent yet"):
            shape_from_config_json(path)

    def test_text_config_all_sliding_layer_types_raises(self, tmp_path) -> None:
        # Uniform but window-capped: still not representable, KV does
        # not scale with the full context.
        config = self._text_config()
        config["text_config"]["layer_types"] = ["sliding_attention"] * 4
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="cannot represent yet"):
            shape_from_config_json(path)

    def test_text_config_non_list_layer_types_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["layer_types"] = "full_attention"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a list of strings"):
            shape_from_config_json(path)

    def test_text_config_k_eq_v_true_raises(self, tmp_path) -> None:
        # Gemma 4 12B/26B-A4B/31B store one tensor per token on global
        # layers. `kv_cache_bytes` prices an independent K and V pair.
        config = self._text_config()
        config["text_config"]["attention_k_eq_v"] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="attention_k_eq_v"):
            shape_from_config_json(path)

    def test_text_config_non_bool_k_eq_v_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["attention_k_eq_v"] = "false"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="attention_k_eq_v must be a boolean"):
            shape_from_config_json(path)

    def test_text_config_shared_kv_layers_raises(self, tmp_path) -> None:
        # E2B declares 20 shared-KV layers; those own no fresh cache.
        config = self._text_config()
        config["text_config"]["num_kv_shared_layers"] = 20
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="shared-KV layers"):
            shape_from_config_json(path)

    def test_text_config_negative_shared_kv_layers_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["num_kv_shared_layers"] = -1
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            shape_from_config_json(path)

    def test_text_config_global_head_dim_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["global_head_dim"] = 512
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="split local/global"):
            shape_from_config_json(path)

    def test_text_config_global_kv_heads_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["num_global_key_value_heads"] = 4
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="split local/global"):
            shape_from_config_json(path)

    def test_text_config_missing_field_names_the_nested_key(self, tmp_path) -> None:
        config = self._text_config()
        del config["text_config"]["num_key_value_heads"]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError,
            match=r"config\.json: text_config\.num_key_value_heads",
        ):
            shape_from_config_json(path)

    def test_integer_literal_past_the_digit_limit_names_the_file(
        self, tmp_path
    ) -> None:
        # `json.loads` raises a plain `ValueError` above
        # `sys.get_int_max_str_digits`, which escaped both named
        # clauses and reported CPython's remedy with no file (#287).
        path = tmp_path / "config.json"
        path.write_text(
            '{"num_hidden_layers": ' + "9" * 5000 + ', "num_key_value_heads": 8, '
            '"num_attention_heads": 32, "hidden_size": 4096}'
        )

        with pytest.raises(ValueError, match=r"config\.json: cannot parse JSON"):
            shape_from_config_json(path)
