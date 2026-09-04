from __future__ import annotations

import json
from pathlib import Path

import pytest

from vramfit.adapters.outbound.hf_config import (
    config_claims_vision,
    shape_from_config_json,
)
from vramfit.domain.budget import ModelShape


def _kv_heads(shape: ModelShape) -> tuple[int, ...]:
    return tuple(layer.kv_heads for layer in shape.kv_layers)


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

        assert _kv_heads(shape) == (8, 8, 8)
        assert shape.kv_layers[0].head_dim == 128

    def test_decilm_config_handles_varying_gqa_group_size(self, tmp_path) -> None:
        config = self._decilm_config()
        config["block_configs"][1]["attention"]["n_heads_in_group"] = 16
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (8, 4, 8)

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

        assert _kv_heads(shape) == (8, 8)

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

        assert _kv_heads(shape) == (8,) * 32
        assert shape.kv_layers[0].head_dim == 128

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

        assert shape.kv_layers[0].head_dim == 64

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

        assert shape.kv_layers[0].head_dim == 2**63 - 1

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

        assert _kv_heads(shape) == (2, 2, 2, 2)
        assert shape.kv_layers[0].head_dim == 128

    def test_text_config_not_an_object_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"] = "gemma4_text"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match=r'config\.json: "text_config" must be a JSON object'
        ):
            shape_from_config_json(path)

    def test_text_config_beside_top_level_layer_count_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["num_hidden_layers"] = 4
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match=r"config\.json: .*decoder config is ambiguous"
        ):
            shape_from_config_json(path)

    def test_text_config_beside_block_configs_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["block_configs"] = self._decilm_config()["block_configs"]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match=r"config\.json: .*decoder config is ambiguous"
        ):
            shape_from_config_json(path)

    def test_text_config_mixed_layer_types_parses_per_layer(self, tmp_path) -> None:
        # The Gemma 4 31B pattern: 50 sliding layers, 10 global, 5:1,
        # split local/global geometry, the k_eq_v KV-head override.
        config = self._text_config()
        config["text_config"].update(
            {
                "num_hidden_layers": 60,
                "num_key_value_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_global_key_value_heads": 4,
                "attention_k_eq_v": True,
                "sliding_window": 1024,
                "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10,
            }
        )
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        sliding = [layer for layer in shape.kv_layers if layer.window is not None]
        top = [layer for layer in shape.kv_layers if layer.window is None]
        assert len(shape.kv_layers) == 60
        assert len(sliding) == 50
        assert {(s.kv_heads, s.head_dim, s.window, s.kv_tensors) for s in sliding} == {
            (16, 256, 1024, 2)
        }
        assert {(g.kv_heads, g.head_dim, g.kv_tensors) for g in top} == {(4, 512, 2)}
        assert not any(layer.shares_kv for layer in shape.kv_layers)

    def test_text_config_all_sliding_layer_types_parses_capped(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["layer_types"] = ["sliding_attention"] * 4
        config["text_config"]["sliding_window"] = 512
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert all(layer.window == 512 for layer in shape.kv_layers)

    def test_sliding_layer_types_without_a_window_raises(self, tmp_path) -> None:
        # A sliding layer with no declared window has no KV price.
        config = self._text_config()
        config["text_config"]["layer_types"] = ["sliding_attention"] * 4
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match="declares sliding layers with no active sliding window"
        ):
            shape_from_config_json(path)

    def test_sliding_layer_types_with_switch_off_raises(self, tmp_path) -> None:
        # `use_sliding_window: false` disables the declared window, so
        # the sliding entries have no KV price.
        config = self._text_config()
        config["text_config"]["layer_types"] = ["sliding_attention"] * 4
        config["text_config"]["sliding_window"] = 512
        config["text_config"]["use_sliding_window"] = False
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match="declares sliding layers with no active sliding window"
        ):
            shape_from_config_json(path)

    def test_unknown_layer_type_raises(self, tmp_path) -> None:
        # `layers_block_type` hybrids stay separately ticketed (#427);
        # an unknown `layer_types` entry refuses rather than pricing
        # a mechanism this reader does not model.
        config = self._text_config()
        config["text_config"]["layer_types"] = ["full_attention"] * 3 + [
            "linear_attention"
        ]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match="declares a layer type this reader does not model"
        ):
            shape_from_config_json(path)

    def test_text_config_empty_layer_types_raises(self, tmp_path) -> None:
        # An empty list declares nothing about the layers, so it does
        # not prove uniformity.
        config = self._text_config()
        config["text_config"]["layer_types"] = []
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="one type per hidden layer"):
            shape_from_config_json(path)

    def test_text_config_short_layer_types_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["layer_types"] = ["full_attention"]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="one type per hidden layer"):
            shape_from_config_json(path)

    def test_text_config_non_list_layer_types_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["layer_types"] = "full_attention"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a list of strings"):
            shape_from_config_json(path)

    def test_text_config_k_eq_v_true_still_prices_kv_pairs(self, tmp_path) -> None:
        # attention_k_eq_v gates the KV-head override only. The ruled
        # runtime allocates the K and V caches on every layer and
        # fills V with K on global layers, so each layer prices two
        # KV tensors (#431).
        config = self._text_config()
        config["text_config"]["attention_k_eq_v"] = True
        config["text_config"]["layer_types"] = ["sliding_attention"] * 2 + [
            "full_attention"
        ] * 2
        config["text_config"]["sliding_window"] = 512
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert [layer.kv_tensors for layer in shape.kv_layers] == [2, 2, 2, 2]

    def test_text_config_non_bool_k_eq_v_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["attention_k_eq_v"] = "false"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="attention_k_eq_v must be a boolean"):
            shape_from_config_json(path)

    def test_text_config_shared_kv_layers_mark_the_tail(self, tmp_path) -> None:
        # E2B declares 20 shared-KV layers; the transformers loader
        # marks the last N layers, and those own no fresh cache.
        config = self._text_config()
        config["text_config"]["num_kv_shared_layers"] = 2
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert [layer.shares_kv for layer in shape.kv_layers] == [
            False,
            False,
            True,
            True,
        ]

    def test_shared_kv_layers_covering_the_stack_raises(self, tmp_path) -> None:
        # A share count at or above the layer count leaves no layer
        # that stores KV, so no cache exists to reuse.
        config = self._text_config()
        config["text_config"]["num_kv_shared_layers"] = 4
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="leaves no layer that stores KV"):
            shape_from_config_json(path)

    def test_shared_tail_without_a_same_type_donor_raises(self, tmp_path) -> None:
        # A shared layer reuses the last fresh layer of its own type.
        # A tail that covers every global layer leaves no donor, and
        # pricing it would zero the growth term with no report.
        config = self._text_config()
        config["text_config"]["layer_types"] = ["sliding_attention"] * 2 + [
            "full_attention"
        ] * 2
        config["text_config"]["sliding_window"] = 512
        config["text_config"]["num_kv_shared_layers"] = 2
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match="leaves a shared layer with no earlier layer"
        ):
            shape_from_config_json(path)

    def test_use_sliding_window_null_prices_the_window(self, tmp_path) -> None:
        # A null switch means unset, and unset leaves a declared
        # window active — only the boolean false disables it (#426).
        config = self._text_config()
        config["text_config"]["layer_types"] = ["sliding_attention"] * 3 + [
            "full_attention"
        ]
        config["text_config"]["sliding_window"] = 512
        config["text_config"]["use_sliding_window"] = None
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert [layer.window for layer in shape.kv_layers] == [512, 512, 512, None]

    def test_text_config_negative_shared_kv_layers_raises(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["num_kv_shared_layers"] = -1
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            shape_from_config_json(path)

    def test_text_config_boolean_shared_kv_layers_raises(self, tmp_path) -> None:
        # `bool` subclasses `int`, so `true` would read as one shared
        # layer and refuse for the wrong reason (#348).
        config = self._text_config()
        config["text_config"]["num_kv_shared_layers"] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            shape_from_config_json(path)

    def test_text_config_global_head_dim_widens_global_layers(self, tmp_path) -> None:
        config = self._text_config()
        config["text_config"]["global_head_dim"] = 512
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert all(layer.head_dim == 512 for layer in shape.kv_layers)

    def test_text_config_mistyped_global_head_dim_raises(self, tmp_path) -> None:
        # `0` is falsy but declared; a type refusal beats a
        # split-geometry refusal that misnames the problem (#426).
        config = self._text_config()
        config["text_config"]["global_head_dim"] = 0
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match="global_head_dim must be a positive integer or null"
        ):
            shape_from_config_json(path)

    def test_global_kv_heads_apply_under_k_eq_v(self, tmp_path) -> None:
        # The transformers Gemma 4 loader gates the global KV-head
        # override on `attention_k_eq_v`. The reader prices what the
        # runtime loads.
        config = self._text_config()
        config["text_config"]["num_global_key_value_heads"] = 4
        config["text_config"]["attention_k_eq_v"] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (4, 4, 4, 4)

    def test_ungated_global_kv_heads_mismatch_raises(self, tmp_path) -> None:
        # The transformers loader discards the override when
        # `attention_k_eq_v` is off. This reader never silently
        # discards a declared geometry value, so it refuses.
        config = self._text_config()
        config["text_config"]["num_global_key_value_heads"] = 4
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="its attention_k_eq_v setting disables"):
            shape_from_config_json(path)

    def test_ungated_global_kv_heads_matching_the_base_parses(self, tmp_path) -> None:
        # An ungated override equal to the base count changes nothing,
        # so there is no declared value to discard.
        config = self._text_config()
        config["text_config"]["num_global_key_value_heads"] = 2
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (2, 2, 2, 2)

    def test_text_config_null_global_kv_heads_parses(self, tmp_path) -> None:
        # The official E2B config carries the key as null. Null spells
        # "unset" in HF configs, so it declares no split geometry.
        config = self._text_config()
        config["text_config"]["num_global_key_value_heads"] = None
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (2, 2, 2, 2)

    def test_top_level_mixed_layer_types_parse_per_layer(self, tmp_path) -> None:
        # The geometry parse runs in the llama parse, so a text-only
        # release that publishes the same decoder at the top level
        # prices the same per-layer stack.
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "sliding_window": 4096,
                    "layer_types": ["sliding_attention", "full_attention"] * 2,
                }
            )
        )

        shape = shape_from_config_json(path)

        assert [layer.window for layer in shape.kv_layers] == [4096, None, 4096, None]

    def test_top_level_all_full_layer_types_parses(self, tmp_path) -> None:
        # A recent transformers dump serializes layer_types for a
        # plain uniform stack. One entry per layer, all full: proven
        # uniform, admitted.
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "layer_types": ["full_attention"] * 4,
                }
            )
        )

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (2, 2, 2, 2)

    def test_top_level_active_sliding_window_raises(self, tmp_path) -> None:
        # A window can be declared without layer_types. Reading the
        # stack as all-global overstates the KV cache with no report.
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "sliding_window": 4096,
                }
            )
        )

        with pytest.raises(ValueError, match="declares windowed attention"):
            shape_from_config_json(path)

    def test_disabled_sliding_window_parses(self, tmp_path) -> None:
        # Qwen-family configs carry the window value with the switch
        # off. That stack is uniform.
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "sliding_window": 4096,
                    "use_sliding_window": False,
                }
            )
        )

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (2, 2, 2, 2)

    def test_null_sliding_window_parses(self, tmp_path) -> None:
        # Mistral-family configs use null for no window.
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "sliding_window": None,
                }
            )
        )

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (2, 2, 2, 2)

    def test_boolean_sliding_window_raises(self, tmp_path) -> None:
        # `bool` subclasses `int`, so `true` would read as a 1-token
        # window (#348).
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "sliding_window": True,
                }
            )
        )

        with pytest.raises(ValueError, match="non-negative integer or null"):
            shape_from_config_json(path)

    def test_negative_sliding_window_raises(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "sliding_window": -1,
                }
            )
        )

        with pytest.raises(ValueError, match="non-negative integer or null"):
            shape_from_config_json(path)

    def test_non_bool_use_sliding_window_raises(self, tmp_path) -> None:
        # A string "false" is not the boolean carve-out. Refusing it as
        # a type error beats a misleading windowed-attention refusal.
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 8,
                    "hidden_size": 1024,
                    "sliding_window": 4096,
                    "use_sliding_window": "false",
                }
            )
        )

        with pytest.raises(
            ValueError, match='"use_sliding_window" must be a boolean or null'
        ):
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

    def test_text_config_e2b_geometry_prices_the_fresh_stack(self, tmp_path) -> None:
        # The official E2B fields: window 512, 20 shared-KV layers,
        # K=V off, null global KV heads, global head width 512.
        config = self._text_config()
        config["text_config"].update(
            {
                "num_hidden_layers": 35,
                "num_key_value_heads": 1,
                "num_attention_heads": 8,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_global_key_value_heads": None,
                "attention_k_eq_v": False,
                "num_kv_shared_layers": 20,
                "sliding_window": 512,
                "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
            }
        )
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        fresh = [layer for layer in shape.kv_layers if not layer.shares_kv]
        assert len(shape.kv_layers) == 35
        assert len(fresh) == 15
        assert sum(1 for layer in fresh if layer.window == 512) == 12
        assert sum(1 for layer in fresh if layer.window is None) == 3
        assert all(layer.kv_tensors == 2 for layer in shape.kv_layers)

    def test_bidirectional_all_raises(self, tmp_path) -> None:
        # The transformers loader rescales the stored window when
        # `use_bidirectional_attention` is "all", so the stored value
        # is not the runtime window.
        config = self._text_config()
        config["text_config"]["use_bidirectional_attention"] = "all"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="declares bidirectional attention"):
            shape_from_config_json(path)

    def test_unknown_bidirectional_value_raises(self, tmp_path) -> None:
        # Only null and "vision" are known to leave the stored window
        # untouched. An unknown value could rescale it like "all".
        config = self._text_config()
        config["text_config"]["use_bidirectional_attention"] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="declares bidirectional attention"):
            shape_from_config_json(path)

    def test_bidirectional_vision_parses(self, tmp_path) -> None:
        # The 31B config carries "vision", which changes masks inside
        # image spans, not KV allocation.
        config = self._text_config()
        config["text_config"]["use_bidirectional_attention"] = "vision"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        shape = shape_from_config_json(path)

        assert _kv_heads(shape) == (2, 2, 2, 2)

    def test_nested_block_configs_raises(self, tmp_path) -> None:
        # A NAS decoder inside `text_config` would flatten its no_op
        # blocks to a wrong KV price in the llama-style parse (#426).
        config = self._text_config()
        config["text_config"]["block_configs"] = self._decilm_config()["block_configs"]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(
            ValueError, match=r"text_config\.block_configs declares a NAS decoder"
        ):
            shape_from_config_json(path)

    def test_decilm_layer_types_raises(self, tmp_path) -> None:
        # The DeciLM parse prices every kept block as a global K and V
        # pair, so a per-layer pattern beside it refuses (#426).
        config = self._decilm_config()
        config["layer_types"] = ["sliding_attention"]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match=r'"layer_types" beside "block_configs"'):
            shape_from_config_json(path)

    def test_decilm_active_sliding_window_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["sliding_window"] = 4096
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="declares windowed attention"):
            shape_from_config_json(path)

    def test_decilm_k_eq_v_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["attention_k_eq_v"] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="declares K=V storage"):
            shape_from_config_json(path)

    def test_decilm_shared_kv_layers_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["num_kv_shared_layers"] = 2
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="declares KV sharing"):
            shape_from_config_json(path)

    def test_decilm_global_head_dim_raises(self, tmp_path) -> None:
        config = self._decilm_config()
        config["global_head_dim"] = 512
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="split local/global"):
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

    def test_config_nested_past_the_recursion_limit_names_the_file(
        self, tmp_path
    ) -> None:
        # Deep nesting exhausts the decoder's stack. `RecursionError` is
        # no `ValueError`, so it escaped every caller (#478).
        path = tmp_path / "config.json"
        path.write_text('{"a": ' + "[" * 100_000 + "]" * 100_000 + "}")

        with pytest.raises(ValueError, match=r"config\.json: JSON nests too deeply"):
            shape_from_config_json(path)


@pytest.mark.unit
class TestConfigClaimsVision:
    def test_vision_config_present_claims_vision(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "text_config": {
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 2,
                        "num_attention_heads": 4,
                        "head_dim": 4,
                    },
                    "vision_config": {"hidden_size": 1152},
                }
            )
        )

        assert config_claims_vision(path) is True

    def test_vision_config_absent_claims_no_vision(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 4,
                    "hidden_size": 16,
                }
            )
        )

        assert config_claims_vision(path) is False

    def test_invalid_json_raises_naming_the_file(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{not json")

        with pytest.raises(ValueError, match=r"config\.json: invalid JSON"):
            config_claims_vision(path)

    def test_duplicate_key_raises_like_the_shape_read(self, tmp_path) -> None:
        # The claim read shares the shape read's refusals, so the two
        # reads of one file cannot disagree on validity (#283).
        path = tmp_path / "config.json"
        path.write_text('{"vision_config": {}, "vision_config": {}}')

        with pytest.raises(ValueError, match="twice"):
            config_claims_vision(path)

    def test_config_nested_past_the_recursion_limit_names_the_file(
        self, tmp_path
    ) -> None:
        # The claim read shares the shape read's nesting refusal (#478).
        path = tmp_path / "config.json"
        path.write_text('{"a": ' + "[" * 100_000 + "]" * 100_000 + "}")

        with pytest.raises(ValueError, match=r"config\.json: JSON nests too deeply"):
            config_claims_vision(path)

    def test_null_vision_config_claims_no_vision(self, tmp_path) -> None:
        # The claim is a declared object — a null would otherwise
        # license a vision line the card never priced.
        path = tmp_path / "config.json"
        path.write_text('{"vision_config": null}')

        assert config_claims_vision(path) is False

    def test_real_gemma_config_claims_vision(self) -> None:
        # The file ADR-0030 measured — the claim read pins to the
        # record's own target.
        config = Path(__file__).parents[2] / "data" / "gemma-4-31b" / "config.json"

        assert config_claims_vision(config) is True
