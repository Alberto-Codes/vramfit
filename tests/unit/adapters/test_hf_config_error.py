"""Every HF config refusal raises `HfConfigError` under the root (#474)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vramfit.adapters.outbound.hf_config import (
    config_claims_vision,
    shape_from_config_json,
)
from vramfit.adapters.outbound.hf_kv_geometry import (
    INT_MAX,
    HfConfigError,
    bounded_int,
    kv_layers_from_decoder,
    refuse_decilm_geometry,
)
from vramfit.domain.errors import VramfitError


def _llama() -> dict[str, Any]:
    return {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 64,
    }


def _text_config() -> dict[str, Any]:
    return {
        "text_config": {
            **_llama(),
            "layer_types": ["full_attention"] * 4,
            "attention_k_eq_v": False,
            "num_kv_shared_layers": 0,
        }
    }


def _decilm() -> dict[str, Any]:
    return {
        "num_attention_heads": 8,
        "hidden_size": 64,
        "block_configs": [{"attention": {"n_heads_in_group": 4, "no_op": False}}],
    }


def _with_text(**fields: Any) -> dict[str, Any]:
    config = _text_config()
    config["text_config"].update(fields)
    return config


def _with_llama(**fields: Any) -> dict[str, Any]:
    return {**_llama(), **fields}


def _with_decilm(block: dict[str, Any] | None, **top: Any) -> dict[str, Any]:
    config = _decilm()
    if block is not None:
        config["block_configs"] = [block]
    config.update(top)
    return config


_RAW_CASES: dict[str, str] = {
    "invalid-json": "{oops",
    "duplicate-key": '{"a": 1, "a": 2}',
    "huge-int-literal": '{"num_hidden_layers": ' + "9" * 5000 + "}",
    "top-level-array": "[]",
}

_CONFIG_CASES: dict[str, dict[str, Any]] = {
    "missing-field": {"num_hidden_layers": 4},
    "layer-count-past-int64": _with_llama(num_hidden_layers=2**63),
    "non-dividing-heads": _with_llama(num_attention_heads=3, hidden_size=64),
    "bool-field": _with_llama(num_hidden_layers=True),
    "bad-head-dim": _with_llama(head_dim=0),
    "text-config-not-object": {"text_config": 1},
    "text-config-beside-layers": {**_text_config(), "num_hidden_layers": 4},
    "text-config-beside-blocks": {**_text_config(), "block_configs": []},
    "block-configs-not-list": _with_decilm(None, block_configs={}),
    "block-without-attention": _with_decilm({}),
    "missing-group-size": _with_decilm({"attention": {"n_heads_in_group": None}}),
    "non-bool-no-op": _with_decilm({"attention": {"n_heads_in_group": 4, "no_op": 1}}),
    "non-dividing-group": _with_decilm({"attention": {"n_heads_in_group": 3}}),
    "all-blocks-no-op": _with_decilm(
        {"attention": {"n_heads_in_group": None, "no_op": True}}
    ),
    "decilm-geometry-key": _with_decilm(None, sliding_window=512),
    "layer-types-not-list": _with_text(layer_types="full_attention"),
    "layer-types-short": _with_text(layer_types=["full_attention"]),
    "layer-types-unknown": _with_text(layer_types=["full_attention"] * 3 + ["x"]),
    "sliding-without-window": _with_text(layer_types=["sliding_attention"] * 4),
    "window-negative": _with_text(
        layer_types=["sliding_attention"] * 4, sliding_window=-1
    ),
    "window-switch-not-bool": _with_text(sliding_window=512, use_sliding_window="no"),
    "k-eq-v-not-bool": _with_text(attention_k_eq_v="false"),
    "shared-layers-negative": _with_text(num_kv_shared_layers=-1),
    "shared-layers-cover-stack": _with_text(num_kv_shared_layers=4),
    "global-head-dim-zero": _with_text(global_head_dim=0),
}


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.json"
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    return path


@pytest.mark.unit
class TestHfConfigError:
    def test_class_keeps_value_error_and_joins_the_root(self) -> None:
        assert issubclass(HfConfigError, ValueError)
        assert issubclass(HfConfigError, VramfitError)

    @pytest.mark.parametrize("text", list(_RAW_CASES.values()), ids=list(_RAW_CASES))
    @pytest.mark.parametrize(
        "reader",
        [shape_from_config_json, config_claims_vision],
        ids=["shape", "vision"],
    )
    def test_unparseable_file_raises_under_the_root(
        self, tmp_path: Path, reader: Callable[[Path], object], text: str
    ) -> None:
        with pytest.raises(VramfitError) as info:
            reader(_write(tmp_path, text))
        assert type(info.value) is HfConfigError

    def test_invalid_utf8_raises_under_the_root(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_bytes(b"\xff\xfe{}")
        with pytest.raises(VramfitError) as info:
            shape_from_config_json(path)
        assert type(info.value) is HfConfigError

    @pytest.mark.parametrize(
        "config", list(_CONFIG_CASES.values()), ids=list(_CONFIG_CASES)
    )
    def test_malformed_config_raises_under_the_root(
        self, tmp_path: Path, config: dict[str, Any]
    ) -> None:
        with pytest.raises(VramfitError) as info:
            shape_from_config_json(_write(tmp_path, json.dumps(config)))
        assert type(info.value) is HfConfigError

    def test_bounded_int_past_int64_raises_under_the_root(self, tmp_path: Path) -> None:
        with pytest.raises(VramfitError) as info:
            bounded_int(INT_MAX + 1, "num_hidden_layers", tmp_path / "config.json")
        assert type(info.value) is HfConfigError

    def test_refuse_decilm_geometry_raises_under_the_root(self, tmp_path: Path) -> None:
        with pytest.raises(VramfitError) as info:
            refuse_decilm_geometry({"layer_types": ["x"]}, tmp_path / "config.json")
        assert type(info.value) is HfConfigError

    def test_kv_layers_shared_tail_without_donor_raises_under_the_root(
        self, tmp_path: Path
    ) -> None:
        decoder = {
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
            "sliding_window": 512,
            "num_kv_shared_layers": 1,
        }
        with pytest.raises(VramfitError) as info:
            kv_layers_from_decoder(
                decoder,
                layers=3,
                kv_heads=2,
                head_dim=8,
                path=tmp_path / "config.json",
                prefix="text_config",
            )
        assert type(info.value) is HfConfigError
