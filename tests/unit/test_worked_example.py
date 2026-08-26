"""Worked examples: real configs reproduce the documented budgets.

Checks the stable numbers in ``docs/explanation/vram-budget.md``, the
arithmetic in ADR-0010, and the #423 KV figures against committed real
configs (``tests/data/nemotron-super-49b-v1_5``, and
``tests/data/gemma-4-31b`` — the ``google/gemma-4-31B`` file, fetched
verbatim 2026-08-26). If these fail, either the budget math regressed
or a docs page lies.

Runs in the hermetic unit tier: the configs are committed in-repo, so no
external resource is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.hf_config import shape_from_config_json
from vramfit.domain.budget import (
    ModelShape,
    kv_cache_bytes,
    kv_growth_bytes_per_token,
    kv_window_pool_bytes,
)

pytestmark = pytest.mark.unit

NEMOTRON_CONFIG = (
    Path(__file__).parents[1] / "data" / "nemotron-super-49b-v1_5" / "config.json"
)
GEMMA_31B_CONFIG = Path(__file__).parents[1] / "data" / "gemma-4-31b" / "config.json"

runner = CliRunner()


def nemotron_parameter_count(config: dict) -> int:
    """Count parameters from the NAS block configs (norms included)."""
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = hidden // heads
    vocab = config["vocab_size"]
    params = (2 if not config["tie_word_embeddings"] else 1) * vocab * hidden
    params += hidden  # final norm
    for block in config["block_configs"]:
        attention, ffn = block["attention"], block["ffn"]
        params += 2 * hidden  # input + post-attention norms
        if not attention["no_op"] and not attention["replace_with_linear"]:
            kv_heads = heads // attention["n_heads_in_group"]
            params += 2 * hidden * hidden + 2 * hidden * kv_heads * head_dim
        elif attention["replace_with_linear"]:
            params += hidden * hidden
        if not ffn["no_op"] and not ffn["replace_with_linear"]:
            intermediate = -(-int(2 * ffn["ffn_mult"] * hidden / 3) // 256) * 256
            params += 3 * hidden * intermediate
        elif ffn["replace_with_linear"]:
            params += hidden * hidden
    return params


def test_nemotron_shape_from_real_config_matches_documented_geometry() -> None:
    shape = shape_from_config_json(NEMOTRON_CONFIG)

    assert len(shape.kv_layers) == 49
    assert {layer.kv_heads for layer in shape.kv_layers} == {8}
    assert {layer.head_dim for layer in shape.kv_layers} == {128}
    assert kv_growth_bytes_per_token(shape, "fp16") == 200_704
    assert kv_growth_bytes_per_token(shape, "fp8") == 100_352
    assert kv_window_pool_bytes(shape) == 0


def test_gemma_31b_shape_from_real_config_matches_recorded_geometry() -> None:
    # The #423 figures, verified against the official config on
    # 2026-08-25: 50 sliding layers (window 1024, 16 KV heads, width
    # 256, K and V pair) and 10 global layers (4 KV heads, width 512,
    # one K=V tensor).
    shape = shape_from_config_json(GEMMA_31B_CONFIG)

    sliding = [layer for layer in shape.kv_layers if layer.window is not None]
    top = [layer for layer in shape.kv_layers if layer.window is None]
    assert len(shape.kv_layers) == 60
    assert len(sliding) == 50
    assert {(s.kv_heads, s.head_dim, s.window, s.kv_tensors) for s in sliding} == {
        (16, 256, 1024, 2)
    }
    assert {(g.kv_heads, g.head_dim, g.kv_tensors) for g in top} == {(4, 512, 1)}
    assert not any(layer.shares_kv for layer in shape.kv_layers)


def test_gemma_31b_kv_figures_match_the_recorded_arithmetic() -> None:
    # #423: 40,960 B/token global growth, ~0.781 GiB sliding pool,
    # ~5.78 GiB at 128k and ~10.78 GiB at 256k, fp16, one sequence.
    shape = shape_from_config_json(GEMMA_31B_CONFIG)

    assert kv_growth_bytes_per_token(shape, "fp16") == 40_960
    assert kv_window_pool_bytes(shape, "fp16") == 838_860_800
    assert kv_cache_bytes(shape, context=131_072) / 2**30 == pytest.approx(
        5.78, abs=0.01
    )
    assert kv_cache_bytes(shape, context=262_144) / 2**30 == pytest.approx(
        10.78, abs=0.01
    )


def test_gemma_31b_all_global_pricing_matches_the_documented_comparison() -> None:
    # docs/explanation/vram-budget.md: the same 60 layers priced fully
    # global charge ~0.82 MiB per token against the mixed stack's
    # 40 KiB.
    shape = shape_from_config_json(GEMMA_31B_CONFIG)

    all_global = ModelShape(
        kv_layers=tuple(replace(layer, window=None) for layer in shape.kv_layers)
    )
    per_token = kv_growth_bytes_per_token(all_global, "fp16")
    assert per_token == 860_160
    assert per_token / 2**20 == pytest.approx(0.82, abs=0.01)


def test_nemotron_config_pins_the_documented_parameter_arithmetic() -> None:
    config = json.loads(NEMOTRON_CONFIG.read_text(encoding="utf-8"))

    params = nemotron_parameter_count(config)
    uniform_4bit_gib = params * 4 / 8 / 2**30

    # docs/explanation/vram-budget.md: 80 blocks, ~49B parameters.
    assert len(config["block_configs"]) == 80
    # 49.87B parameters, ~23.2 GiB at uniform 4-bit — over both
    # measured weight budgets (18.94 / 20.47 GiB) per
    # docs/explanation/vram-budget.md and ADR-0010 on main.
    assert params == pytest.approx(49.87e9, rel=0.001)
    assert uniform_4bit_gib == pytest.approx(23.2, abs=0.1)
    assert uniform_4bit_gib > 20.47


@pytest.mark.parametrize(
    ("kv_dtype", "kv_line", "budget_line"),
    [
        ("fp16", "3.06 GiB", "18.94 GiB"),
        ("fp8", "1.53 GiB", "20.47 GiB"),
    ],
    ids=["fp16-kv", "fp8-kv"],
)
def test_budget_nemotron_16k_reproduces_documented_numbers(
    kv_dtype: str, kv_line: str, budget_line: str
) -> None:
    result = runner.invoke(
        app,
        [
            "budget",
            "--model-config",
            str(NEMOTRON_CONFIG),
            "--vram",
            "24GiB",
            "--context",
            "16384",
            "--kv-dtype",
            kv_dtype,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "attention layers      49" in result.output
    assert kv_line in result.output
    assert f"= weight budget       {budget_line}" in result.output


def test_budget_gemma_31b_128k_reports_growth_and_window_pool() -> None:
    result = runner.invoke(
        app,
        [
            "budget",
            "--model-config",
            str(GEMMA_31B_CONFIG),
            "--vram",
            "24GiB",
            "--context",
            "131072",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "attention layers      60" in result.output
    assert "KV grows 40960 bytes/token" in result.output
    assert "+ 800.00 MiB window pool" in result.output
    assert "- KV cache            5.78 GiB" in result.output
