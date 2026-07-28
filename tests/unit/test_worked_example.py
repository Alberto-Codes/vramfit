"""Worked example: the real Nemotron config reproduces the documented budget.

Checks the stable numbers in ``docs/explanation/vram-budget.md`` and the
arithmetic in ADR-0010 against the committed north-star config
(``tests/data/nemotron-super-49b-v1_5``). If these fail, either the
budget math regressed or a docs page lies.

Runs in the hermetic unit tier: the config is committed in-repo, so no
external resource is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.hf_config import shape_from_config_json
from quantfit.domain.budget import kv_bytes_per_token

pytestmark = pytest.mark.unit

NEMOTRON_CONFIG = (
    Path(__file__).parents[1] / "data" / "nemotron-super-49b-v1_5" / "config.json"
)

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

    assert len(shape.kv_heads_per_layer) == 49
    assert set(shape.kv_heads_per_layer) == {8}
    assert shape.head_dim == 128
    assert kv_bytes_per_token(shape, "fp16") == 200_704
    assert kv_bytes_per_token(shape, "fp8") == 100_352


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
