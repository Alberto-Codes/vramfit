"""Worked example: the real Nemotron config reproduces the documented budget.

Checks the stable numbers in ``docs/explanation/vram-budget.md`` against
the committed north-star config (``tests/data/nemotron-super-49b-v1_5``).
If these fail, either the budget math regressed or the docs page lies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.hf_config import shape_from_config_json
from quantfit.domain.budget import kv_bytes_per_token

pytestmark = pytest.mark.integration

NEMOTRON_CONFIG = (
    Path(__file__).parents[1] / "data" / "nemotron-super-49b-v1_5" / "config.json"
)

runner = CliRunner()


def test_nemotron_shape_matches_documented_geometry() -> None:
    shape = shape_from_config_json(NEMOTRON_CONFIG)

    assert len(shape.kv_heads_per_layer) == 49
    assert set(shape.kv_heads_per_layer) == {8}
    assert shape.head_dim == 128
    assert kv_bytes_per_token(shape, "fp16") == 200_704
    assert kv_bytes_per_token(shape, "fp8") == 100_352


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
