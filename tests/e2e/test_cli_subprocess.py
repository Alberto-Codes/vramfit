"""End-to-end: the installed console script, driven via subprocess."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess

import pytest

from quantfit.adapters.outbound.recipe_json import load_recipe
from tests.unit.conftest import make_map

QUANTFIT = shutil.which("quantfit")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(QUANTFIT is None, reason="quantfit console script not on PATH"),
]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    assert QUANTFIT is not None
    return subprocess.run(  # noqa: S603 - fixed executable, test-controlled args
        [QUANTFIT, *args], capture_output=True, text=True, timeout=60, check=False
    )


def test_plan_flow_produces_loadable_recipe(tmp_path) -> None:
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps(
            make_map(
                [
                    ("g0", 160_000, {8: 0.001, 4: 0.01, 3: 0.1, 2: 1.0}),
                    ("g1", 160_000, {8: 0.0, 4: 0.004, 3: 0.02, 2: 0.1}),
                ]
            )
        )
    )
    out = tmp_path / "recipe.json"

    result = run(
        "plan",
        str(map_path),
        "--vram",
        "200000",
        "--kv-headroom",
        "50000",
        "--out",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    recipe = load_recipe(out)
    assert recipe.plan.weight_budget_bytes == 150_000
    assert "planned 2 groups" in result.stdout


def test_budget_flow_prints_breakdown(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "num_hidden_layers": 4,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 4096,
            }
        )
    )

    result = run("budget", "--model-config", str(config), "--vram", "24GiB")

    assert result.returncode == 0, result.stderr
    assert "weight budget" in result.stdout


def test_scan_without_the_extra_reports_the_install_hint(tmp_path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("scan extra installed — the ImportError path cannot trigger")
    calibration = tmp_path / "calib.txt"
    calibration.write_text("calibration text")

    result = run("scan", "some/model", "--calibration", str(calibration))

    assert result.returncode == 1
    assert "quantfit[scan]" in result.stderr + result.stdout


def test_infeasible_plan_exits_one_via_console_script(tmp_path) -> None:
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps(make_map([("g0", 160_000, {8: 0.001, 4: 0.01, 3: 0.1, 2: 1.0})]))
    )

    result = run("plan", str(map_path), "--vram", "10000", "--kv-headroom", "1000")

    assert result.returncode == 1
    assert "no recipe fits" in result.stderr + result.stdout
