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
    assert recipe.runtime == "llama.cpp"
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


def test_pack_flow_with_stub_toolchain_produces_the_packed_file(tmp_path) -> None:
    from quantfit.adapters.outbound.recipe_json import save_recipe
    from quantfit.adapters.outbound.run_log_jsonl import read_run_log
    from quantfit.domain.model import Assignment, PlanMeta, Recipe

    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        'out = sys.argv[sys.argv.index("--outfile") + 1]\n'
        'open(out, "wb").write(b"G" * 1000)\n'
    )
    quantize = checkout / "build" / "bin" / "llama-quantize"
    quantize.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        'open(sys.argv[-3], "wb").write(b"Q" * 500)\n'
    )
    quantize.chmod(0o700)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    recipe = Recipe(
        model_id=str(model_dir),
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=3_000,
            predicted_total_bytes=2_500,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.embed_tokens", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
        ),
        runtime="llama.cpp",
    )
    recipe_path = tmp_path / "recipe.json"
    save_recipe(recipe, recipe_path)
    out = tmp_path / "packed.gguf"

    result = run(
        "pack", str(recipe_path), "--llama-cpp", str(checkout), "--out", str(out)
    )

    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == b"Q" * 500
    assert "margin" in result.stdout
    events = [e["event"] for e in read_run_log(out.with_name("packed.runlog.jsonl"))]
    assert events[0] == "pack_started"
    assert events[-1] == "pack_finished"


def test_infeasible_plan_exits_one_via_console_script(tmp_path) -> None:
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps(make_map([("g0", 160_000, {8: 0.001, 4: 0.01, 3: 0.1, 2: 1.0})]))
    )

    result = run("plan", str(map_path), "--vram", "10000", "--kv-headroom", "1000")

    assert result.returncode == 1
    assert "no recipe fits" in result.stderr + result.stdout
