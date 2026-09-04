"""End-to-end: the installed console script, driven via subprocess."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.conftest import make_map
from vramfit.adapters.outbound.recipe_json import load_recipe

VRAMFIT = shutil.which("vramfit")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(VRAMFIT is None, reason="vramfit console script not on PATH"),
]


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the console script with its working directory pinned.

    `cwd` has no default on purpose. `vramfit scan` defaults `--out`
    to a relative path and writes its run log beside it. A command
    started from the repository root leaves artifacts there (#155). A
    required argument makes a new test choose a directory.

    This guards the subprocess path only. `CliRunner.invoke` does not
    change directory, so an in-process test that omits `--out` still
    writes into the repository root.

    Args:
        *args: Arguments for the console script.
        cwd: Working directory for the subprocess, normally
            `tmp_path`.

    Returns:
        The completed process, with stdout and stderr captured.
    """
    assert VRAMFIT is not None
    return subprocess.run(  # noqa: S603 - fixed executable, test-controlled args
        [VRAMFIT, *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=cwd,
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
        cwd=tmp_path,
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

    result = run(
        "budget", "--model-config", str(config), "--vram", "24GiB", cwd=tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert "weight budget" in result.stdout


def test_capacity_flow_reads_a_planned_recipe(tmp_path) -> None:
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps(make_map([("g0", 160_000, {8: 0.001, 4: 0.01, 3: 0.1, 2: 1.0})]))
    )
    out = tmp_path / "recipe.json"
    planned = run(
        "plan",
        str(map_path),
        "--vram",
        "200000",
        "--kv-headroom",
        "50000",
        "--out",
        str(out),
        cwd=tmp_path,
    )
    assert planned.returncode == 0, planned.stderr

    result = run(
        "capacity",
        str(out),
        "--vram",
        "1GiB",
        "--overhead",
        "0",
        "--attn-layers",
        "2",
        "--kv-heads",
        "2",
        "--head-dim",
        "4",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "= KV headroom" in result.stdout
    assert "max context" in result.stdout


def test_budget_with_an_unrepresentable_layer_count_reports_and_exits_1(
    tmp_path,
) -> None:
    # #314 is about what the operator sees. A unit test pins the
    # exception type, and only the console script proves the halt
    # reaches stderr as an `error:` line (ADR-0011 decision 5).
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "num_hidden_layers": 10**30,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 4096,
            }
        )
    )

    result = run(
        "budget", "--model-config", str(config), "--vram", "24GiB", cwd=tmp_path
    )

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "config.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_budget_with_an_oversized_integer_literal_names_the_file(tmp_path) -> None:
    # #287's `hf_config` row. The digit-limit failure reported
    # CPython's remedy and named no file.
    config = tmp_path / "config.json"
    config.write_text(
        '{"num_hidden_layers": ' + "9" * 5000 + ', "num_key_value_heads": 8, '
        '"num_attention_heads": 32, "hidden_size": 4096}'
    )

    result = run(
        "budget", "--model-config", str(config), "--vram", "24GiB", cwd=tmp_path
    )

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "config.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_scan_without_the_extra_reports_the_install_hint(tmp_path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("scan extra installed — the ImportError path cannot trigger")
    calibration = tmp_path / "calib.txt"
    calibration.write_text("calibration text")

    result = run("scan", "some/model", "--calibration", str(calibration), cwd=tmp_path)

    assert result.returncode == 1
    assert "vramfit[scan]" in result.stderr + result.stdout
    # The halted scan writes a run log beside its default --out. It
    # lands under the pinned cwd and never in the repository root
    # (#155).
    assert (tmp_path / "sensitivity.runlog.jsonl").is_file()


def test_validate_without_the_extra_reports_the_install_hint(tmp_path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("scan extra installed — the ImportError path cannot trigger")
    from vramfit.adapters.outbound.recipe_json import save_recipe
    from vramfit.domain.model import Assignment, PlanMeta, Recipe

    recipe_path = tmp_path / "recipe.json"
    save_recipe(
        Recipe(
            model_id="some/model",
            plan=PlanMeta(
                vram_budget_bytes=4_000,
                kv_headroom_bytes=1_000,
                weight_budget_bytes=3_000,
                predicted_total_bytes=2_500,
                predicted_damage=0.05,
                solver="greedy-damage-per-byte",
                pins={},
                protections={},
                format_overhead=0.05,
                trace=(),
            ),
            assignments=(
                Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
            ),
            runtime=None,
            within_group=None,
            imatrix=None,
            protected_tensors=(),
        ),
        recipe_path,
    )
    calibration = tmp_path / "calib.txt"
    calibration.write_text("calibration text")

    result = run(
        "validate", str(recipe_path), "--calibration", str(calibration), cwd=tmp_path
    )

    assert result.returncode == 1
    assert "vramfit[scan]" in result.stderr + result.stdout


def test_pack_flow_with_stub_toolchain_produces_the_packed_file(tmp_path) -> None:
    # The pack step reads the base GGUF's tensor names (#303), so
    # the stub convert script needs gguf-py. A dev box synced with
    # `uv sync --dev` alone does not carry it, and the pre-push
    # gate collects this test — skip rather than fail (ADR-0009).
    pytest.importorskip("gguf", reason="gguf extra not installed")
    pytest.importorskip("numpy", reason="gguf extra not installed")
    from vramfit.adapters.outbound.recipe_json import save_recipe
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log
    from vramfit.domain.model import Assignment, PlanMeta, Recipe

    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    # The stub writes a real GGUF carrying the recipe's layer 0, so
    # the pack step's override check finds the tensor it addresses.
    (checkout / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        "import numpy as np\n"
        "from gguf import GGUFWriter\n"
        'out = sys.argv[sys.argv.index("--outfile") + 1]\n'
        'writer = GGUFWriter(out, arch="llama")\n'
        'for name in ("token_embd.weight", "blk.0.attn_v.weight"):\n'
        "    writer.add_tensor(name, np.zeros((2, 2), dtype=np.float16))\n"
        "writer.write_header_to_file()\n"
        "writer.write_kv_data_to_file()\n"
        "writer.write_tensors_to_file()\n"
        "writer.close()\n"
    )
    # The stub quantizer writes a real GGUF the way llama-quantize
    # does: the positional ftype stamped as general.file_type, over
    # a Q8_0 embedding whose bytes outweigh the Q4_K layer. The pack
    # step then relabels the file with the modal type (#414).
    quantize = checkout / "build" / "bin" / "llama-quantize"
    quantize.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "import numpy as np\n"
        "from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType\n"
        'writer = GGUFWriter(sys.argv[-3], arch="llama")\n'
        'writer.add_file_type(LlamaFileType[f"MOSTLY_{sys.argv[-2]}"])\n'
        "writer.add_tensor(\n"
        '    "token_embd.weight", np.zeros(20 * 34, dtype=np.uint8),\n'
        "    raw_dtype=GGMLQuantizationType.Q8_0,\n"
        ")\n"
        "writer.add_tensor(\n"
        '    "blk.0.attn_v.weight", np.zeros(144, dtype=np.uint8),\n'
        "    raw_dtype=GGMLQuantizationType.Q4_K,\n"
        ")\n"
        "writer.write_header_to_file()\n"
        "writer.write_kv_data_to_file()\n"
        "writer.write_tensors_to_file()\n"
        "writer.close()\n"
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
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.embed_tokens", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
        ),
        runtime="llama.cpp",
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )
    recipe_path = tmp_path / "recipe.json"
    save_recipe(recipe, recipe_path)
    out = tmp_path / "packed.gguf"

    result = run(
        "pack",
        str(recipe_path),
        "--llama-cpp",
        str(checkout),
        "--out",
        str(out),
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "margin" in result.stdout
    # The floor is nominal 4, so the quantizer stamped Q4_K_S (14).
    # Q8_0 covers the most bytes, so the file now declares Q8_0 (7).
    from gguf import GGUFReader

    field = GGUFReader(str(out)).get_field("general.file_type")
    assert field is not None
    assert field.contents() == 7
    log = read_run_log(out.with_name("packed.runlog.jsonl"))
    packed = next(e for e in log if e["event"] == "model_packed")
    assert packed["base_type"] == "Q4_K_S"
    assert packed["file_type"] == "Q8_0"
    events = [e["event"] for e in read_run_log(out.with_name("packed.runlog.jsonl"))]
    assert events[0] == "pack_started"
    assert events[-1] == "pack_finished"


def test_pack_refuses_an_override_the_base_gguf_cannot_match(tmp_path) -> None:
    # The whole path through the console script: a real base GGUF, the
    # real adapter, and a recipe naming a layer the file does not
    # carry. The quantizer would apply nothing and exit 0 (#303).
    pytest.importorskip("gguf", reason="gguf extra not installed")
    pytest.importorskip("numpy", reason="gguf extra not installed")
    from vramfit.adapters.outbound.recipe_json import save_recipe
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log
    from vramfit.domain.model import Assignment, PlanMeta, Recipe

    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    # The base GGUF carries layer 9 alone.
    (checkout / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        "import numpy as np\n"
        "from gguf import GGUFWriter\n"
        'out = sys.argv[sys.argv.index("--outfile") + 1]\n'
        'writer = GGUFWriter(out, arch="llama")\n'
        'writer.add_tensor("blk.9.attn_v.weight", np.zeros((2, 2), dtype=np.float16))\n'
        "writer.write_header_to_file()\n"
        "writer.write_kv_data_to_file()\n"
        "writer.write_tensors_to_file()\n"
        "writer.close()\n"
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
    recipe_path = tmp_path / "recipe.json"
    save_recipe(
        Recipe(
            model_id=str(model_dir),
            plan=PlanMeta(
                vram_budget_bytes=4_000,
                kv_headroom_bytes=1_000,
                weight_budget_bytes=3_000,
                predicted_total_bytes=2_500,
                predicted_damage=0.05,
                solver="greedy-damage-per-byte",
                pins={},
                protections={},
                format_overhead=0.05,
                trace=(),
            ),
            assignments=(
                Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
            ),
            runtime="llama.cpp",
            within_group=None,
            imatrix=None,
            protected_tensors=(),
        ),
        recipe_path,
    )
    out = tmp_path / "packed.gguf"

    result = run(
        "pack",
        str(recipe_path),
        "--llama-cpp",
        str(checkout),
        "--out",
        str(out),
        cwd=tmp_path,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "carries no tensor for 1 of 1 override patterns" in output
    assert not out.exists()
    log = read_run_log(out.with_name("packed.runlog.jsonl"))
    assert log[-1]["event"] == "pack_halted"
    assert log[-1]["stage"] == "quantize"


def test_pack_refuses_a_scanned_head_the_base_gguf_cannot_match(tmp_path) -> None:
    # The whole path through the console script: a real base GGUF from
    # a checkpoint that tied its head, the real adapter, and a recipe
    # carrying an lm_head group. `--output-tensor-type` binds nothing
    # and the quantizer exits 0 (#306).
    pytest.importorskip("gguf", reason="gguf extra not installed")
    pytest.importorskip("numpy", reason="gguf extra not installed")
    from vramfit.adapters.outbound.recipe_json import save_recipe
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log
    from vramfit.domain.model import Assignment, PlanMeta, Recipe

    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    # The base GGUF carries the embedding and layer 0, and no
    # `output.weight` — what a tied conversion writes.
    (checkout / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        "import numpy as np\n"
        "from gguf import GGUFWriter\n"
        'out = sys.argv[sys.argv.index("--outfile") + 1]\n'
        'writer = GGUFWriter(out, arch="llama")\n'
        'writer.add_tensor("token_embd.weight", np.zeros((2, 2), dtype=np.float16))\n'
        'writer.add_tensor("blk.0.attn_v.weight", np.zeros((2, 2), dtype=np.float16))\n'
        "writer.write_header_to_file()\n"
        "writer.write_kv_data_to_file()\n"
        "writer.write_tensors_to_file()\n"
        "writer.close()\n"
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
    recipe_path = tmp_path / "recipe.json"
    save_recipe(
        Recipe(
            model_id=str(model_dir),
            plan=PlanMeta(
                vram_budget_bytes=4_000,
                kv_headroom_bytes=1_000,
                weight_budget_bytes=3_000,
                predicted_total_bytes=2_500,
                predicted_damage=0.05,
                solver="greedy-damage-per-byte",
                pins={},
                protections={},
                format_overhead=0.05,
                trace=(),
            ),
            assignments=(
                Assignment(
                    group="model.embed_tokens", bits=8, bytes=1_000, damage=0.001
                ),
                Assignment(group="lm_head", bits=4, bytes=800, damage=0.002),
                Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
            ),
            runtime="llama.cpp",
            within_group=None,
            imatrix=None,
            protected_tensors=(),
        ),
        recipe_path,
    )
    out = tmp_path / "packed.gguf"

    result = run(
        "pack",
        str(recipe_path),
        "--llama-cpp",
        str(checkout),
        "--out",
        str(out),
        cwd=tmp_path,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "no target tensor for 1 dedicated flag" in output
    assert "--output-tensor-type" in output
    assert "output.weight" in output
    assert not out.exists()
    log = read_run_log(out.with_name("packed.runlog.jsonl"))
    assert log[-1]["event"] == "pack_halted"
    assert log[-1]["stage"] == "quantize"


def test_pack_refuses_an_exclusion_the_imatrix_cannot_match(tmp_path) -> None:
    # The whole path through the console script: a real base GGUF, a
    # real imatrix, the real reader, and a recipe excluding a tensor
    # the matrix never priced. The quantizer would erase no row and
    # exit 0 (#309).
    pytest.importorskip("gguf", reason="gguf extra not installed")
    np = pytest.importorskip("numpy", reason="gguf extra not installed")
    from gguf import GGUFWriter

    from vramfit.adapters.outbound.recipe_json import save_recipe
    from vramfit.adapters.outbound.run_log_jsonl import read_run_log
    from vramfit.domain.model import (
        Assignment,
        PlanMeta,
        ProtectedTensor,
        Recipe,
    )

    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        "import numpy as np\n"
        "from gguf import GGUFWriter\n"
        'out = sys.argv[sys.argv.index("--outfile") + 1]\n'
        'writer = GGUFWriter(out, arch="llama")\n'
        'for name in ("blk.0.attn_v.weight", "blk.1.attn_v.weight"):\n'
        "    writer.add_tensor(name, np.zeros((2, 2), dtype=np.float16))\n"
        "writer.write_header_to_file()\n"
        "writer.write_kv_data_to_file()\n"
        "writer.write_tensors_to_file()\n"
        "writer.close()\n"
    )
    quantize = checkout / "build" / "bin" / "llama-quantize"
    quantize.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        'open(sys.argv[-3], "wb").write(b"Q" * 500)\n'
    )
    quantize.chmod(0o700)

    # The matrix prices layer 1 alone, and the recipe excludes layer 0.
    imatrix = tmp_path / "model.imatrix.gguf"
    writer = GGUFWriter(imatrix, arch="imatrix")
    writer.add_type("imatrix")
    writer.add_tensor("blk.1.attn_v.weight.in_sum2", np.ones((1, 4), dtype=np.float32))
    writer.add_tensor("blk.1.attn_v.weight.counts", np.array([[5.0]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    recipe_path = tmp_path / "recipe.json"
    save_recipe(
        Recipe(
            model_id=str(model_dir),
            plan=PlanMeta(
                vram_budget_bytes=4_000,
                kv_headroom_bytes=1_000,
                weight_budget_bytes=3_000,
                predicted_total_bytes=2_500,
                predicted_damage=0.05,
                solver="greedy-damage-per-byte",
                pins={"*.self_attn.v_proj.weight": 5},
                protections={"*.self_attn.v_proj.weight": 5},
                format_overhead=0.05,
                trace=(),
                imatrix_exclusions=("model.layers.0.*",),
            ),
            assignments=(
                Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
            ),
            runtime="llama.cpp",
            # ADR-0020 ties the recorded imatrix to the assisted fit.
            within_group="kquant-imx",
            imatrix=str(imatrix),
            protected_tensors=(
                ProtectedTensor(
                    "model.layers.0.self_attn.v_proj.weight", 5, exclude_imatrix=True
                ),
            ),
        ),
        recipe_path,
    )
    out = tmp_path / "packed.gguf"

    result = run(
        "pack",
        str(recipe_path),
        "--llama-cpp",
        str(checkout),
        "--out",
        str(out),
        "--imatrix",
        str(imatrix),
        cwd=tmp_path,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "carries no row for 1 of 1 recipe exclusions" in output
    assert "blk.0.attn_v.weight" in output
    assert not out.exists()
    log = read_run_log(out.with_name("packed.runlog.jsonl"))
    assert log[-1]["event"] == "pack_halted"
    assert log[-1]["stage"] == "quantize"


def test_infeasible_plan_exits_one_via_console_script(tmp_path) -> None:
    map_path = tmp_path / "sensitivity.json"
    map_path.write_text(
        json.dumps(make_map([("g0", 160_000, {8: 0.001, 4: 0.01, 3: 0.1, 2: 1.0})]))
    )

    result = run(
        "plan", str(map_path), "--vram", "10000", "--kv-headroom", "1000", cwd=tmp_path
    )

    assert result.returncode == 1
    assert "no recipe fits" in result.stderr + result.stdout
