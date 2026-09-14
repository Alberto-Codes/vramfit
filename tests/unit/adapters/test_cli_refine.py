"""The ``vramfit refine`` command, driven with the verified fakes.

The packer and meter seams are monkeypatched, so the command's input
refusals, its sidecar write, and its reporting are the unit under
test. The measure loop itself is covered in
`tests/unit/adapters/test_refine_loop.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import (
    MemoryRecipePacker,
    MemoryRuntimeDivergenceMeter,
    stack_row_widths,
)
from vramfit.adapters.inbound import cli_refine
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.recipe_json import save_recipe
from vramfit.adapters.outbound.sensitivity_map_json import save_sensitivity_map
from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    ScanMeta,
    SensitivityMap,
)

runner = CliRunner()

G = [f"model.layers.{i}.mixer.experts.up_proj" for i in range(4)]
PRECISIONS = (8, 4, 2)
SIZE_AT = {8: 800, 4: 400, 2: 200}
CHUNKS = (0.3, 0.3, 0.3, 0.3)


def _map() -> SensitivityMap:
    return SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="/work/calibration.txt",
            calibration_tokens=131072,
            precisions=PRECISIONS,
            group_by="stack",
            started_at="2026-07-27T00:00:00Z",
        ),
        groups=tuple(
            LayerGroup(
                name=name,
                tensors=(f"{name}.weight",),
                bytes_fp16=1600,
                sensitivity={8: 0.0, 4: 0.1 * (i + 1), 2: 0.4 * (i + 1)},
            )
            for i, name in enumerate(G)
        ),
    )


def _recipe(bits: dict[str, int]) -> Recipe:
    assignments = tuple(
        Assignment(group=n, bits=b, bytes=SIZE_AT[b], damage=0.1)
        for n, b in bits.items()
    )
    plan = PlanMeta(
        vram_budget_bytes=10**9,
        kv_headroom_bytes=0,
        weight_budget_bytes=10**9,
        predicted_total_bytes=sum(a.bytes for a in assignments),
        predicted_damage=1.0,
        solver="greedy-damage-per-byte",
        pins={},
        protections={},
        format_overhead=0.0,
        trace=(),
    )
    return Recipe(
        model_id="test/model",
        plan=plan,
        assignments=assignments,
        runtime="llama.cpp",
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """Lay out a runnable refine invocation with both seams faked."""
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "base.logits").write_bytes(b"logits")
    (tmp_path / "wiki.test.raw").write_text("text")
    save_sensitivity_map(_map(), tmp_path / "map.json")
    save_recipe(_recipe({G[0]: 2, G[1]: 2, G[2]: 4, G[3]: 4}), tmp_path / "r.json")

    monkeypatch.setattr(
        cli_refine, "_resolve_row_widths", lambda recipe, model_dir: stack_row_widths(G)
    )
    monkeypatch.setattr(
        cli_refine,
        "LlamaCppPacker",
        lambda **kwargs: MemoryRecipePacker(
            packed_bytes=500, has_base=True, row_widths=stack_row_widths(G)
        ),
    )
    monkeypatch.setattr(
        cli_refine,
        "LlamaCppDivergenceMeter",
        lambda **kwargs: MemoryRuntimeDivergenceMeter(default=CHUNKS),
    )
    return tmp_path


def _invoke(tmp_path: Path, *extra: str, recipe: str = "r.json"):
    return runner.invoke(
        app,
        [
            "refine",
            str(tmp_path / recipe),
            "--map",
            str(tmp_path / "map.json"),
            "--llama-cpp",
            str(tmp_path / "llama.cpp"),
            "--base-logits",
            str(tmp_path / "base.logits"),
            "--eval-text",
            str(tmp_path / "wiki.test.raw"),
            "--runtime-build",
            "b10362",
            "--hardware",
            "H100 SXM",
            "--model",
            str(tmp_path / "ckpt"),
            "--out-dir",
            str(tmp_path / "arms"),
            *extra,
        ],
    )


def test_refine_writes_a_sidecar_beside_the_recipe(workspace) -> None:
    result = _invoke(workspace, "--limit", "2")

    assert result.exit_code == 0, result.output
    sidecar = json.loads((workspace / "r.refinement.json").read_text())
    assert sidecar["vramfit_schema"] == 1
    assert len(sidecar["arms"]) == 2


def test_refine_records_the_frame_it_measured_in(workspace) -> None:
    _invoke(workspace, "--limit", "1")

    frame = json.loads((workspace / "r.refinement.json").read_text())["frame"]
    assert frame["runtime_build"] == "b10362"
    assert frame["hardware"] == "H100 SXM"
    assert frame["corpus"]["file"].endswith("wiki.test.raw")


def test_refine_records_the_stated_bar(workspace) -> None:
    _invoke(workspace, "--limit", "1", "--bar", "4.0")

    assert json.loads((workspace / "r.refinement.json").read_text())["bar"] == 4.0


def test_refine_reports_no_winner_when_nothing_clears_the_bar(workspace) -> None:
    result = _invoke(workspace, "--limit", "2")

    assert "no arm cleared 7.8 sigma" in result.output
    assert json.loads((workspace / "r.refinement.json").read_text())["winner"] is None


def test_refine_declines_a_recipe_with_no_legal_swap(workspace) -> None:
    save_recipe(_recipe({G[0]: 4, G[1]: 4, G[2]: 4}), workspace / "flat.json")

    result = _invoke(workspace, recipe="flat.json")

    assert result.exit_code == 0, result.output
    assert "declined:" in result.output
    sidecar = json.loads((workspace / "flat.refinement.json").read_text())
    assert sidecar["declined"]
    assert sidecar["control"] is None
    assert sidecar["arms"] == []


def test_refine_refuses_a_missing_base_logits_file(workspace) -> None:
    (workspace / "base.logits").unlink()

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "--base-logits" in result.output


def test_refine_refuses_a_missing_eval_text(workspace) -> None:
    (workspace / "wiki.test.raw").unlink()

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "--eval-text" in result.output


def test_refine_refuses_a_missing_model_directory(workspace) -> None:
    result = runner.invoke(
        app,
        [
            "refine",
            str(workspace / "r.json"),
            "--map",
            str(workspace / "map.json"),
            "--llama-cpp",
            str(workspace / "llama.cpp"),
            "--base-logits",
            str(workspace / "base.logits"),
            "--eval-text",
            str(workspace / "wiki.test.raw"),
            "--runtime-build",
            "b10362",
            "--hardware",
            "H100 SXM",
            "--model",
            str(workspace / "absent"),
        ],
    )

    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_refine_writes_a_run_log(workspace) -> None:
    _invoke(workspace, "--limit", "1")

    events = (workspace / "r.refinement.runlog.jsonl").read_text()
    assert "refine_started" in events
    assert "refine_finished" in events
