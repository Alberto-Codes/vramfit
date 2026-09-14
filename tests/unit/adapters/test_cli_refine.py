"""The ``vramfit refine`` command, driven with the verified fakes.

The packer and meter seams are monkeypatched, so the command's input
refusals, its sidecar write, and its reporting are the unit under
test. The measure loop itself is covered in
`tests/unit/adapters/test_refine_loop.py`.
"""

from __future__ import annotations

import json
from hashlib import sha256
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
from vramfit.adapters.outbound.run_log_jsonl import read_run_log
from vramfit.adapters.outbound.sensitivity_map_json import save_sensitivity_map
from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    ScanMeta,
    SensitivityMap,
)

pytestmark = pytest.mark.unit

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


def _recipe(bits: dict[str, int], imatrix: str | None = None) -> Recipe:
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
        within_group=None if imatrix is None else "kquant-imx",
        imatrix=imatrix,
        protected_tensors=(),
    )


def _build_llama_cpp(root: Path) -> None:
    """Lay out the checkout `refine` pre-flights before any tool runs."""
    bin_dir = root / "build" / "bin"
    bin_dir.mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# convert")
    for name in ("llama-quantize", "llama-perplexity"):
        binary = bin_dir / name
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """Lay out a runnable refine invocation with both seams faked."""
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "base.logits").write_bytes(b"logits")
    (tmp_path / "wiki.test.raw").write_text("text")
    _build_llama_cpp(tmp_path / "llama.cpp")
    save_sensitivity_map(_map(), tmp_path / "map.json")
    save_recipe(_recipe({G[0]: 2, G[1]: 2, G[2]: 4, G[3]: 4}), tmp_path / "r.json")

    monkeypatch.setattr(
        cli_refine, "_resolve_row_widths", lambda recipe, model_dir: stack_row_widths(G)
    )
    monkeypatch.setattr(
        cli_refine,
        "LlamaCppPacker",
        lambda **kwargs: MemoryRecipePacker(
            packed_bytes=500,
            has_base=True,
            row_widths=stack_row_widths(G),
            out_path=kwargs["out_path"],
        ),
    )
    monkeypatch.setattr(
        cli_refine,
        "LlamaCppDivergenceMeter",
        lambda **kwargs: MemoryRuntimeDivergenceMeter(default=CHUNKS),
    )
    return tmp_path


@pytest.fixture
def wiring_log(monkeypatch) -> list[str]:
    """Record every seam the command constructs, in order.

    A pre-flight that runs after the packer or the meter is built has
    already paid for the convert, so the refusal tests assert this
    stays empty.
    """
    built: list[str] = []

    def record(name, factory):
        def build(**kwargs):
            built.append(name)
            return factory(**kwargs)

        monkeypatch.setattr(cli_refine, name, build)

    record(
        "LlamaCppPacker",
        lambda **kwargs: MemoryRecipePacker(
            packed_bytes=500,
            has_base=True,
            row_widths=stack_row_widths(G),
            out_path=kwargs["out_path"],
        ),
    )
    record(
        "LlamaCppDivergenceMeter",
        lambda **kwargs: MemoryRuntimeDivergenceMeter(default=CHUNKS),
    )
    return built


def _invoke(tmp_path: Path, *extra: str, recipe: str = "r.json", bar: str = "7.8"):
    return runner.invoke(
        app,
        [
            "refine",
            str(tmp_path / recipe),
            "--bar",
            bar,
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


def test_refine_names_the_evaluation_corpus_by_content(workspace) -> None:
    _invoke(workspace, "--limit", "1")

    corpus = json.loads((workspace / "r.refinement.json").read_text())["frame"][
        "corpus"
    ]
    assert corpus["sha256"] == sha256(b"text").hexdigest()
    assert corpus["size_bytes"] == 4
    assert corpus["provenance"] == "measured"


def test_refine_records_a_different_corpus_differently(workspace) -> None:
    _invoke(workspace, "--limit", "1")
    first = json.loads((workspace / "r.refinement.json").read_text())["frame"]

    (workspace / "wiki.test.raw").write_text("other text")
    _invoke(workspace, "--limit", "1")
    second = json.loads((workspace / "r.refinement.json").read_text())["frame"]

    assert first["corpus"]["file"] == second["corpus"]["file"]
    assert first["corpus"]["sha256"] != second["corpus"]["sha256"]


def test_refine_refuses_an_empty_evaluation_corpus(workspace) -> None:
    (workspace / "wiki.test.raw").write_text("")

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "holds no bytes" in result.output


def test_refine_refuses_an_unstated_bar(workspace) -> None:
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
            str(workspace / "ckpt"),
        ],
    )

    assert result.exit_code != 0
    assert not (workspace / "r.refinement.json").exists()


def test_refine_records_the_stated_bar(workspace) -> None:
    _invoke(workspace, "--limit", "1", bar="4.0")

    assert json.loads((workspace / "r.refinement.json").read_text())["bar"] == 4.0


def test_refine_reports_no_winner_when_nothing_clears_the_bar(workspace) -> None:
    result = _invoke(workspace, "--limit", "2")

    assert "cleared 7.8 sigma" in result.output
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
            "--bar",
            "7.8",
            "--model",
            str(workspace / "absent"),
        ],
    )

    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_refine_reports_an_unpairable_measurement_cleanly(workspace, monkeypatch):
    """One chunk has no spread, so `compare` refuses (PairedError)."""
    monkeypatch.setattr(
        cli_refine,
        "LlamaCppDivergenceMeter",
        lambda **kwargs: MemoryRuntimeDivergenceMeter(default=(0.3,)),
    )

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "error:" in result.output
    assert "chunks" in result.output
    assert not (workspace / "r.refinement.json").exists()


def test_refine_writes_a_run_log(workspace) -> None:
    _invoke(workspace, "--limit", "1")

    events = read_run_log(workspace / "r.refinement.runlog.jsonl")
    names = [line["event"] for line in events]
    assert names[0] == "refine_started"
    assert names[-1] == "refine_finished"
    assert events[0]["arms"] == 1
    assert events[0]["bar"] == 7.8
    assert "arm_measured" in names


def test_refine_reports_an_empty_runtime_build_cleanly(workspace) -> None:
    result = _invoke(workspace, "--limit", "1", "--runtime-build", "")

    assert result.exit_code == 1
    assert "error:" in result.output
    assert "runtime_build" in result.output


def test_refine_reports_a_failed_sidecar_write_cleanly(workspace, monkeypatch) -> None:
    def refuse(self, sidecar):
        raise OSError("read-only file system")

    monkeypatch.setattr(cli_refine.JsonRefinementSidecarFile, "save", refuse)

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "error:" in result.output
    assert "read-only file system" in result.output


def test_refine_names_the_evaluated_arms_rather_than_claiming_a_verdict(
    workspace,
) -> None:
    result = _invoke(workspace, "--limit", "2")

    assert "no arm among the 2 evaluated of a neighbourhood of 4" in result.output
    assert "the recipe stands" not in result.output


def test_refine_records_the_neighbourhood_the_arms_were_drawn_from(
    workspace,
) -> None:
    _invoke(workspace, "--limit", "2")

    sidecar = json.loads((workspace / "r.refinement.json").read_text())
    assert len(sidecar["arms"]) == 2
    assert sidecar["neighbourhood_moves"] == 4


def test_refine_refuses_a_checkout_missing_llama_perplexity(
    workspace, wiring_log
) -> None:
    (workspace / "llama.cpp" / "build" / "bin" / "llama-perplexity").unlink()

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "llama-perplexity" in result.output
    assert wiring_log == []
    assert not (workspace / "arms").exists()


def test_refine_refuses_a_checkout_missing_the_convert_script(
    workspace, wiring_log
) -> None:
    (workspace / "llama.cpp" / "convert_hf_to_gguf.py").unlink()

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "convert_hf_to_gguf.py" in result.output
    assert wiring_log == []
    assert not (workspace / "arms").exists()


def test_refine_refuses_a_tool_that_cannot_execute(workspace, wiring_log) -> None:
    binary = workspace / "llama.cpp" / "build" / "bin" / "llama-quantize"
    binary.chmod(0o644)

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "not executable" in result.output
    assert wiring_log == []
    assert not (workspace / "arms").exists()


def test_refine_refuses_a_missing_imatrix_before_any_tool_runs(
    workspace, wiring_log
) -> None:
    result = _invoke(
        workspace, "--limit", "1", "--imatrix", str(workspace / "absent.imatrix")
    )

    assert result.exit_code == 1
    assert "--imatrix" in result.output
    assert wiring_log == []
    assert not (workspace / "arms").exists()


def test_refine_accepts_an_imatrix_that_exists(workspace) -> None:
    (workspace / "m.imatrix").write_bytes(b"imatrix")

    result = _invoke(
        workspace, "--limit", "1", "--imatrix", str(workspace / "m.imatrix")
    )

    assert result.exit_code == 0, result.output
    assert (workspace / "r.refinement.json").is_file()


def test_refine_refuses_a_missing_sidecar_directory_before_any_tool_runs(
    workspace, wiring_log
) -> None:
    result = _invoke(
        workspace, "--limit", "1", "--out", str(workspace / "results" / "r.json")
    )

    assert result.exit_code == 1
    assert "--out" in result.output
    assert wiring_log == []


def test_refine_refuses_a_missing_runlog_directory_before_any_tool_runs(
    workspace, wiring_log
) -> None:
    result = _invoke(
        workspace, "--limit", "1", "--runlog", str(workspace / "logs" / "r.jsonl")
    )

    assert result.exit_code == 1
    assert "--runlog" in result.output
    assert wiring_log == []


def test_refine_refuses_a_sidecar_directory_it_cannot_write(
    workspace, wiring_log
) -> None:
    locked = workspace / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        result = _invoke(workspace, "--limit", "1", "--out", str(locked / "r.json"))

        assert result.exit_code == 1
        assert "not writable" in result.output
        assert wiring_log == []
    finally:
        locked.chmod(0o755)


def test_refine_wires_the_tools_the_preflight_checked(workspace, monkeypatch) -> None:
    """The checked paths are the paths that run."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        cli_refine,
        "LlamaCppPacker",
        lambda **kwargs: (
            seen.update(kwargs)
            or MemoryRecipePacker(
                packed_bytes=500, has_base=True, row_widths=stack_row_widths(G)
            )
        ),
    )
    monkeypatch.setattr(
        cli_refine,
        "LlamaCppDivergenceMeter",
        lambda **kwargs: (
            seen.update(kwargs) or MemoryRuntimeDivergenceMeter(default=CHUNKS)
        ),
    )
    tools = cli_refine.LlamaCppTools.under(workspace / "llama.cpp")

    _invoke(workspace, "--limit", "1")

    assert seen["convert_script"] == tools.convert_script
    assert seen["quantize_bin"] == tools.quantize_bin
    assert seen["perplexity_bin"] == tools.perplexity_bin


def test_refine_refuses_a_sidecar_path_that_is_a_directory(
    workspace, wiring_log
) -> None:
    """`--out arms` means "into arms/" to an operator, not "onto it"."""
    (workspace / "arms").mkdir()

    result = _invoke(workspace, "--limit", "1", "--out", str(workspace / "arms"))

    assert result.exit_code == 1
    assert "is a directory" in result.output
    assert wiring_log == []


def test_refine_refuses_a_runlog_path_that_is_a_directory(
    workspace, wiring_log
) -> None:
    (workspace / "logs").mkdir()

    result = _invoke(workspace, "--limit", "1", "--runlog", str(workspace / "logs"))

    assert result.exit_code == 1
    assert "is a directory" in result.output
    assert wiring_log == []


def test_refine_refuses_an_out_dir_that_is_a_file(workspace, wiring_log) -> None:
    blocked = workspace / "base-f16.gguf"
    blocked.write_bytes(b"gguf")

    result = _invoke(workspace, "--limit", "1", "--out-dir", str(blocked))

    assert result.exit_code == 1
    assert "error: --out-dir" in result.output
    # A clean halt exits through typer; an unguarded OSError would
    # surface the OSError itself here.
    assert isinstance(result.exception, SystemExit)
    assert wiring_log == []


def test_refine_accepts_an_out_inside_an_uncreated_arm_directory(workspace) -> None:
    """--out-dir is created first, so --out inside it resolves."""
    result = _invoke(
        workspace, "--limit", "1", "--out", str(workspace / "arms" / "refine.json")
    )

    assert result.exit_code == 0, result.output
    assert (workspace / "arms" / "refine.json").is_file()


def test_refine_refuses_out_naming_the_uncreated_arm_directory(
    workspace, wiring_log
) -> None:
    """`--out arms` before `arms` exists must refuse, not pay for a pass."""
    assert not (workspace / "arms").exists()

    result = _invoke(workspace, "--limit", "1", "--out", str(workspace / "arms"))

    assert result.exit_code == 1
    assert "is a directory" in result.output
    assert wiring_log == []


def test_refine_refuses_a_map_that_priced_another_model(workspace, wiring_log) -> None:
    other = _map()
    save_sensitivity_map(
        SensitivityMap(
            model_id="test/49b",
            scan=other.scan,
            groups=other.groups,
        ),
        workspace / "other.json",
    )

    result = runner.invoke(
        app,
        [
            "refine",
            str(workspace / "r.json"),
            "--bar",
            "7.8",
            "--map",
            str(workspace / "other.json"),
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
            str(workspace / "ckpt"),
        ],
    )

    assert result.exit_code == 1
    assert "test/49b" in result.output
    assert "test/model" in result.output
    assert wiring_log == []
    assert not (workspace / "r.refinement.json").exists()


def test_refine_proceeds_when_the_map_priced_this_recipe(workspace) -> None:
    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 0, result.output
    sidecar = json.loads((workspace / "r.refinement.json").read_text())
    assert sidecar["model_id"] == "test/model"


def test_refine_records_the_imatrix_it_packed_with(workspace) -> None:
    matrix = workspace / "m.imatrix"
    matrix.write_bytes(b"imatrix bytes")

    _invoke(workspace, "--limit", "1", "--imatrix", str(matrix))

    frame = json.loads((workspace / "r.refinement.json").read_text())["frame"]
    assert frame["imatrix"]["file"].endswith("m.imatrix")
    assert frame["imatrix"]["sha256"] == sha256(b"imatrix bytes").hexdigest()
    assert frame["imatrix"]["size_bytes"] == len(b"imatrix bytes")


def test_refine_records_an_unassisted_pass_as_a_null_imatrix(workspace) -> None:
    _invoke(workspace, "--limit", "1")

    frame = json.loads((workspace / "r.refinement.json").read_text())["frame"]
    assert "imatrix" in frame
    assert frame["imatrix"] is None


def test_an_assisted_and_an_unassisted_pass_do_not_serialize_alike(
    workspace,
) -> None:
    matrix = workspace / "m.imatrix"
    matrix.write_bytes(b"imatrix bytes")

    _invoke(workspace, "--limit", "1")
    unassisted = json.loads((workspace / "r.refinement.json").read_text())["frame"]
    _invoke(workspace, "--limit", "1", "--imatrix", str(matrix))
    assisted = json.loads((workspace / "r.refinement.json").read_text())["frame"]

    assert unassisted != assisted


def test_refine_warns_when_an_assisted_recipe_packs_without_its_matrix(
    workspace,
) -> None:
    save_recipe(
        _recipe({G[0]: 2, G[1]: 2, G[2]: 4, G[3]: 4}, imatrix="30b.imatrix"),
        workspace / "assisted.json",
    )

    result = _invoke(workspace, "--limit", "1", recipe="assisted.json")

    assert result.exit_code == 0, result.output
    assert "warning:" in result.output
    assert "30b.imatrix" in result.output


def test_refine_refuses_an_empty_imatrix(workspace, wiring_log) -> None:
    matrix = workspace / "m.imatrix"
    matrix.write_bytes(b"")

    result = _invoke(workspace, "--limit", "1", "--imatrix", str(matrix))

    assert result.exit_code == 1
    assert "holds no bytes" in result.output
    assert wiring_log == []


def test_refine_deletes_each_arm_pack_after_measuring_it(workspace) -> None:
    """A 16-arm pass on the 30B target would otherwise need 336 GiB.

    The fake packer writes the file the real adapter writes, so the
    absence asserted here is a file that existed and was removed.
    """
    _invoke(workspace, "--limit", "2")

    assert (workspace / "arms").is_dir()
    assert list((workspace / "arms").glob("*.gguf")) == []


def test_refine_records_the_reference_logits_it_measured_against(
    workspace,
) -> None:
    """Every divergence is computed against these bytes."""
    _invoke(workspace, "--limit", "1")

    frame = json.loads((workspace / "r.refinement.json").read_text())["frame"]
    assert frame["reference"]["file"].endswith("base.logits")
    assert frame["reference"]["sha256"] == sha256(b"logits").hexdigest()
    assert frame["reference"]["size_bytes"] == len(b"logits")


def test_two_passes_over_different_reference_logits_do_not_serialize_alike(
    workspace,
) -> None:
    """A rebuilt base.logits behind one path is a different frame."""
    _invoke(workspace, "--limit", "1")
    first = json.loads((workspace / "r.refinement.json").read_text())["frame"]

    (workspace / "base.logits").write_bytes(b"rebuilt logits")
    _invoke(workspace, "--limit", "1")
    second = json.loads((workspace / "r.refinement.json").read_text())["frame"]

    assert first["reference"]["file"] == second["reference"]["file"]
    assert first["reference"]["sha256"] != second["reference"]["sha256"]
    assert first != second


def test_refine_refuses_empty_reference_logits(workspace, wiring_log) -> None:
    (workspace / "base.logits").write_bytes(b"")

    result = _invoke(workspace, "--limit", "1")

    assert result.exit_code == 1
    assert "--base-logits" in result.output
    assert "holds no bytes" in result.output
    assert wiring_log == []
