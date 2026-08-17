"""The reconstruction gate of ``vramfit pack`` (ADR-0022).

Drives the command with the verified fakes: the packer seam packs
nothing real, and the checker seam returns configured measurements
per packed file, so the gate's verdicts and refusals are the unit
under test.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import (
    MemoryImatrixCountSource,
    MemoryRecipePacker,
    MemoryReconstructionChecker,
)
from vramfit.adapters.inbound import cli_pack, cli_pack_check, cli_pack_imatrix
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.recipe_json import save_recipe
from vramfit.adapters.outbound.run_log_jsonl import read_run_log
from vramfit.domain.model import (
    Assignment,
    PlanMeta,
    ProtectedTensor,
    Recipe,
)

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture(autouse=True)
def count_source(monkeypatch) -> MemoryImatrixCountSource:
    # Every --imatrix pack reads the matrix's counts (ADR-0026
    # decision 5), and these tests' matrix files are not GGUF.
    fake = MemoryImatrixCountSource()
    monkeypatch.setattr(cli_pack_imatrix, "_build_count_source", lambda *args: fake)
    return fake


WEIGHT_BUDGET = 3_000
PROTECTED_HF = "model.layers.0.self_attn.v_proj.weight"
PROTECTED_GGUF = "blk.0.attn_v.weight"


def make_protected_recipe(model_id: str, protected: bool = True) -> Recipe:
    return Recipe(
        model_id=model_id,
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=WEIGHT_BUDGET,
            predicted_total_bytes=2_500,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            protections={"*.self_attn.v_proj.weight": 5} if protected else {},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.layers.0", bits=3, bytes=1_000, damage=0.01),
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(ProtectedTensor(PROTECTED_HF, 5),) if protected else (),
    )


@pytest.fixture
def llama_cpp_dir(tmp_path: Path) -> Path:
    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "convert_hf_to_gguf.py").touch()
    (checkout / "build" / "bin" / "llama-quantize").touch()
    return checkout


@pytest.fixture
def imatrix_path(tmp_path: Path) -> Path:
    path = tmp_path / "imatrix.gguf"
    path.touch()
    return path


def save_protected_recipe(tmp_path: Path, protected: bool = True) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    path = tmp_path / "recipe.json"
    save_recipe(make_protected_recipe(str(model_dir), protected=protected), path)
    return path


def save_excluded_recipe(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    recipe = make_protected_recipe(str(model_dir))
    recipe = replace(
        recipe,
        plan=replace(recipe.plan, imatrix_exclusions=(PROTECTED_HF,)),
        protected_tensors=(ProtectedTensor(PROTECTED_HF, 5, exclude_imatrix=True),),
    )
    path = tmp_path / "recipe.json"
    save_recipe(recipe, path)
    return path


def patch_checkers(
    monkeypatch, protected_rmse: float, reference_rmse: float
) -> list[Path]:
    built: list[Path] = []

    def builder(packed: Path, base: Path):
        built.append(packed)
        rmse = reference_rmse if "reconstruction-ref" in packed.name else protected_rmse
        return MemoryReconstructionChecker(errors={PROTECTED_GGUF: rmse})

    monkeypatch.setattr(cli_pack_check, "_build_checker", builder)
    return built


def invoke_pack(recipe_path: Path, llama_cpp_dir: Path, out: Path, *extra: str):
    return runner.invoke(
        app,
        [
            "pack",
            str(recipe_path),
            "--llama-cpp",
            str(llama_cpp_dir),
            "--out",
            str(out),
            *extra,
        ],
    )


def events_of(out: Path) -> list[str]:
    return [
        line["event"]
        for line in read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
    ]


class TestReconstructionGate:
    def test_passing_check_reports_and_continues(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        recipe_path = save_protected_recipe(tmp_path)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        patch_checkers(monkeypatch, protected_rmse=0.001, reference_rmse=0.004)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 0, result.output
        assert "reconstruction check passed" in result.output
        assert "reconstruction_checked" in events_of(out)
        # The reference pack ran with the protections stripped.
        assert len(fake.packed) == 2
        assert fake.packed[1].protected_tensors == ()

    def test_collapse_names_the_tensor_and_halts(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        recipe_path = save_protected_recipe(tmp_path)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        # The G1 collapse signature: worse at the protected type.
        patch_checkers(monkeypatch, protected_rmse=0.0241, reference_rmse=0.0048)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 1
        assert "fit collapse" in result.output
        assert PROTECTED_HF in result.output
        assert "COLLAPSED" in result.output
        assert f"kept at {out}" in result.output
        # The refusal suggests the ADR-0023 remedy verbatim.
        assert f'--exclude-imatrix "{PROTECTED_HF}"' in result.output
        events = events_of(out)
        assert "reconstruction_checked" in events
        assert events[-1] == "pack_halted"

    def test_excluded_recipe_still_faces_the_gate(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        # An exclusion changes the fit, not the verdict path: the
        # gate still measures the excluded tensor (ADR-0023).
        recipe_path = save_excluded_recipe(tmp_path)
        fake = MemoryRecipePacker(
            packed_bytes=WEIGHT_BUDGET - 100,
            imatrix=str(imatrix_path),
            # The matrix prices the excluded tensor, so the #309
            # refusal stays out of this suite's way. Leaving it unset
            # would skip a check the real adapter always runs.
            imatrix_entry_names=(PROTECTED_GGUF,),
        )
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        patch_checkers(monkeypatch, protected_rmse=0.00164, reference_rmse=0.0048)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 0, result.output
        assert "imatrix exclusions applied" in result.output
        assert PROTECTED_GGUF in result.output
        assert "reconstruction check passed" in result.output
        # The reference pack strips the exclusions with the
        # protections, so its packer sees no marked pair.
        assert fake.packed[1].plan.imatrix_exclusions == ()

    def test_collapse_on_excluded_pair_offers_only_the_drop_remedy(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        # The exclusion already failed for this tensor — suggesting
        # the same flag again would send the user in a circle
        # (ADR-0023).
        recipe_path = save_excluded_recipe(tmp_path)
        fake = MemoryRecipePacker(
            packed_bytes=WEIGHT_BUDGET - 100,
            imatrix=str(imatrix_path),
            # The matrix prices the excluded tensor, so the #309
            # refusal stays out of this suite's way. Leaving it unset
            # would skip a check the real adapter always runs.
            imatrix_entry_names=(PROTECTED_GGUF,),
        )
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        patch_checkers(monkeypatch, protected_rmse=0.0241, reference_rmse=0.0048)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 1
        assert "fit collapse" in result.output
        assert "already failed" in result.output
        assert "--exclude-imatrix" not in result.output

    def test_excluded_recipe_without_imatrix_warns(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        recipe_path = save_excluded_recipe(tmp_path)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(recipe_path, llama_cpp_dir, out)

        assert result.exit_code == 0, result.output
        assert "exclusions change nothing" in result.output

    def test_protections_with_zero_pairs_skips_with_a_note(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        # Every floor was a per-tensor no-op at plan time (issue #59):
        # the record survives, no pair does, and the silence would
        # read as a gated pack.
        model_dir = tmp_path / "model"
        model_dir.mkdir(exist_ok=True)
        recipe = make_protected_recipe(str(model_dir))
        recipe = replace(recipe, protected_tensors=())
        recipe_path = tmp_path / "recipe.json"
        save_recipe(recipe, recipe_path)
        fake = MemoryRecipePacker(
            packed_bytes=WEIGHT_BUDGET - 100,
            imatrix=str(imatrix_path),
            # The matrix prices the excluded tensor, so the #309
            # refusal stays out of this suite's way. Leaving it unset
            # would skip a check the real adapter always runs.
            imatrix_entry_names=(PROTECTED_GGUF,),
        )
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 0, result.output
        assert "resolved no pairs" in result.output
        assert "reconstruction_checked" not in events_of(out)
        assert len(fake.packed) == 1

    def test_protected_pack_without_imatrix_skips_with_a_note(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        recipe_path = save_protected_recipe(tmp_path)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(recipe_path, llama_cpp_dir, out)

        assert result.exit_code == 0, result.output
        assert "reconstruction check skipped" in result.output
        assert "reconstruction_checked" not in events_of(out)
        assert len(fake.packed) == 1

    def test_unprotected_pack_never_runs_the_gate(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        recipe_path = save_protected_recipe(tmp_path, protected=False)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        built = patch_checkers(monkeypatch, 0.001, 0.004)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 0, result.output
        assert "reconstruction" not in result.output
        assert built == []

    def test_reference_pack_failure_halts_with_the_stage_named(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        recipe_path = save_protected_recipe(tmp_path)
        main = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        packers = iter([main, MemoryRecipePacker(fail_stage="quantize", has_base=True)])
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: next(packers))
        patch_checkers(monkeypatch, 0.001, 0.004)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 1
        events = events_of(out)
        assert events[-1] == "pack_halted"


class TestProtectedPreflight:
    def test_unmappable_protected_tensor_fails_before_any_stage(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        recipe = make_protected_recipe(str(model_dir))
        recipe = replace(
            recipe,
            protected_tensors=(
                ProtectedTensor("model.layers.0.mlp.experts.0.up_proj.weight", 5),
            ),
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(recipe, recipe_path)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)

        result = invoke_pack(recipe_path, llama_cpp_dir, tmp_path / "p.gguf")

        assert result.exit_code == 1
        assert "no GGUF mapping" in result.output
        # Nothing ran: the failure costs milliseconds, not a convert.
        assert fake.packed == []

    def test_unmappable_floor_fails_before_any_stage(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        recipe = make_protected_recipe(str(model_dir))
        recipe = replace(recipe, protected_tensors=(ProtectedTensor(PROTECTED_HF, 7),))
        recipe_path = tmp_path / "recipe.json"
        save_recipe(recipe, recipe_path)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)

        result = invoke_pack(recipe_path, llama_cpp_dir, tmp_path / "p.gguf")

        assert result.exit_code == 1
        assert "no GGUF type maps 7-bit" in result.output
        assert fake.packed == []

    def test_nan_measurement_still_records_the_halt(
        self, tmp_path, monkeypatch, llama_cpp_dir, imatrix_path
    ) -> None:
        recipe_path = save_protected_recipe(tmp_path)
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)
        patch_checkers(monkeypatch, protected_rmse=float("nan"), reference_rmse=0.004)
        out = tmp_path / "packed.gguf"

        result = invoke_pack(
            recipe_path, llama_cpp_dir, out, "--imatrix", str(imatrix_path)
        )

        assert result.exit_code == 1
        events = events_of(out)
        # The NaN must not kill the run log (ADR-0011): the event and
        # the halt both land.
        assert "reconstruction_checked" in events
        assert events[-1] == "pack_halted"
