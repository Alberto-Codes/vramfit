from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantfit.adapters.inbound import cli_pack
from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.recipe_json import save_recipe
from quantfit.adapters.outbound.run_log_jsonl import read_run_log
from quantfit.domain.model import Assignment, PlanMeta, Recipe
from tests.fakes import MemoryRecipePacker

runner = CliRunner()

pytestmark = pytest.mark.unit

WEIGHT_BUDGET = 3_000


def make_recipe(model_id: str) -> Recipe:
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
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.embed_tokens", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
        ),
        runtime=None,
    )


@pytest.fixture
def llama_cpp_dir(tmp_path: Path) -> Path:
    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "convert_hf_to_gguf.py").touch()
    (checkout / "build" / "bin" / "llama-quantize").touch()
    return checkout


@pytest.fixture
def recipe_path(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path = tmp_path / "recipe.json"
    save_recipe(make_recipe(str(model_dir)), path)
    return path


def patch_packer(monkeypatch, fake: MemoryRecipePacker) -> None:
    monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: fake)


def events_of(out: Path) -> list[str]:
    return [
        line["event"]
        for line in read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
    ]


class TestPackCommand:
    def test_happy_path_emits_the_full_event_sequence(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        fake = MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100)
        patch_packer(monkeypatch, fake)
        out = tmp_path / "packed.gguf"

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "margin" in result.output
        assert len(fake.packed) == 1
        assert events_of(out) == [
            "pack_started",
            "gguf_converted",
            "model_packed",
            "size_checked",
            "pack_finished",
        ]

    def test_size_check_records_the_margin(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100))
        out = tmp_path / "packed.gguf"

        runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
            ],
        )

        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        checked = next(line for line in log if line["event"] == "size_checked")
        assert checked["fits"] is True
        assert checked["margin_bytes"] == 100
        assert checked["weight_budget_bytes"] == WEIGHT_BUDGET

    def test_over_budget_pack_exits_1_and_halts_at_size_check(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET + 1))
        out = tmp_path / "packed.gguf"

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 1
        assert "exceeds the weight budget" in result.output
        assert events_of(out)[-1] == "pack_halted"

    def test_convert_failure_exits_1_and_halts_at_convert(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(fail_stage="convert"))
        out = tmp_path / "packed.gguf"

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 1
        assert "error: convert failed" in result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        assert log[-1]["event"] == "pack_halted"
        assert log[-1]["stage"] == "convert"

    def test_quantize_failure_exits_1_and_halts_at_quantize(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(fail_stage="quantize"))
        out = tmp_path / "packed.gguf"

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 1
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        assert log[-1]["stage"] == "quantize"

    def test_missing_recipe_file_exits_1(self, tmp_path, llama_cpp_dir) -> None:
        result = runner.invoke(
            app,
            [
                "pack",
                str(tmp_path / "absent.json"),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
            ],
        )

        assert result.exit_code == 1
        assert "error:" in result.output

    def test_unbuilt_llama_cpp_dir_is_a_usage_error(
        self, tmp_path, recipe_path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(tmp_path / "empty"),
                "--out",
                str(tmp_path / "packed.gguf"),
            ],
        )

        assert result.exit_code == 2
        assert "build the tools first" in result.output

    def test_remote_model_id_without_model_option_exits_1(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker())
        path = tmp_path / "recipe.json"
        save_recipe(make_recipe("hf-org/hf-model"), path)

        result = runner.invoke(
            app,
            [
                "pack",
                str(path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
            ],
        )

        assert result.exit_code == 1
        assert "pass --model" in result.output

    def test_options_reach_the_packer_builder_in_the_right_slots(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        seen: dict[str, object] = {}

        def recorder(model_dir, base_gguf, out, llama_cpp, python_bin, threads):
            seen.update(
                model_dir=model_dir,
                base_gguf=base_gguf,
                out=out,
                llama_cpp=llama_cpp,
                python_bin=python_bin,
                threads=threads,
            )
            return MemoryRecipePacker(packed_bytes=100)

        monkeypatch.setattr(cli_pack, "_build_packer", recorder)
        model_dir = tmp_path / "other-model"
        model_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--model",
                str(model_dir),
                "--base-gguf",
                str(tmp_path / "custom-f16.gguf"),
                "--python-bin",
                str(tmp_path / "python3"),
                "--threads",
                "3",
            ],
        )

        assert result.exit_code == 0, result.output
        assert seen == {
            "model_dir": model_dir,
            "base_gguf": tmp_path / "custom-f16.gguf",
            "out": tmp_path / "packed.gguf",
            "llama_cpp": llama_cpp_dir,
            "python_bin": tmp_path / "python3",
            "threads": 3,
        }

    def test_unmappable_recipe_exits_1_and_halts_at_quantize(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker())
        model_dir = tmp_path / "model5"
        model_dir.mkdir()
        recipe = Recipe(
            model_id=str(model_dir),
            plan=make_recipe(str(model_dir)).plan,
            assignments=(
                Assignment(group="model.layers.0", bits=7, bytes=500, damage=0.01),
            ),
            runtime=None,
        )
        path = tmp_path / "recipe5.json"
        save_recipe(recipe, path)
        out = tmp_path / "packed.gguf"

        result = runner.invoke(
            app,
            ["pack", str(path), "--llama-cpp", str(llama_cpp_dir), "--out", str(out)],
        )

        assert result.exit_code == 1
        assert "no GGUF base type maps 7-bit" in result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        assert log[-1]["event"] == "pack_halted"
        assert log[-1]["stage"] == "quantize"

    def test_events_share_one_run_id(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        out = tmp_path / "packed.gguf"

        runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
            ],
        )

        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        assert len({line["run_id"] for line in log}) == 1
