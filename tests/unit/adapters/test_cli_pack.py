from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantfit.adapters.inbound import cli_pack, cli_pack_smoke
from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.recipe_json import save_recipe
from quantfit.adapters.outbound.run_log_jsonl import read_run_log
from quantfit.domain.model import Assignment, PlanMeta, Recipe
from tests.fakes import MemoryRecipePacker, MemorySmokeTester

runner = CliRunner()

pytestmark = pytest.mark.unit

WEIGHT_BUDGET = 3_000


def make_recipe(
    model_id: str,
    within_group: str | None = None,
    imatrix: str | None = None,
) -> Recipe:
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
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.embed_tokens", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="model.layers.0", bits=4, bytes=500, damage=0.01),
        ),
        runtime=None,
        within_group=within_group,
        imatrix=imatrix,
        protected_tensors=(),
    )


@pytest.fixture
def llama_cpp_dir(tmp_path: Path) -> Path:
    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "convert_hf_to_gguf.py").touch()
    (checkout / "build" / "bin" / "llama-quantize").touch()
    (checkout / "build" / "bin" / "llama-perplexity").touch()
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


def patch_smoke_tester(monkeypatch, fake: MemorySmokeTester) -> None:
    monkeypatch.setattr(cli_pack_smoke, "_build_smoke_tester", lambda *args: fake)


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

    def test_assisted_recipe_without_imatrix_warns(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        # An assisted-priced recipe packed without its imatrix ships
        # a frame the map never priced (ADR-0020) — say so.
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100))
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(
                str(model_dir),
                within_group="kquant-imx",
                imatrix="/runs/map.imatrix.gguf",
            ),
            recipe_path,
        )
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
        assert "warning" in result.output
        assert "will not match the map's frame" in result.output

    def test_assisted_recipe_with_a_different_imatrix_warns(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100))
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        recorded = tmp_path / "map.imatrix.gguf"
        recorded.write_bytes(b"GGUF")
        other = tmp_path / "other.imatrix.gguf"
        other.write_bytes(b"GGUF")
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(
                str(model_dir), within_group="kquant-imx", imatrix=str(recorded)
            ),
            recipe_path,
        )
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
                "--imatrix",
                str(other),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "differs from the recipe" in result.output

    def test_assisted_recipe_with_its_own_imatrix_does_not_warn(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 100))
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        recorded = tmp_path / "map.imatrix.gguf"
        recorded.write_bytes(b"GGUF")
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(
                str(model_dir), within_group="kquant-imx", imatrix=str(recorded)
            ),
            recipe_path,
        )
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
                "--imatrix",
                str(recorded),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "will not match" not in result.output
        assert "differs from the recipe" not in result.output

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

    def test_llama_cpp_recipe_from_file_packs(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        # The composed product flow: plan records "llama.cpp" by
        # default, the saved schema-2 recipe reloads, and pack
        # accepts it.
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        recipe = replace(make_recipe(str(model_dir)), runtime="llama.cpp")
        path = tmp_path / "recipe.json"
        save_recipe(recipe, path)
        out = tmp_path / "packed.gguf"

        result = runner.invoke(
            app,
            ["pack", str(path), "--llama-cpp", str(llama_cpp_dir), "--out", str(out)],
        )

        assert result.exit_code == 0, result.output

    def test_foreign_runtime_recipe_exits_1_and_halts(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        recipe = replace(make_recipe(str(model_dir)), runtime="vllm")
        path = tmp_path / "recipe.json"
        save_recipe(recipe, path)
        out = tmp_path / "packed.gguf"

        result = runner.invoke(
            app,
            ["pack", str(path), "--llama-cpp", str(llama_cpp_dir), "--out", str(out)],
        )

        assert result.exit_code == 1
        assert "packs for llama.cpp" in result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        assert log[-1]["event"] == "pack_halted"
        # The rejection happens inside packer.pack, so the halt is
        # attributed to the quantize stage.
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

        def recorder(
            model_dir, base_gguf, out, llama_cpp, python_bin, threads, imatrix
        ):
            seen.update(
                model_dir=model_dir,
                base_gguf=base_gguf,
                out=out,
                llama_cpp=llama_cpp,
                python_bin=python_bin,
                threads=threads,
                imatrix=imatrix,
            )
            return MemoryRecipePacker(packed_bytes=100)

        monkeypatch.setattr(cli_pack, "_build_packer", recorder)
        model_dir = tmp_path / "other-model"
        model_dir.mkdir()
        imatrix_path = tmp_path / "imatrix.gguf"
        imatrix_path.touch()

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
                "--imatrix",
                str(imatrix_path),
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
            "imatrix": imatrix_path,
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
            within_group=None,
            imatrix=None,
            protected_tensors=(),
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


class TestPackSmokeAndImatrix:
    def test_missing_imatrix_file_is_a_usage_error(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--imatrix",
                str(tmp_path / "absent.gguf"),
            ],
        )

        assert result.exit_code == 2
        assert "is not a file" in result.output

    def test_without_smoke_text_warns_that_the_model_is_unproven(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
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
        assert "unproven" in result.output
        assert "smoke_tested" not in events_of(out)

    def test_passing_smoke_emits_the_event_and_finishes(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        tester = MemorySmokeTester(perplexity=9.5)
        patch_smoke_tester(monkeypatch, tester)
        out = tmp_path / "packed.gguf"
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
                "--smoke-text",
                str(smoke_text),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "passed" in result.output
        assert tester.runs == 1
        assert events_of(out) == [
            "pack_started",
            "gguf_converted",
            "model_packed",
            "size_checked",
            "smoke_tested",
            "pack_finished",
        ]

    def test_failing_smoke_exits_1_and_halts_at_smoke(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        patch_smoke_tester(monkeypatch, MemorySmokeTester(perplexity=1_020_627.87))
        out = tmp_path / "packed.gguf"
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
                "--smoke-text",
                str(smoke_text),
            ],
        )

        assert result.exit_code == 1
        assert "FAILED" in result.output
        assert "the file is kept" in result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        assert log[-1]["event"] == "pack_halted"
        assert log[-1]["stage"] == "smoke"
        smoked = next(line for line in log if line["event"] == "smoke_tested")
        assert smoked["passed"] is False

    def test_nan_smoke_records_null_perplexity_and_fails(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        patch_smoke_tester(monkeypatch, MemorySmokeTester(perplexity=float("nan")))
        out = tmp_path / "packed.gguf"
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
                "--smoke-text",
                str(smoke_text),
            ],
        )

        assert result.exit_code == 1
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        smoked = next(line for line in log if line["event"] == "smoke_tested")
        assert smoked["perplexity"] is None
        assert smoked["passed"] is False

    def test_smoke_tool_failure_exits_1_and_halts_at_smoke(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        patch_smoke_tester(monkeypatch, MemorySmokeTester(fail=True))
        out = tmp_path / "packed.gguf"
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
                "--smoke-text",
                str(smoke_text),
            ],
        )

        assert result.exit_code == 1
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        assert log[-1]["event"] == "pack_halted"
        assert log[-1]["stage"] == "smoke"

    def test_smoke_text_without_perplexity_tool_is_a_usage_error(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        checkout = tmp_path / "llama.cpp-no-ppl"
        (checkout / "build" / "bin").mkdir(parents=True)
        (checkout / "convert_hf_to_gguf.py").touch()
        (checkout / "build" / "bin" / "llama-quantize").touch()
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(checkout),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--smoke-text",
                str(smoke_text),
            ],
        )

        assert result.exit_code == 2
        assert "llama-perplexity" in result.output

    def test_smoke_over_budget_pack_skips_the_smoke_test(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET + 1))
        tester = MemorySmokeTester(perplexity=9.5)
        patch_smoke_tester(monkeypatch, tester)
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--smoke-text",
                str(smoke_text),
            ],
        )

        assert result.exit_code == 1
        assert tester.runs == 0


class TestSmokeWiring:
    def test_smoke_options_reach_the_tester_builder_in_the_right_slots(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        seen: dict[str, object] = {}

        def recorder(llama_cpp, out, smoke_text, chunks, threads):
            seen.update(
                llama_cpp=llama_cpp,
                out=out,
                smoke_text=smoke_text,
                chunks=chunks,
                threads=threads,
            )
            return MemorySmokeTester(perplexity=9.5)

        monkeypatch.setattr(cli_pack_smoke, "_build_smoke_tester", recorder)
        out = tmp_path / "packed.gguf"
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
                "--smoke-text",
                str(smoke_text),
                "--smoke-chunks",
                "3",
                "--threads",
                "5",
            ],
        )

        assert result.exit_code == 0, result.output
        assert seen == {
            "llama_cpp": llama_cpp_dir,
            "out": out,
            "smoke_text": smoke_text,
            "chunks": 3,
            "threads": 5,
        }

    def test_custom_smoke_threshold_reaches_the_verdict(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        patch_smoke_tester(monkeypatch, MemorySmokeTester(perplexity=9.5))
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--smoke-text",
                str(smoke_text),
                "--smoke-threshold",
                "5",
            ],
        )

        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_passing_smoke_records_the_full_event_payload(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        patch_smoke_tester(monkeypatch, MemorySmokeTester(perplexity=9.5))
        out = tmp_path / "packed.gguf"
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
                "--smoke-text",
                str(smoke_text),
                "--smoke-chunks",
                "3",
            ],
        )

        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        smoked = next(line for line in log if line["event"] == "smoke_tested")
        assert smoked["perplexity"] == 9.5
        assert smoked["threshold"] == 1000.0
        assert smoked["chunks"] == 3
        assert smoked["passed"] is True

    def test_model_packed_event_records_the_imatrix(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        imatrix_path = tmp_path / "imatrix.gguf"
        imatrix_path.touch()
        patch_packer(
            monkeypatch,
            MemoryRecipePacker(packed_bytes=100, imatrix=str(imatrix_path)),
        )
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
                "--imatrix",
                str(imatrix_path),
            ],
        )

        assert result.exit_code == 0, result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        packed = next(line for line in log if line["event"] == "model_packed")
        assert packed["imatrix"] == str(imatrix_path)

    def test_smoke_threshold_zero_is_a_usage_error(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--smoke-text",
                str(smoke_text),
                "--smoke-threshold",
                "0",
            ],
        )

        assert result.exit_code == 2
        assert "must be positive" in result.output

    def test_missing_smoke_text_file_is_a_usage_error(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--smoke-text",
                str(tmp_path / "absent.txt"),
            ],
        )

        assert result.exit_code == 2
        assert "is not a file" in result.output

    def test_infinite_smoke_threshold_is_a_usage_error(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(tmp_path / "packed.gguf"),
                "--smoke-text",
                str(smoke_text),
                "--smoke-threshold",
                "inf",
            ],
        )

        assert result.exit_code == 2
        assert "must be positive and finite" in result.output

    def test_pack_finished_records_whether_the_model_was_smoked(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=100))
        patch_smoke_tester(monkeypatch, MemorySmokeTester(perplexity=9.5))
        out = tmp_path / "packed.gguf"
        smoke_text = tmp_path / "smoke.txt"
        smoke_text.write_text("calibration text")

        runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(llama_cpp_dir),
                "--out",
                str(out),
                "--smoke-text",
                str(smoke_text),
            ],
        )

        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        finished = next(line for line in log if line["event"] == "pack_finished")
        assert finished["smoked"] is True

    def test_unsmoked_pack_finished_records_smoked_false(
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
        finished = next(line for line in log if line["event"] == "pack_finished")
        assert finished["smoked"] is False

    def test_uncovered_imatrix_tensors_are_warned_and_recorded(
        self, tmp_path, monkeypatch, llama_cpp_dir, recipe_path
    ) -> None:
        imatrix_path = tmp_path / "imatrix.gguf"
        imatrix_path.touch()
        patch_packer(
            monkeypatch,
            MemoryRecipePacker(
                packed_bytes=100,
                imatrix=str(imatrix_path),
                imatrix_uncovered=("token_embd.weight",),
            ),
        )
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
                "--imatrix",
                str(imatrix_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "did not cover" in result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        packed = next(line for line in log if line["event"] == "model_packed")
        assert packed["imatrix_uncovered"] == ["token_embd.weight"]
