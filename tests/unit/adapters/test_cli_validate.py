from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantfit.adapters.inbound import cli_validate
from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.recipe_json import save_recipe
from quantfit.adapters.outbound.run_log_jsonl import read_run_log
from quantfit.domain.model import Assignment, PlanMeta, Recipe
from quantfit.domain.scan import GroupSpec
from tests.fakes import MemoryDamageMeter

runner = CliRunner()

pytestmark = pytest.mark.unit

SPECS = (
    GroupSpec(name="model.layers.0", tensors=("model.layers.0.w",), bytes_fp16=1000),
    GroupSpec(name="model.layers.1", tensors=("model.layers.1.w",), bytes_fp16=2000),
)
DAMAGES = {
    ("model.layers.0", 8): 0.001,
    ("model.layers.0", 4): 0.01,
    ("model.layers.1", 8): 0.002,
    ("model.layers.1", 4): 0.02,
}


def make_recipe(groups_bits_damage: tuple[tuple[str, int, float], ...]) -> Recipe:
    return Recipe(
        model_id="test/model",
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=3_000,
            predicted_total_bytes=2_500,
            predicted_damage=sum(damage for _, _, damage in groups_bits_damage),
            solver="greedy-damage-per-byte",
            pins={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=tuple(
            Assignment(group=group, bits=bits, bytes=500, damage=damage)
            for group, bits, damage in groups_bits_damage
        ),
        runtime=None,
    )


DEFAULT_RECIPE = (
    ("model.layers.0", 4, 0.01),
    ("model.layers.1", 8, 0.002),
)


@pytest.fixture
def recipe_path(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.json"
    save_recipe(make_recipe(DEFAULT_RECIPE), path)
    return path


def install_meter(monkeypatch, meter: MemoryDamageMeter) -> None:
    def build(
        model,
        calibration,
        *,
        max_tokens,
        group_by,
        device,
        trust_remote_code,
        gpu_memory,
    ):
        return meter

    monkeypatch.setattr(cli_validate, "_build_meter", build)


def invoke_validate(tmp_path: Path, recipe_path: Path, *extra: str):
    calibration = tmp_path / "calib.txt"
    calibration.write_text("calibration text")
    return runner.invoke(
        app,
        ["validate", str(recipe_path), "--calibration", str(calibration), *extra],
    )


def events_of(path: Path) -> list[dict]:
    return list(read_run_log(path))


class TestValidateCommand:
    def test_happy_path_reports_predicted_measured_and_gap(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        meter = MemoryDamageMeter(
            specs=SPECS, damages=dict(DAMAGES), interaction_damage=0.003
        )
        install_meter(monkeypatch, meter)

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 0, result.output
        assert "summed marginal damage (predicted)  0.012000" in result.output
        assert "whole-recipe damage (measured)      0.015000" in result.output
        assert "gap +0.003000 (+25.0 % of predicted)" in result.output
        assert meter.recipe_calls == [{"model.layers.0": 4, "model.layers.1": 8}]

    def test_happy_path_emits_the_full_event_sequence(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        install_meter(
            monkeypatch,
            MemoryDamageMeter(
                specs=SPECS, damages=dict(DAMAGES), interaction_damage=0.003
            ),
        )

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 0, result.output
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        assert [e["event"] for e in events] == [
            "validation_started",
            "meter_built",
            "validation_finished",
        ]
        finished = events[-1]
        assert finished["predicted_damage"] == pytest.approx(0.012)
        assert finished["measured_damage"] == pytest.approx(0.015)
        assert finished["gap"] == pytest.approx(0.003)
        assert finished["ratio"] == pytest.approx(1.25)

    def test_explicit_runlog_path_wins_over_the_default(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        runlog = tmp_path / "custom.runlog.jsonl"

        result = invoke_validate(tmp_path, recipe_path, "--runlog", str(runlog))

        assert result.exit_code == 0, result.output
        assert runlog.exists()
        assert not recipe_path.with_name("recipe.validation.runlog.jsonl").exists()

    def test_missing_recipe_file_exits_one(self, tmp_path, monkeypatch) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(tmp_path, tmp_path / "no-such-recipe.json")

        assert result.exit_code == 1
        assert "error:" in result.output

    def test_recipe_with_unknown_group_exits_one_and_logs_group_match(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe((("model.layers.9", 4, 0.01),)),
            recipe_path,
        )

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 1
        assert "model.layers.9" in result.output
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        assert events[-1]["event"] == "validation_halted"
        assert events[-1]["stage"] == "group_match"

    def test_recipe_missing_a_discovered_group_exits_one(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe((("model.layers.0", 4, 0.01),)), recipe_path)

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 1
        assert "model.layers.1" in result.output

    def test_measurement_failure_exits_one_and_logs_measure_stage(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        # No damage configured for the 8-bit cell — the fake raises.
        damages = {("model.layers.0", 4): 0.01}
        install_meter(monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages))

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 1
        assert "validation halted" in result.output
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        assert events[-1]["event"] == "validation_halted"
        assert events[-1]["stage"] == "measure"

    def test_gpu_memory_without_device_auto_is_a_usage_error(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(
            tmp_path, recipe_path, "--device", "cpu", "--gpu-memory", "17GiB"
        )

        assert result.exit_code == 2
        assert "--gpu-memory requires --device auto" in result.output

    def test_bad_group_by_is_a_usage_error(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(tmp_path, recipe_path, "--group-by", "module")

        assert result.exit_code == 2
        assert "--group-by" in result.output

    def test_missing_runlog_directory_is_a_usage_error(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(
            tmp_path,
            recipe_path,
            "--runlog",
            str(tmp_path / "no-such-dir" / "v.jsonl"),
        )

        assert result.exit_code == 2
        assert "--runlog" in result.output
