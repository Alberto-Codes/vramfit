"""The size-check stage compares packed bytes against the prediction.

The budget margin alone reported publication #2 as a fit while its
prediction missed two classes in opposite directions (#409). The
stage now prints the prediction delta beside the margin and warns
past the tolerance, and never refuses on it (ADR-0012 decision 4,
amended 2026-09-04).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import MemoryRecipePacker
from tests.unit.adapters.test_cli_pack import (
    WEIGHT_BUDGET,
    events_of,
    make_recipe,
    patch_packer,
)
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.inbound.cli_pack_size import _predicted_report
from vramfit.adapters.outbound.recipe_json import save_recipe
from vramfit.adapters.outbound.run_log_jsonl import read_run_log

runner = CliRunner()

pytestmark = pytest.mark.unit

PREDICTED = 2_500


@pytest.fixture
def llama_cpp_dir(tmp_path: Path) -> Path:
    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "convert_hf_to_gguf.py").touch()
    (checkout / "build" / "bin" / "llama-quantize").touch()
    return checkout


def save_recipe_with_prediction(tmp_path: Path, predicted: int) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    recipe = make_recipe(str(model_dir))
    recipe = replace(recipe, plan=replace(recipe.plan, predicted_total_bytes=predicted))
    path = tmp_path / "recipe.json"
    save_recipe(recipe, path)
    return path


def run_pack(tmp_path: Path, llama_cpp_dir: Path, recipe_path: Path):
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
    return result, out


class TestPredictedReport:
    def test_delta_inside_tolerance_reports_within(self) -> None:
        report = _predicted_report(10_000, 10_050)

        assert report.warns is False
        assert "+50 B (+0.50%)" in report.line
        assert "within" in report.line
        assert report.event["predicted_delta_bytes"] == 50
        assert report.event["predicted_within_tolerance"] is True

    def test_delta_above_tolerance_warns_outside(self) -> None:
        report = _predicted_report(10_000, 10_200)

        assert report.warns is True
        assert report.line.startswith("warning:")
        assert "+200 B (+2.00%)" in report.line
        assert "OUTSIDE" in report.line
        assert report.event["predicted_within_tolerance"] is False

    def test_undershoot_carries_a_negative_sign(self) -> None:
        report = _predicted_report(10_000, 9_800)

        assert "-200 B (-2.00%)" in report.line
        assert report.event["predicted_delta_bytes"] == -200

    @pytest.mark.parametrize("predicted", [None, 0], ids=["missing", "zero"])
    def test_absent_prediction_reports_the_absence(self, predicted) -> None:
        report = _predicted_report(predicted, 10_000)

        assert report.warns is False
        assert "predicted bytes absent" in report.line
        assert report.event == {
            "predicted_total_bytes": None,
            "predicted_delta_bytes": None,
            "predicted_delta_fraction": None,
            "predicted_within_tolerance": None,
        }


class TestSizeCheckStage:
    def test_inside_tolerance_prints_the_line_and_records_the_delta(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        recipe_path = save_recipe_with_prediction(tmp_path, PREDICTED)
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=PREDICTED + 10))

        result, out = run_pack(tmp_path, llama_cpp_dir, recipe_path)

        assert result.exit_code == 0, result.output
        assert "delta +10 B (+0.40%)" in result.output
        assert "warning: predicted" not in result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        checked = next(line for line in log if line["event"] == "size_checked")
        assert checked["predicted_total_bytes"] == PREDICTED
        assert checked["predicted_delta_bytes"] == 10
        assert checked["predicted_delta_fraction"] == pytest.approx(0.004)
        assert checked["predicted_within_tolerance"] is True

    def test_above_tolerance_warns_and_does_not_halt(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        recipe_path = save_recipe_with_prediction(tmp_path, PREDICTED)
        packed = PREDICTED + 100
        assert packed <= WEIGHT_BUDGET
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=packed))

        result, out = run_pack(tmp_path, llama_cpp_dir, recipe_path)

        assert result.exit_code == 0, result.output
        assert "warning: predicted" in result.output
        assert "OUTSIDE" in result.output
        assert "pack_halted" not in events_of(out)
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        checked = next(line for line in log if line["event"] == "size_checked")
        assert checked["predicted_within_tolerance"] is False

    def test_zero_prediction_prints_the_absence_and_packs(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        recipe_path = save_recipe_with_prediction(tmp_path, 0)
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET - 1))

        result, out = run_pack(tmp_path, llama_cpp_dir, recipe_path)

        assert result.exit_code == 0, result.output
        assert "predicted bytes absent" in result.output
        log = read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        checked = next(line for line in log if line["event"] == "size_checked")
        assert checked["predicted_total_bytes"] is None
        assert checked["fits"] is True

    def test_over_budget_pack_still_halts_with_the_prediction_line(
        self, tmp_path, monkeypatch, llama_cpp_dir
    ) -> None:
        recipe_path = save_recipe_with_prediction(tmp_path, PREDICTED)
        patch_packer(monkeypatch, MemoryRecipePacker(packed_bytes=WEIGHT_BUDGET + 1))

        result, out = run_pack(tmp_path, llama_cpp_dir, recipe_path)

        assert result.exit_code == 1
        assert "delta +" in result.output
        assert events_of(out)[-1] == "pack_halted"
