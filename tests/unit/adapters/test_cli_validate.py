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


def make_recipe(
    groups_bits_damage: tuple[tuple[str, int, float], ...],
    within_group: str | None = "rtn-block32",
    imatrix: str | None = None,
) -> Recipe:
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
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=tuple(
            Assignment(group=group, bits=bits, bytes=500, damage=damage)
            for group, bits, damage in groups_bits_damage
        ),
        runtime=None,
        within_group=within_group,
        imatrix=imatrix,
        protected_tensors=(),
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


def install_meter(monkeypatch, meter: MemoryDamageMeter) -> list[dict]:
    builds: list[dict] = []

    def build(model, calibration, **options):
        builds.append({"model": model, **options})
        return meter

    monkeypatch.setattr(cli_validate, "_build_meter", build)
    return builds


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

    def test_default_model_is_the_recipes_model_id(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        builds = install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 0, result.output
        assert builds == [
            {
                "model": "test/model",
                "max_tokens": 131072,
                "group_by": "layer",
                "device": "auto",
                "trust_remote_code": False,
                "gpu_memory": None,
                "within_group": "rtn",
                "imatrix": None,
            }
        ]
        assert "warning" not in result.output

    def test_model_override_reaches_the_builder_and_warns(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        builds = install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(tmp_path, recipe_path, "--model", "other/model")

        assert result.exit_code == 0, result.output
        assert builds[0]["model"] == "other/model"
        assert "warning" in result.output
        assert "test/model" in result.output

    def test_zero_prediction_reports_the_gap_without_a_percentage(
        self, tmp_path, monkeypatch
    ) -> None:
        damages = {("model.layers.0", 8): 0.0, ("model.layers.1", 8): 0.0}
        install_meter(monkeypatch, MemoryDamageMeter(specs=SPECS, damages=damages))
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe((("model.layers.0", 8, 0.0), ("model.layers.1", 8, 0.0))),
            recipe_path,
        )

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 0, result.output
        assert "gap +0.000000" in result.output
        assert "of predicted" not in result.output
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        assert events[-1]["ratio"] is None

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
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        assert events[-1]["event"] == "validation_halted"
        assert events[-1]["stage"] == "group_match"

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

    def test_kquant_recipe_resolves_the_builder_method_by_itself(
        self, tmp_path, monkeypatch
    ) -> None:
        # DEFAULT_RECIPE assigns 4- and 8-bit — inside kquant coverage.
        # No flag: the recipe's recorded method decides (ADR-0019).
        builds = install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe(DEFAULT_RECIPE, within_group="kquant-ref"), recipe_path)

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 0, result.output
        assert builds[0]["within_group"] == "kquant"
        assert "warning" not in result.output

    def test_matching_explicit_method_flag_is_accepted(
        self, tmp_path, monkeypatch
    ) -> None:
        builds = install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe(DEFAULT_RECIPE, within_group="kquant-ref"), recipe_path)

        result = invoke_validate(tmp_path, recipe_path, "--within-group", "kquant")

        assert result.exit_code == 0, result.output
        assert builds[0]["within_group"] == "kquant"

    def test_method_flag_contradicting_the_recipe_is_a_usage_error(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        # The recipe records rtn-block32 — a kquant pass would measure
        # a frame the map never priced (ADR-0019).
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(tmp_path, recipe_path, "--within-group", "kquant")

        assert result.exit_code == 2
        assert "must match" in result.output

    def test_assisted_recipe_without_imatrix_is_a_usage_error(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(
                DEFAULT_RECIPE, within_group="kquant-imx", imatrix="/runs/im.gguf"
            ),
            recipe_path,
        )

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 2
        assert "--imatrix" in result.output
        assert "assisted" in result.output

    def test_assisted_recipe_with_imatrix_builds_an_assisted_meter(
        self, tmp_path, monkeypatch
    ) -> None:
        class ImatrixAwareFake(MemoryDamageMeter):
            imatrix_covered_count = 2
            imatrix_uncovered: tuple[str, ...] = ()

        builds = install_meter(
            monkeypatch, ImatrixAwareFake(specs=SPECS, damages=dict(DAMAGES))
        )
        imatrix = tmp_path / "im.gguf"
        imatrix.write_bytes(b"GGUF")
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(
                DEFAULT_RECIPE, within_group="kquant-imx", imatrix=str(imatrix)
            ),
            recipe_path,
        )

        result = invoke_validate(tmp_path, recipe_path, "--imatrix", str(imatrix))

        assert result.exit_code == 0, result.output
        assert builds[0]["within_group"] == "kquant"
        assert builds[0]["imatrix"] == imatrix.resolve()
        assert "warning" not in result.output
        assert "imatrix covers 2 of 2 parameters" in result.output
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        started = events[0]
        assert started["within_group"] == "kquant-imx"
        assert started["imatrix"] == str(imatrix.resolve())
        assert started["recipe_within_group"] == "kquant-imx"

    def test_wrong_imatrix_file_warns_about_the_recorded_one(
        self, tmp_path, monkeypatch
    ) -> None:
        # Token matching cannot see file identity — a different file
        # contaminates the additivity comparison, so the command says
        # so (ADR-0020).
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recorded = tmp_path / "map.gguf"
        recorded.write_bytes(b"GGUF")
        other = tmp_path / "other.gguf"
        other.write_bytes(b"GGUF")
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(
                DEFAULT_RECIPE, within_group="kquant-imx", imatrix=str(recorded)
            ),
            recipe_path,
        )

        result = invoke_validate(tmp_path, recipe_path, "--imatrix", str(other))

        assert result.exit_code == 0, result.output
        assert "warning" in result.output
        assert "differs from the map" in result.output

    def test_imatrix_against_an_unassisted_recipe_is_a_usage_error(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe(DEFAULT_RECIPE, within_group="kquant-ref"), recipe_path)
        imatrix = tmp_path / "im.gguf"
        imatrix.write_bytes(b"GGUF")

        result = invoke_validate(tmp_path, recipe_path, "--imatrix", str(imatrix))

        assert result.exit_code == 2
        assert "must match" in result.output

    def test_imatrix_against_a_rtn_recipe_names_the_record(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        # The recipe records rtn-block32 — the refusal must name the
        # record, not tell the user to add --within-group kquant and
        # walk into a second failure.
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        imatrix = tmp_path / "im.gguf"
        imatrix.write_bytes(b"GGUF")

        result = invoke_validate(tmp_path, recipe_path, "--imatrix", str(imatrix))

        assert result.exit_code == 2
        assert "rtn-block32" in result.output
        assert "must match" in result.output

    def test_imatrix_with_the_rtn_method_is_a_usage_error(
        self, tmp_path, monkeypatch
    ) -> None:
        # Without a record there is no provenance to name — the
        # flag-level pairing rule answers.
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe(DEFAULT_RECIPE, within_group=None), recipe_path)
        imatrix = tmp_path / "im.gguf"
        imatrix.write_bytes(b"GGUF")

        result = invoke_validate(tmp_path, recipe_path, "--imatrix", str(imatrix))

        assert result.exit_code == 2
        assert "requires --within-group kquant" in result.output

    def test_missing_imatrix_file_is_a_usage_error(self, tmp_path, monkeypatch) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(
                DEFAULT_RECIPE, within_group="kquant-imx", imatrix="/runs/im.gguf"
            ),
            recipe_path,
        )

        result = invoke_validate(
            tmp_path, recipe_path, "--imatrix", str(tmp_path / "no-such.gguf")
        )

        assert result.exit_code == 2
        assert "is not a file" in result.output

    def test_unknown_recorded_method_is_a_usage_error(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_recipe(DEFAULT_RECIPE, within_group="future-method"), recipe_path
        )

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 2
        assert "future-method" in result.output

    def test_recipe_without_provenance_warns_and_defaults_to_rtn(
        self, tmp_path, monkeypatch
    ) -> None:
        builds = install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe(DEFAULT_RECIPE, within_group=None), recipe_path)

        result = invoke_validate(tmp_path, recipe_path)

        assert result.exit_code == 0, result.output
        assert builds[0]["within_group"] == "rtn"
        assert "warning: the recipe does not record" in result.output

    def test_recipe_without_provenance_accepts_an_explicit_kquant_flag(
        self, tmp_path, monkeypatch
    ) -> None:
        # The legacy path: no record to enforce against, so the flag
        # decides — and the runlog must label the frame it measured.
        builds = install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe(DEFAULT_RECIPE, within_group=None), recipe_path)

        result = invoke_validate(tmp_path, recipe_path, "--within-group", "kquant")

        assert result.exit_code == 0, result.output
        assert "warning: the recipe does not record" in result.output
        assert builds[0]["within_group"] == "kquant"
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        assert events[0]["within_group"] == "kquant-ref"

    def test_recipe_without_provenance_accepts_kquant_with_imatrix(
        self, tmp_path, monkeypatch
    ) -> None:
        builds = install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(make_recipe(DEFAULT_RECIPE, within_group=None), recipe_path)
        imatrix = tmp_path / "im.gguf"
        imatrix.write_bytes(b"GGUF")

        result = invoke_validate(
            tmp_path, recipe_path, "--within-group", "kquant", "--imatrix", str(imatrix)
        )

        assert result.exit_code == 0, result.output
        assert builds[0]["within_group"] == "kquant"
        assert builds[0]["imatrix"] == imatrix
        events = events_of(recipe_path.with_name("recipe.validation.runlog.jsonl"))
        assert events[0]["within_group"] == "kquant-imx"

    def test_unknown_within_group_is_a_usage_error(
        self, tmp_path, monkeypatch, recipe_path
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )

        result = invoke_validate(tmp_path, recipe_path, "--within-group", "awq")

        assert result.exit_code == 2
        assert "--within-group" in result.output

    def test_kquant_with_uncovered_recipe_bits_is_a_usage_error(
        self, tmp_path, monkeypatch
    ) -> None:
        install_meter(
            monkeypatch, MemoryDamageMeter(specs=SPECS, damages=dict(DAMAGES))
        )
        recipe = make_recipe(
            (("model.layers.0", 6, 0.01), ("model.layers.1", 8, 0.0)),
            within_group=None,
        )
        path = tmp_path / "recipe6.json"
        save_recipe(recipe, path)

        result = invoke_validate(tmp_path, path, "--within-group", "kquant")

        assert result.exit_code == 2
        assert "kquant covers" in result.output
        assert "[6]" in result.output

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
