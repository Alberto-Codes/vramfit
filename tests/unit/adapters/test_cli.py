from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from quantfit import __version__
from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.recipe_json import load_recipe
from tests.unit.conftest import make_map

runner = CliRunner()

CURVE = {8: 0.001, 4: 0.010, 3: 0.100, 2: 1.000}


@pytest.mark.unit
def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.output == f"quantfit {__version__}\n"


@pytest.mark.unit
def test_scan_command_unimplemented_exits_nonzero() -> None:
    result = runner.invoke(app, ["scan"])

    assert result.exit_code == 1
    assert "not implemented" in result.output


@pytest.mark.unit
def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "scan" in result.output
    assert "plan" in result.output
    assert "budget" in result.output


@pytest.mark.unit
class TestPlanCommand:
    def _write_map(self, tmp_path, groups=None):
        raw = make_map(groups or [("g0", 160_000, CURVE), ("g1", 160_000, CURVE)])
        path = tmp_path / "sensitivity.json"
        path.write_text(json.dumps(raw))
        return path

    def test_writes_valid_recipe_file(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "200000",
                "--kv-headroom",
                "50000",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        assert recipe.plan.weight_budget_bytes == 150_000
        assert recipe.plan.vram_budget_bytes == 200_000
        assert "planned 2 groups" in result.output

    def test_pin_flag_reaches_solver(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "500000",
                "--kv-headroom",
                "1000",
                "--pin",
                "g1=4",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        by_group = {a.group: a.bits for a in recipe.assignments}
        assert by_group["g1"] == 4
        assert recipe.plan.pins == {"g1": 4}

    def test_infeasible_budget_exits_one_and_reports_gap(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            ["plan", str(map_path), "--vram", "40000", "--kv-headroom", "1000"],
        )

        assert result.exit_code == 1
        assert "no recipe fits" in result.output
        assert "over" in result.output

    def test_malformed_pin_exits_two(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            ["plan", str(map_path), "--vram", "200000", "--pin", "g0:8"],
        )

        assert result.exit_code == 2
        assert "pattern=bits" in result.output

    def test_unmatched_pin_exits_one(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "200000",
                "--kv-headroom",
                "1000",
                "--pin",
                "nope*=8",
            ],
        )

        assert result.exit_code == 1
        assert "matches no group" in result.output

    def test_invalid_map_exits_one_with_artifact_error(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"quantfit_schema": 99}')

        result = runner.invoke(
            app, ["plan", str(path), "--vram", "200000", "--kv-headroom", "1000"]
        )

        assert result.exit_code == 1
        assert "unsupported schema version" in result.output

    def test_missing_map_file_exits_one(self, tmp_path) -> None:
        result = runner.invoke(
            app,
            [
                "plan",
                str(tmp_path / "absent.json"),
                "--vram",
                "200000",
                "--kv-headroom",
                "1000",
            ],
        )

        assert result.exit_code == 1
        assert "absent.json" in result.output

    def test_headroom_swallowing_vram_exits_one(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            ["plan", str(map_path), "--vram", "1GiB", "--kv-headroom", "2GiB"],
        )

        assert result.exit_code == 1
        assert "nothing for weights" in result.output

    def test_malformed_vram_exits_two(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(app, ["plan", str(map_path), "--vram", "wat"])

        assert result.exit_code == 2


@pytest.mark.unit
class TestBudgetCommand:
    def test_manual_shape_flags_print_weight_budget(self) -> None:
        result = runner.invoke(
            app,
            [
                "budget",
                "--vram",
                "24GiB",
                "--attn-layers",
                "49",
                "--kv-heads",
                "8",
                "--head-dim",
                "128",
                "--context",
                "16384",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "attention layers      49" in result.output
        assert "200704 bytes/token" in result.output
        assert "weight budget" in result.output

    def test_model_config_path_parses_decilm(self, tmp_path) -> None:
        block = {"attention": {"n_heads_in_group": 8, "no_op": False}, "ffn": {}}
        no_op = {"attention": {"no_op": True}, "ffn": {}}
        config = {
            "num_attention_heads": 64,
            "hidden_size": 8192,
            "block_configs": [block, no_op, block],
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        result = runner.invoke(app, ["budget", "--model-config", str(path)])

        assert result.exit_code == 0, result.output
        assert "attention layers      2" in result.output

    def test_both_shape_sources_exits_two(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{}")

        result = runner.invoke(
            app,
            ["budget", "--model-config", str(path), "--attn-layers", "49"],
        )

        assert result.exit_code == 2

    def test_neither_shape_source_exits_two(self) -> None:
        result = runner.invoke(app, ["budget", "--vram", "24GiB"])

        assert result.exit_code == 2

    def test_unknown_kv_dtype_exits_two(self) -> None:
        result = runner.invoke(
            app,
            [
                "budget",
                "--attn-layers",
                "1",
                "--kv-heads",
                "1",
                "--head-dim",
                "1",
                "--kv-dtype",
                "int4",
            ],
        )

        assert result.exit_code == 2

    def test_negative_weight_budget_exits_one(self) -> None:
        result = runner.invoke(
            app,
            [
                "budget",
                "--vram",
                "1GiB",
                "--attn-layers",
                "49",
                "--kv-heads",
                "8",
                "--head-dim",
                "128",
                "--context",
                "16384",
            ],
        )

        assert result.exit_code == 1
        assert "nothing left for weights" in result.output

    @pytest.mark.parametrize(
        "flag",
        ["--attn-layers", "--kv-heads", "--head-dim", "--context", "--sequences"],
    )
    def test_nonpositive_int_option_exits_two(self, flag: str) -> None:
        args = ["budget", "--attn-layers", "49", "--kv-heads", "8", "--head-dim", "128"]
        i = args.index(flag) + 1 if flag in args else None
        if i is not None:
            args[i] = "0"
        else:
            args += [flag, "0"]

        result = runner.invoke(app, args)

        assert result.exit_code == 2

    def test_unreadable_config_exits_one(self, tmp_path) -> None:
        result = runner.invoke(
            app, ["budget", "--model-config", str(tmp_path / "absent.json")]
        )

        assert result.exit_code == 1
