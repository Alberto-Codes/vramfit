from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.unit.conftest import make_map
from vramfit import __version__
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.recipe_json import load_recipe
from vramfit.domain.solver import DEFAULT_FORMAT_OVERHEAD, DEFAULT_RESIDUAL_OVERHEAD

runner = CliRunner()

CURVE = {8: 0.001, 4: 0.010, 3: 0.100, 2: 1.000}


@pytest.mark.unit
def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.output == f"vramfit {__version__}\n"


@pytest.mark.unit
def test_scan_command_without_arguments_exits_with_usage_error() -> None:
    result = runner.invoke(app, ["scan"])

    assert result.exit_code == 2
    assert "Missing" in result.output


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

    def test_unknown_map_field_warns_and_still_plans(self, tmp_path) -> None:
        # The rendered line, not the interpreter's own warning format,
        # which names a vramfit source file and tells the operator
        # nothing about their map (#261).
        raw = make_map([("g0", 160_000, CURVE), ("g1", 160_000, CURVE)])
        raw["notes"] = "hand note"
        map_path = tmp_path / "sensitivity.json"
        map_path.write_text(json.dumps(raw))
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
        # Assert on stderr, not the combined stream: ADR-0011 puts the
        # warning on the human channel, and stdout carries the report
        # line a caller pipes.
        assert result.stderr == (
            "warning: $.notes: vramfit does not know this field. A save drops it.\n"
        )
        assert "planned 2 groups" in result.stdout
        assert "UnknownArtifactFieldWarning" not in result.output

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

    def test_runtime_vllm_filters_the_candidate_set(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)
        out = tmp_path / "recipe.json"

        # 90_000 forces downgrades below 8-bit for both groups; with
        # vLLM's {8, 4} the solver must stop at 4, never at 3 or 2.
        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "100000",
                "--kv-headroom",
                "10000",
                "--runtime",
                "vllm",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        assert recipe.runtime == "vllm"
        assert all(a.bits in {8, 4} for a in recipe.assignments)
        assert "for vllm" in result.output
        # The narrowing is reported, never silent.
        assert "[3, 2] dropped" in result.output

    def test_runtime_dropping_nothing_prints_no_narrowing_line(self, tmp_path) -> None:
        raw = make_map([("g0", 160_000, {8: 0.001, 4: 0.010})], precisions=(8, 4))
        map_path = tmp_path / "sensitivity.json"
        map_path.write_text(json.dumps(raw))
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "100000",
                "--kv-headroom",
                "10000",
                "--runtime",
                "vllm",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "dropped" not in result.output

    def test_default_runtime_is_llama_cpp(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "400000",
                "--kv-headroom",
                "50000",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert load_recipe(out).runtime == "llama.cpp"

    def test_default_overhead_is_the_residual_for_llama_cpp(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "400000",
                "--kv-headroom",
                "50000",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        # Default runtime llama.cpp has an effective-bits table
        # (ADR-0014): sizes are per-type and the overhead shrinks to
        # the residual.
        assert recipe.plan.format_overhead == DEFAULT_RESIDUAL_OVERHEAD
        assert recipe.assignments[0].bytes == 85_425  # ceil(160000*8.5/16*1.005)

    def test_default_overhead_is_the_scalar_for_vllm(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "400000",
                "--kv-headroom",
                "50000",
                "--runtime",
                "vllm",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        # vLLM has no effective-bits table — an omitted
        # --format-overhead must reach the solver as None and resolve
        # to the scalar, not be substituted eagerly by the CLI.
        assert load_recipe(out).plan.format_overhead == DEFAULT_FORMAT_OVERHEAD

    def test_explicit_format_overhead_is_recorded_verbatim(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "400000",
                "--kv-headroom",
                "50000",
                "--format-overhead",
                "0.03",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert load_recipe(out).plan.format_overhead == 0.03

    def test_unknown_runtime_exits_two(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            ["plan", str(map_path), "--vram", "400000", "--runtime", "tgi"],
        )

        assert result.exit_code == 2
        assert "unknown runtime" in result.output

    def test_infeasible_budget_exits_one_and_reports_gap(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            ["plan", str(map_path), "--vram", "40000", "--kv-headroom", "1000"],
        )

        assert result.exit_code == 1
        assert "no recipe fits" in result.output
        assert "over" in result.output

    @pytest.mark.parametrize("overhead", ["nan", "inf"], ids=["nan", "inf"])
    def test_non_finite_format_overhead_exits_two(self, tmp_path, overhead) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "200000",
                "--format-overhead",
                overhead,
            ],
        )

        assert result.exit_code == 2
        assert "must be finite" in result.output

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
        path.write_text('{"vramfit_schema": 99}')

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

    def test_repeated_pin_pattern_keeps_last_position_and_value(self, tmp_path) -> None:
        map_path = self._write_map(
            tmp_path, groups=[("a1", 160_000, CURVE), ("b1", 160_000, CURVE)]
        )
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
                "a*=8",
                "--pin",
                "*1=4",
                "--pin",
                "a*=2",
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        by_group = {a.group: a.bits for a in recipe.assignments}
        # a* was repeated last, so it must override *1 for a1.
        assert by_group["a1"] == 2
        assert by_group["b1"] == 4

    @pytest.mark.parametrize("bad", ["g0:8", "g0=--4", "g0=²", "g0=-4", "g0=0"])
    def test_malformed_pin_variants_exit_two(self, tmp_path, bad: str) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app, ["plan", str(map_path), "--vram", "200000", "--pin", bad]
        )

        assert result.exit_code == 2
        assert "pattern=bits" in result.output

    def test_unwritable_out_exits_one_with_message(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "500000",
                "--kv-headroom",
                "1000",
                "--out",
                str(tmp_path / "no_such_dir" / "recipe.json"),
            ],
        )

        assert result.exit_code == 1
        assert "error:" in result.output

    def test_negative_format_overhead_exits_two(self, tmp_path) -> None:
        map_path = self._write_map(tmp_path)

        result = runner.invoke(
            app,
            ["plan", str(map_path), "--vram", "200000", "--format-overhead", "-0.1"],
        )

        assert result.exit_code == 2

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


class TestPlanProtect:
    def _write_protected_map(self, tmp_path):
        raw = make_map(
            [("model.layers.0", 160_000, CURVE), ("model.layers.1", 160_000, CURVE)]
        )
        for entry in raw["groups"]:
            name = entry["name"]
            entry["tensors"] = [
                f"{name}.self_attn.v_proj.weight",
                f"{name}.mlp.down_proj.weight",
            ]
            entry["tensor_bytes"] = {
                f"{name}.self_attn.v_proj.weight": 32_000,
                f"{name}.mlp.down_proj.weight": 128_000,
            }
        path = tmp_path / "sensitivity.json"
        path.write_text(json.dumps(raw))
        return path

    def _plan(self, map_path, out, *extra: str):
        return runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "500000",
                "--kv-headroom",
                "1000",
                "--out",
                str(out),
                *extra,
            ],
        )

    def test_protect_flag_reaches_the_recipe(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = self._plan(
            map_path,
            out,
            "--protect",
            "*.self_attn.v_proj.weight=5",
            "--pin",
            "model.layers.0=3",
        )

        assert result.exit_code == 0, result.output
        assert "1 protected tensors" in result.output
        recipe = load_recipe(out)
        assert dict(recipe.plan.protections) == {"*.self_attn.v_proj.weight": 5}
        resolved = {p.tensor: p.bits for p in recipe.protected_tensors}
        # Layer 0 is pinned at 3-bit, so its v_proj rises to the floor;
        # layer 1 stays at 8-bit above the floor, so its pair drops —
        # a no-op pair would falsely fail the reconstruction gate.
        assert resolved == {"model.layers.0.self_attn.v_proj.weight": 5}

    def test_noop_protection_warns(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = self._plan(map_path, out, "--protect", "*.self_attn.v_proj.weight=3")

        assert result.exit_code == 0, result.output
        assert "no-op" in result.output
        # The dead rule warns once, per pattern — its tensors do not
        # warn again individually.
        assert "per-tensor no-op" not in result.output

    def test_effective_protection_does_not_warn(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = self._plan(
            map_path,
            out,
            "--protect",
            "*.self_attn.v_proj.weight=5",
            "--pin",
            "model.layers.0=3",
            "--pin",
            "model.layers.1=3",
        )

        assert result.exit_code == 0, result.output
        assert "no-op" not in result.output

    def test_partial_noop_protection_warns_per_tensor(self, tmp_path) -> None:
        # The glob lifts layer 0's floor and no-ops on layer 1 — the
        # per-pattern warning is blind to the partial case (issue #59).
        map_path = self._write_protected_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = self._plan(
            map_path,
            out,
            "--protect",
            "*.self_attn.v_proj.weight=5",
            "--pin",
            "model.layers.0=3",
        )

        assert result.exit_code == 0, result.output
        assert "per-tensor no-op" in result.output
        assert "model.layers.1.self_attn.v_proj.weight" in result.output
        assert "drops the pair" in result.output
        # The rule lifted a real floor, so the pattern warning stays out.
        assert "is a no-op — every tensor" not in result.output

    def test_dropped_pair_warns_about_its_lost_exclusion(self, tmp_path) -> None:
        # The exclusion glob matches layer 0's surviving pair and
        # layer 1's dropped one — the survivor keeps the pattern
        # alive, and the dropped mark must warn (issue #59).
        map_path = self._write_protected_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = self._plan(
            map_path,
            out,
            "--protect",
            "*.self_attn.v_proj.weight=5",
            "--pin",
            "model.layers.0=3",
            "--exclude-imatrix",
            "*.self_attn.v_proj.weight",
        )

        assert result.exit_code == 0, result.output
        assert "imatrix exclusion drops with it" in result.output
        recipe = load_recipe(out)
        marks = {p.tensor: p.exclude_imatrix for p in recipe.protected_tensors}
        assert marks == {"model.layers.0.self_attn.v_proj.weight": True}

    def test_exclusion_riding_only_dropped_pairs_exits_one(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)

        result = self._plan(
            map_path,
            tmp_path / "r.json",
            "--protect",
            "*.self_attn.v_proj.weight=5",
            "--pin",
            "model.layers.0=3",
            "--exclude-imatrix",
            "model.layers.1.self_attn.v_proj.weight",
        )

        assert result.exit_code == 1
        assert "every protected tensor it matches drops" in result.output

    def test_unmatched_protection_exits_one(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)

        result = self._plan(map_path, tmp_path / "r.json", "--protect", "*.nope=5")

        assert result.exit_code == 1
        assert "matches no tensor" in result.output

    def test_unservable_floor_exits_one_naming_the_table(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)

        result = self._plan(
            map_path, tmp_path / "r.json", "--protect", "*.v_proj.weight=7"
        )

        assert result.exit_code == 1
        assert "cannot serve 7-bit" in result.output

    def test_map_without_tensor_sizes_exits_one_naming_the_field(
        self, tmp_path
    ) -> None:
        raw = make_map(
            [("model.layers.0", 160_000, CURVE), ("model.layers.1", 160_000, CURVE)]
        )
        for entry in raw["groups"]:
            entry["tensors"] = [
                f"{entry['name']}.self_attn.v_proj.weight",
                f"{entry['name']}.mlp.down_proj.weight",
            ]
        map_path = tmp_path / "sensitivity.json"
        map_path.write_text(json.dumps(raw))

        result = self._plan(
            map_path, tmp_path / "r.json", "--protect", "*.v_proj.weight=5"
        )

        assert result.exit_code == 1
        assert "tensor_bytes" in result.output

    def test_malformed_protect_exits_two(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)

        result = self._plan(map_path, tmp_path / "r.json", "--protect", "v_proj.weight")

        assert result.exit_code == 2
        assert "pattern=bits" in result.output

    def test_exclude_imatrix_marks_the_matched_pair(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = self._plan(
            map_path,
            out,
            "--protect",
            "*.self_attn.v_proj.weight=5",
            "--pin",
            "model.layers.0=3",
            "--exclude-imatrix",
            "model.layers.0.*",
        )

        assert result.exit_code == 0, result.output
        assert "(1 imatrix-excluded)" in result.output
        recipe = load_recipe(out)
        assert recipe.plan.imatrix_exclusions == ("model.layers.0.*",)
        # Layer 1's pair drops as a per-tensor no-op (issue #59), so
        # the pinned layer 0 carries the only mark.
        marks = {p.tensor: p.exclude_imatrix for p in recipe.protected_tensors}
        assert marks == {"model.layers.0.self_attn.v_proj.weight": True}

    def test_exclude_imatrix_without_protect_exits_one(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)

        result = self._plan(
            map_path, tmp_path / "r.json", "--exclude-imatrix", "model.layers.0.*"
        )

        assert result.exit_code == 1
        assert "require protections" in result.output

    def test_exclude_imatrix_overreaching_glob_warns(self, tmp_path) -> None:
        map_path = self._write_protected_map(tmp_path)
        out = tmp_path / "recipe.json"

        result = self._plan(
            map_path,
            out,
            "--protect",
            "model.layers.0.self_attn.v_proj.weight=5",
            "--pin",
            "model.layers.0=3",
            "--exclude-imatrix",
            "*.self_attn.v_proj.weight",
        )

        assert result.exit_code == 0, result.output
        assert "their imatrix rows stay" in result.output
        assert "model.layers.1.self_attn.v_proj.weight" in result.output

    def test_exclude_imatrix_matching_unprotected_tensor_exits_one(
        self, tmp_path
    ) -> None:
        map_path = self._write_protected_map(tmp_path)

        result = self._plan(
            map_path,
            tmp_path / "r.json",
            "--protect",
            "*.self_attn.v_proj.weight=5",
            "--exclude-imatrix",
            "*.mlp.down_proj.weight",
        )

        assert result.exit_code == 1
        assert "only unprotected" in result.output
