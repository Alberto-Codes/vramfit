"""What `plan --checkpoint` does when the two surfaces merge differently.

`vramfit scan` names groups from the loaded `transformers` model and
`vramfit plan --checkpoint` names them from the safetensors headers.
Qwen3-MoE on `transformers` 5.16.1 loads gate and up as one
``mlp.experts.gate_up_proj`` parameter, and the checkpoint keeps
``gate_proj`` and ``up_proj`` apart (issue #576). The defect survived
until a finished scan met a real checkpoint, so these tests drive both
surfaces at once: a merged map against a split checkpoint.

The decisive test is `test_a_budget_only_the_expert_mass_can_meet_solves`.
Its budget is unreachable while gate and up hold at reference
precision, so removing the reconciliation fails it with the refusal
the run reported.

`test_the_merged_measurement_counts_once_in_the_prediction` pins the
other half: one perturbation measured one curve, so the recipe states
it once.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.conftest import make_map
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.recipe_json import load_recipe

pytestmark = pytest.mark.unit

runner = CliRunner()

LAYERS = 2
EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 128
# One expert's projection at BF16. Gate, up, and down are the same
# size: each is HIDDEN x INTERMEDIATE elements at two bytes.
EXPERT_BYTES = HIDDEN * INTERMEDIATE * 2
# One attention projection at BF16, HIDDEN x HIDDEN.
ATTENTION_BYTES = HIDDEN * HIDDEN * 2
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")

MERGED_CURVE = {8: 0.001, 4: 0.010, 3: 0.100, 2: 1.000}
DOWN_CURVE = {8: 0.002, 4: 0.020, 3: 0.200, 2: 2.000}
ATTENTION_CURVE = {8: 0.003, 4: 0.030, 3: 0.300, 2: 3.000}

# Every weight the checkpoint holds, at reference precision.
MODEL_BYTES = LAYERS * (3 * EXPERTS * EXPERT_BYTES + 4 * ATTENTION_BYTES)
# Between the two minima. Reaching it needs every expert group down
# at 2-bit, which only the reconciled names allow: gate and up held
# at reference precision alone cost 2 * LAYERS * EXPERTS *
# EXPERT_BYTES, which is over this budget on its own.
TIGHT_BUDGET = MODEL_BYTES // 4
HEADROOM = 50_000


def merged_group(layer: int) -> str:
    """The group name the loaded model reports for gate and up."""
    return f"model.layers.{layer}.mlp.experts.gate_up_proj"


def split_groups(layer: int) -> tuple[str, str]:
    """The group names the checkpoint keeps apart."""
    return (
        f"model.layers.{layer}.mlp.experts.gate_proj",
        f"model.layers.{layer}.mlp.experts.up_proj",
    )


def write_map(tmp_path: Path, *, merged: bool = True) -> Path:
    """Write the map a scan of this checkpoint produces.

    Args:
        tmp_path: The test's directory.
        merged: True for the surface a merging `transformers` reports,
            False for the split surface an older one reports.

    Returns:
        Where the map was written.
    """
    groups: list[tuple[str, int, dict[int, float]]] = []
    for layer in range(LAYERS):
        stack = EXPERTS * EXPERT_BYTES
        if merged:
            groups.append((merged_group(layer), 2 * stack, MERGED_CURVE))
        else:
            groups.extend((name, stack, MERGED_CURVE) for name in split_groups(layer))
        groups.append(
            (f"model.layers.{layer}.mlp.experts.down_proj", stack, DOWN_CURVE)
        )
        groups.extend(
            (f"model.layers.{layer}.self_attn.{name}", ATTENTION_BYTES, ATTENTION_CURVE)
            for name in ATTENTION
        )
    raw = make_map(groups)
    raw["scan"]["group_by"] = "stack"
    path = tmp_path / f"sensitivity-{'merged' if merged else 'split'}.json"
    path.write_text(json.dumps(raw))
    return path


def write_checkpoint(tmp_path: Path) -> Path:
    """Write the one-shard checkpoint, experts stored one per tensor."""
    entries: dict[str, tuple[int, list[int]]] = {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            stem = f"model.layers.{layer}.mlp.experts.{expert}"
            for name in ("gate_proj", "up_proj"):
                entries[f"{stem}.{name}.weight"] = (
                    EXPERT_BYTES,
                    [INTERMEDIATE, HIDDEN],
                )
            entries[f"{stem}.down_proj.weight"] = (EXPERT_BYTES, [HIDDEN, INTERMEDIATE])
        for name in ATTENTION:
            entries[f"model.layers.{layer}.self_attn.{name}.weight"] = (
                ATTENTION_BYTES,
                [HIDDEN, HIDDEN],
            )
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir(parents=True, exist_ok=True)
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for name, (span, shape) in entries.items():
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + span],
        }
        offset += span
    blob = json.dumps(header).encode("utf-8")
    (model_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return model_dir


def plan(map_path: Path, out: Path, budget: int, *extra: str):
    """Run `vramfit plan` for a weight budget, returning the result."""
    return runner.invoke(
        app,
        [
            "plan",
            str(map_path),
            "--vram",
            str(budget + HEADROOM),
            "--kv-headroom",
            str(HEADROOM),
            "--out",
            str(out),
            *extra,
        ],
    )


class TestAMergedMapAgainstASplitCheckpoint:
    """The name gap issue #576 measured, driven from both sides."""

    def test_a_budget_only_the_expert_mass_can_meet_solves(self, tmp_path) -> None:
        # The check that matters. Gate and up are two thirds of the
        # expert mass. Held at reference precision they cost more than
        # this whole budget, so a plan that does not reconcile the
        # names refuses here rather than solving.
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path),
            out,
            TIGHT_BUDGET,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        assigned = {a.group: a.bits for a in recipe.assignments}
        for layer in range(LAYERS):
            for name in split_groups(layer):
                assert assigned[name] < 16, f"{name} held at reference precision"
        assert recipe.plan.predicted_total_bytes <= TIGHT_BUDGET

    def test_the_recipe_names_the_checkpoints_projections_and_not_the_merged_one(
        self, tmp_path
    ) -> None:
        # `pack` addresses `ffn_gate_exps` and `ffn_up_exps` by name.
        # A recipe naming the merged parameter reaches neither.
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path),
            out,
            MODEL_BYTES,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
        )

        assert result.exit_code == 0, result.output
        groups = {a.group for a in load_recipe(out).assignments}
        for layer in range(LAYERS):
            assert merged_group(layer) not in groups
            assert set(split_groups(layer)) <= groups

    def test_every_checkpoint_group_is_measured_so_none_holds_at_reference(
        self, tmp_path
    ) -> None:
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path),
            out,
            MODEL_BYTES,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
        )

        assert result.exit_code == 0, result.output
        # Gate and up fold onto the one group the scan measured, so
        # the priced set holds one expert group less per layer than
        # the checkpoint spells.
        priced = LAYERS * (2 + len(ATTENTION))
        assert (
            f"checkpoint holds {priced} groups: {priced} measured by "
            f"the map, 0 held at reference precision" in result.output
        )

    def test_the_command_reports_the_reconciliation_it_made(self, tmp_path) -> None:
        # Reader-facing provenance: the pair's damage curve was
        # measured on the merged parameter, not on either half.
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path),
            out,
            MODEL_BYTES,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
        )

        assert result.exit_code == 0, result.output
        assert f"reconciled {LAYERS} merged projections" in result.output
        assert merged_group(0) in result.output
        assert "one measured damage curve counts once" in result.output

    def test_the_merged_measurement_counts_once_in_the_prediction(
        self, tmp_path
    ) -> None:
        # The scan perturbed gate and up together, so one curve
        # covers both. Charging it to each half would inflate
        # `predicted_damage` and `validate`'s ratio with it.
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path),
            out,
            MODEL_BYTES,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        top = max(MERGED_CURVE)
        expected = LAYERS * (
            MERGED_CURVE[top] + DOWN_CURVE[top] + len(ATTENTION) * ATTENTION_CURVE[top]
        )
        assert recipe.plan.predicted_damage == pytest.approx(expected)
        rows = {a.group: a for a in recipe.assignments}
        gate, up = split_groups(0)
        assert rows[gate].damage == MERGED_CURVE[top]
        assert rows[up].damage == 0.0

    def test_the_split_rows_share_one_precision_and_the_merged_prediction(
        self, tmp_path
    ) -> None:
        # One measurement prices the pair, so the solver moves it as
        # one unit and the two rows sum to what it predicted.
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path),
            out,
            TIGHT_BUDGET,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
        )

        assert result.exit_code == 0, result.output
        recipe = load_recipe(out)
        rows = {a.group: a for a in recipe.assignments}
        for layer in range(LAYERS):
            gate, up = split_groups(layer)
            assert rows[gate].bits == rows[up].bits
        assert sum(a.bytes for a in recipe.assignments) == (
            recipe.plan.predicted_total_bytes
        )

    def test_a_pin_on_the_experts_now_reaches_gate_and_up(self, tmp_path) -> None:
        # "Scan them to spend it" became followable, and so did a pin
        # over the same names.
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path),
            out,
            MODEL_BYTES,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
            "--pin",
            "model.layers.*.mlp.experts.*=4",
        )

        assert result.exit_code == 0, result.output
        assigned = {a.group: a.bits for a in load_recipe(out).assignments}
        for layer in range(LAYERS):
            for name in split_groups(layer):
                assert assigned[name] == 4


class TestSurfacesThatAlreadyAgree:
    """The reconciliation stays out of the way when it is not needed."""

    def test_a_split_map_plans_unchanged_and_reports_no_split(self, tmp_path) -> None:
        # An older `transformers` names gate and up apart, and the
        # same checkpoint then needs no reconciliation.
        out = tmp_path / "recipe.json"

        result = plan(
            write_map(tmp_path, merged=False),
            out,
            TIGHT_BUDGET,
            "--checkpoint",
            str(write_checkpoint(tmp_path)),
        )

        assert result.exit_code == 0, result.output
        assert "reconciled" not in result.output
        groups = {a.group for a in load_recipe(out).assignments}
        assert set(split_groups(0)) <= groups

    def test_without_a_checkpoint_the_merged_map_plans_its_own_names(
        self, tmp_path
    ) -> None:
        # No checkpoint means no authority on names, so nothing splits.
        out = tmp_path / "recipe.json"

        result = plan(write_map(tmp_path), out, MODEL_BYTES, "--runtime", "vllm")

        assert result.exit_code == 0, result.output
        groups = {a.group for a in load_recipe(out).assignments}
        assert merged_group(0) in groups
        assert "reconciled" not in result.output
