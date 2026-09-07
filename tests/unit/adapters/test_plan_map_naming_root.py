"""What `plan --checkpoint` does with each naming root a map can carry.

The scan normalizes no root, so a map repeats the root its module tree
names (ADR-0029, amended 2026-09-06, issue #552). `docs/reference/cli.md`
states what the plan then does. These pin that text against the command.

A `backbone.`-rooted map planned against its own checkpoint is the
#564 defect. The tests below hold the current behavior, not the wanted
one: they pin the doubled recipe and the over-budget refusal so a fix
to #564 fails them loudly.
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

CURVE = {8: 0.001, 4: 0.010, 3: 0.100, 2: 1.000}
LAYERS = 3
LAYER_BYTES = 160_000
HEAD_BYTES = 40_000
MODEL_BYTES = LAYERS * LAYER_BYTES + HEAD_BYTES


def write_map(tmp_path: Path, root: str) -> Path:
    """Write a layer map rooted at `root`, keeping a root-less head."""
    raw = make_map(
        [(f"{root}layers.{i}", LAYER_BYTES, CURVE) for i in range(LAYERS)]
        + [("lm_head", HEAD_BYTES, CURVE)]
    )
    for group in raw["groups"]:
        if group["name"] != "lm_head":
            group["tensors"] = [f"{group['name']}.mlp.up_proj.weight"]
    path = tmp_path / f"sensitivity-{root.strip('.') or 'root'}.json"
    path.write_text(json.dumps(raw))
    return path


def write_checkpoint(tmp_path: Path, root: str) -> Path:
    """Write the one-shard checkpoint that map was scanned from."""
    entries = {f"{root}layers.{i}.mlp.up_proj.weight": LAYER_BYTES for i in range(LAYERS)}
    entries["lm_head.weight"] = HEAD_BYTES
    model_dir = tmp_path / f"checkpoint-{root.strip('.') or 'root'}"
    model_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    header = {}
    for name, span in entries.items():
        header[name] = {
            "dtype": "BF16",
            "shape": [span // 200, 100],
            "data_offsets": [offset, offset + span],
        }
        offset += span
    blob = json.dumps(header).encode("utf-8")
    (model_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return model_dir


def plan(map_path: Path, model_dir: Path, out: Path, vram: int, *extra: str):
    return runner.invoke(
        app,
        [
            "plan",
            str(map_path),
            "--checkpoint",
            str(model_dir),
            "--vram",
            str(vram),
            "--kv-headroom",
            "50000",
            "--out",
            str(out),
            *extra,
        ],
    )


class TestBackboneRootedMap:
    """The root the coverage match never reconciles (#564)."""

    def test_its_own_checkpoint_warns_and_prices_every_decoder_layer_twice(
        self, tmp_path
    ) -> None:
        map_path = write_map(tmp_path, "backbone.")
        model_dir = write_checkpoint(tmp_path, "backbone.")
        out = tmp_path / "recipe.json"

        result = plan(map_path, model_dir, out, 2_000_000)

        assert result.exit_code == 0, result.output
        assert (
            f"the checkpoint does not carry {LAYERS} of the map's groups"
            in result.stderr
        )
        groups = {a.group for a in load_recipe(out).assignments}
        for i in range(LAYERS):
            assert f"backbone.layers.{i}" in groups
            assert f"model.layers.{i}" in groups

    def test_a_root_less_group_matches_so_no_total_miss_refusal_fires(
        self, tmp_path
    ) -> None:
        # `lm_head` carries no root, so the coverage match finds it by
        # exact string. That keeps the checkpoint out of the total-miss
        # refusal, and the doubled plan proceeds.
        map_path = write_map(tmp_path, "backbone.")
        model_dir = write_checkpoint(tmp_path, "backbone.")
        out = tmp_path / "recipe.json"

        result = plan(map_path, model_dir, out, 2_000_000)

        assert result.exit_code == 0, result.output
        assert "carries none of the map's groups" not in result.output
        assert (
            f"checkpoint holds {LAYERS + 1} groups: 1 measured by the map, "
            f"{LAYERS} held at reference precision" in result.stdout
        )

    def test_pinned_at_reference_the_recipe_reserves_about_twice_the_model(
        self, tmp_path
    ) -> None:
        map_path = write_map(tmp_path, "backbone.")
        model_dir = write_checkpoint(tmp_path, "backbone.")
        out = tmp_path / "recipe.json"

        result = plan(map_path, model_dir, out, 4_000_000, "--pin", "*=16")

        assert result.exit_code == 0, result.output
        reserved = load_recipe(out).plan.predicted_total_bytes
        assert 1.9 <= reserved / MODEL_BYTES <= 2.1

    def test_a_budget_below_the_doubled_floor_refuses_naming_only_held_groups(
        self, tmp_path
    ) -> None:
        map_path = write_map(tmp_path, "backbone.")
        model_dir = write_checkpoint(tmp_path, "backbone.")
        out = tmp_path / "recipe.json"

        result = plan(map_path, model_dir, out, 500_000)

        assert result.exit_code == 1
        refusal = next(
            line for line in result.output.splitlines() if line.startswith("error:")
        )
        assert "no recipe fits" in refusal
        assert (
            f"The checkpoint holds {LAYERS} groups the map does not measure" in refusal
        )
        assert "Scan them to spend it" in refusal
        # The refusal counts the held groups and requests a scan of
        # them. It names neither the root nor the coverage mismatch
        # that made them held, so the reader cannot reach #564 from it.
        assert "root" not in refusal
        assert "does not carry" not in refusal
        assert not out.exists()


class TestModelRootedMap:
    """The llama-family root, where the coverage match lines up."""

    def test_its_own_checkpoint_covers_every_group_and_holds_none(
        self, tmp_path
    ) -> None:
        map_path = write_map(tmp_path, "model.")
        model_dir = write_checkpoint(tmp_path, "model.")
        out = tmp_path / "recipe.json"

        result = plan(map_path, model_dir, out, 500_000)

        assert result.exit_code == 0, result.output
        assert "does not carry" not in result.stderr
        assert (
            f"checkpoint holds {LAYERS + 1} groups: {LAYERS + 1} measured by the "
            "map, 0 held at reference precision" in result.stdout
        )
        reserved = load_recipe(out).plan.predicted_total_bytes
        assert reserved < MODEL_BYTES
