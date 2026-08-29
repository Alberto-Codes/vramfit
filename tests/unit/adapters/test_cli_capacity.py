"""The ``vramfit capacity`` command: the budget ledger run in reverse.

Drives the command over saved recipe artifacts and manual or config
shapes, so the ledger lines, the inverse readouts, and the refusals
are the unit under test (#422).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.recipe_json import save_recipe
from vramfit.domain.model import Assignment, PlanMeta, Recipe

runner = CliRunner()

pytestmark = pytest.mark.unit

# The manual shape below: 2 layers x 2 heads x 4 wide x 2 tensors x
# 2 bytes = 64 bytes per token at fp16.
SHAPE_OPTIONS = ["--attn-layers", "2", "--kv-heads", "2", "--head-dim", "4"]


def save_capacity_recipe(
    tmp_path: Path,
    vram_budget_bytes: int = 24 * 2**30,
    predicted_total_bytes: int = 15 * 2**30,
) -> Path:
    recipe = Recipe(
        model_id="test-model",
        plan=PlanMeta(
            vram_budget_bytes=vram_budget_bytes,
            kv_headroom_bytes=4 * 2**30,
            weight_budget_bytes=vram_budget_bytes - 4 * 2**30,
            predicted_total_bytes=predicted_total_bytes,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.layers.0", bits=4, bytes=1_000, damage=0.01),
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )
    path = tmp_path / "recipe.json"
    save_recipe(recipe, path)
    return path


def test_capacity_reports_the_ledger_and_max_context(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--overhead",
            "3GiB",
            *SHAPE_OPTIONS,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "attention layers      2  (KV grows 64 bytes/token, fp16)" in result.output
    assert "VRAM total            24.00 GiB" in result.output
    assert "- weights (recipe)    15.00 GiB" in result.output
    assert "- runtime overhead    3.00 GiB" in result.output
    assert "= KV headroom         6.00 GiB" in result.output
    # 6 GiB over 64 bytes/token, floored.
    assert f"max context           {6 * 2**30 // 64} tokens  (1 sequence)" in (
        result.output
    )


def test_capacity_defaults_vram_to_the_recipe_record(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(app, ["capacity", str(recipe), *SHAPE_OPTIONS])

    assert result.exit_code == 0, result.output
    assert "VRAM total            24.00 GiB" in result.output


def test_capacity_negative_headroom_errors_after_the_ledger(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app, ["capacity", str(recipe), "--vram", "16GiB", *SHAPE_OPTIONS]
    )

    assert result.exit_code == 1, result.output
    assert "= KV headroom         -1.00 GiB" in result.output
    assert "error: the recipe leaves nothing for KV cache" in result.output


def test_capacity_sequences_split_the_context_reading(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--sequences",
            "2",
            *SHAPE_OPTIONS,
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"max context           {7 * 2**30 // 128} tokens  (2 sequences)" in (
        result.output
    )


def test_capacity_context_option_adds_the_sequence_line(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--context",
            "16384",
            *SHAPE_OPTIONS,
        ],
    )

    assert result.exit_code == 0, result.output
    # 7 GiB over 64 x 16384 bytes per sequence, floored.
    per_sequence = 64 * 16384
    assert (
        f"max sequences         {7 * 2**30 // per_sequence}  (at 16384 tokens)"
        in result.output
    )


def test_capacity_tokens_per_image_adds_the_image_line(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--tokens-per-image",
            "256",
            *SHAPE_OPTIONS,
        ],
    )

    assert result.exit_code == 0, result.output
    images = 7 * 2**30 // 64 // 256
    assert (
        f"image capacity        {images} images  (256 tokens per image, 1 sequence)"
        in result.output
    )


def test_capacity_image_line_reads_per_the_sequences_split(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--sequences",
            "2",
            "--tokens-per-image",
            "256",
            *SHAPE_OPTIONS,
        ],
    )

    assert result.exit_code == 0, result.output
    images = 7 * 2**30 // 128 // 256
    assert (
        f"image capacity        {images} images  (256 tokens per image, 2 sequences)"
        in result.output
    )


def test_capacity_sliding_only_config_reports_unbounded(tmp_path: Path) -> None:
    # Every layer saturates at its window, so past the pool the
    # context is not KV-limited.
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "layer_types": ["sliding_attention"] * 4,
            "sliding_window": 512,
            "attention_k_eq_v": False,
            "num_kv_shared_layers": 0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--model-config",
            str(config_path),
            "--tokens-per-image",
            "256",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "max context           unbounded  (1 sequence)" in result.output
    assert (
        "image capacity        unbounded  (256 tokens per image, 1 sequence)"
        in result.output
    )


def test_capacity_missing_recipe_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["capacity", str(tmp_path / "absent.json"), *SHAPE_OPTIONS]
    )

    assert result.exit_code == 1
    assert "error:" in result.output


def test_capacity_invalid_recipe_errors(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text("{not json")

    result = runner.invoke(app, ["capacity", str(path), *SHAPE_OPTIONS])

    assert result.exit_code == 1
    assert "error:" in result.output


def test_capacity_both_shape_sources_rejected(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        ["capacity", str(recipe), "--model-config", "config.json", *SHAPE_OPTIONS],
    )

    assert result.exit_code == 2
    assert "not both" in result.output


def test_capacity_no_shape_source_rejected(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(app, ["capacity", str(recipe)])

    assert result.exit_code == 2
    assert "give --model-config" in result.output


def test_capacity_malformed_overhead_rejected_before_recipe_io(
    tmp_path: Path,
) -> None:
    # A usage error reports even when the recipe is also missing, as
    # in budget and plan.
    result = runner.invoke(
        app,
        [
            "capacity",
            str(tmp_path / "absent.json"),
            "--overhead",
            "many",
            *SHAPE_OPTIONS,
        ],
    )

    assert result.exit_code == 2
    assert "--overhead" in result.output


def test_capacity_unknown_dtype_rejected(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app, ["capacity", str(recipe), "--kv-dtype", "fp4", *SHAPE_OPTIONS]
    )

    assert result.exit_code == 2
    assert "unknown dtype" in result.output


def test_capacity_malformed_vram_rejected(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app, ["capacity", str(recipe), "--vram", "many", *SHAPE_OPTIONS]
    )

    assert result.exit_code == 2
    assert "--vram" in result.output


def write_composite_config(tmp_path: Path, claims_vision: bool) -> Path:
    # The same 64 bytes/token geometry as SHAPE_OPTIONS, declared by
    # a composite config so the card can claim vision.
    decoder = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_attention_heads": 4,
        "head_dim": 4,
    }
    config: dict = {"text_config": decoder}
    if claims_vision:
        config["vision_config"] = {"hidden_size": 1152}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_capacity_vision_line_subtracts_from_the_headroom(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)
    config = write_composite_config(tmp_path, claims_vision=True)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--overhead",
            "3GiB",
            "--vision-line",
            "1GiB",
            "--model-config",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "- vision line         1.00 GiB  (measured, ADR-0030)" in result.output
    assert "= KV headroom         5.00 GiB" in result.output
    # 5 GiB over 64 bytes/token, floored.
    assert f"max context           {5 * 2**30 // 64} tokens  (1 sequence)" in (
        result.output
    )


def test_capacity_vision_claim_without_a_line_states_the_gap(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)
    config = write_composite_config(tmp_path, claims_vision=True)

    result = runner.invoke(
        app, ["capacity", str(recipe), "--model-config", str(config)]
    )

    assert result.exit_code == 0, result.output
    assert (
        "vision                claimed — no --vision-line supplied, "
        "nothing subtracted" in result.output
    )


def test_capacity_no_vision_claim_states_the_absence(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)
    config = write_composite_config(tmp_path, claims_vision=False)

    result = runner.invoke(
        app, ["capacity", str(recipe), "--model-config", str(config)]
    )

    assert result.exit_code == 0, result.output
    assert "vision                none claimed — nothing subtracted" in result.output


def test_capacity_vision_line_without_a_claim_states_it_does_not_apply(
    tmp_path: Path,
) -> None:
    # A card that claims no vision subtracts nothing and states the
    # absence (ADR-0030 decision 3) — the supplied option draws a
    # note, and the headroom keeps all three original terms.
    recipe = save_capacity_recipe(tmp_path)
    config = write_composite_config(tmp_path, claims_vision=False)

    result = runner.invoke(
        app,
        [
            "capacity",
            str(recipe),
            "--vram",
            "24GiB",
            "--overhead",
            "3GiB",
            "--vision-line",
            "1GiB",
            "--model-config",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--vision-line does not apply" in result.output
    assert "= KV headroom         6.00 GiB" in result.output


def test_capacity_vision_line_with_a_manual_shape_rejected(tmp_path: Path) -> None:
    recipe = save_capacity_recipe(tmp_path)

    result = runner.invoke(
        app,
        ["capacity", str(recipe), "--vision-line", "1GiB", *SHAPE_OPTIONS],
    )

    assert result.exit_code == 2
    assert "needs --model-config" in result.output
