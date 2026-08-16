from __future__ import annotations

from typing import Any

import pytest


def make_map(
    groups: list[tuple[str, int, dict[int, float]]],
    precisions: tuple[int, ...] = (8, 4, 3, 2),
    model_id: str = "test/model",
) -> dict[str, Any]:
    """Build a raw sensitivity-map dict for tests.

    Args:
        groups: Tuples of (name, bytes_fp16, {bits: damage}).
        precisions: Candidate precisions recorded in scan metadata.
        model_id: Model identifier recorded in the map.

    Returns:
        A dict accepted by map_from_dict.
    """
    return {
        "vramfit_schema": 3,
        "model_id": model_id,
        "scan": {
            "metric": "kl_divergence",
            "calibration": "wikitext",
            "calibration_tokens": 131072,
            "precisions": list(precisions),
            "group_by": "layer",
            "started_at": "2026-07-27T00:00:00Z",
        },
        "groups": [
            {
                "name": name,
                "tensors": [f"{name}.weight"],
                "bytes_fp16": bytes_fp16,
                "sensitivity": {str(bits): damage for bits, damage in curve.items()},
            }
            for name, bytes_fp16, curve in groups
        ],
    }


@pytest.fixture
def nemotron_like_map() -> dict[str, Any]:
    """A small heterogeneous map echoing the target model's structure."""
    return make_map(
        [
            ("model.layers.0.self_attn", 1600, {8: 0.001, 4: 0.050, 3: 0.30, 2: 0.90}),
            ("model.layers.0.mlp", 8000, {8: 0.000, 4: 0.004, 3: 0.02, 2: 0.10}),
            ("model.layers.1.mlp", 4000, {8: 0.000, 4: 0.002, 3: 0.05, 2: 0.04}),
            ("model.embed", 2000, {8: 0.002, 4: 0.080, 3: 0.40, 2: 1.50}),
        ]
    )


def make_recipe_dict() -> dict:
    return {
        "vramfit_schema": 6,
        "model_id": "test/model",
        "runtime": "llama.cpp",
        "plan": {
            "vram_budget_bytes": 100,
            "kv_headroom_bytes": 10,
            "weight_budget_bytes": 90,
            "predicted_total_bytes": 80,
            "predicted_damage": 0.5,
            "solver": "greedy-damage-per-byte",
            "pins": {"g*": 8},
            "protections": {},
            "imatrix_exclusions": [],
            "format_overhead": 0.05,
            "trace": [
                {
                    "step": 1,
                    "group": "g1",
                    "from_bits": 8,
                    "to_bits": 4,
                    "damage_delta": 0.1,
                    "bytes_freed": 20,
                    "ratio": 0.005,
                }
            ],
        },
        "assignments": [
            {"group": "g0", "bits": 8, "bytes": 40, "damage": 0.0},
            {"group": "g1", "bits": 4, "bytes": 40, "damage": 0.5},
        ],
        "protected_tensors": [],
    }
