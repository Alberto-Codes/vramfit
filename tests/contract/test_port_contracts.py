"""Verified-fake contract suites (ADR-0009).

Each port's suite runs against BOTH the real adapter and its in-memory
fake, proving identical Protocol behavior — so unit tests that rely on a
fake are relying on something proven faithful.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from quantfit.adapters.outbound.hf_config import HfConfigFile
from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.recipe_json import JsonRecipeFile, load_recipe
from quantfit.adapters.outbound.sensitivity_map_json import (
    JsonSensitivityMapFile,
    map_from_dict,
    save_sensitivity_map,
)
from quantfit.domain.budget import ModelShape
from quantfit.domain.model import Recipe, SensitivityMap
from quantfit.domain.solver import solve
from quantfit.ports.outbound import (
    ModelShapeSource,
    RecipeSink,
    SensitivityMapSource,
)
from tests.fakes import (
    MemoryModelShapeSource,
    MemoryRecipeSink,
    MemorySensitivityMapSource,
)
from tests.unit.conftest import make_map


def sample_map() -> SensitivityMap:
    return map_from_dict(
        make_map(
            [
                ("g0", 1600, {8: 0.001, 4: 0.01, 3: 0.1, 2: 1.0}),
                ("g1", 8000, {8: 0.0, 4: 0.004, 3: 0.02, 2: 0.1}),
            ]
        )
    )


def sample_recipe() -> Recipe:
    return solve(
        sample_map(),
        weight_budget_bytes=5000,
        vram_budget_bytes=6000,
        kv_headroom_bytes=1000,
        pins={"g0": 8},
    )


# --- SensitivityMapSource -------------------------------------------------- #


def _real_map_source(tmp_path: Path, map_: SensitivityMap | None):
    path = tmp_path / "map.json"
    if map_ is None:
        path.write_text("{}")
    else:
        save_sensitivity_map(map_, path)
    return JsonSensitivityMapFile(path)


def _fake_map_source(tmp_path: Path, map_: SensitivityMap | None):
    return MemorySensitivityMapSource(map_)


@pytest.mark.contract
@pytest.mark.parametrize(
    "build", [_real_map_source, _fake_map_source], ids=["real-json", "fake-memory"]
)
class TestSensitivityMapSourceContract:
    def test_load_returns_the_configured_map(self, build, tmp_path) -> None:
        expected = sample_map()
        source: SensitivityMapSource = build(tmp_path, expected)

        assert source.load() == expected

    def test_load_without_valid_map_raises_artifact_error(
        self, build, tmp_path
    ) -> None:
        source: SensitivityMapSource = build(tmp_path, None)

        with pytest.raises(ArtifactError):
            source.load()

    def test_load_is_repeatable(self, build, tmp_path) -> None:
        source: SensitivityMapSource = build(tmp_path, sample_map())

        assert source.load() == source.load()


@pytest.mark.contract
@pytest.mark.parametrize(
    "build",
    [
        lambda tmp: JsonSensitivityMapFile(tmp / "absent.json"),
        lambda tmp: MemorySensitivityMapSource(None),
    ],
    ids=["real-json", "fake-memory"],
)
class TestSensitivityMapSourceMissingBackingContract:
    def test_missing_backing_raises_artifact_error(self, build, tmp_path) -> None:
        source: SensitivityMapSource = build(tmp_path)

        with pytest.raises(ArtifactError):
            source.load()


# --- RecipeSink ------------------------------------------------------------ #


def _real_recipe_sink(tmp_path: Path) -> tuple[RecipeSink, Callable[[], Recipe]]:
    path = tmp_path / "recipe.json"
    sink = JsonRecipeFile(path)
    return sink, lambda: load_recipe(path)


def _fake_recipe_sink(tmp_path: Path) -> tuple[RecipeSink, Callable[[], Recipe]]:
    sink = MemoryRecipeSink()
    return sink, lambda: sink.last


@pytest.mark.contract
@pytest.mark.parametrize(
    "build", [_real_recipe_sink, _fake_recipe_sink], ids=["real-json", "fake-memory"]
)
class TestRecipeSinkContract:
    def test_saved_recipe_reads_back_equal(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)
        recipe = sample_recipe()

        sink.save(recipe)

        assert readback() == recipe

    def test_second_save_wins(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)
        first = sample_recipe()
        second = solve(
            sample_map(),
            weight_budget_bytes=9000,
            vram_budget_bytes=10000,
            kv_headroom_bytes=1000,
        )

        sink.save(first)
        sink.save(second)

        assert readback() == second


# --- ModelShapeSource ------------------------------------------------------ #

SAMPLE_SHAPE = ModelShape.uniform(attn_layers=4, kv_heads=8, head_dim=128)

_SAMPLE_CONFIG: dict[str, Any] = {
    "num_hidden_layers": 4,
    "num_key_value_heads": 8,
    "num_attention_heads": 32,
    "hidden_size": 4096,
}


def _real_shape_source(tmp_path: Path, shape: ModelShape | None):
    path = tmp_path / "config.json"
    if shape is None:
        path.write_text("[]")
    else:
        path.write_text(json.dumps(_SAMPLE_CONFIG))
    return HfConfigFile(path)


def _fake_shape_source(tmp_path: Path, shape: ModelShape | None):
    return MemoryModelShapeSource(shape)


@pytest.mark.contract
@pytest.mark.parametrize(
    "build", [_real_shape_source, _fake_shape_source], ids=["real-json", "fake-memory"]
)
class TestModelShapeSourceContract:
    def test_shape_returns_the_configured_geometry(self, build, tmp_path) -> None:
        source: ModelShapeSource = build(tmp_path, SAMPLE_SHAPE)

        assert source.shape() == SAMPLE_SHAPE

    def test_shape_without_valid_config_raises_value_error(
        self, build, tmp_path
    ) -> None:
        source: ModelShapeSource = build(tmp_path, None)

        with pytest.raises(ValueError):
            source.shape()


@pytest.mark.contract
@pytest.mark.parametrize(
    "build",
    [
        lambda tmp: HfConfigFile(tmp / "absent.json"),
        lambda tmp: MemoryModelShapeSource(None),
    ],
    ids=["real-json", "fake-memory"],
)
class TestModelShapeSourceMissingBackingContract:
    def test_missing_backing_raises_os_or_value_error(self, build, tmp_path) -> None:
        source: ModelShapeSource = build(tmp_path)

        with pytest.raises((OSError, ValueError)):
            source.shape()
