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

from tests.fakes import (
    MemoryModelShapeSource,
    MemoryRecipeSink,
    MemoryRunLog,
    MemoryScanCheckpointStore,
    MemorySensitivityMapSink,
    MemorySensitivityMapSource,
)
from tests.unit.conftest import make_map
from vramfit.adapters.outbound.hf_config import HfConfigFile
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.recipe_json import JsonRecipeFile, load_recipe
from vramfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile, read_run_log
from vramfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from vramfit.adapters.outbound.sensitivity_map_json import (
    JsonSensitivityMapFile,
    load_sensitivity_map,
    map_from_dict,
    save_sensitivity_map,
)
from vramfit.domain.budget import ModelShape
from vramfit.domain.model import Recipe, ScanMeta, SensitivityMap
from vramfit.domain.scan import Measurement, scan_fingerprint
from vramfit.domain.solver import solve
from vramfit.ports.outbound import (
    ModelShapeSource,
    RecipeSink,
    ScanCheckpointStore,
    SensitivityMapSink,
    SensitivityMapSource,
)


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
        protections={},
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
    def test_load_returns_the_configured_geometry(self, build, tmp_path) -> None:
        source: ModelShapeSource = build(tmp_path, SAMPLE_SHAPE)

        assert source.load() == SAMPLE_SHAPE

    def test_load_without_valid_config_raises_value_error(
        self, build, tmp_path
    ) -> None:
        source: ModelShapeSource = build(tmp_path, None)

        with pytest.raises(ValueError):
            source.load()


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
            source.load()


# --- SensitivityMapSink ----------------------------------------------------- #


def _real_map_sink(
    tmp_path: Path,
) -> tuple[SensitivityMapSink, Callable[[], SensitivityMap]]:
    path = tmp_path / "map.json"
    return JsonSensitivityMapFile(path), lambda: load_sensitivity_map(path)


def _fake_map_sink(
    tmp_path: Path,
) -> tuple[SensitivityMapSink, Callable[[], SensitivityMap]]:
    sink = MemorySensitivityMapSink()
    return sink, lambda: sink.last


@pytest.mark.contract
@pytest.mark.parametrize(
    "build", [_real_map_sink, _fake_map_sink], ids=["real-json", "fake-memory"]
)
class TestSensitivityMapSinkContract:
    def test_saved_map_reads_back_equal(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)
        map_ = sample_map()

        sink.save(map_)

        assert readback() == map_

    def test_second_save_wins(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)
        first = sample_map()
        second = map_from_dict(
            make_map([("solo", 4000, {8: 0.02, 4: 0.2, 3: 0.4, 2: 2.0})])
        )

        sink.save(first)
        sink.save(second)

        assert readback() == second


# --- ScanCheckpointStore ---------------------------------------------------- #

# The real fingerprint shape the CLI persists, method token included.
FINGERPRINT = scan_fingerprint(
    "test/model",
    ScanMeta(
        metric="kl_divergence",
        calibration="calib.txt",
        calibration_tokens=1024,
        precisions=(8, 4),
        group_by="layer",
        started_at="unused",
    ),
)


def _real_checkpoint_store(tmp_path: Path) -> ScanCheckpointStore:
    return JsonScanCheckpointFile(tmp_path / "scan.checkpoint.json")


def _fake_checkpoint_store(tmp_path: Path) -> ScanCheckpointStore:
    return MemoryScanCheckpointStore()


@pytest.mark.contract
@pytest.mark.parametrize(
    "build",
    [_real_checkpoint_store, _fake_checkpoint_store],
    ids=["real-json", "fake-memory"],
)
class TestScanCheckpointStoreContract:
    def test_fresh_store_loads_empty(self, build, tmp_path) -> None:
        store: ScanCheckpointStore = build(tmp_path)

        assert store.load(FINGERPRINT) == ()

    def test_appended_measurements_read_back_in_order(self, build, tmp_path) -> None:
        store: ScanCheckpointStore = build(tmp_path)
        first = Measurement(group="g0", bits=8, damage=0.001)
        second = Measurement(group="g0", bits=4, damage=0.02)

        store.append(FINGERPRINT, first)
        store.append(FINGERPRINT, second)

        assert store.load(FINGERPRINT) == (first, second)

    def test_load_with_different_fingerprint_raises_value_error(
        self, build, tmp_path
    ) -> None:
        store: ScanCheckpointStore = build(tmp_path)
        store.append(FINGERPRINT, Measurement(group="g0", bits=8, damage=0.001))

        with pytest.raises(ValueError, match="different scan"):
            store.load("another|scan")

    def test_append_with_different_fingerprint_raises_value_error(
        self, build, tmp_path
    ) -> None:
        store: ScanCheckpointStore = build(tmp_path)
        store.append(FINGERPRINT, Measurement(group="g0", bits=8, damage=0.001))

        with pytest.raises(ValueError, match="different scan"):
            store.append("another|scan", Measurement(group="g0", bits=4, damage=0.1))

    def test_load_is_repeatable(self, build, tmp_path) -> None:
        store: ScanCheckpointStore = build(tmp_path)
        store.append(FINGERPRINT, Measurement(group="g0", bits=8, damage=0.001))

        assert store.load(FINGERPRINT) == store.load(FINGERPRINT)


# --- RunLogSink ------------------------------------------------------------- #


def _real_run_log(tmp_path: Path):
    path = tmp_path / "scan.runlog.jsonl"
    sink = JsonlRunLogFile(path)
    return sink, lambda: [
        (
            e["event"],
            {k: v for k, v in e.items() if k not in ("event", "ts", "vramfit_runlog")},
        )
        for e in read_run_log(path)
    ]


def _fake_run_log(tmp_path: Path):
    sink = MemoryRunLog()
    return sink, lambda: [(event, dict(fields)) for event, fields in sink.events]


@pytest.mark.contract
@pytest.mark.parametrize(
    "build", [_real_run_log, _fake_run_log], ids=["real-jsonl", "fake-memory"]
)
class TestRunLogSinkContract:
    def test_events_read_back_in_emit_order(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)

        sink.emit("scan_started", {"model": "test/model"})
        sink.emit("cell_measured", {"group": "g0", "bits": 4, "damage": 0.01})

        assert readback() == [
            ("scan_started", {"model": "test/model"}),
            ("cell_measured", {"group": "g0", "bits": 4, "damage": 0.01}),
        ]

    def test_empty_fields_are_allowed(self, build, tmp_path) -> None:
        sink, readback = build(tmp_path)

        sink.emit("scan_finished", {})

        assert readback() == [("scan_finished", {})]
