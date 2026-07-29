"""Offload-aware scanning (ADR-0015): stubs, shards, and dispatch scale.

The stub tier pins the weights-map resolution behavior without
accelerate. The shard tier exercises the safetensors restore path on
tmp files. The GPU tier loads the offload-scale synthetic checkpoint
under a cap, engaging real accelerate dispatch — the regression the
PR #13 review demanded: a capped model goes from "refuses" to
"measures correctly".
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from typer.testing import CliRunner

from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.scan.offload import (
    open_shard_reader,
    resolve_offloaded_params,
)
from quantfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map
from tests.conftest import CALIBRATION_TEXT, OFFLOAD_GPU_CAP

pytestmark = [pytest.mark.integration, pytest.mark.slow]

runner = CliRunner()


class _Hook:
    def __init__(self, weights_map) -> None:
        self.weights_map = weights_map


class _CloningMap(dict):
    """A weights map whose reads return fresh copies — unstable storage."""

    def __getitem__(self, key):
        return super().__getitem__(key).clone()


def _ghost_model(weights_map) -> torch.nn.Module:
    """A stub model with one real module and one offloaded ("ghost") module."""
    model = torch.nn.Module()
    model.resident = torch.nn.Linear(4, 4)
    model.ghost = torch.nn.Module()
    model.ghost.weight = torch.nn.Parameter(torch.empty(4, 4, device="meta"))
    if weights_map is not None:
        model.ghost._hf_hook = _Hook(weights_map)
    return model


GHOST_GROUPS = {"resident": ["resident.weight"], "ghost": ["ghost.weight"]}


class TestResolveOffloadedParams:
    def test_model_without_meta_params_resolves_to_nothing(self) -> None:
        model = torch.nn.Module()
        model.resident = torch.nn.Linear(4, 4)

        backing = resolve_offloaded_params(model, {"resident": ["resident.weight"]})

        assert backing == {}

    def test_meta_param_with_stable_weights_map_resolves(self) -> None:
        tensor = torch.randn(4, 4)
        model = _ghost_model({"ghost.weight": tensor})

        backing = resolve_offloaded_params(model, GHOST_GROUPS)

        assert backing == {"ghost.weight": tensor}
        assert backing["ghost.weight"].data_ptr() == tensor.data_ptr()

    def test_meta_param_without_hook_is_refused(self) -> None:
        model = _ghost_model(None)

        with pytest.raises(ValueError, match=r"offloaded beyond host RAM.*ghost"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_unstable_weights_map_storage_is_refused(self) -> None:
        model = _ghost_model(_CloningMap({"ghost.weight": torch.randn(4, 4)}))

        with pytest.raises(ValueError, match="offloaded beyond host RAM"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_wrong_shape_backing_tensor_is_refused(self) -> None:
        model = _ghost_model({"ghost.weight": torch.randn(2, 2)})

        with pytest.raises(ValueError, match="offloaded beyond host RAM"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_missing_weights_map_entry_is_refused(self) -> None:
        model = _ghost_model({"other.weight": torch.randn(4, 4)})

        with pytest.raises(ValueError, match="offloaded beyond host RAM"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_meta_backing_tensor_is_refused(self) -> None:
        ghost = torch.empty(4, 4, device="meta")
        model = _ghost_model({"ghost.weight": ghost})

        with pytest.raises(ValueError, match="offloaded beyond host RAM"):
            resolve_offloaded_params(model, GHOST_GROUPS)


class TestShardReader:
    def test_open_on_single_file_layout_finds_every_tensor(
        self, tiny_model_dir
    ) -> None:
        reader = open_shard_reader(str(tiny_model_dir))

        assert reader is not None
        assert reader.verify({"model.embed_tokens.weight": (512, 32)}) is None

    def test_open_on_indexed_layout_maps_tensors_to_shards(self, tmp_path) -> None:
        from safetensors.torch import save_file

        save_file({"a": torch.randn(2, 2)}, tmp_path / "m-00001.safetensors")
        save_file({"b": torch.randn(3, 3)}, tmp_path / "m-00002.safetensors")
        index = {"weight_map": {"a": "m-00001.safetensors", "b": "m-00002.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        reader = open_shard_reader(str(tmp_path))

        assert reader is not None
        assert reader.verify({"a": (2, 2), "b": (3, 3)}) is None

    def test_open_on_a_hub_id_returns_none(self) -> None:
        assert open_shard_reader("org/model-that-is-not-a-path") is None

    def test_open_on_a_directory_without_safetensors_returns_none(
        self, tmp_path
    ) -> None:
        assert open_shard_reader(str(tmp_path)) is None

    def test_verify_names_a_missing_tensor(self, tmp_path) -> None:
        from safetensors.torch import save_file

        save_file({"a": torch.randn(2, 2)}, tmp_path / "model.safetensors")
        reader = open_shard_reader(str(tmp_path))

        assert reader is not None
        problem = reader.verify({"gone": (2, 2)})
        assert problem is not None
        assert "gone" in problem

    def test_verify_names_a_shape_mismatch(self, tmp_path) -> None:
        from safetensors.torch import save_file

        save_file({"a": torch.randn(2, 2)}, tmp_path / "model.safetensors")
        reader = open_shard_reader(str(tmp_path))

        assert reader is not None
        problem = reader.verify({"a": (4, 4)})
        assert problem is not None
        assert "shape" in problem

    def test_read_into_restores_a_mutated_tensor(self, tmp_path) -> None:
        from safetensors.torch import save_file

        original = torch.randn(4, 4)
        save_file({"a": original}, tmp_path / "model.safetensors")
        reader = open_shard_reader(str(tmp_path))
        assert reader is not None
        live = original.clone()
        live.zero_()

        reader.read_into({"a": live})

        assert torch.equal(live, original)

    def test_read_into_casts_to_the_live_dtype(self, tmp_path) -> None:
        from safetensors.torch import save_file

        original = torch.randn(4, 4)
        save_file({"a": original}, tmp_path / "model.safetensors")
        reader = open_shard_reader(str(tmp_path))
        assert reader is not None
        live = torch.zeros(4, 4, dtype=torch.bfloat16)

        reader.read_into({"a": live})

        assert live.dtype == torch.bfloat16
        assert torch.equal(live, original.to(torch.bfloat16))


@pytest.fixture(scope="module")
def offload_meter(offload_model_dir, tmp_path_factory):
    """A meter over the offload-scale checkpoint, capped to force dispatch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from quantfit.adapters.outbound.scan.meter import TorchDamageMeter

    calibration = tmp_path_factory.mktemp("offload-calib") / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    return TorchDamageMeter(
        str(offload_model_dir),
        calibration,
        max_tokens=256,
        device="auto",
        max_gpu_memory=OFFLOAD_GPU_CAP,
    )


@pytest.mark.gpu
class TestOffloadedMeter:
    def test_cap_engages_dispatch_and_every_group_resolves(self, offload_meter) -> None:
        # If this fails the synthetic checkpoint no longer triggers
        # accelerate dispatch — grow it, do not skip: every other test
        # in this class silently loses its subject otherwise.
        assert offload_meter.offloaded_group_count > 0
        assert offload_meter._offloaded

    def test_offloaded_group_measures_nonzero_damage(self, offload_meter) -> None:
        group = next(
            name
            for name, members in offload_meter._groups.items()
            if any(m in offload_meter._offloaded for m in members)
        )

        damage = offload_meter.measure(group, 2)

        assert damage > 0.0

    def test_offloaded_measurement_is_deterministic_and_restores(
        self, offload_meter
    ) -> None:
        group = next(
            name
            for name, members in offload_meter._groups.items()
            if any(m in offload_meter._offloaded for m in members)
        )

        first = offload_meter.measure(group, 2)
        second = offload_meter.measure(group, 2)

        assert first == second

    def test_capped_damage_matches_uncapped(
        self, offload_meter, offload_model_dir, tmp_path
    ) -> None:
        # The regression the PR #13 review demanded: pre-#13, an
        # offloaded group measured exactly zero. The capped meter must
        # agree with an all-GPU meter on the same cell.
        group = next(
            name
            for name, members in offload_meter._groups.items()
            if any(m in offload_meter._offloaded for m in members)
        )
        capped = offload_meter.measure(group, 2)
        from quantfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        uncapped = TorchDamageMeter(
            str(offload_model_dir), calibration, max_tokens=256, device="cuda"
        )

        assert uncapped.offloaded_group_count == 0
        assert capped == pytest.approx(uncapped.measure(group, 2), rel=0.15)
        assert capped > 0.0

    def test_measure_recipe_restores_offloaded_originals_from_shards(
        self, offload_meter
    ) -> None:
        group = offload_meter.groups()[0].name
        before = offload_meter.measure(group, 2)
        recipe = {spec.name: 4 for spec in offload_meter.groups()}

        damage = offload_meter.measure_recipe(recipe)

        assert damage >= 0.0
        assert offload_meter._shards is not None
        assert offload_meter.measure(group, 2) == before

    def test_measure_recipe_without_local_shards_refuses_before_perturbing(
        self, offload_meter, monkeypatch
    ) -> None:
        group = offload_meter.groups()[0].name
        before = offload_meter.measure(group, 2)
        monkeypatch.setattr(offload_meter, "model_id", "org/hub-only-model")
        monkeypatch.setattr(offload_meter, "_shards", None)
        recipe = {spec.name: 4 for spec in offload_meter.groups()}

        with pytest.raises(ValueError, match="local safetensors directory"):
            offload_meter.measure_recipe(recipe)

        assert offload_meter.measure(group, 2) == before


@pytest.mark.gpu
def test_scan_cli_measures_a_capped_model(offload_model_dir, tmp_path) -> None:
    # Before ADR-0015 this exact invocation halted at meter build with
    # "groups were offloaded".
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    calibration = tmp_path / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    out = tmp_path / "sensitivity.json"

    result = runner.invoke(
        app,
        [
            "scan",
            str(offload_model_dir),
            "--calibration",
            str(calibration),
            "--out",
            str(out),
            "--precisions",
            "8,2",
            "--max-tokens",
            "128",
            "--device",
            "auto",
            "--gpu-memory",
            "120MiB",
        ],
    )

    assert result.exit_code == 0, result.output
    map_ = load_sensitivity_map(out)
    assert all(group.sensitivity[2] > 0.0 for group in map_.groups)
    runlog = json.loads(
        next(
            line
            for line in (tmp_path / "sensitivity.runlog.jsonl").read_text().splitlines()
            if '"meter_built"' in line
        )
    )
    assert runlog["offloaded_groups"] > 0
