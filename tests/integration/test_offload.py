"""Offload-aware scanning at dispatch scale (ADR-0015).

The hermetic stub tier lives in ``tests/unit/adapters/test_scan_offload.py``.
This GPU tier loads the offload-scale synthetic checkpoint under a
cap, engaging real accelerate dispatch — the regression the PR #13
review demanded: a capped model goes from "refuses" to "measures
correctly".
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from typer.testing import CliRunner

from tests.conftest import CALIBRATION_TEXT, OFFLOAD_GPU_CAP
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map

pytestmark = [pytest.mark.integration, pytest.mark.slow]

runner = CliRunner()


@pytest.fixture(scope="module")
def offload_meter(offload_model_dir, tmp_path_factory):
    """A meter over the offload-scale checkpoint, capped to force dispatch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

    calibration = tmp_path_factory.mktemp("offload-calib") / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    return TorchDamageMeter(
        str(offload_model_dir),
        calibration,
        max_tokens=256,
        device="auto",
        max_gpu_memory=OFFLOAD_GPU_CAP,
    )


def _offloaded_group(meter) -> str:
    return next(
        name
        for name, members in meter._groups.items()
        if any(m in meter._offloaded for m in members)
    )


def _fully_offloaded_group(meter) -> str:
    return next(
        name
        for name, members in meter._groups.items()
        if members and all(m in meter._offloaded for m in members)
    )


def _two_distinct_offloaded_groups(meter) -> set[str]:
    """The fully offloaded group plus a second offloaded group when one exists.

    The first ``any``-offloaded group can be the fully offloaded one,
    so a naive pair could collapse to a single element and lose the
    two-group coverage silently.
    """
    fully = _fully_offloaded_group(meter)
    second = next(
        (
            name
            for name, members in meter._groups.items()
            if name != fully and any(m in meter._offloaded for m in members)
        ),
        fully,
    )
    return {fully, second}


@pytest.mark.gpu
class TestOffloadedMeter:
    def test_cap_engages_dispatch_and_every_group_resolves(self, offload_meter) -> None:
        # If this fails the synthetic checkpoint no longer triggers
        # accelerate dispatch — grow it, do not skip: every other test
        # in this class silently loses its subject otherwise.
        assert offload_meter.offloaded_group_count > 0
        assert offload_meter._offloaded

    def test_offloaded_group_measures_nonzero_damage(self, offload_meter) -> None:
        damage = offload_meter.measure(_offloaded_group(offload_meter), 2)

        assert damage > 0.0

    def test_fully_offloaded_group_damage_scales_with_bits(self, offload_meter) -> None:
        # The PR #13 poison detector: if perturbing a group whose every
        # tensor lives in the weights map silently no-opped, 2-bit and
        # 8-bit damage would both sit at the fp16-cache noise floor
        # and roughly agree. Real perturbation separates them by
        # orders of magnitude.
        group = _fully_offloaded_group(offload_meter)

        coarse = offload_meter.measure(group, 2)
        fine = offload_meter.measure(group, 8)

        assert coarse > 5 * fine

    def test_offloaded_measurement_is_deterministic_and_restores(
        self, offload_meter
    ) -> None:
        group = _offloaded_group(offload_meter)

        first = offload_meter.measure(group, 2)
        second = offload_meter.measure(group, 2)

        assert first == second

    def test_capped_damage_matches_uncapped(
        self, offload_meter, offload_model_dir, tmp_path
    ) -> None:
        # The regression the PR #13 review demanded: pre-#13, an
        # offloaded group measured exactly zero. The capped meter must
        # agree with an all-GPU meter on the same cells — including a
        # group with no resident member to hide behind.
        groups = _two_distinct_offloaded_groups(offload_meter)
        capped = {group: offload_meter.measure(group, 2) for group in groups}
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        uncapped = TorchDamageMeter(
            str(offload_model_dir), calibration, max_tokens=256, device="cuda"
        )

        assert uncapped.offloaded_group_count == 0
        for group in groups:
            assert capped[group] > 0.0
            assert capped[group] == pytest.approx(uncapped.measure(group, 2), rel=0.15)

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
