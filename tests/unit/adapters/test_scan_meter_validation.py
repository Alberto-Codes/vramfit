"""The meter's input-validation order.

The contract suite pins group names before bits. This suite pins the
next edge: bits validate before shard planning, so a bad input never
spends shard I/O (`reader.verify` opens every shard) before it
refuses. The meter is assembled without a model load, the
`test_scan_meter_summaries` pattern. Skips cleanly where the scan
extra is absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
pytest.importorskip("transformers", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

pytestmark = pytest.mark.unit

UP = "model.layers.0.mixer.experts.up_proj"
DENSE = "model.layers.1.mixer.out_proj.weight"


def build_meter() -> TorchDamageMeter:
    # Two offloaded groups against a hub-style model id: shard
    # planning would refuse with the "not a local safetensors
    # directory" message, so a bits error proves bits checked first.
    meter = TorchDamageMeter.__new__(TorchDamageMeter)
    meter._poisoned = False
    meter._groups = {"model.layers.0": [UP], "model.layers.1": [DENSE]}
    meter._offloaded = {
        UP: torch.zeros(2, 4, 8),
        DENSE: torch.zeros(4, 8),
    }
    meter.model_id = "hub/never-local"
    meter._shards = None
    return meter


class TestValidationOrder:
    def test_measure_recipe_checks_bits_before_shard_planning(self) -> None:
        with pytest.raises(ValueError, match="bits must be at least"):
            build_meter().measure_recipe({"model.layers.0": 1, "model.layers.1": 1})

    def test_measure_recipe_still_plans_shards_for_valid_bits(self) -> None:
        with pytest.raises(ValueError, match="not a local safetensors"):
            build_meter().measure_recipe({"model.layers.0": 4, "model.layers.1": 4})
