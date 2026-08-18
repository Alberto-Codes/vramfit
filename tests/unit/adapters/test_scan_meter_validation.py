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

from pathlib import Path
from typing import cast

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
pytest.importorskip("transformers", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.meter import TorchDamageMeter
from vramfit.adapters.outbound.scan.within_group import WithinGroupMethod

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


class TestWithinGroupDispatch:
    """The meter routes each method and names what refused.

    A scan spends hours over hundreds of cells, so a refusal must
    name the tensor that stopped it, not the type alone.
    """

    def _meter(self, method: WithinGroupMethod) -> TorchDamageMeter:
        meter = build_meter()
        meter._within_group = method
        meter._imatrix_weights = {}
        meter._block_size = 32
        return meter

    def test_q0_prices_a_routed_expert_row(self) -> None:
        # 2688 refuses every 256-element super-block type. The q0
        # method's blocks of 64 and 32 both divide it (#159).
        torch.manual_seed(0)
        param = torch.randn(2, 2688)

        result = self._meter("q0")._quantize_dequantize(param, 2, UP)

        assert result.shape == param.shape

    def test_kquant_refusal_names_the_parameter(self) -> None:
        param = torch.randn(2, 2688)

        with pytest.raises(ValueError, match="does not divide the row length") as exc:
            self._meter("kquant")._quantize_dequantize(param, 2, UP)

        assert str(exc.value).startswith(f"{UP}: ")

    def test_an_unknown_method_refuses_before_the_model_loads(self) -> None:
        # The guard is the first statement in __init__, so neither
        # the calibration file nor the model is ever read. A silent
        # RTN fallback under a mistyped method would corrupt every
        # damage the meter measures.
        unknown = cast(WithinGroupMethod, "gptq")

        with pytest.raises(ValueError, match="within_group must be one of"):
            TorchDamageMeter(
                "hub/never-local",
                Path("never-read.txt"),
                max_tokens=8,
                within_group=unknown,
            )
