from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantfit.adapters.outbound.recipe_json import recipe_from_dict, recipe_to_dict
from quantfit.adapters.outbound.sensitivity_map_json import (
    map_from_dict,
    map_to_dict,
)
from quantfit.domain.solver import group_bytes, solve
from tests.strategies import raw_sensitivity_maps


@pytest.mark.unit
class TestArtifactRoundTripProperties:
    @given(raw=raw_sensitivity_maps())
    def test_map_round_trip_is_identity(self, raw: dict[str, Any]) -> None:
        map_ = map_from_dict(raw)

        assert map_from_dict(map_to_dict(map_)) == map_

    @given(raw=raw_sensitivity_maps(), data=st.data())
    def test_solver_recipes_round_trip_through_json(self, raw, data) -> None:
        map_ = map_from_dict(raw)
        ceiling = sum(
            group_bytes(g.bytes_fp16, map_.scan.precisions[0], 0.05)
            for g in map_.groups
        )
        floor = sum(
            group_bytes(g.bytes_fp16, map_.scan.precisions[-1], 0.05)
            for g in map_.groups
        )
        budget = data.draw(st.integers(min_value=floor, max_value=ceiling))

        recipe = solve(
            map_,
            weight_budget_bytes=budget,
            vram_budget_bytes=budget + 1000,
            kv_headroom_bytes=1000,
        )

        assert recipe_from_dict(recipe_to_dict(recipe)) == recipe
