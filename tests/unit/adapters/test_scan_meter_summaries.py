"""The meter's count-summary wiring (ADR-0026 decision 4).

The pieces around the wiring have their own suites — the selection
rule, the reduction, the `GroupSpec` ride-through, and the JSON round
trip. These checks pin the meter's connective code: the per-group
pooling and the `groups()` handoff. They import the meter module
without loading a model, and skip cleanly where the scan extra is
absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
pytest.importorskip("transformers", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.discovery import group_count_summaries
from vramfit.adapters.outbound.scan.meter import TorchDamageMeter
from vramfit.domain.model import ImatrixCountSummary

pytestmark = pytest.mark.unit

UP = "model.layers.0.mixer.experts.up_proj"
DOWN = "model.layers.0.mixer.experts.down_proj"
DENSE = "model.layers.0.mixer.out_proj.weight"


class TestGroupCountSummaries:
    def test_covered_group_pools_its_stacks_and_nothing_else(self) -> None:
        counts = {UP: (3, 9), DOWN: (5, 7), DENSE: 421_370}

        summaries = group_count_summaries(counts, {"model.layers.0": [UP, DOWN, DENSE]})

        summary = summaries["model.layers.0"]
        assert (summary.min, summary.median, summary.max) == (3, 6.0, 9)

    def test_group_without_a_resolved_stack_records_nothing(self) -> None:
        summaries = group_count_summaries(
            {UP: (3, 9), DENSE: 421_370},
            {"model.layers.0": [UP, DOWN], "model.layers.1": [DENSE]},
        )

        assert summaries == {}

    def test_empty_resolution_records_nothing_and_never_raises(self) -> None:
        assert group_count_summaries({}, {"model.layers.0": [UP]}) == {}


class TestGroupsHandoff:
    def build_meter(self) -> TorchDamageMeter:
        # The wiring under test sits after the model load, so the
        # meter is assembled without one: groups(), _param, and the
        # summary store are the only state it reads.
        meter = TorchDamageMeter.__new__(TorchDamageMeter)
        meter._groups = {"model.layers.0": [UP], "model.layers.1": [DENSE]}
        meter._offloaded = {
            UP: torch.zeros(2, 4, 8),
            DENSE: torch.zeros(4, 8),
        }
        meter._imatrix_count_summaries = {
            "model.layers.0": ImatrixCountSummary(min=3, median=6.0, max=9)
        }
        return meter

    def test_groups_attach_each_summary_to_its_own_group(self) -> None:
        specs = {spec.name: spec for spec in self.build_meter().groups()}

        summary = specs["model.layers.0"].imatrix_counts
        assert summary is not None
        assert (summary.min, summary.median, summary.max) == (3, 6.0, 9)

    def test_groups_leave_a_summaryless_group_absent(self) -> None:
        specs = {spec.name: spec for spec in self.build_meter().groups()}

        assert specs["model.layers.1"].imatrix_counts is None
