from __future__ import annotations

from typing import Any

import pytest

from vramfit.domain.model import ImatrixCountSummary, ScanMeta
from vramfit.domain.scan import (
    GroupSpec,
    Measurement,
    assemble_map,
    group_key,
    matches_a_layer,
    plan_measurements,
    scan_fingerprint,
    summarize_imatrix_counts,
)

pytestmark = pytest.mark.unit


def make_meta(**overrides) -> ScanMeta:
    fields: dict[str, Any] = {
        "metric": "kl_divergence",
        "calibration": "calib.txt",
        "calibration_tokens": 1024,
        "precisions": (8, 4),
        "group_by": "layer",
        "started_at": "2026-07-28T00:00:00Z",
    }
    fields.update(overrides)
    return ScanMeta(**fields)


SPECS = (
    GroupSpec(name="g0", tensors=("g0.w",), bytes_fp16=1000),
    GroupSpec(name="g1", tensors=("g1.w", "g1.v"), bytes_fp16=2000),
)


class TestSpecAndMeasurementInvariants:
    def test_group_spec_with_zero_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="bytes_fp16"):
            GroupSpec(name="g", tensors=("w",), bytes_fp16=0)

    def test_group_spec_without_tensors_raises(self) -> None:
        with pytest.raises(ValueError, match="tensors"):
            GroupSpec(name="g", tensors=(), bytes_fp16=1)

    @pytest.mark.parametrize("bits", [0, -1], ids=["zero", "negative"])
    def test_measurement_with_nonpositive_bits_raises(self, bits: int) -> None:
        with pytest.raises(ValueError, match="bits"):
            Measurement(group="g", bits=bits, damage=0.0)

    @pytest.mark.parametrize(
        "damage", [-0.1, float("nan"), float("inf")], ids=["negative", "nan", "inf"]
    )
    def test_measurement_with_invalid_damage_raises(self, damage: float) -> None:
        with pytest.raises(ValueError, match="damage"):
            Measurement(group="g", bits=4, damage=damage)

    def test_measurement_with_zero_damage_is_valid(self) -> None:
        assert Measurement(group="g", bits=8, damage=0.0).damage == 0.0


class TestPlanMeasurements:
    def test_fresh_scan_covers_the_full_grid_in_order(self) -> None:
        todo = plan_measurements(SPECS, (8, 4))

        assert todo == (("g0", 8), ("g0", 4), ("g1", 8), ("g1", 4))

    def test_done_cells_are_skipped_without_reordering(self) -> None:
        todo = plan_measurements(SPECS, (8, 4), done=[("g0", 4), ("g0", 8)])

        assert todo == (("g1", 8), ("g1", 4))

    def test_complete_scan_leaves_nothing_to_do(self) -> None:
        grid = plan_measurements(SPECS, (8, 4))

        assert plan_measurements(SPECS, (8, 4), done=grid) == ()

    def test_done_cell_outside_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="different scan"):
            plan_measurements(SPECS, (8, 4), done=[("g9", 8)])

    def test_done_precision_outside_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="different scan"):
            plan_measurements(SPECS, (8, 4), done=[("g0", 3)])

    def test_duplicate_done_cell_raises(self) -> None:
        with pytest.raises(ValueError, match="appears twice"):
            plan_measurements(SPECS, (8, 4), done=[("g0", 8), ("g0", 8)])

    def test_duplicate_spec_names_raise(self) -> None:
        twins = (SPECS[0], SPECS[0])

        with pytest.raises(ValueError, match="unique"):
            plan_measurements(twins, (8, 4))

    def test_generator_precisions_cover_every_group(self) -> None:
        todo = plan_measurements(SPECS, iter((8, 4)))

        assert todo == (("g0", 8), ("g0", 4), ("g1", 8), ("g1", 4))


class TestScanFingerprint:
    def test_same_scan_yields_same_fingerprint(self) -> None:
        assert scan_fingerprint("m", make_meta()) == scan_fingerprint("m", make_meta())

    def test_started_at_does_not_change_the_fingerprint(self) -> None:
        earlier = make_meta(started_at="2026-07-28T00:00:00Z")
        later = make_meta(started_at="2026-07-28T09:00:00Z")

        assert scan_fingerprint("m", earlier) == scan_fingerprint("m", later)

    @pytest.mark.parametrize(
        "override",
        [
            {"metric": "perplexity_delta"},
            {"calibration": "other.txt"},
            {"calibration_tokens": 2048},
            {"precisions": (8, 4, 3, 2)},
            {"group_by": "tensor"},
            {"within_group": "kquant-ref"},
        ],
        ids=["metric", "calibration", "tokens", "precisions", "group-by", "method"],
    )
    def test_identity_field_changes_the_fingerprint(self, override: dict) -> None:
        assert scan_fingerprint("m", make_meta()) != scan_fingerprint(
            "m", make_meta(**override)
        )

    def test_model_id_changes_the_fingerprint(self) -> None:
        assert scan_fingerprint("a", make_meta()) != scan_fingerprint("b", make_meta())

    def test_method_changes_the_fingerprint(self) -> None:
        assert scan_fingerprint("m", make_meta()) != scan_fingerprint(
            "m", make_meta(within_group="awq-block32")
        )

    def test_fingerprint_format_is_pinned(self) -> None:
        # On-disk checkpoints key on this exact string — a format
        # drift silently invalidates every resumable scan in flight.
        # The trailing field is the imatrix path, empty when
        # unassisted (ADR-0020).
        expected = "m|kl_divergence|calib.txt|1024|layer|8,4|rtn-block32|"

        assert scan_fingerprint("m", make_meta()) == expected

    def test_assisted_fingerprint_carries_the_imatrix_path(self) -> None:
        meta = make_meta(within_group="kquant-imx", imatrix="im.gguf")

        fingerprint = scan_fingerprint("m", meta)

        assert fingerprint.endswith("|kquant-imx|im.gguf")

    def test_imatrix_path_changes_the_fingerprint(self) -> None:
        # Two assisted scans with different imatrix files must never
        # share a checkpoint (ADR-0020).
        left = make_meta(within_group="kquant-imx", imatrix="a.gguf")
        right = make_meta(within_group="kquant-imx", imatrix="b.gguf")

        assert scan_fingerprint("m", left) != scan_fingerprint("m", right)

    def test_empty_within_group_raises(self) -> None:
        with pytest.raises(ValueError, match="within_group"):
            make_meta(within_group="")

    def test_assisted_token_without_imatrix_raises(self) -> None:
        with pytest.raises(ValueError, match="imatrix"):
            make_meta(within_group="kquant-imx")

    def test_imatrix_without_the_assisted_token_raises(self) -> None:
        with pytest.raises(ValueError, match="kquant-imx"):
            make_meta(within_group="kquant-ref", imatrix="im.gguf")

    def test_empty_imatrix_raises(self) -> None:
        with pytest.raises(ValueError, match="imatrix"):
            make_meta(within_group="kquant-imx", imatrix="")

    def test_separator_in_field_values_cannot_collide(self) -> None:
        injected = scan_fingerprint("m|kl_divergence", make_meta(metric="x"))
        honest = scan_fingerprint("m", make_meta(metric="kl_divergence|x"))

        assert injected != honest

    def test_backslash_in_field_values_cannot_collide(self) -> None:
        left = scan_fingerprint("m\\", make_meta(metric="|x"))
        right = scan_fingerprint("m", make_meta(metric="\\|x"))

        assert left != right


def complete_measurements() -> list[Measurement]:
    return [
        Measurement(group="g0", bits=8, damage=0.001),
        Measurement(group="g0", bits=4, damage=0.01),
        Measurement(group="g1", bits=8, damage=0.0),
        Measurement(group="g1", bits=4, damage=0.2),
    ]


class TestAssembleMap:
    def test_complete_measurements_assemble_into_a_valid_map(self) -> None:
        map_ = assemble_map("m", make_meta(), SPECS, complete_measurements())

        assert map_.model_id == "m"
        assert [g.name for g in map_.groups] == ["g0", "g1"]
        assert map_.groups[0].sensitivity == {8: 0.001, 4: 0.01}
        assert map_.groups[1].tensors == ("g1.w", "g1.v")
        assert map_.groups[1].bytes_fp16 == 2000

    def test_measurement_order_does_not_matter(self) -> None:
        forward = assemble_map("m", make_meta(), SPECS, complete_measurements())
        backward = assemble_map(
            "m", make_meta(), SPECS, list(reversed(complete_measurements()))
        )

        assert forward == backward

    def test_missing_cell_raises_naming_the_group(self) -> None:
        with pytest.raises(ValueError, match=r'"g1".*missing'):
            assemble_map("m", make_meta(), SPECS, complete_measurements()[:3])

    def test_duplicate_cell_raises(self) -> None:
        measurements = [*complete_measurements(), Measurement("g0", 8, 0.5)]

        with pytest.raises(ValueError, match="duplicate"):
            assemble_map("m", make_meta(), SPECS, measurements)

    def test_unknown_group_raises(self) -> None:
        measurements = [*complete_measurements(), Measurement("g9", 8, 0.5)]

        with pytest.raises(ValueError, match='unknown group "g9"'):
            assemble_map("m", make_meta(), SPECS, measurements)


class TestGroupKey:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("model.layers.0.mlp.experts.7.up_proj.weight", "model.layers.0"),
            ("model.embed_tokens.weight", "model.embed_tokens"),
            ("transformer.h.3.attn.c_attn.weight", "transformer.h.3"),
        ],
        ids=["moe-member", "outside-any-layer", "gpt2-style"],
    )
    def test_layer_key_returns_the_decoder_prefix(self, name, expected) -> None:
        assert group_key(name, "layer") == expected

    def test_tensor_key_keeps_the_expert_index(self) -> None:
        name = "model.layers.0.mlp.experts.7.up_proj.weight"

        assert group_key(name, "tensor") == "model.layers.0.mlp.experts.7.up_proj"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (
                "model.layers.0.mlp.experts.7.up_proj.weight",
                "model.layers.0.mlp.experts.up_proj",
            ),
            (
                "backbone.layers.3.mixer.experts.57.down_proj.weight",
                "backbone.layers.3.mixer.experts.down_proj",
            ),
            (
                "model.layers.1.block_sparse_moe.experts.0.w1.weight",
                "model.layers.1.block_sparse_moe.experts.w1",
            ),
        ],
        ids=["qwen-style", "nemotron-style", "mixtral-style"],
    )
    def test_stack_key_removes_the_routed_expert_index(self, name, expected) -> None:
        assert group_key(name, "stack") == expected

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj.weight",
        ],
        ids=[
            "shared-expert",
            "shared-experts-plural",
            "router",
            "attention",
            "already-fused-3d",
        ],
    )
    def test_stack_key_leaves_unindexed_names_alone(self, name) -> None:
        # Only a routed-expert index collapses. A shared expert, a
        # router, and an already-fused 3D stack each stay their own
        # group, which is what the pack addresses (#159).
        assert group_key(name, "stack") == group_key(name, "tensor")

    def test_matches_a_layer_separates_layer_weights_from_edges(self) -> None:
        assert matches_a_layer("model.layers.0.mlp.up_proj.weight")
        assert not matches_a_layer("model.embed_tokens.weight")


class TestSummarizeImatrixCounts:
    def test_pooled_reduction_spans_all_vectors(self) -> None:
        summary = summarize_imatrix_counts([(3, 9), (5, 7)])

        assert (summary.min, summary.median, summary.max) == (3, 6.0, 9)

    def test_median_is_a_float_for_an_odd_pool(self) -> None:
        # statistics.median returns an int here — one field must not
        # write two JSON types (the PR #195 round-trip trap).
        summary = summarize_imatrix_counts([(1, 2, 3)])

        assert summary.median == 2.0
        assert isinstance(summary.median, float)

    def test_zero_counts_pool_like_any_other(self) -> None:
        # Decision 1 trusts any nonzero count, and the summary is
        # provenance — a zero stays visible as the minimum.
        summary = summarize_imatrix_counts([(0, 8)])

        assert summary.min == 0

    def test_empty_pool_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="no counts"):
            summarize_imatrix_counts([])

    def test_summary_rides_through_assemble_map(self) -> None:
        summary = ImatrixCountSummary(min=3, median=6.0, max=9)
        specs = (
            GroupSpec(
                name="g0", tensors=("g0.w",), bytes_fp16=1000, imatrix_counts=summary
            ),
            GroupSpec(name="g1", tensors=("g1.w",), bytes_fp16=2000),
        )
        measurements = [
            Measurement(group, bits, 0.1) for group in ("g0", "g1") for bits in (8, 4)
        ]

        map_ = assemble_map("m", make_meta(), specs, measurements)

        assert map_.groups[0].imatrix_counts == summary
        assert map_.groups[1].imatrix_counts is None


class TestImatrixCountSummaryInvariants:
    def test_ordered_summary_constructs(self) -> None:
        summary = ImatrixCountSummary(min=426, median=18114.0, max=192191)

        assert summary.min <= summary.median <= summary.max

    def test_integer_median_coerces_to_float(self) -> None:
        # An in-memory constructor must not reintroduce the
        # two-JSON-types defect the JSON writer guards against.
        summary = ImatrixCountSummary(min=1, median=2, max=3)

        assert summary.median == 2.0
        assert isinstance(summary.median, float)

    def test_unordered_summary_rejected(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            ImatrixCountSummary(min=10, median=5.0, max=20)

    def test_median_above_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            ImatrixCountSummary(min=1, median=30.0, max=20)

    def test_negative_min_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            ImatrixCountSummary(min=-1, median=0.0, max=1)

    def test_non_finite_median_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            ImatrixCountSummary(min=0, median=float("nan"), max=1)
