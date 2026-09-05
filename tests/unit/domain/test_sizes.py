"""Reference-byte arithmetic, root reconciliation, group aggregation."""

from __future__ import annotations

from typing import ClassVar, Literal

import pytest

from vramfit.domain.model import LayerGroup, ScanMeta, SensitivityMap
from vramfit.domain.sizes import (
    CHECKPOINT_ROOTS,
    SizeSourceError,
    TensorSize,
    discovered_group_bytes,
    discovered_group_rows,
    held_class_overlaps,
    reconcile_root,
    reference_bytes,
    uncovered_groups,
)

pytestmark = pytest.mark.unit

# One routed expert of the 30B target's up projection: 2688 x 1856 at
# bf16. The map records the 128-tensor stack at 1,277,165,568 bytes.
EXPERT_BYTES = 2688 * 1856 * 2
EXPERT_STACK_BYTES = 1_277_165_568


class TestTensorSize:
    def test_empty_dtype_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            TensorSize(dtype="", bytes=8, rows=4)

    def test_zero_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TensorSize(dtype="BF16", bytes=0, rows=4)


class TestReferenceBytes:
    def test_bf16_tensor_keeps_its_stored_size(self) -> None:
        size = TensorSize(dtype="BF16", bytes=EXPERT_BYTES, rows=4)

        assert reference_bytes("w", size) == EXPERT_BYTES

    def test_fp32_tensor_halves_to_the_reference(self) -> None:
        size = TensorSize(dtype="F32", bytes=16, rows=4)

        assert reference_bytes("w", size) == 8

    def test_fp8_tensor_doubles_to_the_reference(self) -> None:
        size = TensorSize(dtype="F8_E4M3", bytes=8, rows=4)

        assert reference_bytes("w", size) == 16

    def test_unknown_dtype_raises_size_source_error(self) -> None:
        size = TensorSize(dtype="I8", bytes=8, rows=4)

        with pytest.raises(SizeSourceError, match="no reference size"):
            reference_bytes("w", size)

    def test_partial_element_raises_size_source_error(self) -> None:
        size = TensorSize(dtype="F32", bytes=6, rows=3)

        with pytest.raises(SizeSourceError, match="whole number"):
            reference_bytes("w", size)


class TestReconcileRoot:
    def test_backbone_root_becomes_the_map_root(self) -> None:
        name = "backbone.layers.1.mixer.experts.0.up_proj.weight"

        assert reconcile_root(name) == ("model.layers.1.mixer.experts.0.up_proj.weight")

    def test_map_root_passes_through(self) -> None:
        assert reconcile_root("model.layers.0.mlp") == "model.layers.0.mlp"

    def test_a_rootless_tensor_passes_through(self) -> None:
        assert reconcile_root("lm_head.weight") == "lm_head.weight"

    def test_an_unknown_root_over_layers_refuses(self) -> None:
        # #177: a prefix wildcard mapped a vision tower's layers.5
        # onto the decoder's blk.5. The table refuses instead.
        with pytest.raises(SizeSourceError, match="root the table does not carry"):
            reconcile_root("vision_tower.layers.5.attn.q_proj.weight")

    def test_the_table_names_both_roots_the_target_uses(self) -> None:
        assert set(CHECKPOINT_ROOTS) == {"backbone.", "model."}


class TestDiscoveredGroupBytes:
    def test_expert_tensors_sum_into_one_stack_group(self) -> None:
        sizes = {
            f"backbone.layers.1.mixer.experts.{i}.up_proj.weight": TensorSize(
                dtype="BF16", bytes=EXPERT_BYTES, rows=256
            )
            for i in range(128)
        }

        groups = discovered_group_bytes(sizes, "stack")

        assert groups == {"model.layers.1.mixer.experts.up_proj": EXPERT_STACK_BYTES}

    def test_layer_granularity_sums_a_whole_layer(self) -> None:
        sizes = {
            "backbone.layers.0.mlp.up_proj.weight": TensorSize("BF16", 8, 256),
            "backbone.layers.0.mlp.down_proj.weight": TensorSize("BF16", 16, 256),
        }

        groups = discovered_group_bytes(sizes, "layer")

        assert groups == {"model.layers.0": 24}

    def test_an_empty_source_discovers_no_group(self) -> None:
        assert discovered_group_bytes({}, "stack") == {}

    def test_an_fp32_tensor_groups_at_its_reference_size(self) -> None:
        sizes = {"backbone.layers.0.mlp.up_proj.weight": TensorSize("F32", 16, 256)}

        groups = discovered_group_bytes(sizes, "layer")

        assert groups == {"model.layers.0": 8}

    def test_an_unknown_dtype_refuses_rather_than_undercounting(self) -> None:
        sizes = {"backbone.layers.0.mlp.up_proj.weight": TensorSize("I8", 8, 256)}

        with pytest.raises(SizeSourceError, match="no reference size"):
            discovered_group_bytes(sizes, "layer")

    def test_layer_granularity_keys_an_unquantizable_class_by_tensor(self) -> None:
        # The Nemotron-H shape under `scan`'s default granularity. The
        # meter skips the Mamba conv1d and the router at discovery
        # (#204), so the map's layer group holds no bytes for them.
        # Folded into that covered group, their bytes would leave the
        # plan while the pack holds them at F32 (#409).
        sizes = {
            "backbone.layers.0.mixer.in_proj.weight": TensorSize("BF16", 8, 256),
            "backbone.layers.0.mixer.conv1d.weight": TensorSize("BF16", 4, 256),
            "backbone.layers.1.mixer.gate.weight": TensorSize("BF16", 2, 256),
            "backbone.layers.1.mixer.experts.0.up_proj.weight": TensorSize(
                "BF16", 16, 256
            ),
        }

        groups = discovered_group_bytes(sizes, "layer")

        assert groups == {
            "model.layers.0": 8,
            "model.layers.0.mixer.conv1d": 4,
            "model.layers.1": 16,
            "model.layers.1.mixer.gate": 2,
        }

    @pytest.mark.parametrize("group_by", ["layer", "tensor", "stack"])
    def test_an_unquantizable_class_keys_the_same_under_every_granularity(
        self, group_by: Literal["layer", "tensor", "stack"]
    ) -> None:
        sizes = {"backbone.layers.3.mixer.conv1d.weight": TensorSize("F32", 8, 256)}

        groups = discovered_group_bytes(sizes, group_by)

        assert groups == {"model.layers.3.mixer.conv1d": 4}


class TestDiscoveredGroupRows:
    def test_a_stack_reports_the_width_its_experts_share(self) -> None:
        sizes = {
            f"backbone.layers.1.mixer.experts.{i}.up_proj.weight": TensorSize(
                dtype="BF16", bytes=EXPERT_BYTES, rows=2688
            )
            for i in range(128)
        }

        assert discovered_group_rows(sizes, "stack") == {
            "model.layers.1.mixer.experts.up_proj": 2688
        }

    def test_a_whole_layer_group_reports_no_width(self) -> None:
        # A layer group holds classes of several widths, so no single
        # width describes it and it keeps the ADR-0012 k-quant table.
        sizes = {
            "backbone.layers.0.mlp.up_proj.weight": TensorSize("BF16", 8, 2048),
            "backbone.layers.0.mlp.down_proj.weight": TensorSize("BF16", 16, 768),
        }

        assert discovered_group_rows(sizes, "layer") == {}

    def test_two_widths_in_one_group_refuse_rather_than_pick_one(self) -> None:
        # One group packs under one type. Keeping the first width
        # would misprice the second silently, which is the class #515
        # exists to remove.
        sizes = {
            "backbone.layers.1.mixer.experts.0.up_proj.weight": TensorSize(
                "BF16", 8, 2048
            ),
            "backbone.layers.1.mixer.experts.1.up_proj.weight": TensorSize(
                "BF16", 8, 2688
            ),
        }

        with pytest.raises(SizeSourceError, match="rows of 2048 and 2688"):
            discovered_group_rows(sizes, "stack")

    def test_a_foreign_root_refuses_naming_the_table(self) -> None:
        sizes = {"vision_tower.layers.0.attn.q_proj.weight": TensorSize("BF16", 8, 256)}

        with pytest.raises(SizeSourceError, match="root the table does not carry"):
            discovered_group_rows(sizes, "tensor")

    def test_an_unquantizable_class_keys_by_its_own_name_under_layer(self) -> None:
        # `discovered_group_bytes` holds such a tensor by its own name
        # under every granularity (#409, #204), and the width read
        # mirrors that grouping so the two agree on one name set.
        sizes = {"backbone.layers.0.mixer.conv1d.weight": TensorSize("F32", 16, 4)}

        assert discovered_group_rows(sizes, "layer") == {
            "model.layers.0.mixer.conv1d": 4
        }

    def test_an_empty_source_measures_no_width(self) -> None:
        assert discovered_group_rows({}, "stack") == {}


class TestUncoveredGroups:
    def test_groups_outside_the_map_are_returned_in_name_order(self) -> None:
        discovered = {"c": 3, "a": 1, "b": 2}

        assert uncovered_groups(discovered, ["b"]) == (("a", 1), ("c", 3))

    def test_a_fully_covered_source_yields_nothing(self) -> None:
        assert uncovered_groups({"a": 1}, ["a"]) == ()

    def test_a_map_group_the_source_lacks_is_not_invented(self) -> None:
        assert uncovered_groups({}, ["a"]) == ()


def _map(
    groups: list[tuple[str, tuple[str, ...]]],
    group_by: Literal["layer", "tensor", "stack"],
) -> SensitivityMap:
    return SensitivityMap(
        model_id="test/model",
        scan=ScanMeta("kl_divergence", "wikitext", 1, (8,), group_by, "t"),
        groups=tuple(
            LayerGroup(name=name, tensors=tensors, bytes_fp16=8, sensitivity={8: 0.0})
            for name, tensors in groups
        ),
    )


class TestHeldClassOverlaps:
    """A pre-#204 layer map folds a tensor the source holds by itself."""

    HYBRID: ClassVar[dict[str, TensorSize]] = {
        "backbone.layers.0.mixer.in_proj.weight": TensorSize("BF16", 8, 256),
        "backbone.layers.0.mixer.conv1d.weight": TensorSize("BF16", 4, 256),
        "backbone.layers.0.mixer.gate.weight": TensorSize("BF16", 2, 256),
    }

    def test_a_layer_group_folding_refused_classes_is_reported(self) -> None:
        map_ = _map(
            [
                (
                    "model.layers.0",
                    (
                        "model.layers.0.mixer.in_proj.weight",
                        "model.layers.0.mixer.conv1d.weight",
                        "model.layers.0.mixer.gate.weight",
                    ),
                )
            ],
            "layer",
        )
        discovered = discovered_group_bytes(self.HYBRID, "layer")

        assert held_class_overlaps(discovered, map_) == (
            (
                "model.layers.0",
                (
                    "model.layers.0.mixer.conv1d.weight",
                    "model.layers.0.mixer.gate.weight",
                ),
            ),
        )

    def test_a_layer_group_scanned_after_the_skip_is_clean(self) -> None:
        map_ = _map(
            [("model.layers.0", ("model.layers.0.mixer.in_proj.weight",))], "layer"
        )
        discovered = discovered_group_bytes(self.HYBRID, "layer")

        assert held_class_overlaps(discovered, map_) == ()

    def test_a_tensor_map_carrying_the_class_covers_it(self) -> None:
        # The class is its own covered group, so the source's group
        # is not uncovered and nothing prices twice.
        map_ = _map(
            [
                (
                    "model.layers.0.mixer.in_proj",
                    ("model.layers.0.mixer.in_proj.weight",),
                ),
                (
                    "model.layers.0.mixer.conv1d",
                    ("model.layers.0.mixer.conv1d.weight",),
                ),
            ],
            "tensor",
        )
        discovered = discovered_group_bytes(self.HYBRID, "tensor")

        assert held_class_overlaps(discovered, map_) == ()

    def test_a_stack_map_is_unaffected(self) -> None:
        # The checked-in maps use stack granularity: a stack group
        # lists only its own projection's experts.
        sizes = {
            "backbone.layers.1.mixer.experts.0.up_proj.weight": TensorSize(
                "BF16", 8, 256
            ),
            "backbone.layers.1.mixer.experts.1.up_proj.weight": TensorSize(
                "BF16", 8, 256
            ),
            "backbone.layers.1.mixer.conv1d.weight": TensorSize("BF16", 4, 256),
        }
        map_ = _map(
            [
                (
                    "model.layers.1.mixer.experts.up_proj",
                    (
                        "model.layers.1.mixer.experts.0.up_proj.weight",
                        "model.layers.1.mixer.experts.1.up_proj.weight",
                    ),
                )
            ],
            "stack",
        )
        discovered = discovered_group_bytes(sizes, "stack")

        assert held_class_overlaps(discovered, map_) == ()

    def test_a_folded_tensor_the_checkpoint_lacks_is_not_reported(self) -> None:
        # No source group exists to price it a second time.
        map_ = _map(
            [
                (
                    "model.layers.0",
                    (
                        "model.layers.0.mixer.in_proj.weight",
                        "model.layers.0.mixer.conv1d.weight",
                    ),
                )
            ],
            "layer",
        )
        sizes = {"backbone.layers.0.mixer.in_proj.weight": TensorSize("BF16", 8, 256)}
        discovered = discovered_group_bytes(sizes, "layer")

        assert held_class_overlaps(discovered, map_) == ()
