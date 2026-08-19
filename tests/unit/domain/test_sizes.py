"""Reference-byte arithmetic, root reconciliation, group aggregation."""

from __future__ import annotations

import pytest

from vramfit.domain.sizes import (
    CHECKPOINT_ROOTS,
    SizeSourceError,
    TensorSize,
    discovered_group_bytes,
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
            TensorSize(dtype="", bytes=8)

    def test_zero_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TensorSize(dtype="BF16", bytes=0)


class TestReferenceBytes:
    def test_bf16_tensor_keeps_its_stored_size(self) -> None:
        size = TensorSize(dtype="BF16", bytes=EXPERT_BYTES)

        assert reference_bytes("w", size) == EXPERT_BYTES

    def test_fp32_tensor_halves_to_the_reference(self) -> None:
        size = TensorSize(dtype="F32", bytes=16)

        assert reference_bytes("w", size) == 8

    def test_fp8_tensor_doubles_to_the_reference(self) -> None:
        size = TensorSize(dtype="F8_E4M3", bytes=8)

        assert reference_bytes("w", size) == 16

    def test_unknown_dtype_raises_size_source_error(self) -> None:
        size = TensorSize(dtype="I8", bytes=8)

        with pytest.raises(SizeSourceError, match="no reference size"):
            reference_bytes("w", size)

    def test_partial_element_raises_size_source_error(self) -> None:
        size = TensorSize(dtype="F32", bytes=6)

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
                dtype="BF16", bytes=EXPERT_BYTES
            )
            for i in range(128)
        }

        groups = discovered_group_bytes(sizes, "stack")

        assert groups == {"model.layers.1.mixer.experts.up_proj": EXPERT_STACK_BYTES}

    def test_layer_granularity_sums_a_whole_layer(self) -> None:
        sizes = {
            "backbone.layers.0.mlp.up_proj.weight": TensorSize("BF16", 8),
            "backbone.layers.0.mlp.down_proj.weight": TensorSize("BF16", 16),
        }

        groups = discovered_group_bytes(sizes, "layer")

        assert groups == {"model.layers.0": 24}

    def test_an_empty_source_discovers_no_group(self) -> None:
        assert discovered_group_bytes({}, "stack") == {}

    def test_an_fp32_tensor_groups_at_its_reference_size(self) -> None:
        sizes = {"backbone.layers.0.mlp.up_proj.weight": TensorSize("F32", 16)}

        groups = discovered_group_bytes(sizes, "layer")

        assert groups == {"model.layers.0": 8}

    def test_an_unknown_dtype_refuses_rather_than_undercounting(self) -> None:
        sizes = {"backbone.layers.0.mlp.up_proj.weight": TensorSize("I8", 8)}

        with pytest.raises(SizeSourceError, match="no reference size"):
            discovered_group_bytes(sizes, "layer")


class TestUncoveredGroups:
    def test_groups_outside_the_map_are_returned_in_name_order(self) -> None:
        discovered = {"c": 3, "a": 1, "b": 2}

        assert uncovered_groups(discovered, ["b"]) == (("a", 1), ("c", 3))

    def test_a_fully_covered_source_yields_nothing(self) -> None:
        assert uncovered_groups({"a": 1}, ["a"]) == ()

    def test_a_map_group_the_source_lacks_is_not_invented(self) -> None:
        assert uncovered_groups({}, ["a"]) == ()
