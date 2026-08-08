from __future__ import annotations

import pytest

from quantfit.domain.model import LayerGroup, ProtectedTensor, ScanMeta, SensitivityMap
from quantfit.domain.protection import (
    ProtectionError,
    expand_protections,
    noop_protection_patterns,
    protected_group_bytes,
    resolve_protected,
)

CURVE = {8: 0.0, 4: 0.1, 3: 0.2, 2: 0.4}


def make_layer_map(with_sizes: bool = True) -> SensitivityMap:
    def layer(index: int) -> LayerGroup:
        tensors = (
            f"model.layers.{index}.self_attn.v_proj.weight",
            f"model.layers.{index}.mlp.down_proj.weight",
        )
        return LayerGroup(
            name=f"model.layers.{index}",
            tensors=tensors,
            bytes_fp16=1_000,
            sensitivity=CURVE,
            tensor_bytes={tensors[0]: 200, tensors[1]: 800} if with_sizes else {},
        )

    return SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="wikitext",
            calibration_tokens=1024,
            precisions=(8, 4, 3, 2),
            group_by="layer",
            started_at="2026-08-08T00:00:00Z",
        ),
        groups=(
            layer(0),
            layer(1),
            LayerGroup(
                name="model.embed_tokens",
                tensors=("model.embed_tokens.weight",),
                bytes_fp16=500,
                sensitivity=CURVE,
                tensor_bytes={"model.embed_tokens.weight": 500} if with_sizes else {},
            ),
        ),
    )


@pytest.mark.unit
class TestExpandProtections:
    def test_pattern_resolves_to_matched_tensor_floors(self) -> None:
        floors = expand_protections(
            {"*.self_attn.v_proj.weight": 5}, make_layer_map(), runtime="llama.cpp"
        )

        assert floors == {
            "model.layers.0.self_attn.v_proj.weight": 5,
            "model.layers.1.self_attn.v_proj.weight": 5,
        }

    def test_later_pattern_overrides_earlier_for_overlapping_tensors(self) -> None:
        floors = expand_protections(
            {"*.self_attn.v_proj.weight": 5, "model.layers.0.*": 6},
            make_layer_map(),
            runtime="llama.cpp",
        )

        assert floors["model.layers.0.self_attn.v_proj.weight"] == 6
        assert floors["model.layers.1.self_attn.v_proj.weight"] == 5

    def test_unservable_floor_rejected_through_capability_table(self) -> None:
        with pytest.raises(ProtectionError, match="cannot serve 7-bit"):
            expand_protections(
                {"*.self_attn.v_proj.weight": 7},
                make_layer_map(),
                runtime="llama.cpp",
            )

    def test_no_runtime_accepts_any_positive_floor(self) -> None:
        floors = expand_protections(
            {"*.self_attn.v_proj.weight": 7}, make_layer_map(), runtime=None
        )

        assert set(floors.values()) == {7}

    def test_no_match_pattern_rejected(self) -> None:
        with pytest.raises(ProtectionError, match="matches no tensor"):
            expand_protections(
                {"*.nope.weight": 5}, make_layer_map(), runtime="llama.cpp"
            )

    def test_single_tensor_group_rejected_with_pin_pointer(self) -> None:
        with pytest.raises(ProtectionError, match="--pin"):
            expand_protections(
                {"model.embed_tokens.weight": 5},
                make_layer_map(),
                runtime="llama.cpp",
            )

    def test_map_without_tensor_bytes_rejected_naming_the_field(self) -> None:
        with pytest.raises(ProtectionError, match="tensor_bytes"):
            expand_protections(
                {"*.self_attn.v_proj.weight": 5},
                make_layer_map(with_sizes=False),
                runtime="llama.cpp",
            )


@pytest.mark.unit
class TestProtectedGroupBytes:
    def price(self, bytes_fp16: int, bits: int) -> int:
        return bytes_fp16 * bits // 16

    def test_group_without_protected_tensors_prices_as_one_piece(self) -> None:
        group = make_layer_map().groups[0]

        assert protected_group_bytes(group, 4, {}, self.price) == self.price(1_000, 4)

    def test_protected_tensor_prices_at_floor_when_candidate_is_lower(self) -> None:
        group = make_layer_map().groups[0]
        floors = {group.tensors[0]: 5}

        priced = protected_group_bytes(group, 3, floors, self.price)

        assert priced == self.price(800, 3) + self.price(200, 5)

    def test_protected_tensor_prices_at_candidate_when_it_meets_the_floor(
        self,
    ) -> None:
        group = make_layer_map().groups[0]
        floors = {group.tensors[0]: 5}

        priced = protected_group_bytes(group, 8, floors, self.price)

        assert priced == self.price(800, 8) + self.price(200, 8)

    def test_fully_protected_group_prices_no_plain_piece(self) -> None:
        group = make_layer_map().groups[0]
        floors = {group.tensors[0]: 5, group.tensors[1]: 5}

        priced = protected_group_bytes(group, 3, floors, self.price)

        assert priced == self.price(200, 5) + self.price(800, 5)


@pytest.mark.unit
class TestResolveProtected:
    def test_resolved_bits_are_max_of_assignment_and_floor(self) -> None:
        map_ = make_layer_map()
        floors = {
            "model.layers.0.self_attn.v_proj.weight": 5,
            "model.layers.1.self_attn.v_proj.weight": 5,
        }
        state = {"model.layers.0": 3, "model.layers.1": 8, "model.embed_tokens": 8}

        resolved = resolve_protected(map_, state, floors)

        assert resolved == (
            ProtectedTensor("model.layers.0.self_attn.v_proj.weight", 5),
            ProtectedTensor("model.layers.1.self_attn.v_proj.weight", 8),
        )


@pytest.mark.unit
class TestNoopProtectionPatterns:
    def test_floor_at_or_below_every_assignment_is_named(self) -> None:
        map_ = make_layer_map()
        protections = {"*.self_attn.v_proj.weight": 4}
        floors = expand_protections(protections, map_, runtime=None)
        state = {"model.layers.0": 8, "model.layers.1": 4, "model.embed_tokens": 8}

        assert noop_protection_patterns(protections, map_, state, floors) == (
            "*.self_attn.v_proj.weight",
        )

    def test_floor_above_any_assignment_is_not_named(self) -> None:
        map_ = make_layer_map()
        protections = {"*.self_attn.v_proj.weight": 5}
        floors = expand_protections(protections, map_, runtime=None)
        state = {"model.layers.0": 3, "model.layers.1": 8, "model.embed_tokens": 8}

        assert noop_protection_patterns(protections, map_, state, floors) == ()

    def test_pattern_fully_overridden_by_later_rule_is_named(self) -> None:
        # A dead rule governs no tensor and changed nothing — silence
        # would read as protection applied (ADR-0022).
        map_ = make_layer_map()
        protections = {
            "model.layers.*.self_attn.v_proj.weight": 4,
            "model.layers.*.v_proj.weight": 5,
        }
        floors = expand_protections(protections, map_, runtime=None)
        state = {"model.layers.0": 3, "model.layers.1": 3, "model.embed_tokens": 8}

        noop = noop_protection_patterns(protections, map_, state, floors)

        assert "model.layers.*.self_attn.v_proj.weight" in noop
        assert "model.layers.*.v_proj.weight" not in noop
