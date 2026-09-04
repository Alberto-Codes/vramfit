from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import pytest

from vramfit.domain.errors import VramfitError
from vramfit.domain.runtime import (
    CONVERT_DTYPE_BITS,
    EFFECTIVE_BITS,
    EXPERT_STACK_EFFECTIVE_BITS,
    PASSTHROUGH_BITS,
    RUNTIME_CAPABILITIES,
    UNQUANTIZABLE_CLASS_FILTERS,
    RuntimeCapabilityError,
    effective_bits,
    expert_stack_effective_bits,
    missing_unquantizable_module,
    passthrough_bits,
    rows_refuse_super_block,
    servable_precisions,
    unquantizable_class,
    unquantizable_filter,
)

pytestmark = pytest.mark.unit


class TestServeablePrecisions:
    def test_llama_cpp_serves_the_full_scan_set(self) -> None:
        assert servable_precisions((8, 6, 5, 4, 3, 2), "llama.cpp") == (
            8,
            6,
            5,
            4,
            3,
            2,
        )

    def test_vllm_filters_to_its_kernel_set(self) -> None:
        assert servable_precisions((8, 4, 3, 2), "vllm") == (8, 4)

    def test_filter_preserves_descending_order(self) -> None:
        assert servable_precisions((8, 5, 2), "llama.cpp") == (8, 5, 2)

    def test_unknown_runtime_raises_capability_error(self) -> None:
        with pytest.raises(RuntimeCapabilityError, match='unknown runtime "tgi"'):
            servable_precisions((8, 4), "tgi")

    def test_empty_intersection_raises_capability_error(self) -> None:
        with pytest.raises(RuntimeCapabilityError, match="serves none"):
            servable_precisions((3, 2), "vllm")

    def test_capability_error_inherits_the_vramfit_root(self) -> None:
        assert issubclass(RuntimeCapabilityError, VramfitError)
        assert issubclass(RuntimeCapabilityError, ValueError)

    def test_capability_table_is_read_only(self) -> None:
        table = cast("MutableMapping[str, frozenset[int]]", RUNTIME_CAPABILITIES)

        with pytest.raises(TypeError):
            table["new"] = frozenset()


class TestEffectiveBits:
    def test_llama_cpp_table_matches_the_kquant_block_layouts(self) -> None:
        # Exact ADR-0014 constants, verified against packed files.
        assert effective_bits("llama.cpp") == {
            16: 16.0,
            8: 8.5,
            6: 6.5625,
            5: 5.5,
            4: 4.5,
            3: 3.4375,
            2: 2.625,
        }

    def test_runtime_without_a_measured_table_returns_none(self) -> None:
        assert effective_bits("vllm") is None

    def test_none_runtime_returns_none(self) -> None:
        assert effective_bits(None) is None

    def test_unknown_runtime_returns_none(self) -> None:
        # Name validation stays with servable_precisions.
        assert effective_bits("tgi") is None

    def test_every_table_covers_its_runtime_capability_exactly(self) -> None:
        # The solver indexes the table with any servable candidate, so
        # a table must cover its capability set — no more, no less.
        for runtime, table in EFFECTIVE_BITS.items():
            assert set(table) == RUNTIME_CAPABILITIES[runtime]

    def test_effective_bits_never_fall_below_nominal_bits(self) -> None:
        # Block scales cost extra bits — a table entry below nominal
        # would mean a type stores weights for free. The F16
        # passthrough is the one equality: `F16` stores two bytes per
        # weight and carries no block scale (ADR-0029 decision 4).
        for table in EFFECTIVE_BITS.values():
            assert all(spent >= bits for bits, spent in table.items())
            assert all(spent > bits for bits, spent in table.items() if bits != 16)

    def test_effective_bits_tables_are_read_only(self) -> None:
        outer = cast("MutableMapping[str, object]", EFFECTIVE_BITS)
        with pytest.raises(TypeError):
            outer["new"] = {}
        inner = cast("MutableMapping[int, float]", EFFECTIVE_BITS["llama.cpp"])
        with pytest.raises(TypeError):
            inner[16] = 16.5


class TestExpertStackEffectiveBits:
    def test_llama_cpp_table_carries_the_adr_0028_rows(self) -> None:
        # Q8_0 at 8.50, Q4_0 at 4.50, Q2_0 at 2.25 — exact
        # block-layout constants for the expert-stack type table. The
        # 16 row is the ADR-0029 passthrough, which has no block to
        # divide and so costs the same on a stack row.
        assert expert_stack_effective_bits("llama.cpp") == {
            16: 16.0,
            8: 8.5,
            4: 4.5,
            2: 2.25,
        }

    def test_table_has_no_3_bit_row(self) -> None:
        # No GGUF type lands between 2.25 and 4.25 bits per weight on
        # the stack rows — pack refuses nominal 3 there (ADR-0028
        # decision 2).
        assert 3 not in EXPERT_STACK_EFFECTIVE_BITS["llama.cpp"]

    def test_none_runtime_returns_none(self) -> None:
        assert expert_stack_effective_bits(None) is None

    def test_runtime_without_a_table_returns_none(self) -> None:
        assert expert_stack_effective_bits("vllm") is None

    def test_stack_bits_never_exceed_the_dense_entry(self) -> None:
        # The stack table exists because k-quants are unreachable on
        # the stack rows — every replacement row must not price above
        # its dense counterpart, or the solver would overshoot.
        for runtime, table in EXPERT_STACK_EFFECTIVE_BITS.items():
            for bits, spent in table.items():
                assert spent <= EFFECTIVE_BITS[runtime][bits]

    def test_stack_tables_are_read_only(self) -> None:
        outer = cast("MutableMapping[str, object]", EXPERT_STACK_EFFECTIVE_BITS)
        with pytest.raises(TypeError):
            outer["new"] = {}
        inner = cast(
            "MutableMapping[int, float]", EXPERT_STACK_EFFECTIVE_BITS["llama.cpp"]
        )
        with pytest.raises(TypeError):
            inner[3] = 3.0


@pytest.mark.unit
class TestSuperBlockRefusedClasses:
    """The layer classes the ADR-0028 table reaches (2026-08-20)."""

    @pytest.mark.parametrize(
        "group",
        [
            "model.layers.3.mixer.in_proj",
            "backbone.layers.3.mixer.out_proj",
            "model.layers.1.mixer.shared_experts.down_proj",
            "model.layers.2.mixer.q_proj",
        ],
    )
    def test_a_nemotron_h_dense_class_routes_to_the_stack_table(
        self, group: str
    ) -> None:
        assert rows_refuse_super_block(group)

    @pytest.mark.parametrize(
        "group",
        [
            "model.layers.3",
            "model.layers.3.self_attn.q_proj",
            "model.layers.3.mixer.gate",
            "model.layers.3.mixer.conv1d",
            "model.embeddings",
        ],
    )
    def test_other_groups_keep_their_own_table(self, group: str) -> None:
        assert not rows_refuse_super_block(group)


@pytest.mark.unit
class TestUnquantizableClassFilters:
    """The copied llama-quantize filter contract (2026-08-20)."""

    def test_the_router_names_its_upstream_filter(self) -> None:
        group = "model.layers.3.mixer.gate"
        assert unquantizable_filter(group, "llama.cpp") == "ffn_gate_inp.weight"

    def test_the_conv1d_names_its_upstream_filter(self) -> None:
        group = "backbone.layers.9.mixer.conv1d"
        assert unquantizable_filter(group, "llama.cpp") == "ssm_conv1d"

    @pytest.mark.parametrize(
        ("group", "runtime"),
        [
            ("model.layers.3.mixer.gate", None),
            ("model.layers.3.mixer.gate", "vllm"),
            ("model.layers.3.mixer.in_proj", "llama.cpp"),
            ("model.layers.3", "llama.cpp"),
            ("mixer.gate", "llama.cpp"),
        ],
        ids=["no-runtime", "no-table", "quantizable", "layer", "no-layer-prefix"],
    )
    def test_everything_else_carries_no_filter(
        self, group: str, runtime: str | None
    ) -> None:
        assert unquantizable_filter(group, runtime) is None

    def test_the_filter_tables_are_read_only(self) -> None:
        outer = cast("MutableMapping[str, object]", UNQUANTIZABLE_CLASS_FILTERS)
        with pytest.raises(TypeError):
            outer["new"] = {}
        inner = cast(
            "MutableMapping[str, str]", UNQUANTIZABLE_CLASS_FILTERS["llama.cpp"]
        )
        with pytest.raises(TypeError):
            inner["mixer.norm"] = "x"


@pytest.mark.unit
class TestPassthroughBits:
    """The passthrough prices at what the packed file holds (#409)."""

    @pytest.mark.parametrize(
        "group",
        ["model.layers.3.mixer.gate", "backbone.layers.9.mixer.conv1d"],
        ids=["router", "conv1d"],
    )
    def test_an_unquantizable_class_prices_at_the_convert_dtype(
        self, group: str
    ) -> None:
        # `convert_hf_to_gguf.py` writes both classes at float32
        # whatever `--outtype` asks, and the quantizer never touches
        # them, so the packed file stores four bytes per weight.
        assert passthrough_bits(group, "llama.cpp") == 32.0

    @pytest.mark.parametrize(
        ("group", "runtime"),
        [
            ("model.layers.3.mixer.in_proj", "llama.cpp"),
            ("model.layers.3.mixer.experts.up_proj", "llama.cpp"),
            ("model.layers.3", "llama.cpp"),
            ("model.embeddings", "llama.cpp"),
            ("model.layers.3.mixer.gate", None),
            ("model.layers.3.mixer.gate", "vllm"),
        ],
        ids=["dense-class", "stack", "layer", "no-layer", "no-runtime", "no-table"],
    )
    def test_everything_else_spends_f16(self, group: str, runtime: str | None) -> None:
        assert passthrough_bits(group, runtime) == PASSTHROUGH_BITS == 16.0

    def test_the_convert_dtype_table_names_every_unquantizable_class(self) -> None:
        # A class one table names, the other names too: a refused
        # class with no dtype row would price at 16.0 again.
        for runtime, filters in UNQUANTIZABLE_CLASS_FILTERS.items():
            assert set(CONVERT_DTYPE_BITS[runtime]) == set(filters)

    def test_the_convert_dtype_table_is_read_only(self) -> None:
        inner = cast("MutableMapping[str, float]", CONVERT_DTYPE_BITS["llama.cpp"])
        with pytest.raises(TypeError):
            inner["mixer.gate"] = 16.0


@pytest.mark.unit
class TestUnquantizableClass:
    """The scan-side skip, keyed by class under any runtime (#204)."""

    def test_every_30b_conv1d_cell_is_named(self) -> None:
        # The 23 `mixer.conv1d.weight` parameters the scan discovered
        # on the 30B target, shaped (6144, 1, 4). Each is 3-D, so the
        # rank gate admits it, and the class rule is what skips it.
        names = [f"model.layers.{n}.mixer.conv1d" for n in range(52) if n % 2 == 0][:23]
        assert len(names) == 23
        assert {unquantizable_class(name) for name in names} == {"mixer.conv1d"}

    def test_the_router_is_named_too(self) -> None:
        assert unquantizable_class("backbone.layers.1.mixer.gate") == "mixer.gate"

    @pytest.mark.parametrize(
        "group",
        [
            "model.layers.3.mixer.in_proj",
            "model.layers.3.mixer.experts.up_proj",
            "model.layers.3",
            "model.embeddings",
            "mixer.conv1d",
        ],
    )
    def test_everything_else_is_kept(self, group: str) -> None:
        assert unquantizable_class(group) is None


@pytest.mark.unit
class TestMissingUnquantizableModule:
    """A map naming a refused class's module and not the class (#409)."""

    def test_a_post_skip_hybrid_map_names_the_module(self) -> None:
        # The 30B target scanned since #204: the Mamba mixer's
        # in_proj and the MoE mixer's experts reach the map, and the
        # conv1d and the router never do.
        tensors = [
            "model.layers.0.mixer.in_proj.weight",
            "model.layers.1.mixer.experts.up_proj.weight",
        ]

        assert missing_unquantizable_module(tensors) == "mixer"

    def test_a_layer_map_names_the_module_through_its_members(self) -> None:
        tensors = [
            "model.layers.0.mixer.in_proj.weight",
            "model.layers.0.mixer.out_proj.weight",
        ]

        assert missing_unquantizable_module(tensors) == "mixer"

    @pytest.mark.parametrize("carried", ["mixer.conv1d", "mixer.gate"])
    def test_a_map_carrying_a_refused_class_prices_it_itself(
        self, carried: str
    ) -> None:
        tensors = [
            "model.layers.0.mixer.in_proj.weight",
            f"model.layers.0.{carried}.weight",
        ]

        assert missing_unquantizable_module(tensors) is None

    def test_a_llama_map_names_no_such_module(self) -> None:
        tensors = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.embed_tokens.weight",
        ]

        assert missing_unquantizable_module(tensors) is None

    def test_a_whole_layer_tensor_name_is_not_a_class(self) -> None:
        assert missing_unquantizable_module(["model.layers.0.weight"]) is None

    def test_an_empty_map_names_none(self) -> None:
        assert missing_unquantizable_module([]) is None
