from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import pytest

from vramfit.domain.errors import VramfitError
from vramfit.domain.runtime import (
    EFFECTIVE_BITS,
    EXPERT_STACK_EFFECTIVE_BITS,
    RUNTIME_CAPABILITIES,
    RuntimeCapabilityError,
    effective_bits,
    expert_stack_effective_bits,
    servable_precisions,
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
