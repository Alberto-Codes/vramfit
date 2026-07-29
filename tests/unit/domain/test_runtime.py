from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import pytest

from quantfit.domain.errors import QuantfitError
from quantfit.domain.runtime import (
    EFFECTIVE_BITS,
    RUNTIME_CAPABILITIES,
    RuntimeCapabilityError,
    effective_bits,
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

    def test_capability_error_inherits_the_quantfit_root(self) -> None:
        assert issubclass(RuntimeCapabilityError, QuantfitError)
        assert issubclass(RuntimeCapabilityError, ValueError)

    def test_capability_table_is_read_only(self) -> None:
        table = cast("MutableMapping[str, frozenset[int]]", RUNTIME_CAPABILITIES)

        with pytest.raises(TypeError):
            table["new"] = frozenset()


class TestEffectiveBits:
    def test_llama_cpp_table_matches_the_kquant_block_layouts(self) -> None:
        # Exact ADR-0014 constants, verified against packed files.
        assert effective_bits("llama.cpp") == {
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

    def test_effective_bits_always_exceed_nominal_bits(self) -> None:
        # Block scales cost extra bits — a table entry below nominal
        # would mean a type stores weights for free.
        for table in EFFECTIVE_BITS.values():
            assert all(spent > bits for bits, spent in table.items())

    def test_effective_bits_tables_are_read_only(self) -> None:
        outer = cast("MutableMapping[str, object]", EFFECTIVE_BITS)
        with pytest.raises(TypeError):
            outer["new"] = {}
        inner = cast("MutableMapping[int, float]", EFFECTIVE_BITS["llama.cpp"])
        with pytest.raises(TypeError):
            inner[16] = 16.5
