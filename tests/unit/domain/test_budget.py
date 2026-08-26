from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import kv_shapes
from vramfit.domain.budget import (
    Budget,
    KVLayer,
    ModelShape,
    format_size,
    kv_cache_bytes,
    kv_growth_bytes_per_token,
    kv_window_pool_bytes,
    parse_size,
)

NEMOTRON_SHAPE = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)

# Gemma 4 31B (#423): 50 sliding layers at window 1024 plus 10 global
# K=V layers, verified against the official config on 2026-08-25.
GEMMA_31B_SHAPE = ModelShape(
    kv_layers=(
        (KVLayer(kv_heads=16, head_dim=256, window=1024),) * 50
        + (KVLayer(kv_heads=4, head_dim=512, kv_tensors=1),) * 10
    )
)


@pytest.mark.unit
class TestParseSize:
    def test_gib_returns_binary_bytes(self) -> None:
        assert parse_size("24GiB") == 24 * 2**30

    def test_gb_returns_decimal_bytes(self) -> None:
        assert parse_size("24GB") == 24 * 10**9

    def test_bare_integer_is_bytes(self) -> None:
        assert parse_size("1073741824") == 2**30

    def test_fractional_size_with_unit(self) -> None:
        assert parse_size("1.5GiB") == int(1.5 * 2**30)

    def test_size_past_a_64_bit_byte_count_rejected(self) -> None:
        # `--vram 20000000000GiB` used to solve, write a recipe, and
        # then fail when that recipe was read back. Refusing here
        # names the option the operator typed (#260).
        with pytest.raises(ValueError, match="does not fit a 64-bit byte count"):
            parse_size("20000000000GiB")

    def test_size_that_floats_to_infinity_refuses_as_value_error(self) -> None:
        # A long enough digit string floats to infinity and reaches
        # `int()` before the bound runs. `OverflowError` is not a
        # `ValueError`, so the CLI would show a traceback (#260).
        with pytest.raises(ValueError, match="does not fit a 64-bit byte count"):
            parse_size("9" * 400)

    def test_refusal_caps_the_size_it_repeats_back(self) -> None:
        with pytest.raises(ValueError) as caught:
            parse_size("9" * 400)

        assert len(str(caught.value)) < 120

    def test_size_inside_the_bound_still_parses(self) -> None:
        # `parse_size` runs the number through a float, so it cannot
        # land on 2**63 - 1 exactly. 2**62 is representable and well
        # past any real card.
        assert parse_size(str(2**62)) == 2**62

    def test_whitespace_and_case_tolerated(self) -> None:
        assert parse_size(" 4 gib ") == 4 * 2**30

    @pytest.mark.parametrize("bad", ["", "GiB", "12XB", "1.2.3GiB", "-4GiB"])
    def test_malformed_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError, match="not a recognizable size"):
            parse_size(bad)


@pytest.mark.unit
class TestFormatSize:
    def test_renders_gib_with_two_decimals(self) -> None:
        assert format_size(2 * 2**30) == "2.00 GiB"

    def test_renders_small_values_as_bytes(self) -> None:
        assert format_size(512) == "512 B"

    def test_round_trips_with_parse_size(self) -> None:
        assert parse_size(format_size(4 * 2**20)) == 4 * 2**20


@pytest.mark.unit
class TestKvMath:
    def test_growth_per_token_nemotron_shape_matches_hand_calc(self) -> None:
        # 2 (K+V) x 49 layers x 8 kv_heads x 128 head_dim x 2 bytes
        assert kv_growth_bytes_per_token(NEMOTRON_SHAPE) == 200_704

    def test_growth_per_token_fp8_is_half_of_fp16(self) -> None:
        fp16 = kv_growth_bytes_per_token(NEMOTRON_SHAPE, "fp16")

        assert kv_growth_bytes_per_token(NEMOTRON_SHAPE, "fp8") == fp16 // 2

    def test_unknown_dtype_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            kv_growth_bytes_per_token(NEMOTRON_SHAPE, "int4")

    def test_cache_scales_linearly_with_context_and_sequences(self) -> None:
        base = kv_cache_bytes(NEMOTRON_SHAPE, context=1024)

        assert kv_cache_bytes(NEMOTRON_SHAPE, context=2048) == 2 * base
        assert kv_cache_bytes(NEMOTRON_SHAPE, context=1024, sequences=3) == 3 * base

    def test_heterogeneous_heads_and_widths_sum_per_layer(self) -> None:
        shape = ModelShape(
            kv_layers=(
                KVLayer(kv_heads=8, head_dim=128),
                KVLayer(kv_heads=4, head_dim=64),
            )
        )

        assert kv_growth_bytes_per_token(shape) == 2 * (8 * 128 + 4 * 64) * 2

    def test_uniform_pool_is_zero_and_matches_growth_times_context(self) -> None:
        assert kv_window_pool_bytes(NEMOTRON_SHAPE) == 0
        assert kv_cache_bytes(NEMOTRON_SHAPE, context=16384) == (
            kv_growth_bytes_per_token(NEMOTRON_SHAPE) * 16384
        )

    def test_gemma_31b_growth_matches_recorded_figure(self) -> None:
        # 10 global layers x 4 kv_heads x 512 head_dim x 1 tensor x 2 bytes
        assert kv_growth_bytes_per_token(GEMMA_31B_SHAPE) == 40_960

    def test_gemma_31b_window_pool_matches_recorded_figure(self) -> None:
        # 50 sliding layers x 16 kv_heads x 256 head_dim x 2 x 2 x 1024
        assert kv_window_pool_bytes(GEMMA_31B_SHAPE) == 838_860_800

    def test_gemma_31b_cache_at_128k_matches_recorded_figure(self) -> None:
        cache = kv_cache_bytes(GEMMA_31B_SHAPE, context=131_072)

        assert cache == 40_960 * 131_072 + 838_860_800
        assert cache / 2**30 == pytest.approx(5.78, abs=0.01)

    def test_sliding_layer_stops_growing_at_its_window(self) -> None:
        shape = ModelShape(kv_layers=(KVLayer(kv_heads=2, head_dim=64, window=1024),))
        at_window = kv_cache_bytes(shape, context=1024)

        assert kv_cache_bytes(shape, context=4096) == at_window
        assert at_window == kv_window_pool_bytes(shape)

    def test_context_below_the_window_prices_only_the_context(self) -> None:
        shape = ModelShape(kv_layers=(KVLayer(kv_heads=2, head_dim=64, window=1024),))

        assert kv_cache_bytes(shape, context=256) == 2 * 2 * 64 * 2 * 256

    def test_k_eq_v_layer_stores_one_tensor_per_token(self) -> None:
        pair = ModelShape(kv_layers=(KVLayer(kv_heads=4, head_dim=128),))
        single = ModelShape(
            kv_layers=(KVLayer(kv_heads=4, head_dim=128, kv_tensors=1),)
        )

        assert kv_cache_bytes(single, context=1024) == (
            kv_cache_bytes(pair, context=1024) // 2
        )

    def test_shared_layer_allocates_nothing(self) -> None:
        fresh = KVLayer(kv_heads=4, head_dim=128)
        shape = ModelShape(
            kv_layers=(fresh, KVLayer(kv_heads=4, head_dim=128, shares_kv=True))
        )

        assert kv_cache_bytes(shape, context=1024) == kv_cache_bytes(
            ModelShape(kv_layers=(fresh,)), context=1024
        )
        assert kv_growth_bytes_per_token(shape) == kv_growth_bytes_per_token(
            ModelShape(kv_layers=(fresh,))
        )

    def test_shared_sliding_layer_adds_no_window_pool(self) -> None:
        shape = ModelShape(
            kv_layers=(KVLayer(kv_heads=2, head_dim=64, window=512, shares_kv=True),)
        )

        assert kv_window_pool_bytes(shape) == 0
        assert kv_cache_bytes(shape, context=4096) == 0

    def test_mixed_stack_below_the_window_sums_partial_terms(self) -> None:
        # At half the window, the sliding layers hold 512 tokens each
        # while the global layers hold the same 512-token context.
        cache = kv_cache_bytes(GEMMA_31B_SHAPE, context=512)

        assert cache == 40_960 * 512 + 50 * 16 * 256 * 2 * 2 * 512

    def test_mixed_stack_sequences_multiply_pool_and_growth(self) -> None:
        one = kv_cache_bytes(GEMMA_31B_SHAPE, context=8192)

        assert kv_cache_bytes(GEMMA_31B_SHAPE, context=8192, sequences=4) == 4 * one


@pytest.mark.unit
class TestKvMathProperties:
    @given(
        shape=kv_shapes(),
        contexts=st.tuples(
            st.integers(min_value=1, max_value=1 << 20),
            st.integers(min_value=1, max_value=1 << 20),
        ),
    )
    def test_cache_is_monotone_in_context(
        self, shape: ModelShape, contexts: tuple[int, int]
    ) -> None:
        lo, hi = sorted(contexts)

        assert kv_cache_bytes(shape, context=lo) <= kv_cache_bytes(shape, context=hi)

    @given(
        shape=kv_shapes(),
        context=st.integers(min_value=1, max_value=1 << 20),
        sequences=st.integers(min_value=1, max_value=64),
    )
    def test_cache_is_linear_in_sequences(
        self, shape: ModelShape, context: int, sequences: int
    ) -> None:
        one = kv_cache_bytes(shape, context=context)

        assert kv_cache_bytes(shape, context=context, sequences=sequences) == (
            sequences * one
        )

    @given(shape=kv_shapes(), context=st.integers(min_value=1, max_value=1 << 20))
    def test_saturated_cache_is_growth_times_context_plus_pool(
        self, shape: ModelShape, context: int
    ) -> None:
        windows = [
            layer.window for layer in shape.kv_layers if layer.window is not None
        ]
        saturated = max(windows, default=1)
        context = max(context, saturated)

        assert kv_cache_bytes(shape, context=context) == (
            kv_growth_bytes_per_token(shape) * context + kv_window_pool_bytes(shape)
        )

    @given(
        shape=kv_shapes(),
        context=st.integers(min_value=1, max_value=1 << 20),
        data=st.data(),
    )
    def test_marking_a_layer_shared_never_raises_the_cache(
        self, shape: ModelShape, context: int, data: st.DataObject
    ) -> None:
        index = data.draw(st.integers(min_value=0, max_value=len(shape.kv_layers) - 1))
        flipped = ModelShape(
            kv_layers=tuple(
                replace(layer, shares_kv=True) if i == index else layer
                for i, layer in enumerate(shape.kv_layers)
            )
        )

        assert kv_cache_bytes(flipped, context=context) <= kv_cache_bytes(
            shape, context=context
        )

    @given(shape=kv_shapes(), context=st.integers(min_value=1, max_value=1 << 20))
    def test_all_shared_stack_prices_at_zero(
        self, shape: ModelShape, context: int
    ) -> None:
        all_shared = ModelShape(
            kv_layers=tuple(replace(layer, shares_kv=True) for layer in shape.kv_layers)
        )

        assert kv_cache_bytes(all_shared, context=context) == 0


@pytest.mark.unit
class TestBudget:
    def test_weight_budget_subtracts_kv_and_overhead(self) -> None:
        ledger = Budget(
            vram_total_bytes=parse_size("24GiB"),
            kv_cache_bytes=parse_size("3GiB"),
            runtime_overhead_bytes=parse_size("2GiB"),
        )

        assert ledger.weight_budget_bytes == parse_size("19GiB")

    def test_weight_budget_negative_when_overcommitted(self) -> None:
        ledger = Budget(
            vram_total_bytes=parse_size("8GiB"),
            kv_cache_bytes=parse_size("6GiB"),
            runtime_overhead_bytes=parse_size("4GiB"),
        )

        assert ledger.weight_budget_bytes < 0
