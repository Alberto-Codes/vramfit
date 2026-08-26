from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import kv_shapes
from vramfit.domain.budget import (
    KV_WINDOW_PAD_TOKENS,
    KVLayer,
    ModelShape,
    kv_cache_bytes,
    kv_growth_bytes_per_token,
    kv_window_pool_bytes,
)
from vramfit.domain.capacity import image_capacity, max_context_tokens, max_sequences

pytestmark = pytest.mark.unit

# Two uniform global layers: 2 heads x 4 wide x 2 tensors x 2 bytes,
# 64 bytes per token at fp16.
UNIFORM = ModelShape.uniform(attn_layers=2, kv_heads=2, head_dim=4)
TOKEN_BYTES = 64

# One global layer (16 B/token) plus one sliding layer (16 B/token,
# window 8). The runtime pads the window with `KV_WINDOW_PAD_TOKENS`
# (#431), so the layer saturates at 520 tokens: an 8,320-byte pool.
MIXED = ModelShape(
    kv_layers=(
        KVLayer(kv_heads=1, head_dim=4),
        KVLayer(kv_heads=1, head_dim=4, window=8),
    )
)

SLIDING_ONLY = ModelShape(kv_layers=(KVLayer(kv_heads=1, head_dim=4, window=8),))

ALL_SHARED = ModelShape(
    kv_layers=(KVLayer(kv_heads=1, head_dim=4, shares_kv=True),) * 2
)


class TestMaxContextTokens:
    def test_uniform_shape_matches_closed_form_division(self) -> None:
        assert max_context_tokens(UNIFORM, 1000) == 1000 // TOKEN_BYTES

    def test_exact_boundary_headroom_returns_the_boundary(self) -> None:
        assert max_context_tokens(UNIFORM, 10 * TOKEN_BYTES) == 10

    def test_one_byte_short_of_the_next_token_stays_at_the_boundary(self) -> None:
        assert max_context_tokens(UNIFORM, 11 * TOKEN_BYTES - 1) == 10

    def test_mixed_stack_past_saturation_prices_growth_plus_pool(self) -> None:
        # Past the padded window (520 tokens): cost = 16 x context + 8320.
        headroom = 16 * 1000 + 8320

        assert max_context_tokens(MIXED, headroom) == 1000

    def test_mixed_stack_below_the_window_prices_both_terms(self) -> None:
        # Below the window both layers grow: cost = 32 x context.
        assert max_context_tokens(MIXED, 32 * 5) == 5

    def test_negative_headroom_returns_zero(self) -> None:
        assert max_context_tokens(UNIFORM, -1) == 0

    def test_zero_headroom_returns_zero(self) -> None:
        assert max_context_tokens(UNIFORM, 0) == 0

    def test_headroom_below_one_token_returns_zero(self) -> None:
        assert max_context_tokens(UNIFORM, TOKEN_BYTES - 1) == 0

    def test_sliding_only_stack_with_saturated_fit_is_unbounded(self) -> None:
        pool = kv_window_pool_bytes(SLIDING_ONLY)

        assert max_context_tokens(SLIDING_ONLY, pool) is None

    def test_sliding_only_stack_with_tight_headroom_stays_finite(self) -> None:
        # One sliding layer at 16 B/token: five tokens fit inside the
        # 8-token window.
        assert max_context_tokens(SLIDING_ONLY, 16 * 5) == 5

    def test_sliding_only_stack_stays_finite_inside_the_padding(self) -> None:
        # The runtime pads the 8-token window to 520 cells (#431), so
        # a headroom one token short of the pool reads 519.
        pool = kv_window_pool_bytes(SLIDING_ONLY)

        assert max_context_tokens(SLIDING_ONLY, pool - 1) == 519

    def test_sequences_split_the_headroom(self) -> None:
        assert max_context_tokens(UNIFORM, 1000, sequences=2) == 1000 // (
            2 * TOKEN_BYTES
        )

    def test_unknown_dtype_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            max_context_tokens(UNIFORM, 1000, kv_dtype="fp4")


class TestMaxSequences:
    def test_fixed_context_matches_closed_form_division(self) -> None:
        # 640 bytes per sequence at 10 tokens.
        assert max_sequences(UNIFORM, 2000, context=10) == 3

    def test_exact_boundary_headroom_returns_the_boundary(self) -> None:
        assert max_sequences(UNIFORM, 3 * 640, context=10) == 3

    def test_one_byte_short_of_the_next_sequence_stays(self) -> None:
        assert max_sequences(UNIFORM, 3 * 640 - 1, context=10) == 2

    def test_mixed_stack_prices_the_saturated_window(self) -> None:
        # At 1000 tokens one sequence costs 16 x 1000 + 8320 = 24320.
        assert max_sequences(MIXED, 2 * 24320, context=1000) == 2

    def test_negative_headroom_returns_zero(self) -> None:
        assert max_sequences(UNIFORM, -1, context=10) == 0

    def test_headroom_below_one_sequence_returns_zero(self) -> None:
        assert max_sequences(UNIFORM, 639, context=10) == 0

    def test_all_shared_stack_is_unbounded(self) -> None:
        assert max_sequences(ALL_SHARED, 0, context=10) is None

    def test_unknown_dtype_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            max_sequences(UNIFORM, 1000, context=10, kv_dtype="fp4")


class TestImageCapacity:
    def test_divides_tokens_by_the_image_token_cost(self) -> None:
        assert image_capacity(1000, image_token_cost=256) == 3

    def test_exact_multiple_returns_the_full_count(self) -> None:
        assert image_capacity(1024, image_token_cost=256) == 4

    def test_zero_tokens_carry_no_images(self) -> None:
        assert image_capacity(0, image_token_cost=256) == 0

    @pytest.mark.parametrize("cost", [0, -1], ids=["zero", "negative"])
    def test_non_positive_cost_rejected(self, cost: int) -> None:
        with pytest.raises(ValueError, match="image token cost must be positive"):
            image_capacity(1000, image_token_cost=cost)

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            image_capacity(-1, image_token_cost=256)


class TestCapacityProperties:
    @given(
        shape=kv_shapes(),
        headroom=st.integers(min_value=0, max_value=1 << 40),
        sequences=st.integers(min_value=1, max_value=4),
    )
    def test_max_context_is_exact_at_the_fit_boundary(
        self, shape: ModelShape, headroom: int, sequences: int
    ) -> None:
        result = max_context_tokens(shape, headroom, "fp16", sequences)

        if result is None:
            assert kv_growth_bytes_per_token(shape) == 0
            assert kv_window_pool_bytes(shape) * sequences <= headroom
            # Unbounded means the cache really fits past every
            # padded window.
            windows = [
                layer.window for layer in shape.kv_layers if layer.window is not None
            ]
            probe = max(windows, default=1) + KV_WINDOW_PAD_TOKENS + 1
            assert kv_cache_bytes(shape, probe, "fp16", sequences) <= headroom
        else:
            assert kv_cache_bytes(shape, result, "fp16", sequences) <= headroom
            assert kv_cache_bytes(shape, result + 1, "fp16", sequences) > headroom

    @given(
        shape=kv_shapes(),
        headroom=st.integers(min_value=0, max_value=1 << 40),
        extra=st.integers(min_value=0, max_value=1 << 40),
        sequences=st.integers(min_value=1, max_value=4),
    )
    def test_more_headroom_never_reduces_max_context(
        self, shape: ModelShape, headroom: int, extra: int, sequences: int
    ) -> None:
        smaller = max_context_tokens(shape, headroom, "fp16", sequences)
        larger = max_context_tokens(shape, headroom + extra, "fp16", sequences)

        # None means unbounded, which dominates every finite reading.
        if larger is not None:
            assert smaller is not None
            assert smaller <= larger

    @given(
        shape=kv_shapes(),
        headroom=st.integers(min_value=0, max_value=1 << 40),
        context=st.integers(min_value=1, max_value=1 << 20),
    )
    def test_max_sequences_is_exact_at_the_fit_boundary(
        self, shape: ModelShape, headroom: int, context: int
    ) -> None:
        result = max_sequences(shape, headroom, context)

        if result is None:
            assert kv_cache_bytes(shape, context) == 0
        else:
            assert kv_cache_bytes(shape, context, "fp16", result) <= headroom
            assert kv_cache_bytes(shape, context, "fp16", result + 1) > headroom

    @given(
        shape=kv_shapes(),
        headroom=st.integers(min_value=0, max_value=1 << 40),
        extra=st.integers(min_value=0, max_value=1 << 40),
        context=st.integers(min_value=1, max_value=1 << 20),
    )
    def test_more_headroom_never_reduces_max_sequences(
        self, shape: ModelShape, headroom: int, extra: int, context: int
    ) -> None:
        smaller = max_sequences(shape, headroom, context)
        larger = max_sequences(shape, headroom + extra, context)

        if larger is not None:
            assert smaller is not None
            assert smaller <= larger
