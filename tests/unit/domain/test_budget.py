from __future__ import annotations

import pytest

from vramfit.domain.budget import (
    Budget,
    ModelShape,
    format_size,
    kv_bytes_per_token,
    kv_cache_bytes,
    parse_size,
)

NEMOTRON_SHAPE = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)


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
    def test_bytes_per_token_nemotron_shape_matches_hand_calc(self) -> None:
        # 2 (K+V) x 49 layers x 8 kv_heads x 128 head_dim x 2 bytes
        assert kv_bytes_per_token(NEMOTRON_SHAPE) == 200_704

    def test_bytes_per_token_fp8_is_half_of_fp16(self) -> None:
        fp16 = kv_bytes_per_token(NEMOTRON_SHAPE, "fp16")

        assert kv_bytes_per_token(NEMOTRON_SHAPE, "fp8") == fp16 // 2

    def test_unknown_dtype_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            kv_bytes_per_token(NEMOTRON_SHAPE, "int4")

    def test_cache_scales_linearly_with_context_and_sequences(self) -> None:
        base = kv_cache_bytes(NEMOTRON_SHAPE, context=1024)

        assert kv_cache_bytes(NEMOTRON_SHAPE, context=2048) == 2 * base
        assert kv_cache_bytes(NEMOTRON_SHAPE, context=1024, sequences=3) == 3 * base

    def test_heterogeneous_shape_sums_per_layer_heads(self) -> None:
        shape = ModelShape(kv_heads_per_layer=(8, 4), head_dim=128)

        assert kv_bytes_per_token(shape) == 2 * 128 * (8 + 4) * 2


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
