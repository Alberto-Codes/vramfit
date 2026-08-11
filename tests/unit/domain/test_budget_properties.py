from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vramfit.domain.budget import format_size, parse_size


@pytest.mark.unit
class TestSizeProperties:
    @given(n=st.integers(min_value=0, max_value=2**50))
    def test_format_then_parse_is_close(self, n: int) -> None:
        # format_size rounds to two decimals of a binary unit, so the
        # round trip is exact below 1 KiB and within rounding above it.
        result = parse_size(format_size(n))

        assert abs(result - n) <= max(1, int(0.006 * n))

    @given(n=st.integers(min_value=0, max_value=1023))
    def test_sub_kib_round_trip_is_exact(self, n: int) -> None:
        assert parse_size(format_size(n)) == n

    @given(gib=st.integers(min_value=1, max_value=1024))
    def test_gib_strings_parse_to_exact_binary_bytes(self, gib: int) -> None:
        assert parse_size(f"{gib}GiB") == gib * 2**30
