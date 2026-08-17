"""Exclusion matching against a supplied entry list (#309).

`unmatched_exclusions` is pure, so it needs no file and no gguf-py.
`check_exclusion_match` reads one, and the empty-sequence path is the
only branch that returns before the read — the rest of that function
runs against a written imatrix in `test_exclusion_match_gguf.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vramfit.adapters.outbound.gguf.exclusion_match import (
    check_exclusion_match,
    unmatched_exclusions,
)

pytestmark = pytest.mark.unit

ENTRIES = (
    "blk.0.attn_v.weight",
    "blk.1.attn_v.weight",
    "blk.1.ffn_down.weight",
)


class TestUnmatchedExclusions:
    def test_a_name_an_entry_carries_matches(self) -> None:
        assert unmatched_exclusions(("blk.1.attn_v.weight",), ENTRIES) == ()

    def test_a_name_no_entry_carries_reports(self) -> None:
        assert unmatched_exclusions(("blk.9.attn_v.weight",), ENTRIES) == (
            "blk.9.attn_v.weight",
        )

    def test_the_quantizers_substring_rule_applies(self) -> None:
        # `it->first.find(name)` searches rather than anchoring, so a
        # partial name matches every entry that carries it. ADR-0023
        # owns that over-deletion and this check must not refuse it.
        assert unmatched_exclusions(("attn_v",), ENTRIES) == ()

    def test_a_repeated_unmatched_name_reports_once(self) -> None:
        names = ("blk.9.attn_v.weight", "blk.9.attn_v.weight")
        assert unmatched_exclusions(names, ENTRIES) == ("blk.9.attn_v.weight",)

    def test_the_report_keeps_recipe_order(self) -> None:
        names = ("blk.9.attn_v.weight", "blk.1.attn_v.weight", "blk.8.attn_v.weight")
        assert unmatched_exclusions(names, ENTRIES) == (
            "blk.9.attn_v.weight",
            "blk.8.attn_v.weight",
        )

    def test_no_exclusions_report_nothing(self) -> None:
        assert unmatched_exclusions((), ENTRIES) == ()

    def test_an_empty_matrix_reports_every_exclusion(self) -> None:
        assert unmatched_exclusions(("blk.0.attn_v.weight",), ()) == (
            "blk.0.attn_v.weight",
        )


class TestCheckExclusionMatch:
    def test_no_exclusions_reads_no_file(self, tmp_path: Path) -> None:
        # The path does not exist. A read would raise before the
        # assertion, so this pins the early return.
        check_exclusion_match((), tmp_path / "absent.imatrix.gguf")
