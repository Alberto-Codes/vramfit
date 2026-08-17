"""Override matching against the base GGUF's tensor names (#303).

The pure matching runs everywhere. The one function that opens a
GGUF is exercised through a stub here and against a written file in
the gguf-guarded case at the bottom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vramfit.adapters.outbound.gguf import override_match
from vramfit.adapters.outbound.gguf.override_match import (
    check_overrides_match,
    unmatched_patterns,
)
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import TypeOverride

pytestmark = pytest.mark.unit

_DECODER_NAMES = (
    "token_embd.weight",
    "blk.0.attn_v.weight",
    "blk.1.attn_v.weight",
    "blk.11.attn_v.weight",
    "output.weight",
)


class TestUnmatchedPatterns:
    def test_pattern_matching_a_tensor_reports_nothing(self) -> None:
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        assert unmatched_patterns(overrides, _DECODER_NAMES) == ()

    def test_pattern_matching_no_tensor_reports_itself(self) -> None:
        overrides = (TypeOverride(r"blk\.99\.", "q4_k"),)
        assert unmatched_patterns(overrides, _DECODER_NAMES) == (r"blk\.99\.",)

    def test_escaped_dot_does_not_reach_a_longer_index(self) -> None:
        # `blk.1.` unescaped would match `blk.11.` through the wildcard
        # dot. Only `blk.11.attn_v.weight` is present, so an escaped
        # `blk\.1\.` must report unmatched.
        overrides = (TypeOverride(r"blk\.1\.", "q4_k"),)
        names = ("blk.11.attn_v.weight",)
        assert unmatched_patterns(overrides, names) == (r"blk\.1\.",)

    def test_uppercase_pattern_lowercases_before_the_search(self) -> None:
        # quantize.cpp:332 lower-cases the pattern before it compiles.
        overrides = (TypeOverride(r"BLK\.0\.ATTN_V\.", "q4_k"),)
        assert unmatched_patterns(overrides, _DECODER_NAMES) == ()

    def test_pattern_searches_rather_than_anchors(self) -> None:
        # llama-quant.cpp:694 uses regex_search, so a mid-name pattern
        # matches.
        overrides = (TypeOverride(r"attn_v\.", "q4_k"),)
        assert unmatched_patterns(overrides, _DECODER_NAMES) == ()

    def test_repeated_unmatched_pattern_reports_once(self) -> None:
        overrides = (
            TypeOverride(r"blk\.99\.", "q4_k"),
            TypeOverride(r"blk\.99\.", "q8_0"),
        )
        assert unmatched_patterns(overrides, _DECODER_NAMES) == (r"blk\.99\.",)

    def test_unmatched_patterns_keep_override_order(self) -> None:
        overrides = (
            TypeOverride(r"blk\.98\.", "q4_k"),
            TypeOverride(r"blk\.0\.", "q4_k"),
            TypeOverride(r"blk\.99\.", "q4_k"),
        )
        assert unmatched_patterns(overrides, _DECODER_NAMES) == (
            r"blk\.98\.",
            r"blk\.99\.",
        )

    def test_empty_name_list_reports_every_pattern(self) -> None:
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        assert unmatched_patterns(overrides, ()) == (r"blk\.0\.",)

    def test_uncompilable_pattern_raises_pack_error(self) -> None:
        overrides = (TypeOverride("blk[", "q4_k"),)
        with pytest.raises(PackError, match="does not compile"):
            unmatched_patterns(overrides, _DECODER_NAMES)


class TestCheckOverridesMatch:
    def test_empty_overrides_never_open_the_base_gguf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_: Path) -> tuple[str, ...]:
            raise AssertionError("the base GGUF must stay unopened")

        monkeypatch.setattr(override_match, "base_tensor_names", explode)
        check_overrides_match((), Path("base.gguf"))

    def test_every_override_matching_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: _DECODER_NAMES
        )
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        check_overrides_match(overrides, Path("base.gguf"))

    def test_unmatched_override_names_the_pattern_and_the_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: _DECODER_NAMES
        )
        overrides = (
            TypeOverride(r"blk\.0\.", "q4_k"),
            TypeOverride(r"blk\.99\.", "q4_k"),
        )
        with pytest.raises(PackError) as caught:
            check_overrides_match(overrides, Path("model-f16.gguf"))
        message = str(caught.value)
        assert r"blk\.99\." in message
        assert r"blk\.0\." not in message
        assert "model-f16.gguf" in message
        assert "no tensor for 1 of 2 override patterns" in message

    def test_recipe_naming_absent_layers_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A recipe carrying more layers than the checkpoint, or one
        # packed against the wrong checkpoint: every override
        # addresses an index the file does not carry, so the
        # quantizer would apply none of them and exit 0.
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: ("blk.0.attn_v.weight",)
        )
        overrides = (
            TypeOverride(r"blk\.7\.", "q4_k"),
            TypeOverride(r"blk\.8\.", "q2_k"),
        )
        with pytest.raises(PackError, match="no tensor for 2 of 2"):
            check_overrides_match(overrides, Path("model-f16.gguf"))

    def test_prefixed_tensor_tree_still_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The quantizer searches rather than anchors, so `blk\.0\.`
        # matches `v.blk.0.attn_v.weight` as a substring. This check
        # therefore passes a foreign root whose GGUF names carry the
        # decoder's spelling inside them. #236 owns that case, and
        # this test pins the boundary so a later reader does not
        # mistake it for coverage.
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: ("v.blk.0.attn_v.weight",)
        )
        check_overrides_match((TypeOverride(r"blk\.0\.", "q4_k"),), Path("base.gguf"))
