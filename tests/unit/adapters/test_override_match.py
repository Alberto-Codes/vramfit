"""Override matching against the base GGUF's tensor names (#303, #306, #307).

The pure matching runs everywhere. The one function that opens a
GGUF is exercised through a stub here and against a written file in
the gguf-guarded case at the bottom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vramfit.adapters.outbound.gguf import override_match
from vramfit.adapters.outbound.gguf.override_match import (
    check_base_coverage,
    floored_layers,
    unmatched_flags,
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


class TestUnmatchedFlags:
    def test_both_flags_reaching_their_target_report_nothing(self) -> None:
        assert unmatched_flags(_DECODER_NAMES, embedding=True, output=True) == ()

    def test_no_output_weight_reports_the_output_flag_and_its_target(self) -> None:
        # The #306 case: a tied base GGUF carries no `output.weight`,
        # so a scanned lm_head group's flag binds nothing.
        names = ("token_embd.weight", "blk.0.attn_v.weight")
        assert unmatched_flags(names, embedding=True, output=True) == (
            ("--output-tensor-type", ("output.weight",)),
        )

    def test_no_embedding_tensor_reports_the_embedding_flag(self) -> None:
        names = ("blk.0.attn_v.weight", "output.weight")
        assert unmatched_flags(names, embedding=True, output=True) == (
            (
                "--token-embedding-type",
                ("token_embd.weight", "per_layer_token_embd.weight"),
            ),
        )

    def test_per_layer_embedding_alone_satisfies_the_embedding_flag(self) -> None:
        # llama.cpp's `tensor_name_match_token_embd` accepts either
        # name, so this check must accept either too.
        names = ("per_layer_token_embd.weight", "output.weight")
        assert unmatched_flags(names, embedding=True, output=True) == ()

    def test_unemitted_flags_report_nothing_on_an_empty_file(self) -> None:
        assert unmatched_flags((), embedding=False, output=False) == ()

    def test_an_unemitted_embedding_flag_is_not_held(self) -> None:
        # A recipe carrying an lm_head group and no embedding group
        # emits the output flag alone. The embedding tensor is then
        # nobody's target, so its absence refuses nothing.
        names = ("output.weight", "blk.0.attn_v.weight")
        assert unmatched_flags(names, embedding=False, output=True) == ()

    def test_an_unemitted_embedding_flag_leaves_the_output_flag_held(self) -> None:
        # The same recipe against a tied base GGUF. `--pure` keeps
        # llama-quant.cpp:452 dead, so the output flag applies nothing
        # and the head takes the floor.
        names = ("token_embd.weight", "blk.0.attn_v.weight")
        assert unmatched_flags(names, embedding=False, output=True) == (
            ("--output-tensor-type", ("output.weight",)),
        )

    def test_both_unmatched_report_the_embedding_first(self) -> None:
        unmatched = unmatched_flags((), embedding=True, output=True)
        assert [flag for flag, _ in unmatched] == [
            "--token-embedding-type",
            "--output-tensor-type",
        ]

    def test_uppercase_tensor_name_does_not_satisfy_a_flag(self) -> None:
        # The two flags bind through `std::strcmp`, which folds no
        # case. `unmatched_patterns` lower-cases because the tool
        # lower-cases a --tensor-type pattern. Neither applies here.
        names = ("TOKEN_EMBD.WEIGHT", "OUTPUT.WEIGHT")
        assert len(unmatched_flags(names, embedding=True, output=True)) == 2

    def test_a_name_carrying_the_target_does_not_satisfy_a_flag(self) -> None:
        # The comparison is equality and not a search, because the
        # tool compares whole names here.
        names = ("blk.0.output.weight", "v.token_embd.weight")
        assert len(unmatched_flags(names, embedding=True, output=True)) == 2


class TestFlooredLayers:
    def test_every_layer_reached_reports_nothing(self) -> None:
        overrides = (
            TypeOverride(r"blk\.0\.", "q4_k"),
            TypeOverride(r"blk\.1\.", "q4_k"),
            TypeOverride(r"blk\.11\.", "q4_k"),
        )
        assert floored_layers(overrides, _DECODER_NAMES) == ()

    def test_layer_no_override_reaches_reports_its_prefix(self) -> None:
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        assert floored_layers(overrides, _DECODER_NAMES) == ("blk.1.", "blk.11.")

    def test_mtp_block_beyond_the_scanned_range_reports(self) -> None:
        # #256 measured the published 30B builds carrying 48 expert
        # stacks: 46 backbone plus 2 under `blk.52`. A recipe scanned
        # over the backbone alone leaves that block at the floor.
        names = (
            "blk.0.ffn_down_exps.weight",
            "blk.1.ffn_down_exps.weight",
            "blk.52.ffn_down_exps.weight",
        )
        overrides = (
            TypeOverride(r"blk\.0\.ffn_down_exps\.", "q4_0"),
            TypeOverride(r"blk\.1\.ffn_down_exps\.", "q4_0"),
        )
        assert floored_layers(overrides, names) == ("blk.52.",)

    def test_one_reached_tensor_covers_the_whole_layer(self) -> None:
        # An expert-stack recipe addresses one tensor class per layer
        # on purpose. Reporting the rest would name every attention
        # and dense tensor in the model.
        names = ("blk.0.ffn_down_exps.weight", "blk.0.attn_v.weight")
        overrides = (TypeOverride(r"blk\.0\.ffn_down_exps\.", "q4_0"),)
        assert floored_layers(overrides, names) == ()

    def test_escaped_dot_does_not_cover_a_longer_index(self) -> None:
        # The sharpest edge of the coverage rule. `blk\.1\.` unescaped
        # would match `blk.11.attn_v.weight` through the wildcard dot
        # and mark layer 11 covered. Patterns arrive `re.escape`d, so
        # it must not.
        overrides = (TypeOverride(r"blk\.1\.", "q4_k"),)
        assert floored_layers(overrides, _DECODER_NAMES) == ("blk.0.", "blk.11.")

    def test_class_wide_pattern_covers_every_layer_carrying_it(self) -> None:
        # llama-quantize searches rather than anchors, so a caller
        # pattern naming a tensor class alone reaches every layer.
        overrides = (TypeOverride(r"attn_v\.", "q4_k"),)
        assert floored_layers(overrides, _DECODER_NAMES) == ()

    def test_repeated_tensor_name_does_not_double_report(self) -> None:
        names = ("blk.4.attn_v.weight", "blk.4.attn_v.weight")
        assert floored_layers((), names) == ("blk.4.",)

    def test_layers_report_in_index_order_not_file_order(self) -> None:
        names = ("blk.11.attn_v.weight", "blk.2.attn_v.weight")
        assert floored_layers((), names) == ("blk.2.", "blk.11.")

    def test_file_numbering_no_layer_reports_nothing(self) -> None:
        names = ("token_embd.weight", "output.weight")
        assert floored_layers((), names) == ()

    def test_foreign_root_layer_is_not_a_decoder_layer(self) -> None:
        # A vision tower numbers `v.blk.<n>.`. The prefix is anchored,
        # so that tower reports nothing here — #236 owns the root
        # question and this report must not pre-empt it.
        names = ("v.blk.0.attn_v.weight", "blk.0.attn_v.weight")
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        assert floored_layers(overrides, names) == ()

    def test_uncompilable_pattern_raises_pack_error(self) -> None:
        overrides = (TypeOverride("blk[", "q4_k"),)
        with pytest.raises(PackError, match="does not compile"):
            floored_layers(overrides, _DECODER_NAMES)


class TestMissingGgufPy:
    def test_missing_gguf_names_the_gguf_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No CI job runs the suite without gguf-py since the test job
        # gained the group, so this branch needs its own stub. Same
        # shape as the imatrix reader's suite.
        import builtins

        real_import = builtins.__import__

        def no_gguf(name, *args, **kwargs):
            if name == "gguf":
                raise ImportError("No module named 'gguf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_gguf)

        with pytest.raises(PackError, match="gguf extra"):
            override_match.base_tensor_names(Path("base.gguf"))


class TestCheckBaseCoverage:
    def test_empty_overrides_never_open_the_base_gguf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_: Path) -> tuple[str, ...]:
            raise AssertionError("the base GGUF must stay unopened")

        monkeypatch.setattr(override_match, "base_tensor_names", explode)
        assert check_base_coverage((), Path("base.gguf")) == ()

    def test_every_override_matching_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: _DECODER_NAMES
        )
        overrides = (
            TypeOverride(r"blk\.0\.", "q4_k"),
            TypeOverride(r"blk\.1\.", "q4_k"),
            TypeOverride(r"blk\.11\.", "q4_k"),
        )
        assert check_base_coverage(overrides, Path("base.gguf")) == ()

    def test_matching_overrides_still_report_an_unreached_layer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The #307 case: every override matches, so the #303 refusal
        # passes, and the file still carries layers the recipe never
        # addressed.
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: _DECODER_NAMES
        )
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        assert check_base_coverage(overrides, Path("base.gguf")) == (
            "blk.1.",
            "blk.11.",
        )

    def test_the_refusal_wins_over_the_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A recipe can fail both ways at once. The refusal is the
        # actionable one, and it costs no quantize run.
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: _DECODER_NAMES
        )
        overrides = (TypeOverride(r"blk\.99\.", "q4_k"),)
        with pytest.raises(PackError, match="no tensor for 1 of 1"):
            check_base_coverage(overrides, Path("base.gguf"))

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
            check_base_coverage(overrides, Path("model-f16.gguf"))
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
            check_base_coverage(overrides, Path("model-f16.gguf"))

    def test_scanned_head_against_a_tied_base_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #306's concrete case. The recipe carries an lm_head group,
        # so the pack emits --output-tensor-type, and the base GGUF
        # came from a conversion that tied the head. The flag binds
        # nothing, the quantizer exits 0, and the head lands at the
        # --pure floor while PackResult records the recipe's type.
        monkeypatch.setattr(
            override_match,
            "base_tensor_names",
            lambda _: ("token_embd.weight", "blk.0.attn_v.weight"),
        )
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        with pytest.raises(PackError) as caught:
            check_base_coverage(
                overrides,
                Path("model-f16.gguf"),
                embedding_flag=True,
                output_flag=True,
            )
        message = str(caught.value)
        assert "--output-tensor-type" in message
        assert "output.weight" in message
        assert "model-f16.gguf" in message
        assert "no target tensor for 1 dedicated flag" in message
        # The flag that does bind must stay out of the message.
        assert "--token-embedding-type" not in message

    def test_tied_fallback_against_a_tied_base_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The exempt case. Without an lm_head group the embedding
        # assignment drives the output flag, and ADR-0012 decision 2
        # states the flag never applies on a model that ties
        # embeddings. Refusing here would refuse a pack the record
        # sanctions, which is why `output_flag` is False.
        monkeypatch.setattr(
            override_match,
            "base_tensor_names",
            lambda _: ("token_embd.weight", "blk.0.attn_v.weight"),
        )
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        assert (
            check_base_coverage(
                overrides,
                Path("base.gguf"),
                embedding_flag=True,
                output_flag=False,
            )
            == ()
        )

    def test_the_pattern_refusal_wins_over_the_flag_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A recipe can fail both. The pattern mismatch is the coarser
        # one and the likelier cause, so it reports first.
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: ("blk.0.attn_v.weight",)
        )
        overrides = (TypeOverride(r"blk\.99\.", "q4_k"),)
        with pytest.raises(PackError, match="no tensor for 1 of 1"):
            check_base_coverage(
                overrides,
                Path("base.gguf"),
                embedding_flag=True,
                output_flag=True,
            )

    def test_an_emitted_flag_opens_a_base_gguf_no_override_would(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A recipe of embedding and lm_head groups alone drives no
        # pattern override. The flags still need holding, so the
        # empty-override shortcut must not skip the read.
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: ("blk.0.attn_v.weight",)
        )
        with pytest.raises(PackError, match="no target tensor for 2 dedicated flags"):
            check_base_coverage(
                (), Path("base.gguf"), embedding_flag=True, output_flag=True
            )

    def test_an_emitted_flag_alone_reports_no_floored_layer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The read happens for the flags, and `floored_layers` would
        # name every layer against an empty override set. #307 does
        # not cover a recipe that floors the whole model, so the
        # report stays empty.
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: _DECODER_NAMES
        )
        assert (
            check_base_coverage(
                (), Path("base.gguf"), embedding_flag=True, output_flag=True
            )
            == ()
        )

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
        check_base_coverage((TypeOverride(r"blk\.0\.", "q4_k"),), Path("base.gguf"))
