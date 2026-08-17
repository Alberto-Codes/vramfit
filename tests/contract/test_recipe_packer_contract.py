"""RecipePacker contract: the llama.cpp adapter and the memory fake agree.

The real side runs the true adapter code — argument construction,
subprocess handling, error translation — against stub tools that
stand in for the llama.cpp binaries, so the suite stays hermetic
(ADR-0009). The type mapping itself is shared pure code, proven in
its own unit suite. The stubs record their argv, so the real-only
tests below the shared contract pin the exact command lines. Those
tests also pin how the adapter reads the quantizer's output. Both
are seams the verified fake cannot reach, because the fake receives
structured values and never parses a stream.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from tests.fakes import (
    MemoryRecipePacker,
    decoder_imatrix_entry_names,
    decoder_tensor_names,
)
from vramfit.adapters.outbound.gguf import exclusion_match, override_match
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker, TypeFallbackError
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.model import Assignment, PlanMeta, ProtectedTensor, Recipe
from vramfit.domain.pack import TypeOverride
from vramfit.ports.outbound import RecipePacker

pytestmark = pytest.mark.contract

BASE_BYTES = 1_000
PACKED_BYTES = 500

_CONVERT_STUB = f"""\
import json, sys

with open({{argv_log!r}}, "w") as log:
    json.dump(sys.argv[1:], log)
out = sys.argv[sys.argv.index("--outfile") + 1]
with open(out, "wb") as handle:
    handle.write(b"G" * {BASE_BYTES})
"""

_QUANTIZE_STUB = f"""\
#!/usr/bin/env python3
import json, sys

with open({{argv_log!r}}, "w") as log:
    json.dump(sys.argv[1:], log)
print({{extra_line!r}})
with open(sys.argv[-3], "wb") as handle:
    handle.write(b"Q" * {PACKED_BYTES})
"""

UNCOVERED_WARNING = (
    "====== llama_model_quantize_impl: did not find weights for token_embd.weight"
)

# The quantizer reports an excluded tensor's dropped row with the same
# warning — the adapter must file it as intentional (ADR-0023).
EXCLUDED_MISS_WARNING = (
    "====== llama_model_quantize_impl: did not find weights for blk.0.attn_v.weight"
)

# The same miss warning after `run_tool` replaced a byte it could not
# decode. U+FFFD is not whitespace, so `_IMATRIX_MISS` captures it as
# part of the tensor name (#252).
UNDECODABLE_MISS_WARNING = (
    "====== llama_model_quantize_impl: did not find weights for token_embd.wei�ght"
)

SECOND_UNDECODABLE_MISS_WARNING = (
    "====== llama_model_quantize_impl: did not find weights for blk.3.ffn_up�.weight"
)

# The same damage on a name the recipe excludes. An exclusion is an
# exact string, so the ADR-0023 discount cannot match a damaged name.
UNDECODABLE_EXCLUDED_MISS_WARNING = (
    "====== llama_model_quantize_impl: did not find weights for blk.0.attn_�.weight"
)

# The zero-exit type-fallback warning pair, one merged output line
# (`tensor_type_fallback`, llama.cpp src/llama-quant.cpp): the ncols
# report, then the substituted type (ADR-0028 decision 3).
FALLBACK_WARNING = (
    "warning: blk.1.ffn_up_exps.weight            - ncols   2688 not "
    "divisible by 256 (required for type    q3_K) -> falling back to    q4_0"
)
FALLBACK_REWRITE = ("blk.1.ffn_up_exps.weight", "q3_K", "q4_0")

# The rare second shape: the fallback type also misses the row, so
# the quantizer interjects its F16 warning between the pair's halves
# (llama.cpp src/llama-quant.cpp, the unusual-shape branch).
FALLBACK_WARNING_F16 = (
    "warning: blk.2.ssm_in.weight                 - ncols   1857 not "
    "divisible by 256 (required for type    q3_K) "
    "(WARNING: must use F16 due to unusual shape) -> falling back to     f16"
)
FALLBACK_REWRITE_F16 = ("blk.2.ssm_in.weight", "q3_K", "f16")

_FAILING_STUB = """\
#!/usr/bin/env python3
import sys

sys.stderr.write("stub tool exploded\\n")
sys.exit(3)
"""

_SILENT_STUB = """\
#!/usr/bin/env python3
"""


def sample_pack_recipe() -> Recipe:
    return Recipe(
        model_id="test/model",
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=3_000,
            predicted_total_bytes=2_500,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group="model.embed_tokens", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="lm_head", bits=4, bytes=500, damage=0.002),
            Assignment(group="model.layers.0", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="model.layers.1", bits=4, bytes=500, damage=0.01),
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


def stack_pack_recipe() -> Recipe:
    """A `--group-by stack` recipe shaped like the Nemotron target.

    Every name is one the target's checkpoint really carries: the
    embedding at `backbone.embeddings`, layers at
    `backbone.layers.<n>`, and one layer's routed experts fused into
    two stacks (#159, #160, #161). No name here mapped under the
    pre-#180 backend, the embedding included.
    """
    return replace(
        sample_pack_recipe(),
        assignments=(
            Assignment(group="backbone.embeddings", bits=8, bytes=1_000, damage=0.001),
            Assignment(
                group="backbone.layers.1.mixer.experts.up_proj",
                bits=4,
                bytes=900,
                damage=0.01,
            ),
            Assignment(
                group="backbone.layers.1.mixer.experts.down_proj",
                bits=2,
                bytes=900,
                damage=0.03,
            ),
        ),
    )


def excluded_pack_recipe() -> Recipe:
    base = sample_pack_recipe()
    return replace(
        base,
        plan=replace(
            base.plan,
            protections={"*.self_attn.v_proj.weight": 5},
            imatrix_exclusions=("model.layers.0.*",),
        ),
        protected_tensors=(
            ProtectedTensor(
                "model.layers.0.self_attn.v_proj.weight", 5, exclude_imatrix=True
            ),
        ),
    )


def _write_stub(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o700)
    return path


def _real_packer(  # noqa: PLR0913 - the contract fixture surface: one flag per stub behavior
    tmp_path: Path,
    *,
    fail_stage: Literal["convert", "quantize"] | None = None,
    base_exists: bool = False,
    silent_stage: str | None = None,
    with_imatrix: bool = False,
    with_uncovered: bool = False,
    with_excluded_miss: bool = False,
    with_type_fallback: bool = False,
    with_f16_fallback: bool = False,
    with_undecodable_miss: bool = False,
    with_second_undecodable_miss: bool = False,
    with_undecodable_excluded_miss: bool = False,
    with_unreached_exclusion: bool = False,
    with_unmatched_override: bool = False,
) -> RecipePacker:
    # These two configure the fake alone. The real adapter reads both
    # name lists through the module seams the conftest fixtures patch,
    # so a test wanting either refusal re-patches the seam and passes
    # the flag for the other side.
    del with_unreached_exclusion, with_unmatched_override

    # The undecodable-name flags have no fake counterpart. The fake
    # receives structured names and never decodes a stream, so the
    # damaged-name case is real-adapter-only (#252).
    def stub_body(stage: str, template: str) -> str:
        if fail_stage == stage:
            return _FAILING_STUB
        if silent_stage == stage:
            return _SILENT_STUB
        lines = []
        if with_uncovered:
            lines.append(UNCOVERED_WARNING)
        if with_undecodable_miss:
            lines.append(UNDECODABLE_MISS_WARNING)
        if with_second_undecodable_miss:
            lines.append(SECOND_UNDECODABLE_MISS_WARNING)
        if with_undecodable_excluded_miss:
            lines.append(UNDECODABLE_EXCLUDED_MISS_WARNING)
        if with_excluded_miss:
            lines.append(EXCLUDED_MISS_WARNING)
        if with_type_fallback:
            lines.append(FALLBACK_WARNING)
        if with_f16_fallback:
            lines.append(FALLBACK_WARNING_F16)
        return template.format(
            argv_log=str(tmp_path / f"{stage}-argv.json"),
            extra_line="\n".join(lines),
        )

    convert = _write_stub(tmp_path / "convert.py", stub_body("convert", _CONVERT_STUB))
    quantize = _write_stub(
        tmp_path / "llama-quantize", stub_body("quantize", _QUANTIZE_STUB)
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    if base_exists:
        (tmp_path / "base.gguf").write_bytes(b"G" * BASE_BYTES)
    return LlamaCppPacker(
        model_dir=model_dir,
        base_gguf=tmp_path / "base.gguf",
        out_path=tmp_path / "out.gguf",
        convert_script=convert,
        quantize_bin=quantize,
        python_bin=Path(sys.executable),
        threads=1,
        imatrix=tmp_path / "imatrix.gguf" if with_imatrix else None,
    )


# The entries a matrix carrying no row for the excluded tensor holds.
# `excluded_pack_recipe` excludes `blk.0.attn_v.weight`, so a list
# starting at layer 1 reaches it with nothing.
UNREACHED_ENTRY_NAMES = ("blk.1.attn_v.weight",)
# A base GGUF the sample recipe's layer overrides do not address.
UNMATCHED_BASE_NAMES = ("token_embd.weight", "blk.9.attn_v.weight")


def _entry_names(*, with_imatrix: bool, unreached: bool) -> tuple[str, ...] | None:
    """Pick the matrix entry names the fake answers from."""
    if not with_imatrix:
        return None
    return UNREACHED_ENTRY_NAMES if unreached else decoder_imatrix_entry_names()


def _fake_packer(  # noqa: PLR0913 - mirrors _real_packer's fixture surface
    tmp_path: Path,
    *,
    fail_stage: Literal["convert", "quantize"] | None = None,
    base_exists: bool = False,
    silent_stage: str | None = None,
    with_imatrix: bool = False,
    with_uncovered: bool = False,
    with_excluded_miss: bool = False,
    with_type_fallback: bool = False,
    with_f16_fallback: bool = False,
    with_unreached_exclusion: bool = False,
    with_unmatched_override: bool = False,
) -> RecipePacker:
    uncovered = ("token_embd.weight",) if with_uncovered else ()
    if with_excluded_miss:
        uncovered += ("blk.0.attn_v.weight",)
    fallbacks = (FALLBACK_REWRITE,) if with_type_fallback else ()
    if with_f16_fallback:
        fallbacks += (FALLBACK_REWRITE_F16,)
    return MemoryRecipePacker(
        base_bytes=BASE_BYTES,
        packed_bytes=PACKED_BYTES,
        fail_stage=fail_stage,
        has_base=base_exists,
        imatrix=str(tmp_path / "imatrix.gguf") if with_imatrix else None,
        imatrix_uncovered=uncovered,
        type_fallbacks=fallbacks,
        # The same names `base_gguf_names` serves the real adapter, so
        # both sides run the #303 refusal and the #307 report over one
        # tensor list. The matrix's entries are the narrower list the
        # `imatrix_entry_names` fixture serves, for the same reason.
        base_tensor_names=(
            UNMATCHED_BASE_NAMES if with_unmatched_override else decoder_tensor_names()
        ),
        imatrix_entry_names=_entry_names(
            with_imatrix=with_imatrix, unreached=with_unreached_exclusion
        ),
    )


@pytest.mark.parametrize(
    "build", [_real_packer, _fake_packer], ids=["real-subprocess", "fake-memory"]
)
@pytest.mark.usefixtures("base_gguf_names", "imatrix_entry_names")
class TestRecipePackerContract:
    def test_convert_returns_the_base_size(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path)

        assert packer.convert() == BASE_BYTES

    def test_convert_twice_returns_the_same_size(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path)

        assert packer.convert() == packer.convert()

    def test_convert_with_existing_base_skips_the_broken_tool(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, fail_stage="convert", base_exists=True)

        assert packer.convert() == BASE_BYTES

    def test_pack_after_convert_reports_the_real_packed_size(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.packed_bytes == PACKED_BYTES

    def test_pack_carries_the_shared_type_mapping(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.base_type == "Q4_K_S"
        assert result.token_embedding_type == "q8_0"  # noqa: S105 - a ggml type name, not a secret
        assert result.output_tensor_type == "q4_k"
        assert result.overrides == (
            TypeOverride(pattern=r"blk\.0\.", quant_type="q8_0"),
            TypeOverride(pattern=r"blk\.1\.", quant_type="q4_k"),
        )

    def test_pack_carries_the_stack_type_mapping(self, build, tmp_path) -> None:
        # A `--group-by stack` recipe for the Nemotron target packs:
        # the layer index derives from `backbone.layers.<n>`, and
        # each routed-expert stack addresses its fused tensor (#180).
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(stack_pack_recipe())

        assert result.overrides == (
            TypeOverride(pattern=r"blk\.1\.ffn_up_exps\.", quant_type="q4_0"),
            TypeOverride(pattern=r"blk\.1\.ffn_down_exps\.", quant_type="q2_0"),
        )

    def test_pack_with_a_type_fallback_warning_raises_with_the_rewrites(
        self, build, tmp_path
    ) -> None:
        # The quantizer substitutes a type on a zero exit, so the
        # artifact ignores the recipe. The pack halts and keeps the
        # file — never record-and-continue (ADR-0028 decision 3).
        packer: RecipePacker = build(tmp_path, with_type_fallback=True)
        packer.convert()

        with pytest.raises(TypeFallbackError) as caught:
            packer.pack(sample_pack_recipe())

        assert caught.value.rewritten == (FALLBACK_REWRITE,)
        assert "kept at" in str(caught.value)

    def test_pack_with_two_fallback_warnings_carries_both_rewrites_in_order(
        self, build, tmp_path
    ) -> None:
        # Two rewrites in one run, the second through the quantizer's
        # F16 interjection, beside an imatrix miss. The halt wins
        # over the ADR-0016 record-and-continue path, and the payload
        # keeps every triple in output order (ADR-0028 decision 3).
        packer: RecipePacker = build(
            tmp_path,
            with_imatrix=True,
            with_uncovered=True,
            with_type_fallback=True,
            with_f16_fallback=True,
        )
        packer.convert()

        with pytest.raises(TypeFallbackError) as caught:
            packer.pack(sample_pack_recipe())

        assert caught.value.rewritten == (FALLBACK_REWRITE, FALLBACK_REWRITE_F16)

    def test_pack_stack_recipe_binds_the_nemotron_embedding_group(
        self, build, tmp_path
    ) -> None:
        # The target names its embedding `backbone.embeddings`, not
        # `model.embed_tokens`. Missing that name refuses the whole
        # recipe, because `--pure` would drop the embedding to the
        # floor otherwise (#180).
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(stack_pack_recipe())

        assert result.token_embedding_type == "q8_0"  # noqa: S105 - a ggml type name, not a secret

    def test_pack_two_layer_stacks_raises_pack_error_naming_both(
        self, build, tmp_path
    ) -> None:
        # The target carries `mtp.layers.<n>` beside
        # `backbone.layers.<n>`. Both would map to `blk.<n>.`, and
        # the quantizer applies the first match, so the second
        # assignment would vanish. Refuse instead (#183).
        packer: RecipePacker = build(tmp_path, base_exists=True)
        recipe = replace(
            sample_pack_recipe(),
            assignments=(
                Assignment(group="backbone.layers.0", bits=8, bytes=1_000, damage=0.01),
                Assignment(group="mtp.layers.0", bits=4, bytes=500, damage=0.02),
            ),
        )

        with pytest.raises(PackError, match="two layer stacks"):
            packer.pack(recipe)

    def test_pack_unmappable_group_raises_pack_error_naming_it(
        self, build, tmp_path
    ) -> None:
        # The Mamba mixer projection has no GGUF class mapping. The
        # backend refuses by name rather than guessing a tensor.
        packer: RecipePacker = build(tmp_path, base_exists=True)
        recipe = replace(
            sample_pack_recipe(),
            assignments=(
                Assignment(
                    group="backbone.layers.0.mixer.in_proj",
                    bits=4,
                    bytes=500,
                    damage=0.01,
                ),
            ),
        )

        with pytest.raises(PackError, match=r"backbone\.layers\.0\.mixer\.in_proj"):
            packer.pack(recipe)

    def test_pack_without_imatrix_records_none(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_path is None

    def test_pack_with_imatrix_records_the_path(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, with_imatrix=True)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_path == str(tmp_path / "imatrix.gguf")

    def test_pack_with_imatrix_records_uncovered_tensors(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, with_imatrix=True, with_uncovered=True)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_uncovered == ("token_embd.weight",)

    def test_pack_without_imatrix_reports_no_uncovered_tensors(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, with_uncovered=True)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_uncovered == ()

    def test_pack_with_imatrix_records_excluded_tensors(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, with_imatrix=True)
        packer.convert()

        result = packer.pack(excluded_pack_recipe())

        assert result.imatrix_excluded == ("blk.0.attn_v.weight",)

    def test_pack_without_imatrix_records_no_excluded_tensors(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(excluded_pack_recipe())

        assert result.imatrix_excluded == ()

    def test_pack_excluded_miss_not_reported_uncovered(self, build, tmp_path) -> None:
        # The dropped row surfaces as a quantizer miss — intentional,
        # so it must not read as a coverage gap (ADR-0023).
        packer: RecipePacker = build(
            tmp_path, with_imatrix=True, with_uncovered=True, with_excluded_miss=True
        )
        packer.convert()

        result = packer.pack(excluded_pack_recipe())

        assert result.imatrix_uncovered == ("token_embd.weight",)
        assert result.imatrix_excluded == ("blk.0.attn_v.weight",)

    def test_pack_exclusion_reaching_no_imatrix_row_refuses(
        self, build, tmp_path, monkeypatch
    ) -> None:
        # ADR-0009 wants both sides to raise the same error for the
        # same recipe. The real adapter answers from the module seam
        # and the fake from its field, so both take the same names
        # (#309).
        monkeypatch.setattr(
            exclusion_match,
            "imatrix_entry_names",
            lambda _: UNREACHED_ENTRY_NAMES,
        )
        packer: RecipePacker = build(
            tmp_path, with_imatrix=True, with_unreached_exclusion=True
        )
        packer.convert()

        with pytest.raises(PackError) as exc:
            packer.pack(excluded_pack_recipe())

        message = str(exc.value)
        assert "carries no row for 1 of 1 recipe exclusions" in message
        assert '"blk.0.attn_v.weight"' in message
        # The remedy is what the operator acts on, so both sides
        # carry it rather than a bare summary.
        assert "Check the recipe's protected tensors" in message

    def test_pack_failing_both_checks_reports_the_override_refusal(
        self, build, tmp_path, monkeypatch
    ) -> None:
        # The base-GGUF read runs first, so a recipe broken both ways
        # reports the mapping error (#303 before #309).
        monkeypatch.setattr(
            override_match,
            "base_tensor_names",
            lambda _: UNMATCHED_BASE_NAMES,
        )
        monkeypatch.setattr(
            exclusion_match,
            "imatrix_entry_names",
            lambda _: UNREACHED_ENTRY_NAMES,
        )
        packer: RecipePacker = build(
            tmp_path,
            with_imatrix=True,
            with_unreached_exclusion=True,
            with_unmatched_override=True,
        )
        packer.convert()

        with pytest.raises(PackError, match="carries no tensor for"):
            packer.pack(excluded_pack_recipe())

    def test_pack_records_the_layers_no_override_reached(self, build, tmp_path) -> None:
        # The sample recipe addresses layers 0 and 1. Both sides read
        # a 64-layer decoder, so 62 layers took the floor with no
        # word from the quantizer (#307).
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.floored_layers == tuple(f"blk.{index}." for index in range(2, 64))

    def test_pack_counts_a_stack_override_as_covering_its_layer(
        self, build, tmp_path
    ) -> None:
        # The stack recipe addresses layer 1's two expert stacks and
        # nothing else. Those overrides reach two tensors of that
        # layer, so layer 1 is covered and the rest are not. Reporting
        # per tensor instead would name every attention and dense
        # tensor of a layer an expert-stack recipe addresses on
        # purpose.
        packer: RecipePacker = build(tmp_path)
        packer.convert()

        result = packer.pack(stack_pack_recipe())

        assert "blk.1." not in result.floored_layers
        assert "blk.0." in result.floored_layers

    def test_pack_llama_cpp_recipe_is_accepted(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, base_exists=True)

        result = packer.pack(replace(sample_pack_recipe(), runtime="llama.cpp"))

        assert result.packed_bytes == PACKED_BYTES

    def test_pack_foreign_runtime_recipe_raises_pack_error(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, base_exists=True)
        recipe = replace(sample_pack_recipe(), runtime="vllm")

        with pytest.raises(PackError, match=r"packs for llama\.cpp"):
            packer.pack(recipe)

    def test_pack_without_convert_raises_pack_error(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path)

        with pytest.raises(PackError, match="run convert first"):
            packer.pack(sample_pack_recipe())

    def test_convert_failure_raises_pack_error_with_exit_code(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, fail_stage="convert")

        with pytest.raises(PackError, match="convert failed with exit code 3"):
            packer.convert()

    def test_quantize_failure_raises_pack_error_with_exit_code(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, fail_stage="quantize")
        packer.convert()

        with pytest.raises(PackError, match="quantize failed with exit code 3"):
            packer.pack(sample_pack_recipe())


@pytest.mark.usefixtures("base_gguf_names", "imatrix_entry_names")
class TestLlamaCppCommandLines:
    """Real-adapter behavior the fake structurally cannot cover.

    The argv contracts below pin the exact command lines. The
    damaged-name tests pin how the adapter reads a stream the fake
    never sees, because the fake receives structured names (#252).
    """

    def test_pack_with_an_undecodable_miss_name_raises_pack_error(
        self, tmp_path
    ) -> None:
        # `run_tool` replaces an undecodable byte with U+FFFD (#247),
        # and the miss capture takes it as part of the tensor name.
        # Recording a name nobody read as coverage fact is the defect,
        # and dropping it silently would hide a real miss (ADR-0016).
        packer = _real_packer(tmp_path, with_imatrix=True, with_undecodable_miss=True)
        packer.convert()

        with pytest.raises(PackError, match="could not decode"):
            packer.pack(sample_pack_recipe())

    def test_pack_with_an_undecodable_miss_name_names_it_and_the_kept_file(
        self, tmp_path
    ) -> None:
        packer = _real_packer(tmp_path, with_imatrix=True, with_undecodable_miss=True)
        packer.convert()

        with pytest.raises(PackError) as caught:
            packer.pack(sample_pack_recipe())

        # The message escapes the replacement character, so a viewer
        # that cannot render the glyph still reads the byte.
        assert r"token_embd.wei\ufffdght" in str(caught.value)
        assert str(tmp_path / "out.gguf") in str(caught.value)

    def test_pack_with_an_undecodable_miss_name_names_only_the_damaged_one(
        self, tmp_path
    ) -> None:
        # A clean miss beside a damaged one must not reach the record,
        # and must not be blamed in the halt either.
        packer = _real_packer(
            tmp_path, with_imatrix=True, with_uncovered=True, with_undecodable_miss=True
        )
        packer.convert()

        with pytest.raises(PackError) as caught:
            packer.pack(sample_pack_recipe())

        assert r"token_embd.wei\ufffdght" in str(caught.value)
        assert "1 imatrix-miss tensor name" in str(caught.value)
        assert "'token_embd.weight'" not in str(caught.value)

    def test_pack_with_two_undecodable_miss_names_counts_both(self, tmp_path) -> None:
        packer = _real_packer(
            tmp_path,
            with_imatrix=True,
            with_undecodable_miss=True,
            with_second_undecodable_miss=True,
        )
        packer.convert()

        with pytest.raises(PackError, match="2 imatrix-miss tensor names"):
            packer.pack(sample_pack_recipe())

    def test_pack_with_an_undecodable_excluded_name_still_halts(self, tmp_path) -> None:
        # An exclusion is an exact string, so a damaged name can never
        # match one. The adapter cannot tell an intentional miss from
        # a real gap here, so it refuses rather than discount blindly
        # (ADR-0023).
        packer = _real_packer(
            tmp_path, with_imatrix=True, with_undecodable_excluded_miss=True
        )
        packer.convert()

        with pytest.raises(PackError, match="could not decode"):
            packer.pack(excluded_pack_recipe())

    def test_pack_without_imatrix_ignores_an_undecodable_miss_name(
        self, tmp_path
    ) -> None:
        # Without a matrix there is no coverage record to corrupt, so
        # the scan never runs and the pack completes (ADR-0016).
        packer = _real_packer(tmp_path, with_undecodable_miss=True)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_uncovered == ()

    def test_convert_argv_requests_an_f16_outfile(self, tmp_path) -> None:
        packer = _real_packer(tmp_path)

        packer.convert()

        argv = json.loads((tmp_path / "convert-argv.json").read_text())
        assert argv[0] == str(tmp_path / "model")
        assert argv[argv.index("--outfile") + 1] == str(tmp_path / "base.gguf")
        assert argv[argv.index("--outtype") + 1] == "f16"

    def test_quantize_argv_carries_the_full_type_mapping(self, tmp_path) -> None:
        packer = _real_packer(tmp_path)
        packer.convert()

        packer.pack(sample_pack_recipe())

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert argv[0] == "--pure"
        assert argv[argv.index("--token-embedding-type") + 1] == "q8_0"
        assert argv[argv.index("--output-tensor-type") + 1] == "q4_k"
        pairs = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--tensor-type"]
        assert pairs == [r"blk\.0\.=q8_0", r"blk\.1\.=q4_k"]
        assert argv[-4:] == [
            str(tmp_path / "base.gguf"),
            str(tmp_path / "out.gguf"),
            "Q4_K_S",
            "1",
        ]

    def test_quantize_argv_carries_the_stack_patterns(self, tmp_path) -> None:
        packer = _real_packer(tmp_path)
        packer.convert()

        packer.pack(stack_pack_recipe())

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        pairs = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--tensor-type"]
        assert pairs == [
            r"blk\.1\.ffn_up_exps\.=q4_0",
            r"blk\.1\.ffn_down_exps\.=q2_0",
        ]

    def test_quantize_argv_orders_stack_patterns_before_layer_patterns(
        self, tmp_path
    ) -> None:
        # llama-quantize applies the first matching pattern, and
        # `blk\.1\.` also matches `blk.1.ffn_up_exps.weight`. A stack
        # pattern placed after the layer pattern would never apply.
        packer = _real_packer(tmp_path)
        packer.convert()
        recipe = replace(
            sample_pack_recipe(),
            assignments=(
                Assignment(group="model.layers.1", bits=8, bytes=1_000, damage=0.001),
                Assignment(
                    group="model.layers.1.mlp.experts.up_proj",
                    bits=2,
                    bytes=900,
                    damage=0.03,
                ),
            ),
        )

        packer.pack(recipe)

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        pairs = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--tensor-type"]
        assert pairs == [r"blk\.1\.ffn_up_exps\.=q2_0", r"blk\.1\.=q8_0"]

    def test_quantize_argv_orders_protection_then_stack_then_layer(
        self, tmp_path
    ) -> None:
        # ADR-0012 decision 2 rules the three-way priority. The
        # quantizer applies the first matching pattern, and each
        # pattern here matches a superset of the next.
        packer = _real_packer(tmp_path)
        packer.convert()
        recipe = replace(
            sample_pack_recipe(),
            plan=replace(
                sample_pack_recipe().plan,
                protections={"*.self_attn.v_proj.weight": 5},
            ),
            assignments=(
                Assignment(group="model.layers.1", bits=8, bytes=1_000, damage=0.001),
                Assignment(
                    group="model.layers.1.mlp.experts.up_proj",
                    bits=2,
                    bytes=900,
                    damage=0.03,
                ),
            ),
            protected_tensors=(
                ProtectedTensor("model.layers.1.self_attn.v_proj.weight", 5),
            ),
        )

        packer.pack(recipe)

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        pairs = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--tensor-type"]
        assert pairs == [
            r"blk\.1\.attn_v\.=q5_k",
            r"blk\.1\.ffn_up_exps\.=q2_0",
            r"blk\.1\.=q8_0",
        ]

    def test_quantize_argv_with_imatrix_carries_the_flag(self, tmp_path) -> None:
        packer = _real_packer(tmp_path, with_imatrix=True)
        packer.convert()

        packer.pack(sample_pack_recipe())

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert argv[argv.index("--imatrix") + 1] == str(tmp_path / "imatrix.gguf")

    def test_quantize_argv_without_imatrix_omits_the_flag(self, tmp_path) -> None:
        packer = _real_packer(tmp_path)
        packer.convert()

        packer.pack(sample_pack_recipe())

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert "--imatrix" not in argv

    def test_quantize_argv_with_exclusions_carries_the_flags(self, tmp_path) -> None:
        packer = _real_packer(tmp_path, with_imatrix=True)
        packer.convert()

        packer.pack(excluded_pack_recipe())

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert argv[argv.index("--exclude-weights") + 1] == "blk.0.attn_v.weight"

    def test_quantize_argv_without_imatrix_omits_exclude_weights(
        self, tmp_path
    ) -> None:
        packer = _real_packer(tmp_path)
        packer.convert()

        packer.pack(excluded_pack_recipe())

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert "--exclude-weights" not in argv

    def test_quantize_argv_without_lm_head_pins_output_to_the_embedding(
        self, tmp_path
    ) -> None:
        packer = _real_packer(tmp_path)
        packer.convert()
        recipe = sample_pack_recipe()
        tied = replace(
            recipe,
            assignments=tuple(a for a in recipe.assignments if a.group != "lm_head"),
        )

        packer.pack(tied)

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert argv[argv.index("--token-embedding-type") + 1] == "q8_0"
        assert argv[argv.index("--output-tensor-type") + 1] == "q8_0"

    def test_quantize_argv_without_flag_groups_omits_both_flags(self, tmp_path) -> None:
        packer = _real_packer(tmp_path)
        packer.convert()
        recipe = sample_pack_recipe()
        layers_only = replace(
            recipe,
            assignments=tuple(
                a
                for a in recipe.assignments
                if a.group not in ("lm_head", "model.embed_tokens")
            ),
        )

        packer.pack(layers_only)

        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert "--token-embedding-type" not in argv
        assert "--output-tensor-type" not in argv

    def test_convert_writing_no_file_raises_pack_error(self, tmp_path) -> None:
        packer = _real_packer(tmp_path, silent_stage="convert")

        with pytest.raises(PackError, match="cannot be inspected"):
            packer.convert()

    def test_quantize_writing_no_file_raises_pack_error(self, tmp_path) -> None:
        packer = _real_packer(tmp_path, silent_stage="quantize")
        packer.convert()

        with pytest.raises(PackError, match="cannot be inspected"):
            packer.pack(sample_pack_recipe())

    def test_override_matching_no_base_tensor_refuses_before_the_quantizer(
        self, tmp_path, monkeypatch
    ) -> None:
        # The base GGUF carries layer 9 alone, so both of the sample
        # recipe's layer overrides address an absent index. The
        # quantizer would apply neither and exit 0 (#303).
        monkeypatch.setattr(
            override_match,
            "base_tensor_names",
            lambda _: ("token_embd.weight", "blk.9.attn_v.weight"),
        )
        packer = _real_packer(tmp_path)
        packer.convert()

        with pytest.raises(PackError, match="no tensor for 2 of 2"):
            packer.pack(sample_pack_recipe())

        # The refusal runs before the quantizer, so no file survives.
        assert not (tmp_path / "out.gguf").exists()

    def test_exclusion_reaching_no_imatrix_row_refuses_before_the_quantizer(
        self, tmp_path, monkeypatch
    ) -> None:
        # The matrix prices layer 1 alone, so the recipe's exclusion
        # of layer 0 erases no row. The quantizer would exit 0 and
        # report nothing (#309).
        monkeypatch.setattr(
            exclusion_match,
            "imatrix_entry_names",
            lambda _: ("blk.1.attn_v.weight",),
        )
        packer = _real_packer(tmp_path, with_imatrix=True)
        packer.convert()

        with pytest.raises(PackError, match="carries no row for 1 of 1"):
            packer.pack(excluded_pack_recipe())

        assert not (tmp_path / "out.gguf").exists()

    def test_a_matrix_less_pack_reads_no_imatrix_for_exclusions(
        self, tmp_path, monkeypatch
    ) -> None:
        # Decision 4 makes an exclusion inert without a matrix, so
        # the check must not run and refuse one (ADR-0023).
        def refuse(_: Path) -> tuple[str, ...]:
            raise AssertionError("a matrix-less pack read an imatrix")

        monkeypatch.setattr(exclusion_match, "imatrix_entry_names", refuse)
        packer = _real_packer(tmp_path)
        packer.convert()

        result = packer.pack(excluded_pack_recipe())

        assert result.imatrix_excluded == ()
        # The stub above proves no read happened. This proves why: the
        # command carries no exclusion to check.
        argv = json.loads((tmp_path / "quantize-argv.json").read_text())
        assert "--exclude-weights" not in argv
