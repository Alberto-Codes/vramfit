"""RecipePacker contract: the llama.cpp adapter and the memory fake agree.

The real side runs the true adapter code — argument construction,
subprocess handling, error translation — against stub tools that
stand in for the llama.cpp binaries, so the suite stays hermetic
(ADR-0009). The type mapping itself is shared pure code, proven in
its own unit suite. The stubs record their argv, so the real-only
tests below the shared contract pin the exact command lines — the
one seam the verified fake cannot reach.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from tests.fakes import MemoryImatrixCounts, MemoryRecipePacker
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.model import Assignment, PlanMeta, ProtectedTensor, Recipe
from vramfit.domain.pack import TypeOverride, ZeroCountExpert
from vramfit.ports.outbound import RecipePacker

pytestmark = pytest.mark.contract

BASE_BYTES = 1_000
PACKED_BYTES = 500

# Two experts of one stack the router never fired, plus one inside a
# stack a recipe excludes. No such expert exists on the real target
# and corpus (issue #162), so decision 5 only tests against a fake.
STARVED = (
    ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=57),
    ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=4),
    ZeroCountExpert(stack="blk.0.attn_v.weight", expert=1),
)

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
    "====== llama_tensor_get_wanted_type: did not find weights for token_embd.weight"
)

# The quantizer reports an excluded tensor's dropped row with the same
# warning — the adapter must file it as intentional (ADR-0023).
EXCLUDED_MISS_WARNING = (
    "====== llama_tensor_get_wanted_type: did not find weights for blk.0.attn_v.weight"
)

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


def _real_packer(
    tmp_path: Path,
    fail_stage: Literal["convert", "quantize"] | None = None,
    base_exists: bool = False,
    silent_stage: str | None = None,
    with_imatrix: bool = False,
    with_uncovered: bool = False,
    with_excluded_miss: bool = False,
    with_starved: bool = False,
) -> RecipePacker:
    def stub_body(stage: str, template: str) -> str:
        if fail_stage == stage:
            return _FAILING_STUB
        if silent_stage == stage:
            return _SILENT_STUB
        lines = []
        if with_uncovered:
            lines.append(UNCOVERED_WARNING)
        if with_excluded_miss:
            lines.append(EXCLUDED_MISS_WARNING)
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
        # The matrix itself is never written here. Its read is a
        # separate port with its own contract suite (ADR-0026
        # decision 5), so this suite injects the fake and keeps the
        # packer's own behavior hermetic.
        imatrix_counts=MemoryImatrixCounts(experts=STARVED if with_starved else ()),
    )


def _fake_packer(
    tmp_path: Path,
    fail_stage: Literal["convert", "quantize"] | None = None,
    base_exists: bool = False,
    silent_stage: str | None = None,
    with_imatrix: bool = False,
    with_uncovered: bool = False,
    with_excluded_miss: bool = False,
    with_starved: bool = False,
) -> RecipePacker:
    uncovered = ("token_embd.weight",) if with_uncovered else ()
    if with_excluded_miss:
        uncovered += ("blk.0.attn_v.weight",)
    return MemoryRecipePacker(
        base_bytes=BASE_BYTES,
        packed_bytes=PACKED_BYTES,
        fail_stage=fail_stage,
        has_base=base_exists,
        imatrix=str(tmp_path / "imatrix.gguf") if with_imatrix else None,
        imatrix_uncovered=uncovered,
        imatrix_zero_count_experts=STARVED if with_starved else (),
    )


@pytest.mark.parametrize(
    "build", [_real_packer, _fake_packer], ids=["real-subprocess", "fake-memory"]
)
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
            TypeOverride(pattern=r"blk\.1\.ffn_up_exps\.", quant_type="q4_k"),
            TypeOverride(pattern=r"blk\.1\.ffn_down_exps\.", quant_type="q2_k"),
        )

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

    def test_pack_with_imatrix_records_zero_count_experts(
        self, build, tmp_path
    ) -> None:
        # ADR-0026 decision 5. The quantizer warns for a missing
        # tensor and says nothing for an expert its stack covers at
        # a count of zero, so this field is the only report.
        packer: RecipePacker = build(tmp_path, with_imatrix=True, with_starved=True)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_zero_count_experts == (
            ZeroCountExpert(stack="blk.0.attn_v.weight", expert=1),
            ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=4),
            ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=57),
        )

    def test_pack_without_imatrix_reports_no_zero_count_experts(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, with_starved=True)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_zero_count_experts == ()

    def test_pack_zero_count_expert_in_excluded_stack_is_not_reported(
        self, build, tmp_path
    ) -> None:
        # An exclusion drops the stack's whole entry, so every
        # expert in it misses on purpose (ADR-0023). Reporting those
        # would bury the unintentional case this field exists for.
        packer: RecipePacker = build(tmp_path, with_imatrix=True, with_starved=True)
        packer.convert()

        result = packer.pack(excluded_pack_recipe())

        assert result.imatrix_excluded == ("blk.0.attn_v.weight",)
        assert result.imatrix_zero_count_experts == (
            ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=4),
            ZeroCountExpert(stack="blk.1.ffn_up_exps.weight", expert=57),
        )

    def test_pack_with_imatrix_and_no_starved_expert_reports_none(
        self, build, tmp_path
    ) -> None:
        # The measured case on the real target and corpus: 0 cells
        # of 2,944 carry a zero count (issue #162).
        packer: RecipePacker = build(tmp_path, with_imatrix=True)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.imatrix_zero_count_experts == ()

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


class TestLlamaCppCommandLines:
    """Real-adapter argv contracts the fake structurally cannot cover."""

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
            r"blk\.1\.ffn_up_exps\.=q4_k",
            r"blk\.1\.ffn_down_exps\.=q2_k",
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
        assert pairs == [r"blk\.1\.ffn_up_exps\.=q2_k", r"blk\.1\.=q8_0"]

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
            r"blk\.1\.ffn_up_exps\.=q2_k",
            r"blk\.1\.=q8_0",
        ]

    def test_unreadable_imatrix_halts_before_the_quantizer_runs(self, tmp_path) -> None:
        # A failed count read must reach the caller, never collapse
        # into an empty report — that is what a healthy matrix gives
        # (ADR-0026 decision 5). It must also refuse before the
        # quantize stage, not after a half-hour of work.
        source = MemoryImatrixCounts(unreadable=True)
        packer = LlamaCppPacker(
            model_dir=tmp_path / "model",
            base_gguf=tmp_path / "base.gguf",
            out_path=tmp_path / "out.gguf",
            convert_script=_write_stub(tmp_path / "convert.py", _CONVERT_STUB),
            quantize_bin=_write_stub(tmp_path / "llama-quantize", _FAILING_STUB),
            python_bin=Path(sys.executable),
            threads=1,
            imatrix=tmp_path / "imatrix.gguf",
            imatrix_counts=source,
        )
        (tmp_path / "model").mkdir(exist_ok=True)
        (tmp_path / "base.gguf").write_bytes(b"G" * BASE_BYTES)

        with pytest.raises(PackError, match="cannot read the imatrix"):
            packer.pack(sample_pack_recipe())

        assert source.calls == 1
        # The quantize stub always fails. A different message would
        # mean the read ran after it.
        assert not (tmp_path / "out.gguf").exists()

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
