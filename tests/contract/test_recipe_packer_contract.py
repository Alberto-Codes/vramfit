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

from quantfit.adapters.outbound.gguf.pack import LlamaCppPacker
from quantfit.adapters.outbound.gguf.types import PackError
from quantfit.domain.model import Assignment, PlanMeta, Recipe
from quantfit.domain.pack import TypeOverride
from quantfit.ports.outbound import RecipePacker
from tests.fakes import MemoryRecipePacker

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
with open(sys.argv[-3], "wb") as handle:
    handle.write(b"Q" * {PACKED_BYTES})
"""

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
) -> RecipePacker:
    def stub_body(stage: str, template: str) -> str:
        if fail_stage == stage:
            return _FAILING_STUB
        if silent_stage == stage:
            return _SILENT_STUB
        return template.format(argv_log=str(tmp_path / f"{stage}-argv.json"))

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
    )


def _fake_packer(
    tmp_path: Path,
    fail_stage: Literal["convert", "quantize"] | None = None,
    base_exists: bool = False,
    silent_stage: str | None = None,
) -> RecipePacker:
    return MemoryRecipePacker(
        base_bytes=BASE_BYTES,
        packed_bytes=PACKED_BYTES,
        fail_stage=fail_stage,
        has_base=base_exists,
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

    def test_convert_writing_no_file_raises_pack_error(self, tmp_path) -> None:
        packer = _real_packer(tmp_path, silent_stage="convert")

        with pytest.raises(PackError, match="cannot be inspected"):
            packer.convert()

    def test_quantize_writing_no_file_raises_pack_error(self, tmp_path) -> None:
        packer = _real_packer(tmp_path, silent_stage="quantize")
        packer.convert()

        with pytest.raises(PackError, match="cannot be inspected"):
            packer.pack(sample_pack_recipe())
