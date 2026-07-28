"""RecipePacker contract: the llama.cpp adapter and the memory fake agree.

The real side runs the true adapter code — argument construction,
subprocess handling, error translation — against stub tools that
stand in for the llama.cpp binaries, so the suite stays hermetic
(ADR-0009). The type mapping itself is shared pure code, proven in
its own unit suite.
"""

from __future__ import annotations

from pathlib import Path

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
import sys

out = sys.argv[sys.argv.index("--outfile") + 1]
with open(out, "wb") as handle:
    handle.write(b"G" * {BASE_BYTES})
"""

_QUANTIZE_STUB = f"""\
#!/usr/bin/env python3
import sys

with open(sys.argv[-3], "wb") as handle:
    handle.write(b"Q" * {PACKED_BYTES})
"""

_FAILING_STUB = """\
#!/usr/bin/env python3
import sys

sys.stderr.write("stub tool exploded\\n")
sys.exit(3)
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
            Assignment(group="model.layers.0", bits=8, bytes=1_000, damage=0.001),
            Assignment(group="model.layers.1", bits=4, bytes=500, damage=0.01),
        ),
    )


def _write_stub(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o700)
    return path


def _real_packer(tmp_path: Path, fail_stage: str | None) -> RecipePacker:
    import sys

    convert = _write_stub(
        tmp_path / "convert.py",
        _FAILING_STUB if fail_stage == "convert" else _CONVERT_STUB,
    )
    quantize = _write_stub(
        tmp_path / "llama-quantize",
        _FAILING_STUB if fail_stage == "quantize" else _QUANTIZE_STUB,
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    return LlamaCppPacker(
        model_dir=model_dir,
        base_gguf=tmp_path / "base.gguf",
        out_path=tmp_path / "out.gguf",
        convert_script=convert,
        quantize_bin=quantize,
        python_bin=Path(sys.executable),
        threads=1,
    )


def _fake_packer(tmp_path: Path, fail_stage: str | None) -> RecipePacker:
    return MemoryRecipePacker(
        base_bytes=BASE_BYTES, packed_bytes=PACKED_BYTES, fail_stage=fail_stage
    )


@pytest.mark.parametrize(
    "build", [_real_packer, _fake_packer], ids=["real-subprocess", "fake-memory"]
)
class TestRecipePackerContract:
    def test_convert_returns_the_base_size(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, None)

        assert packer.convert() == BASE_BYTES

    def test_convert_twice_returns_the_same_size(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, None)

        assert packer.convert() == packer.convert()

    def test_pack_after_convert_reports_the_real_packed_size(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, None)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.packed_bytes == PACKED_BYTES

    def test_pack_carries_the_shared_type_mapping(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, None)
        packer.convert()

        result = packer.pack(sample_pack_recipe())

        assert result.base_type == "Q4_K_S"
        assert result.token_embedding_type == "q8_0"  # noqa: S105 - a ggml type name, not a secret
        assert result.overrides == (
            TypeOverride(pattern=r"blk\.0\.", ggml_type="q8_0"),
            TypeOverride(pattern=r"blk\.1\.", ggml_type="q4_k"),
        )

    def test_pack_without_convert_raises_pack_error(self, build, tmp_path) -> None:
        packer: RecipePacker = build(tmp_path, None)

        with pytest.raises(PackError, match="run convert first"):
            packer.pack(sample_pack_recipe())

    def test_convert_failure_raises_pack_error_with_exit_code(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, "convert")

        with pytest.raises(PackError, match="convert failed with exit code 3"):
            packer.convert()

    def test_quantize_failure_raises_pack_error_with_exit_code(
        self, build, tmp_path
    ) -> None:
        packer: RecipePacker = build(tmp_path, "quantize")
        packer.convert()

        with pytest.raises(PackError, match="quantize failed with exit code 3"):
            packer.pack(sample_pack_recipe())
