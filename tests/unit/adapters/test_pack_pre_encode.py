"""Checks of the pack adapter's pre-encoding path against stub tools (ADR-0032).

The base and the imatrix are real GGUF files gguf-py writes, so the
adapter's own header reads run for real. The quantizer is a stub
that copies its input to its output and records its argv, which is
what stock ``llama-quantize`` does to a pre-encoded tensor. The
encoder is a stub program that writes sized payloads, so the suite
stays torch-free.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="gguf extra not installed")
pytest.importorskip("gguf", reason="gguf extra not installed")

from gguf import GGUFWriter

from vramfit.adapters.outbound.gguf.header import read_header
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker
from vramfit.adapters.outbound.gguf.q2_0_blocks import Q2_0_TYPE_ID, q2_0_payload_bytes
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.model import (
    Q0_IMX_METHOD,
    Q0_IMX_SUCCESSOR_METHOD,
    Assignment,
    PlanMeta,
    Recipe,
)

pytestmark = pytest.mark.unit

STACK = "blk.0.ffn_down_exps.weight"
UP = "blk.0.ffn_up_exps.weight"
DOWN_GROUP = "model.layers.0.mixer.experts.down_proj"
UP_GROUP = "model.layers.0.mixer.experts.up_proj"
ROW_WIDTHS = {DOWN_GROUP: 128, UP_GROUP: 64}

QUANTIZE_STUB = """\
#!{python}
import json, shutil, sys
with open({argv_log!r}, "w") as log:
    json.dump(sys.argv[1:], log)
shutil.copyfile(sys.argv[-4], sys.argv[-3])
"""

ENCODER_STUB = """\
import hashlib, json, os, sys
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
names = [args[i + 1] for i, a in enumerate(args) if a == "--tensor"]
out = opt("--out-dir")
os.makedirs(out, exist_ok=True)
sizes = json.loads(os.environ["STUB_SIZES"])
report = {"encoder": "stub-encoder", "peak_rss_bytes": 4096, "tensors": {}}
for i, name in enumerate(names):
    path = os.path.join(out, f"{i}.q2_0")
    payload = bytes([0x11 * (i + 1)]) * sizes[name]
    open(path, "wb").write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    report["tensors"][name] = {"payload": path, "bytes": len(payload), "sha256": digest}
json.dump(report, open(opt("--report"), "w"))
open(os.environ["STUB_ARGV"], "w").write(json.dumps(args))
"""


def write_base(path: Path, *, down_width: int = 128) -> None:
    writer = GGUFWriter(path, "llama")
    writer.add_block_count(1)
    writer.add_file_type(1)
    rng = np.random.default_rng(5)
    writer.add_tensor(STACK, rng.standard_normal((2, 4, down_width)).astype(np.float16))
    writer.add_tensor(UP, rng.standard_normal((2, 8, 64)).astype(np.float16))
    writer.add_tensor("blk.0.attn_norm.weight", np.ones(64, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_imatrix(path: Path, names: tuple[str, ...]) -> None:
    writer = GGUFWriter(path, "imatrix")
    writer.add_type("imatrix")
    writer.add_uint32("imatrix.chunk_count", 1)
    for name in names:
        columns = 128 if name == STACK else 64
        writer.add_tensor(f"{name}.in_sum2", np.ones((2, columns), dtype=np.float32))
        writer.add_tensor(f"{name}.counts", np.ones(2, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def recipe(
    method: str = Q0_IMX_SUCCESSOR_METHOD,
    imatrix: str = "m.gguf",
) -> Recipe:
    return Recipe(
        model_id="model",
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
            Assignment(group=DOWN_GROUP, bits=2, bytes=1_000, damage=0.01),
            Assignment(group=UP_GROUP, bits=2, bytes=1_000, damage=0.01),
        ),
        runtime=None,
        within_group=method,
        imatrix=imatrix,
        protected_tensors=(),
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    base = tmp_path / "base.gguf"
    write_base(base)
    imatrix = tmp_path / "m.gguf"
    write_imatrix(imatrix, (STACK, UP))
    argv_log = tmp_path / "quantize-argv.json"
    quantize = tmp_path / "llama-quantize"
    quantize.write_text(
        QUANTIZE_STUB.format(python=sys.executable, argv_log=str(argv_log))
    )
    quantize.chmod(0o700)
    encoder = tmp_path / "encoder.py"
    encoder.write_text(ENCODER_STUB)
    encoder_argv = tmp_path / "encoder-argv.json"
    monkeypatch.setenv("STUB_ARGV", str(encoder_argv))
    monkeypatch.setenv(
        "STUB_SIZES",
        json.dumps({STACK: q2_0_payload_bytes(1024), UP: q2_0_payload_bytes(1024)}),
    )
    return {
        "base": base,
        "imatrix": imatrix,
        "quantize": quantize,
        "quantize_argv": argv_log,
        "encoder": encoder,
        "encoder_argv": encoder_argv,
        "out": tmp_path / "out.gguf",
        "model": tmp_path / "model",
    }


def packer(workspace: dict[str, Path], *, imatrix: bool = True) -> LlamaCppPacker:
    workspace["model"].mkdir(exist_ok=True)
    stack_width = read_header(workspace["base"]).tensors[0].dims[0]
    return LlamaCppPacker(
        model_dir=workspace["model"],
        base_gguf=workspace["base"],
        out_path=workspace["out"],
        convert_script=workspace["base"],
        quantize_bin=workspace["quantize"],
        python_bin=Path(sys.executable),
        threads=2,
        imatrix=workspace["imatrix"] if imatrix else None,
        row_widths={**ROW_WIDTHS, DOWN_GROUP: stack_width},
        encoder_command=(sys.executable, str(workspace["encoder"])),
    )


class TestPreEncodingPack:
    def test_pre_encodes_covered_stacks_and_hands_the_mixed_file_to_the_quantizer(
        self, workspace: dict[str, Path]
    ) -> None:
        p = packer(workspace)
        result = p.pack(recipe())

        argv = json.loads(workspace["quantize_argv"].read_text())
        assert argv[0] == "--pure"
        assert "--allow-requantize" not in argv
        assert argv[-4] == str(p.mixed_gguf)
        assert argv[-3] == str(workspace["out"])
        assert argv[-2] == "Q2_K"
        assert "blk\\.0\\.ffn_down_exps\\.=q2_0" in argv
        assert "blk\\.0\\.ffn_up_exps\\.=q2_0" in argv

        encoder_argv = json.loads(workspace["encoder_argv"].read_text())
        assert encoder_argv[encoder_argv.index("--imatrix") + 1] == str(
            workspace["imatrix"]
        )
        assert encoder_argv[encoder_argv.index("--threads") + 1] == "2"
        assert [
            a for i, a in enumerate(encoder_argv) if encoder_argv[i - 1] == "--tensor"
        ] == [
            STACK,
            UP,
        ]

        assert result.pre_encoded == (STACK, UP)
        assert result.q2_0_encoder == "stub-encoder"
        assert result.pre_encode_cost is not None
        assert result.pre_encode_cost.payload_bytes == 2 * q2_0_payload_bytes(1024)
        assert result.pre_encode_cost.peak_rss_bytes == 4096
        assert result.pre_encode_cost.mixed_gguf_bytes == result.packed_bytes
        assert result.file_type == "Q2_0"
        header = read_header(workspace["out"])
        assert {t.name: t.type_id for t in header.tensors}[STACK] == Q2_0_TYPE_ID
        # The temporary files go only after the payloads verified.
        assert not p.mixed_gguf.exists()
        assert not p.pre_encode_dir.exists()

    def test_stock_q0_imx_recipe_takes_the_stock_path(
        self, workspace: dict[str, Path]
    ) -> None:
        p = packer(workspace)
        result = p.pack(recipe(method=Q0_IMX_METHOD))
        argv = json.loads(workspace["quantize_argv"].read_text())
        assert argv[-4] == str(workspace["base"])
        assert not workspace["encoder_argv"].exists()
        assert result.pre_encoded == ()
        assert result.q2_0_encoder is None
        assert result.pre_encode_cost is None

    def test_successor_recipe_without_imatrix_refuses_before_any_tool_runs(
        self, workspace: dict[str, Path]
    ) -> None:
        # The recipe records the matrix that priced it. A pack that
        # omits --imatrix would ship every Q2_0 tensor stock.
        with pytest.raises(PackError, match="has no --imatrix"):
            packer(workspace, imatrix=False).pack(recipe())
        assert not workspace["quantize_argv"].exists()
        assert not workspace["encoder_argv"].exists()

    def test_rows_outside_the_block_refuse_before_the_preprocessor_writes(
        self, workspace: dict[str, Path]
    ) -> None:
        # 96-wide rows refuse the 256 super-block, so the recipe maps
        # the stack to q2_0, and they do not divide into 64-element
        # blocks either. The quantizer would substitute a type, so the
        # pack refuses before any tool runs (ADR-0032).
        write_base(workspace["base"], down_width=96)
        p = packer(workspace)
        with pytest.raises(PackError, match="rows of 96") as info:
            p.pack(recipe())
        assert "refuses before the preprocessor writes" in str(info.value)
        assert not p.mixed_gguf.exists()
        assert not workspace["encoder_argv"].exists()
        assert not workspace["quantize_argv"].exists()

    def test_uncovered_stack_packs_unassisted(self, workspace: dict[str, Path]) -> None:
        write_imatrix(workspace["imatrix"], (STACK,))
        result = packer(workspace).pack(recipe())
        assert result.pre_encoded == (STACK,)

    def test_lost_payload_refuses_and_keeps_the_file(
        self, workspace: dict[str, Path]
    ) -> None:
        # A quantizer that hands back the float base instead of the
        # mixed file loses every pre-encoded tensor.
        workspace["quantize"].write_text(
            QUANTIZE_STUB.format(
                python=sys.executable, argv_log=str(workspace["quantize_argv"])
            ).replace("sys.argv[-4]", repr(str(workspace["base"])))
        )
        p = packer(workspace)
        with pytest.raises(PackError, match="not Q2_0"):
            p.pack(recipe())
        assert workspace["out"].exists()
        assert p.mixed_gguf.exists()

    def test_encoder_failure_removes_the_stage_temporaries(
        self, workspace: dict[str, Path]
    ) -> None:
        # The encoder writes payloads and then dies. Nothing depends
        # on them, and they are multi-gigabyte on a real model.
        workspace["encoder"].write_text(
            "import os, sys\n"
            "args = sys.argv[1:]\n"
            'out = args[args.index("--out-dir") + 1]\n'
            "os.makedirs(out, exist_ok=True)\n"
            'open(os.path.join(out, "partial.q2_0"), "wb").write(b"\\0" * 64)\n'
            "sys.exit(4)\n"
        )
        p = packer(workspace)
        with pytest.raises(
            PackError, match="pre-encode failed with exit code 4"
        ) as info:
            p.pack(recipe())
        assert str(p.pre_encode_dir) in str(info.value)
        assert str(p.mixed_gguf) in str(info.value)
        assert not p.pre_encode_dir.exists()
        assert not p.mixed_gguf.exists()
        assert not workspace["quantize_argv"].exists()

    def test_quantizer_failure_names_the_kept_mixed_file(
        self, workspace: dict[str, Path]
    ) -> None:
        workspace["quantize"].write_text(
            f"#!{sys.executable}\nimport sys; sys.exit(3)\n"
        )
        p = packer(workspace)
        with pytest.raises(PackError, match="temporary mixed GGUF is kept") as info:
            p.pack(recipe())
        assert "quantize failed with exit code 3" in str(info.value)
        assert p.mixed_gguf.exists()
