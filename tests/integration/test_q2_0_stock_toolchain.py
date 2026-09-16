"""End-to-end checks of the assisted ``Q2_0`` path against a stock llama.cpp build.

Three of ADR-0032's acceptance items need the real toolchain: a tiny
pack-and-load run, the scan reconstructing the blocks pack emits, and
the passthrough probe rerun on the build in use. The build comes from
``VRAMFIT_LLAMA_CPP_BIN``, a directory holding stock ``llama-quantize``
and ``llama-bench``. The suite skips with that reason when the
variable is unset, when a tool is missing, or when the scan extra is
absent (ADR-0009).

The model is a one-layer llama-architecture mixture of two experts,
small enough to pack in under a second. Its down-projection stack
has 64-wide rows, which refuse the 256 super-block, so its nominal-2
assignment maps to ``q2_0`` through the ADR-0028 table: the case the
encoder pre-encodes. Every other tensor has 256-wide rows and packs
stock, so the file mixes one pre-encoded stack with k-quant tensors
the way a real pack does.
"""

# ruff: noqa: E402 - the importorskip guards must run before adapter imports
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")
pytest.importorskip("gguf", reason="scan extra not installed")

from gguf import GGMLQuantizationType, GGUFWriter

from vramfit.adapters.outbound.gguf.header import read_header
from vramfit.adapters.outbound.gguf.mixed_gguf import write_mixed_gguf
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker
from vramfit.adapters.outbound.gguf.pre_encode import (
    ENCODER_BOOTSTRAP,
    run_encoder,
    select_pre_encode_targets,
)
from vramfit.adapters.outbound.gguf.q2_0_blocks import (
    Q2_0_TYPE_ID,
    dequantize_q2_0,
    q2_0_payload_bytes,
)
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.adapters.outbound.scan.imatrix import load_imatrix
from vramfit.adapters.outbound.scan.q2_0_assisted import Q2_0_ENCODER_REVISION
from vramfit.adapters.outbound.scan.within_group import perturb
from vramfit.domain.model import (
    Q0_IMX_SUCCESSOR_METHOD,
    Assignment,
    PlanMeta,
    Recipe,
)
from vramfit.domain.pack import TypeOverride

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TOOLS = os.environ.get("VRAMFIT_LLAMA_CPP_BIN")
N_EMBD = 256
N_FF = 64
N_EXPERT = 2
N_VOCAB = 32
DOWN = "blk.0.ffn_down_exps.weight"
UP = "blk.0.ffn_up_exps.weight"
GATE = "blk.0.ffn_gate_exps.weight"
DOWN_GROUP = "model.layers.0.mlp.experts.down_proj"
# The down stack's 64-wide rows take the ADR-0028 table at nominal 2.
# The gate and up stacks' 256-wide rows take the k-quant table.
STACKS = {
    DOWN_GROUP: DOWN,
    "model.layers.0.mlp.experts.up_proj": UP,
    "model.layers.0.mlp.experts.gate_proj": GATE,
}


def _tool(name: str) -> Path:
    if TOOLS is None:
        pytest.skip("VRAMFIT_LLAMA_CPP_BIN names no stock llama.cpp build directory")
    path = Path(TOOLS) / name
    if not path.is_file():
        pytest.skip(f"{path} is not a file")
    return path


def _write_model(path: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Write a runnable one-layer MoE llama in f16 with a tiny SPM vocab."""
    writer = GGUFWriter(path, "llama")
    writer.add_name("vramfit q2_0 pre-encode fixture")
    writer.add_context_length(64)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(1)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_rope_dimension_count(N_EMBD // 2)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_expert_count(N_EXPERT)
    writer.add_expert_used_count(1)
    writer.add_vocab_size(N_VOCAB)
    writer.add_file_type(1)
    writer.add_tokenizer_model("llama")
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(N_VOCAB - 3)]
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * N_VOCAB)
    writer.add_token_types([2, 3, 3] + [1] * (N_VOCAB - 3))
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)

    def normal(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.05).astype(np.float16)

    tensors = {
        "token_embd.weight": normal(N_VOCAB, N_EMBD),
        "output_norm.weight": np.ones(N_EMBD, dtype=np.float32),
        "blk.0.attn_norm.weight": np.ones(N_EMBD, dtype=np.float32),
        "blk.0.ffn_norm.weight": np.ones(N_EMBD, dtype=np.float32),
        "blk.0.attn_q.weight": normal(N_EMBD, N_EMBD),
        "blk.0.attn_k.weight": normal(N_EMBD, N_EMBD),
        "blk.0.attn_v.weight": normal(N_EMBD, N_EMBD),
        "blk.0.attn_output.weight": normal(N_EMBD, N_EMBD),
        "blk.0.ffn_gate_inp.weight": normal(N_EXPERT, N_EMBD).astype(np.float32),
        GATE: normal(N_EXPERT, N_FF, N_EMBD),
        UP: normal(N_EXPERT, N_FF, N_EMBD),
        DOWN: normal(N_EXPERT, N_EMBD, N_FF),
    }
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def _write_imatrix(
    path: Path, rng: np.random.Generator, *, zero_count_expert: bool
) -> None:
    writer = GGUFWriter(path, "imatrix")
    writer.add_type("imatrix")
    # The stock loader requires the three provenance keys.
    writer.add_array("imatrix.datasets", ["synthetic-fixture"])
    writer.add_uint32("imatrix.chunk_count", 4)
    writer.add_uint32("imatrix.chunk_size", 64)
    for name, columns in ((DOWN, N_FF), (UP, N_EMBD), (GATE, N_EMBD)):
        sums = rng.uniform(0.5, 4.0, size=(N_EXPERT, columns)).astype(np.float32)
        counts = np.array([4.0, 0.0 if zero_count_expert else 4.0], dtype=np.float32)
        writer.add_tensor(f"{name}.in_sum2", sums)
        writer.add_tensor(f"{name}.counts", counts)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _recipe(imatrix: Path) -> Recipe:
    return Recipe(
        model_id="fixture",
        plan=PlanMeta(
            vram_budget_bytes=10**9,
            kv_headroom_bytes=10**6,
            weight_budget_bytes=10**9,
            predicted_total_bytes=10**5,
            predicted_damage=0.1,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=tuple(
            Assignment(
                group=group,
                bits=2 if group == DOWN_GROUP else 4,
                bytes=1_000,
                damage=0.01,
            )
            for group in STACKS
        ),
        runtime="llama.cpp",
        within_group=Q0_IMX_SUCCESSOR_METHOD,
        imatrix=str(imatrix),
        protected_tensors=(),
    )


def _packer(
    tmp_path: Path, base: Path, imatrix: Path, quantize: Path
) -> LlamaCppPacker:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    return LlamaCppPacker(
        model_dir=model_dir,
        base_gguf=base,
        out_path=tmp_path / "packed.gguf",
        convert_script=base,
        quantize_bin=quantize,
        python_bin=Path(sys.executable),
        threads=2,
        imatrix=imatrix,
        row_widths={
            "model.layers.0.mlp.experts.down_proj": N_FF,
            "model.layers.0.mlp.experts.up_proj": N_EMBD,
            "model.layers.0.mlp.experts.gate_proj": N_EMBD,
        },
    )


def _payload(packed: Path, name: str) -> bytes:
    header = read_header(packed)
    info = next(t for t in header.tensors if t.name == name)
    assert info.type_id == Q2_0_TYPE_ID
    with packed.open("rb") as handle:
        handle.seek(header.data_start + info.offset)
        return handle.read(q2_0_payload_bytes(info.elements))


class TestPackAndLoad:
    def test_pack_pre_encodes_every_stack_and_the_stock_runtime_loads_the_file(
        self, tmp_path: Path
    ) -> None:
        quantize = _tool("llama-quantize")
        bench = _tool("llama-bench")
        rng = np.random.default_rng(601)
        base = tmp_path / "base-f16.gguf"
        tensors = _write_model(base, rng)
        imatrix = tmp_path / "imatrix.gguf"
        _write_imatrix(imatrix, rng, zero_count_expert=True)
        packer = _packer(tmp_path, base, imatrix, quantize)

        result = packer.pack(_recipe(imatrix))

        assert result.pre_encoded == (DOWN,)
        assert result.q2_0_encoder == Q2_0_ENCODER_REVISION
        header = read_header(packer.out_path)
        by_type = {t.name: t.type_id for t in header.tensors}
        assert by_type[DOWN] == Q2_0_TYPE_ID
        assert by_type[UP] == GGMLQuantizationType.Q4_K
        assert by_type["blk.0.attn_q.weight"] == GGMLQuantizationType.Q2_K
        assert not packer.mixed_gguf.exists()
        assert not packer.pre_encode_dir.exists()

        # The scan reconstructs the blocks pack emitted: the meter's
        # successor method prices each stack to the bytes' decoding.
        entries = load_imatrix(imatrix)
        weight = torch.from_numpy(tensors[DOWN].astype(np.float32))
        priced = perturb(
            weight, 2, DOWN_GROUP, "q0-successor", 32, entries[DOWN].column_weights
        )
        decoded = dequantize_q2_0(_payload(packer.out_path, DOWN), weight.numel())
        assert np.array_equal(decoded, priced.reshape(-1).numpy())
        # The zero-count expert weighs 1 in both places, so its rows
        # still fit through the encoder rather than the stock path.
        stock = perturb(
            torch.from_numpy(tensors[DOWN].astype(np.float32)), 2, "d", "q0", 32, None
        )
        assert not np.array_equal(
            dequantize_q2_0(_payload(packer.out_path, DOWN), stock.numel()),
            stock.reshape(-1).numpy(),
        )

        # The stock runtime loads the packed file and runs a forward pass.
        run = subprocess.run(  # noqa: S603 - fixed tool path, test-controlled args
            [
                str(bench),
                "-m",
                str(packer.out_path),
                "-p",
                "8",
                "-n",
                "0",
                "-ngl",
                "0",
                "-r",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert run.returncode == 0, run.stdout + run.stderr

    def test_a_pack_without_the_matrix_refuses_before_any_tool_runs(
        self, tmp_path: Path
    ) -> None:
        quantize = _tool("llama-quantize")
        rng = np.random.default_rng(7)
        base = tmp_path / "base-f16.gguf"
        _write_model(base, rng)
        imatrix = tmp_path / "imatrix.gguf"
        _write_imatrix(imatrix, rng, zero_count_expert=False)
        packer = _packer(tmp_path, base, imatrix, quantize)
        # Without --imatrix the pack has nothing to weight the encoder with.
        no_matrix = replace(packer, imatrix=None)
        with pytest.raises(PackError, match="has no --imatrix"):
            no_matrix.pack(_recipe(imatrix))
        assert not packer.out_path.exists()


class TestPassthroughProbe:
    """The ADR-0032 evidence probe, rerun on the build in use.

    The input is the preprocessor's own mixed file: the base with the
    down stack pre-encoded by the real encoder and every other tensor
    still float.
    """

    def test_stock_quantize_preserves_q2_0_and_refuses_a_requantize(
        self, tmp_path: Path
    ) -> None:
        quantize = _tool("llama-quantize")
        rng = np.random.default_rng(32)
        base = tmp_path / "base-f16.gguf"
        _write_model(base, rng)
        imatrix = tmp_path / "imatrix.gguf"
        _write_imatrix(imatrix, rng, zero_count_expert=False)
        targets = select_pre_encode_targets(
            (TypeOverride(r"blk\.0\.ffn_down_exps\.", "q2_0"),),
            read_header(base),
            covered={DOWN},
            excluded=(),
        )
        report = run_encoder(
            (sys.executable, "-c", ENCODER_BOOTSTRAP),
            base_gguf=base,
            imatrix=imatrix,
            targets=targets,
            work_dir=tmp_path / "work",
            threads=1,
        )
        mixed = tmp_path / "mixed.gguf"
        write_mixed_gguf(
            base, mixed, {t.name: (t.payload, Q2_0_TYPE_ID) for t in report.tensors}
        )
        digest = report.tensors[0].sha256
        assert hashlib.sha256(_payload(mixed, DOWN)).hexdigest() == digest

        # Matching override: the payload copies through unchanged, with
        # and without an imatrix, and the float peer quantizes.
        for label, with_matrix in (("matching", False), ("matching-imatrix", True)):
            out = tmp_path / f"{label}.gguf"
            argv = [
                str(quantize),
                "--pure",
                "--tensor-type",
                r"blk\.0\.ffn_down_exps\.weight=q2_0",
                "--tensor-type",
                r"blk\.0\.attn_q\.weight=q4_0",
                str(mixed),
                str(out),
                "Q2_K",
                "1",
            ]
            if with_matrix:
                argv[1:1] = ["--imatrix", str(imatrix)]
            run = subprocess.run(  # noqa: S603 - fixed tool path, test-controlled args
                argv, capture_output=True, text=True, timeout=300, check=False
            )
            assert run.returncode == 0, run.stdout + run.stderr
            assert hashlib.sha256(_payload(out, DOWN)).hexdigest() == digest
            by_name = {t.name: t.type_id for t in read_header(out).tensors}
            assert by_name["blk.0.attn_q.weight"] == GGMLQuantizationType.Q4_0

        # Negative control: a Q4_0 override on the existing Q2_0 tensor
        # refuses, and the pack never passes --allow-requantize.
        run = subprocess.run(  # noqa: S603 - fixed tool path, test-controlled args
            [
                str(quantize),
                "--pure",
                "--tensor-type",
                r"blk\.0\.ffn_down_exps\.weight=q4_0",
                str(mixed),
                str(tmp_path / "mismatch.gguf"),
                "Q2_K",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert run.returncode != 0
        assert "requantizing from type q2_0 is disabled" in run.stdout + run.stderr
