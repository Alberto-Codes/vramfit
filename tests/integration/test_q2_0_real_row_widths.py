"""The assisted ``Q2_0`` path at the 30B target's own row widths.

``test_q2_0_stock_toolchain`` proves the path on a 64-wide expert
stack. The 30B acceptance target carries no such row: its routed
stacks are shaped ``(128, 1856, 2688)`` and ``(128, 2688, 1856)``,
so they hold rows of 2688 and 1856 (ADR-0026, ADR-0028). This
module runs the same path at those two widths before a rented card
is booked. It keeps two experts, because the encoder fits each row
alone and the expert count only multiplies the row count.

The tool comes from ``VRAMFIT_LLAMA_CPP_BIN``, a directory holding
stock ``llama-quantize``. The suite skips with that reason when the
variable is unset, when the tool is missing, or when the scan extra
is absent (ADR-0009). A skip is the honest outcome, and it never
reports a pass. The variable names the build, so the module cannot
pin one. It was measured on b10362 (``4801e3c56``), the campaign
arms' build, where both widths held.
"""

# ruff: noqa: E402 - the importorskip guards must run before adapter imports
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")
pytest.importorskip("gguf", reason="scan extra not installed")

from gguf import GGUFWriter

from vramfit.adapters.outbound.gguf.header import read_header
from vramfit.adapters.outbound.gguf.mixed_gguf import write_mixed_gguf
from vramfit.adapters.outbound.gguf.pre_encode import (
    ENCODER_BOOTSTRAP,
    run_encoder,
    select_pre_encode_targets,
)
from vramfit.adapters.outbound.gguf.q2_0_blocks import (
    Q2_0_TYPE_ID,
    QK2_0,
    dequantize_q2_0,
    q2_0_payload_bytes,
)
from vramfit.adapters.outbound.scan.imatrix import load_imatrix
from vramfit.adapters.outbound.scan.within_group import perturb
from vramfit.domain.pack import TypeOverride

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TOOLS = os.environ.get("VRAMFIT_LLAMA_CPP_BIN")
# The 30B target's own widths. The gate and up stacks hold rows of
# N_EMBD, the down stack rows of N_FF.
N_EMBD = 2688
N_FF = 1856
N_EXPERT = 2
N_VOCAB = 32
DOWN = "blk.0.ffn_down_exps.weight"
UP = "blk.0.ffn_up_exps.weight"
GATE = "blk.0.ffn_gate_exps.weight"
DOWN_GROUP = "model.layers.0.mlp.experts.down_proj"
GATE_GROUP = "model.layers.0.mlp.experts.gate_proj"
# The two pre-encoded stacks and the row width each carries.
ENCODED = ((DOWN, DOWN_GROUP, N_FF), (GATE, GATE_GROUP, N_EMBD))


def _tool(name: str) -> Path:
    if TOOLS is None:
        pytest.skip("VRAMFIT_LLAMA_CPP_BIN names no stock llama.cpp build directory")
    path = Path(TOOLS) / name
    if not path.is_file():
        pytest.skip(f"{path} is not a file")
    return path


def _write_model(path: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Write a one-layer MoE llama in f16 at the target's row widths."""
    writer = GGUFWriter(path, "llama")
    writer.add_name("vramfit q2_0 real-row-width fixture")
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


def _write_imatrix(path: Path, rng: np.random.Generator) -> None:
    writer = GGUFWriter(path, "imatrix")
    writer.add_type("imatrix")
    # The stock loader requires the three provenance keys.
    writer.add_array("imatrix.datasets", ["synthetic-fixture"])
    writer.add_uint32("imatrix.chunk_count", 4)
    writer.add_uint32("imatrix.chunk_size", 64)
    for name, columns in ((DOWN, N_FF), (UP, N_EMBD), (GATE, N_EMBD)):
        sums = rng.uniform(0.5, 4.0, size=(N_EXPERT, columns)).astype(np.float32)
        writer.add_tensor(f"{name}.in_sum2", sums)
        writer.add_tensor(f"{name}.counts", np.array([4.0, 4.0], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _payload(packed: Path, name: str) -> bytes:
    header = read_header(packed)
    info = next(t for t in header.tensors if t.name == name)
    assert info.type_id == Q2_0_TYPE_ID
    with packed.open("rb") as handle:
        handle.seek(header.data_start + info.offset)
        return handle.read(q2_0_payload_bytes(info.elements))


class TestRealRowWidths:
    """The encoder, the preprocessor, and the stock pass at 2688 and 1856."""

    def test_both_target_row_widths_divide_the_block(self) -> None:
        """The premise the whole path rests on, stated as a check."""
        assert N_EMBD % QK2_0 == 0
        assert N_FF % QK2_0 == 0

    def test_the_stock_pass_preserves_both_widths_and_the_scan_matches(
        self, tmp_path: Path
    ) -> None:
        quantize = _tool("llama-quantize")
        rng = np.random.default_rng(2688)
        base = tmp_path / "base-f16.gguf"
        tensors = _write_model(base, rng)
        imatrix = tmp_path / "imatrix.gguf"
        _write_imatrix(imatrix, rng)
        header = read_header(base)
        by_name = {t.name: t for t in header.tensors}
        assert by_name[DOWN].dims[0] == N_FF
        assert by_name[GATE].dims[0] == N_EMBD

        overrides = (
            TypeOverride(r"blk\.0\.ffn_down_exps\.", "q2_0"),
            TypeOverride(r"blk\.0\.ffn_gate_exps\.", "q2_0"),
        )
        targets = select_pre_encode_targets(
            overrides, header, covered={DOWN, GATE}, excluded=()
        )
        assert {t.name for t in targets} == {DOWN, GATE}

        report = run_encoder(
            (sys.executable, "-c", ENCODER_BOOTSTRAP),
            base_gguf=base,
            imatrix=imatrix,
            targets=targets,
            work_dir=tmp_path / "work",
            threads=2,
        )
        mixed = tmp_path / "mixed.gguf"
        write_mixed_gguf(
            base, mixed, {t.name: (t.payload, Q2_0_TYPE_ID) for t in report.tensors}
        )
        digests = {t.name: t.sha256 for t in report.tensors}
        for name, digest in digests.items():
            assert hashlib.sha256(_payload(mixed, name)).hexdigest() == digest

        # The stock pass. `--pure` and no `--allow-requantize`, the
        # way the pack path drives it.
        out = tmp_path / "packed.gguf"
        run = subprocess.run(  # noqa: S603 - fixed tool path, test-controlled args
            [
                str(quantize),
                "--imatrix",
                str(imatrix),
                "--pure",
                "--tensor-type",
                r"blk\.0\.ffn_down_exps\.weight=q2_0",
                "--tensor-type",
                r"blk\.0\.ffn_gate_exps\.weight=q2_0",
                "--tensor-type",
                r"blk\.0\.attn_q\.weight=q4_0",
                str(mixed),
                str(out),
                "Q4_0",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        assert run.returncode == 0, run.stdout + run.stderr

        entries = load_imatrix(imatrix)
        for name, group, width in ENCODED:
            packed_bytes = _payload(out, name)
            # The payload survives the stock pass byte for byte.
            assert hashlib.sha256(packed_bytes).hexdigest() == digests[name], (
                f"the stock pass changed the {width}-wide {name} payload"
            )
            # The scan meter decodes to exactly what the pack emitted.
            weight = torch.from_numpy(tensors[name].astype(np.float32))
            priced = perturb(
                weight, 2, group, "q0-imx2", 32, entries[name].column_weights
            )
            decoded = dequantize_q2_0(packed_bytes, weight.numel())
            assert np.array_equal(decoded, priced.reshape(-1).numpy()), (
                f"the scan meter and the pack disagree on {width}-wide {name}"
            )

    def test_a_mismatched_override_refuses_at_a_target_row_width(
        self, tmp_path: Path
    ) -> None:
        """The negative control, at 1856 rather than 64."""
        quantize = _tool("llama-quantize")
        rng = np.random.default_rng(1856)
        base = tmp_path / "base-f16.gguf"
        _write_model(base, rng)
        imatrix = tmp_path / "imatrix.gguf"
        _write_imatrix(imatrix, rng)
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
            threads=2,
        )
        mixed = tmp_path / "mixed.gguf"
        write_mixed_gguf(
            base, mixed, {t.name: (t.payload, Q2_0_TYPE_ID) for t in report.tensors}
        )
        run = subprocess.run(  # noqa: S603 - fixed tool path, test-controlled args
            [
                str(quantize),
                "--pure",
                "--tensor-type",
                r"blk\.0\.ffn_down_exps\.weight=q4_0",
                str(mixed),
                str(tmp_path / "mismatch.gguf"),
                "Q4_0",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        assert run.returncode != 0
        assert "requantizing from type q2_0 is disabled" in run.stdout + run.stderr
