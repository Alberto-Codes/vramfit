"""The GGUF fixture both assisted ``Q2_0`` integration suites write.

The two suites differ in one thing: the row widths their expert
stacks carry. `test_q2_0_stock_toolchain` runs a 64-wide down stack,
and `test_q2_0_real_row_widths` runs the 30B target's own 2688 and
1856 (ADR-0026, ADR-0028). The GGUF layout is otherwise the same, so
this module owns it once and takes the widths as arguments.

The tool comes from ``VRAMFIT_LLAMA_CPP_BIN``, a directory holding
stock ``llama-quantize`` and ``llama-bench``. `tool` skips with that
reason when the variable is unset or the tool is missing (ADR-0009).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from vramfit.adapters.outbound.gguf.header import read_header
from vramfit.adapters.outbound.gguf.q2_0_blocks import (
    Q2_0_TYPE_ID,
    q2_0_payload_bytes,
)

N_EXPERT = 2
N_VOCAB = 32
DOWN = "blk.0.ffn_down_exps.weight"
UP = "blk.0.ffn_up_exps.weight"
GATE = "blk.0.ffn_gate_exps.weight"
DOWN_GROUP = "model.layers.0.mlp.experts.down_proj"
UP_GROUP = "model.layers.0.mlp.experts.up_proj"
GATE_GROUP = "model.layers.0.mlp.experts.gate_proj"


def tool(name: str) -> Path:
    """Name one stock llama.cpp tool, or skip with the reason.

    The environment read happens here, at call time. A module-level
    read binds the value at import, so a variable set or cleared
    afterwards never reaches the skip decision, and the suite would
    skip against a stale reading of its own documented condition.
    """
    tools = os.environ.get("VRAMFIT_LLAMA_CPP_BIN")
    if tools is None:
        pytest.skip("VRAMFIT_LLAMA_CPP_BIN names no stock llama.cpp build directory")
    path = Path(tools) / name
    if not path.is_file():
        pytest.skip(f"{path} is not a file")
    return path


def write_model(
    path: Path,
    rng: np.random.Generator,
    *,
    n_embd: int,
    n_ff: int,
    name: str,
) -> dict[str, np.ndarray]:
    """Write a runnable one-layer MoE llama in f16 with a tiny SPM vocab.

    The gate and up stacks hold rows of ``n_embd``, and the down
    stack holds rows of ``n_ff``. Those two widths select the type
    table each stack takes (ADR-0028).

    Args:
        path: Where to write the base GGUF.
        rng: Source of the weights.
        n_embd: Embedding length, and the gate and up row width.
        n_ff: Feed-forward length, and the down row width.
        name: The ``general.name`` the file declares.

    Returns:
        Every tensor the file carries, by GGUF name.
    """
    writer = GGUFWriter(path, "llama")
    writer.add_name(name)
    writer.add_context_length(64)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(1)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_rope_dimension_count(n_embd // 2)
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
        "token_embd.weight": normal(N_VOCAB, n_embd),
        "output_norm.weight": np.ones(n_embd, dtype=np.float32),
        "blk.0.attn_norm.weight": np.ones(n_embd, dtype=np.float32),
        "blk.0.ffn_norm.weight": np.ones(n_embd, dtype=np.float32),
        "blk.0.attn_q.weight": normal(n_embd, n_embd),
        "blk.0.attn_k.weight": normal(n_embd, n_embd),
        "blk.0.attn_v.weight": normal(n_embd, n_embd),
        "blk.0.attn_output.weight": normal(n_embd, n_embd),
        "blk.0.ffn_gate_inp.weight": normal(N_EXPERT, n_embd).astype(np.float32),
        GATE: normal(N_EXPERT, n_ff, n_embd),
        UP: normal(N_EXPERT, n_ff, n_embd),
        DOWN: normal(N_EXPERT, n_embd, n_ff),
    }
    for tensor_name, data in tensors.items():
        writer.add_tensor(tensor_name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def write_imatrix(
    path: Path,
    rng: np.random.Generator,
    *,
    n_embd: int,
    n_ff: int,
    zero_count_expert: bool,
) -> None:
    """Write an imatrix covering all three expert stacks.

    Args:
        path: Where to write the imatrix GGUF.
        rng: Source of the column sums.
        n_embd: Column count of the gate and up entries.
        n_ff: Column count of the down entry.
        zero_count_expert: Give the second expert a zero chunk count,
            the case that weighs its rows at 1 (ADR-0026).
    """
    writer = GGUFWriter(path, "imatrix")
    writer.add_type("imatrix")
    # The stock loader requires the three provenance keys.
    writer.add_array("imatrix.datasets", ["synthetic-fixture"])
    writer.add_uint32("imatrix.chunk_count", 4)
    writer.add_uint32("imatrix.chunk_size", 64)
    for name, columns in ((DOWN, n_ff), (UP, n_embd), (GATE, n_embd)):
        sums = rng.uniform(0.5, 4.0, size=(N_EXPERT, columns)).astype(np.float32)
        counts = np.array([4.0, 0.0 if zero_count_expert else 4.0], dtype=np.float32)
        writer.add_tensor(f"{name}.in_sum2", sums)
        writer.add_tensor(f"{name}.counts", counts)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def payload(packed: Path, name: str) -> bytes:
    """Read one Q2_0 tensor's payload bytes out of a packed GGUF."""
    header = read_header(packed)
    info = next(t for t in header.tensors if t.name == name)
    assert info.type_id == Q2_0_TYPE_ID
    with packed.open("rb") as handle:
        handle.seek(header.data_start + info.offset)
        return handle.read(q2_0_payload_bytes(info.elements))
