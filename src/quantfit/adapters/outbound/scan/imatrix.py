"""Imatrix adapter: column weights for assisted pricing (ADR-0020).

Reads the GGUF imatrix artifact that ``llama-imatrix`` writes and
turns it into per-parameter column weights the meter can consume.
The weight formula is ``in_sum2 / counts`` per column — the load
formula in ``llama-quantize`` (``load_imatrix``, checkout e9fa078).
GGUF tensor names map back to HF parameter names through the same
fixed table ``convert_hf_to_gguf.py`` applies to llama-family
models. A parameter without imatrix coverage is reported, not
defaulted — the caller decides, and the meter prices uncovered
tensors unassisted, exactly as ``llama-quantize`` treats a NULL
imatrix row. ``token_embd`` is never covered.

Examples:
    Load weights for the parameters a meter discovered:

    ```python
    covered, uncovered = assisted_weights_for_params(
        Path("model.imatrix.gguf"), {n: p.shape for n, p in params}
    )
    ```

See Also:
    - [quantfit.adapters.outbound.scan.kquant_assisted][]: Consumes
      the column weights.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from gguf import GGUFReader

# HF module suffix -> GGUF tensor stem, llama-family naming.
_SUFFIX_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
_DIRECT_TO_GGUF = {
    "lm_head.weight": "output.weight",
    "model.embed_tokens.weight": "token_embd.weight",
}
_LAYER_PARAM = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")


def gguf_tensor_name(param_name: str) -> str | None:
    """Map an HF parameter name to its GGUF tensor name.

    Args:
        param_name: The HF dotted parameter name.

    Returns:
        The GGUF tensor name, or None when the table has no mapping —
        the caller treats that as uncovered.
    """
    direct = _DIRECT_TO_GGUF.get(param_name)
    if direct is not None:
        return direct
    match = _LAYER_PARAM.match(param_name)
    if match is None:
        return None
    stem = _SUFFIX_TO_GGUF.get(match.group(2))
    if stem is None:
        return None
    return f"blk.{match.group(1)}.{stem}.weight"


def load_imatrix(path: Path) -> dict[str, torch.Tensor]:
    """Read a GGUF imatrix into column weights per GGUF tensor name.

    Args:
        path: The ``.gguf`` imatrix file ``llama-imatrix`` wrote.

    Returns:
        Float32 column weights keyed by GGUF tensor name. A column
        whose chunk count is zero weighs 1, per ``load_imatrix``.

    Raises:
        ValueError: If the file is not an imatrix, or a sums tensor
            arrives without its counts twin.
        OSError: If the file cannot be read.
    """
    reader = GGUFReader(str(path))
    general_type = reader.fields.get("general.type")
    if general_type is None or general_type.contents() != "imatrix":
        raise ValueError(f"{path} is not an imatrix GGUF (general.type mismatch)")

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        data = torch.from_numpy(tensor.data.copy()).to(torch.float32)
        if tensor.name.endswith(".in_sum2"):
            sums[tensor.name.removesuffix(".in_sum2")] = data
        elif tensor.name.endswith(".counts"):
            counts[tensor.name.removesuffix(".counts")] = data

    weights: dict[str, torch.Tensor] = {}
    for name, sum2 in sums.items():
        count = counts.get(name)
        if count is None:
            raise ValueError(f"{path}: {name}.in_sum2 has no counts twin")
        per_expert = sum2.reshape(count.numel(), -1)
        expert_counts = count.reshape(-1, 1)
        weights[name] = torch.where(
            expert_counts > 0, per_expert / expert_counts, torch.ones_like(per_expert)
        ).reshape(-1)
    return weights


def assisted_weights_for_params(
    path: Path, shapes: Mapping[str, Sequence[int]]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Resolve imatrix column weights for a set of parameters.

    Args:
        path: The ``.gguf`` imatrix file.
        shapes: Parameter shapes keyed by HF parameter name.

    Returns:
        ``(covered, uncovered)`` — float32 column weights keyed by
        parameter name, and the names the imatrix does not cover, in
        input order.

    Raises:
        ValueError: If the file is not an imatrix, or a covered
            tensor's weight length does not match the parameter's row
            length — a silent mismatch would price against the wrong
            columns.
        OSError: If the file cannot be read.
    """
    by_gguf_name = load_imatrix(path)
    covered: dict[str, torch.Tensor] = {}
    uncovered: list[str] = []
    for name, shape in shapes.items():
        gguf_name = gguf_tensor_name(name)
        weight = by_gguf_name.get(gguf_name) if gguf_name is not None else None
        if weight is None:
            uncovered.append(name)
            continue
        if weight.numel() != shape[-1]:
            raise ValueError(
                f"imatrix weights for {name} ({gguf_name}) have "
                f"{weight.numel()} entries, the parameter rows have {shape[-1]}"
            )
        covered[name] = weight
    return covered, tuple(uncovered)
