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
imatrix row. ``token_embd`` is never covered. The module also owns
`check_imatrix_weights`, the construction-time gate the meter runs
over any weight source — resolved from a file or passed directly.

Examples:
    Load weights for the parameters a meter discovered:

    ```python
    covered, uncovered = assisted_weights_for_params(
        Path("model.imatrix.gguf"), {n: p.shape for n, p in params}
    )
    ```

See Also:
    - [vramfit.adapters.outbound.scan.kquant_assisted][]: Consumes
      the column weights.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from gguf import GGUFReader

from vramfit.adapters.outbound.scan.kquant import SUPER_BLOCK

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
        ValueError: If the file is not an imatrix, a tensor name
            carries neither known suffix, a sums tensor arrives
            without its counts twin (or the reverse), or the file
            holds no data at all — every malformation here would
            otherwise shrink coverage silently. The unknown-suffix
            refusal is deliberately stricter than the C loader,
            which skips unrecognized tensors: a suffix rename in a
            future imatrix format must fail loudly here, not price
            a 30-hour scan unassisted.
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
        else:
            raise ValueError(
                f"{path}: unexpected tensor {tensor.name} — an imatrix "
                "holds only .in_sum2/.counts pairs"
            )
    if not sums:
        raise ValueError(f"{path}: the imatrix holds no .in_sum2 tensors")
    orphans = sorted(set(counts) - set(sums))
    if orphans:
        raise ValueError(f"{path}: {orphans[0]}.counts has no in_sum2 twin")

    weights: dict[str, torch.Tensor] = {}
    for name, sum2 in sums.items():
        count = counts.get(name)
        if count is None:
            raise ValueError(f"{path}: {name}.in_sum2 has no counts twin")
        if sum2.numel() % count.numel():
            raise ValueError(
                f"{path}: {name}.in_sum2 has {sum2.numel()} entries, "
                f"not divisible by its {count.numel()} counts"
            )
        per_expert = sum2.reshape(count.numel(), -1)
        expert_counts = count.reshape(-1, 1)
        weights[name] = torch.where(
            expert_counts > 0, per_expert / expert_counts, torch.ones_like(per_expert)
        ).reshape(-1)
    return weights


def resolve_assisted_weights(
    by_gguf_name: Mapping[str, torch.Tensor], rows_by_param: Mapping[str, int]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Match loaded imatrix weights to a set of parameters.

    The meter calls this after model load — `load_imatrix` runs
    first, so a malformed file refuses before the load burns
    minutes.

    Args:
        by_gguf_name: Column weights keyed by GGUF tensor name, from
            `load_imatrix`.
        rows_by_param: Row length per HF parameter name.

    Returns:
        ``(covered, uncovered)`` — float32 column weights keyed by
        parameter name, and the names the imatrix does not cover, in
        input order. A parameter whose rows do not divide into
        super-blocks joins the uncovered set: it cannot price
        assisted (ADR-0020), and the fallback beats refusing a
        multi-day scan over one tensor. The run log and the console
        echo report it with the rest of the uncovered names.

    Raises:
        ValueError: If a covered tensor's weight length does not
            match the parameter's row length — a silent mismatch
            would price against the wrong columns — or no parameter
            is covered at all. Zero coverage means the wrong file,
            and a scan run on it would price every cell unassisted
            under the assisted label.
    """
    covered: dict[str, torch.Tensor] = {}
    uncovered: list[str] = []
    for name, rows in rows_by_param.items():
        gguf_name = gguf_tensor_name(name)
        weight = by_gguf_name.get(gguf_name) if gguf_name is not None else None
        if weight is None or rows % SUPER_BLOCK:
            uncovered.append(name)
            continue
        if weight.numel() != rows:
            raise ValueError(
                f"imatrix weights for {name} ({gguf_name}) have "
                f"{weight.numel()} entries, the parameter rows have {rows}"
            )
        covered[name] = weight
    if rows_by_param and not covered:
        raise ValueError(
            f"the imatrix covers none of the {len(rows_by_param)} parameters — "
            "wrong imatrix file for this model?"
        )
    return covered, tuple(uncovered)


def check_imatrix_weights(
    weights: Mapping[str, torch.Tensor], rows_by_param: Mapping[str, int]
) -> None:
    """Refuse imatrix weights that cannot match the model.

    The meter runs this at construction. A typoed name or a
    wrong-length vector would price cells against the wrong
    columns — silently. A misaligned or non-finite weight would
    abort the scan at its first assisted cell, hours in, when
    milliseconds here refuse it up front.

    Args:
        weights: Column weights keyed by HF parameter name.
        rows_by_param: Row length per discovered parameter name.

    Raises:
        ValueError: If a weighted name is not a discovered
            parameter, its weights are not 1-D, its weight length
            does not match the parameter's row length, the rows do
            not divide into super-blocks, or a weight is negative
            or non-finite.
    """
    for name, columns in weights.items():
        if name not in rows_by_param:
            raise ValueError(f'imatrix weights name unknown parameter "{name}"')
        if columns.dim() != 1:
            raise ValueError(
                f"imatrix weights for {name} must be 1-D, got shape "
                f"{tuple(columns.shape)}"
            )
        rows = rows_by_param[name]
        if columns.numel() != rows:
            raise ValueError(
                f"imatrix weights for {name} have {columns.numel()} "
                f"entries, the parameter rows have {rows}"
            )
        if rows % SUPER_BLOCK:
            raise ValueError(
                f"{name} has rows of {rows}, not divisible into "
                f"{SUPER_BLOCK}-element super-blocks — it "
                "cannot price assisted (ADR-0020)"
            )
        if not bool(torch.isfinite(columns).all()) or bool((columns < 0).any()):
            raise ValueError(
                f"imatrix weights for {name} must be finite and non-negative"
            )


def assisted_weights_for_params(
    path: Path, shapes: Mapping[str, Sequence[int]]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Resolve imatrix column weights for a set of parameters.

    `load_imatrix` and `resolve_assisted_weights` in one call, for
    callers that already hold the shapes.

    Args:
        path: The ``.gguf`` imatrix file.
        shapes: Parameter shapes keyed by HF parameter name.

    Returns:
        ``(covered, uncovered)`` — float32 column weights keyed by
        parameter name, and the names the imatrix does not cover, in
        input order.

    Raises:
        ValueError: If the file is not an imatrix, a covered
            tensor's weight length does not match the parameter's
            row length, or no parameter is covered at all.
        OSError: If the file cannot be read.
    """
    rows_by_param = {name: int(shape[-1]) for name, shape in shapes.items()}
    return resolve_assisted_weights(load_imatrix(path), rows_by_param)
