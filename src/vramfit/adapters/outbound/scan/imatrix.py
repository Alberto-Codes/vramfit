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

An imatrix entry holds one row per matrix, and `ImatrixEntry` keeps
that shape. llama.cpp writes ``in_sum2`` as a 2-D tensor of
``[columns, matrices]`` and ``counts`` as one float per matrix
(``imatrix.cpp:595-607``, checkout e9fa078). A dense tensor holds
one matrix. A fused expert stack holds one matrix per expert, and
the HF checkpoint spells those experts as separate parameters. So
`resolve_assisted_weights` reads a stack row by expert index, not
by row length (#177). `resolve_imatrix_counts` reads the same
entries for their counts, which price nothing and record routing
frequency (ADR-0026 decisions 2 and 4).

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
# A routed expert's projection -> its fused GGUF stack stem. The
# module path in front of ".experts.N." varies by family and the
# projection after it does not, so the table keys on the projection
# alone. Mixtral spells its projections "w1", "w2", and "w3", which
# this table omits deliberately — an unmapped name reports uncovered
# and a wrong guess would price against the wrong columns.
_EXPERT_PROJ_TO_GGUF = {
    "gate_proj": "ffn_gate_exps",
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}
_DIRECT_TO_GGUF = {
    "lm_head.weight": "output.weight",
    "model.embed_tokens.weight": "token_embd.weight",
}
# Any module path may precede "layers.N.". Nemotron 3.5 Lightning
# roots its decoder at "backbone.layers.N." (#160), and the domain's
# grouping already reads the prefix the same way.
_LAYER_PARAM = re.compile(r"^(?:[^.]+\.)*layers\.(\d+)\.(.+)\.weight$")
# The routed-expert index, spelled between ".experts." and the
# projection by every family the domain groups (#160, #161).
_EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")


@dataclass(frozen=True, slots=True)
class ImatrixEntry:
    """One imatrix tensor, kept one row per matrix.

    Attributes:
        columns (torch.Tensor): Column weights, shape
            ``(matrices, columns)``, float32. Row ``i`` is matrix
            ``i``'s ``in_sum2 / counts``, or ones where its count is
            zero.
        counts (torch.Tensor): Chunk count per matrix, shape
            ``(matrices,)``, float32.

    Examples:
        Read a fused stack's row for expert 57:

        ```python
        entry = load_imatrix(Path("model.imatrix.gguf"))["blk.3.ffn_up_exps.weight"]
        assert entry.counts.numel() == 128
        columns = entry.columns[57]
        ```
    """

    columns: torch.Tensor
    counts: torch.Tensor


def gguf_tensor_name(param_name: str) -> str | None:
    """Map an HF parameter name to its GGUF tensor name.

    A routed expert maps to the fused stack that holds it, so all
    128 of a projection's experts share one name (#159).

    Args:
        param_name: The HF dotted parameter name.

    Returns:
        The GGUF tensor name, or None when the table has no mapping —
        the caller treats that as uncovered.

    Examples:
        Map one routed expert to its stack:

        ```python
        from vramfit.adapters.outbound.scan.imatrix import gguf_tensor_name

        name = "backbone.layers.3.mixer.experts.57.down_proj.weight"
        assert gguf_tensor_name(name) == "blk.3.ffn_down_exps.weight"
        ```
    """
    direct = _DIRECT_TO_GGUF.get(param_name)
    if direct is not None:
        return direct
    match = _LAYER_PARAM.match(param_name)
    if match is None:
        return None
    suffix = match.group(2)
    if _EXPERT_INDEX.search(suffix) is not None:
        stem = _EXPERT_PROJ_TO_GGUF.get(suffix.rsplit(".", 1)[-1])
    else:
        stem = _SUFFIX_TO_GGUF.get(suffix)
    if stem is None:
        return None
    return f"blk.{match.group(1)}.{stem}.weight"


def expert_index(param_name: str) -> int | None:
    """Read a parameter's routed-expert index.

    Args:
        param_name: The HF dotted parameter name.

    Returns:
        The expert index, or None when the parameter is not one
        routed expert of a stack. A shared expert has no index and
        reads None.
    """
    match = _EXPERT_INDEX.search(param_name)
    return int(match.group(1)) if match is not None else None


def load_imatrix(path: Path) -> dict[str, ImatrixEntry]:
    """Read a GGUF imatrix into one entry per GGUF tensor name.

    Args:
        path: The ``.gguf`` imatrix file ``llama-imatrix`` wrote.

    Returns:
        `ImatrixEntry` per GGUF tensor name, keeping one row per
        matrix. A dense tensor holds one row. A fused expert stack
        holds one row per expert. A column whose chunk count is zero
        weighs 1, per ``load_imatrix``.

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

    entries: dict[str, ImatrixEntry] = {}
    for name, sum2 in sums.items():
        count = counts.get(name)
        if count is None:
            raise ValueError(f"{path}: {name}.in_sum2 has no counts twin")
        matrices = count.numel()
        if sum2.numel() % matrices:
            raise ValueError(
                f"{path}: {name}.in_sum2 has {sum2.numel()} entries, "
                f"not divisible by its {matrices} counts"
            )
        # in_sum2 runs matrix-major, which is what llama-quantize
        # slices per expert (src/llama-quant.cpp:1256-1262).
        per_matrix = sum2.reshape(matrices, -1)
        divisors = count.reshape(matrices, 1)
        entries[name] = ImatrixEntry(
            columns=torch.where(
                divisors > 0, per_matrix / divisors, torch.ones_like(per_matrix)
            ),
            counts=count.reshape(-1),
        )
    return entries


def _matrix_index(entry: ImatrixEntry, param_name: str, gguf_name: str) -> int:
    """Locate one parameter's row inside its imatrix entry.

    Args:
        entry: The entry `load_imatrix` read for `gguf_name`.
        param_name: The HF dotted parameter name.
        gguf_name: The GGUF tensor name `param_name` maps to.

    Returns:
        The row index. A routed expert reads its own index. Every
        other parameter reads row 0.

    Raises:
        ValueError: If the expert index runs past the stack, or a
            parameter that is not a routed expert lands on a stack
            of more than one matrix. Either mismatch means the
            imatrix and the checkpoint describe different models,
            and reading row 0 anyway would price against another
            expert's columns.
    """
    matrices = int(entry.counts.numel())
    index = expert_index(param_name)
    if index is None:
        if matrices != 1:
            raise ValueError(
                f"{gguf_name} holds {matrices} matrices, and {param_name} "
                "carries no expert index — the imatrix does not describe "
                "this checkpoint"
            )
        return 0
    if index >= matrices:
        raise ValueError(
            f"{param_name} is expert {index}, and {gguf_name} holds "
            f"only {matrices} matrices"
        )
    return index


def resolve_assisted_weights(
    by_gguf_name: Mapping[str, ImatrixEntry], rows_by_param: Mapping[str, int]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Match loaded imatrix entries to a set of parameters.

    The meter calls this after model load — `load_imatrix` runs
    first, so a malformed file refuses before the load burns
    minutes. A routed expert reads its own row of the fused stack,
    so all 128 experts of a projection resolve against one entry
    (#177).

    Args:
        by_gguf_name: Entries keyed by GGUF tensor name, from
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
            would price against the wrong columns — if the entry's
            matrix count does not fit the parameter, or if no
            parameter is covered at all. Zero coverage means the
            wrong file, and a scan run on it would price every cell
            unassisted under the assisted label.
    """
    covered: dict[str, torch.Tensor] = {}
    uncovered: list[str] = []
    for name, rows in rows_by_param.items():
        gguf_name = gguf_tensor_name(name)
        entry = by_gguf_name.get(gguf_name) if gguf_name is not None else None
        if entry is None or gguf_name is None or rows % SUPER_BLOCK:
            uncovered.append(name)
            continue
        weight = entry.columns[_matrix_index(entry, name, gguf_name)]
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


def resolve_imatrix_counts(
    by_gguf_name: Mapping[str, ImatrixEntry], param_names: Iterable[str]
) -> dict[str, int]:
    """Match each parameter to its imatrix count.

    A routed expert reads its own count, which is how often the
    router fired it over the calibration corpus (ADR-0026). The
    counts record routing frequency and price nothing, so this
    resolver applies no super-block gate: a stack whose rows refuse
    a k-quant fit still reports its counts.

    Args:
        by_gguf_name: Entries keyed by GGUF tensor name, from
            `load_imatrix`.
        param_names: HF parameter names to read counts for.

    Returns:
        The chunk count per parameter name, in input order, holding
        only the names the imatrix covers. An uncovered name is
        absent rather than zero — zero is a real count that
        ADR-0026 decision 5 reports. Each count rounds to the
        nearest integer, which is what the C loader does with the
        float the file stores (``imatrix-loader.cpp:158``).

    Raises:
        ValueError: If the entry's matrix count does not fit the
            parameter.

    Examples:
        Read one stack's routing distribution:

        ```python
        entries = load_imatrix(Path("model.imatrix.gguf"))
        stack = [
            f"backbone.layers.3.mixer.experts.{i}.up_proj.weight" for i in range(128)
        ]
        counts = resolve_imatrix_counts(entries, stack)
        ```
    """
    resolved: dict[str, int] = {}
    for name in param_names:
        gguf_name = gguf_tensor_name(name)
        entry = by_gguf_name.get(gguf_name) if gguf_name is not None else None
        if entry is None or gguf_name is None:
            continue
        count = entry.counts[_matrix_index(entry, name, gguf_name)]
        resolved[name] = int(count.round().item())
    return resolved


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
            row length, an entry's matrix count does not fit the
            parameter, or no parameter is covered at all.
        OSError: If the file cannot be read.
    """
    rows_by_param = {name: int(shape[-1]) for name, shape in shapes.items()}
    return resolve_assisted_weights(load_imatrix(path), rows_by_param)
