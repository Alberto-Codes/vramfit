"""Imatrix adapter: column weights for assisted pricing (ADR-0020).

Reads the GGUF imatrix artifact that ``llama-imatrix`` writes and
turns it into per-parameter column weights the meter can consume.
The weight formula is ``in_sum2 / counts`` per column — the load
formula in ``llama-quantize`` (``load_imatrix``, checkout e9fa078).
GGUF tensor names map back to HF parameter names through a fixed
table. Each row copies a name ``gguf.TensorNameMap`` carries. The
table covers the llama family, Nemotron-H-MoE (#186), and the
nested ``model.language_model.`` decoder root a composite
checkpoint spells (Gemma 4 31B, #423).
Nemotron-H spells attention, Mamba-2, the router, and the shared
expert under one ``mixer.`` module, so no llama-family name reaches
its dense tensors. A parameter without imatrix coverage is reported, not
defaulted — the caller decides, and the meter prices uncovered
tensors unassisted, exactly as ``llama-quantize`` treats a NULL
imatrix row. ``token_embd`` is never covered. The module also owns
`check_imatrix_weights`, the construction-time gate the meter runs
over any weight source — resolved from a file or passed directly.

An imatrix entry holds one row per matrix, and `ImatrixEntry` keeps
that shape. llama.cpp declares ``in_sum2`` with ``ne`` of
``[columns, matrices]`` (``imatrix.cpp:595-607``, checkout e9fa078).
``ne`` runs fastest axis first, so the buffer reads as
``(matrices, columns)`` in NumPy order. It writes ``counts`` as one
float per matrix.

A dense tensor holds one matrix. An expert stack holds one matrix
per expert. The checkpoint spells those experts as separate
parameters, and transformers may fuse them into one 3D parameter
at load (#202). The resolvers key on the names the loaded model
reports and never construct indexed names — constructed names are
how #177 missed the fusion. An indexed 2D parameter reads its own
row by expert index, not by row length (#177). A fused 3D
parameter reads its whole entry as one expert stack.
`resolve_imatrix_counts` reads the same entries for their counts.
The counts record routing frequency and price nothing (ADR-0026
decisions 4 and 5). `expert_stack_count_vectors` then selects one
group's expert-stack vectors for decision 4's map summary, all or
nothing per group (the #201 amendment). `resolve_assisted_weights`
refuses a fused expert stack by rule (ADR-0026, 2026-08-13
amendment). Under this reader the expert stacks price unassisted.
The q0 reader carries their assisted fit
(ADR-0018, 2026-08-21 amendment).

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

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from gguf import GGUFReader

from vramfit.adapters.outbound.scan.kquant import SUPER_BLOCK

# HF module suffix -> GGUF tensor stem. Each entry copies a name
# ``gguf.TensorNameMap`` already carries, and
# tests/unit/adapters/test_imatrix_names.py pins every row against
# that map.
_SUFFIX_TO_GGUF = {
    # llama-family naming.
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    # Nemotron-H naming (#186). This family spells attention, Mamba-2,
    # the router, and the shared expert under one "mixer." module. A
    # layer is attention or Mamba-2 or MoE, never two at once. So these
    # suffixes never collide inside one layer.
    #
    # These 9 rows cover the MoE variant's dense set, which is what
    # Nemotron 3.5 Lightning holds. A dense-MLP Nemotron-H also spells
    # "mixer.up_proj" and "mixer.down_proj", and Nemotron 3 Super adds
    # "mixer.fc1_latent_proj" and "mixer.fc2_latent_proj". No target
    # needs them yet, so they stay out until a checkpoint proves them.
    "mixer.q_proj": "attn_q",
    "mixer.k_proj": "attn_k",
    "mixer.v_proj": "attn_v",
    "mixer.o_proj": "attn_output",
    "mixer.in_proj": "ssm_in",
    "mixer.out_proj": "ssm_out",
    "mixer.gate": "ffn_gate_inp",
    "mixer.shared_experts.up_proj": "ffn_up_shexp",
    "mixer.shared_experts.down_proj": "ffn_down_shexp",
}
# A routed expert's projection -> its GGUF expert-stack stem. The
# module path in front of ".experts.N." varies by family. The
# projection after it does not, so the table keys on the projection.
# Mixtral spells its projections "w1", "w2", and "w3". This table
# omits them deliberately. An unmapped name reports uncovered, and a
# wrong guess would price against the wrong columns.
_EXPERT_PROJ_TO_GGUF = {
    "gate_proj": "ffn_gate_exps",
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}
# A fused expert stack's module suffix -> its GGUF expert-stack
# stem. transformers 5 fuses a supported MoE model's routed experts
# into one 3D parameter at load, so the loaded module reports no
# indexed expert name (#202). Checkpoints stay indexed, and a
# custom-code model loads unfused, so both layouts persist. #202
# measured the fused names under the model root. The backbone root
# joins because #186 pairs both roots for this family's dense
# names. Nemotron 3.5 Lightning's experts are ungated at two
# matrices each (#160), so a fused gate stays out until a loaded
# model reports one.
_FUSED_EXPERT_TO_GGUF = {
    "mixer.experts.up_proj": "ffn_up_exps",
    "mixer.experts.down_proj": "ffn_down_exps",
}
_DIRECT_TO_GGUF = {
    "lm_head.weight": "output.weight",
    "model.embed_tokens.weight": "token_embd.weight",
    # No nested lm_head row: Gemma 4 ties its head to the embedding,
    # so the loaded model reports one name. An untied composite needs
    # a row here, or its head prices unassisted.
    "model.language_model.embed_tokens.weight": "token_embd.weight",
}
# The decoder roots this table supports. llama-family models root at
# "model.layers.N.", Nemotron 3.5 Lightning at "backbone.layers.N."
# (#160), and a composite checkpoint nests its decoder at
# "model.language_model.layers.N." (Gemma 4 31B, #423). The
# alternation stays closed on purpose. A prefix wildcard would map a
# vision tower's "layers.5" onto the decoder's "blk.5" and price it
# against the wrong columns. Gemma 4's tower roots at
# "model.vision_tower.encoder.layers.N.", which every root here
# rejects. `gguf.TensorNameMap` carries no "language_model" template:
# its converter strips the nesting before it maps, so the root joins
# here and not there.
#
# The closed alternation also keeps the "mixer." suffixes safe. Three
# families spell one of them for a different tensor. phi2 roots at
# "transformer.h.N." and jina at "encoder.layers.N.", and both read
# "mixer.out_proj" as the attention output. plamo2 roots at a doubled
# "model.layers.layers.N." and reads "mixer.in_proj" as the SSM input.
# All three of those roots fail this pattern, so no name here maps
# to two meanings. Check that again before adding a root. The #423
# root addition re-ran the check, and all three still fail.
_ROOTS = r"(?:model|backbone|model\.language_model)"
_LAYER_PARAM = re.compile(rf"^{_ROOTS}\.layers\.(\d+)\.(.+)\.weight$")
# The routed-expert index, spelled between ".experts." and the
# projection by every family the domain groups (#160, #161).
_EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")
# A fused expert stack, as the loaded module reports it. It is a
# raw 3D parameter rather than a Linear, so it carries no ".weight"
# suffix and no ".experts.N." index (#202).
_FUSED_PARAM = re.compile(rf"^{_ROOTS}\.layers\.(\d+)\.(.+)$")
# A fused expert stack shapes as (experts, rows, columns). Any other
# rank under a fused name is a layout mismatch.
_FUSED_STACK_DIMS = 3


@dataclass(frozen=True, slots=True)
class ImatrixEntry:
    """One imatrix tensor, kept one row per matrix.

    Attributes:
        column_weights (torch.Tensor): Shape ``(matrices, columns)``,
            float32. Row ``i`` is matrix ``i``'s ``in_sum2 / counts``,
            or ones where its count is zero.
        counts (torch.Tensor): Chunk count per matrix, shape
            ``(matrices,)``, float32.

    Examples:
        Read an expert stack's row for expert 57:

        ```python
        entry = load_imatrix(Path("model.imatrix.gguf"))["blk.3.ffn_up_exps.weight"]
        assert entry.counts.numel() == 128
        weights = entry.column_weights[57]
        ```
    """

    column_weights: torch.Tensor
    counts: torch.Tensor


def _resolve_name(param_name: str) -> tuple[str, int | None, bool] | None:
    """Map a parameter to its GGUF tensor, its row, and its layout.

    One parse serves every lookup. Parsing the name twice let the
    tensor name and the expert index disagree on the same string.

    Args:
        param_name: The dotted parameter name the loaded model
            reports.

    Returns:
        ``(gguf_name, expert, fused)``, or None when the table has
        no mapping. `expert` is the routed-expert index, or None
        when the parameter is not one expert of an expert stack.
        `fused` is True when the parameter is a whole fused expert
        stack (#202).
    """
    direct = _DIRECT_TO_GGUF.get(param_name)
    if direct is not None:
        return direct, None, False
    match = _LAYER_PARAM.match(param_name)
    if match is None:
        fused = _FUSED_PARAM.match(param_name)
        stem = None if fused is None else _FUSED_EXPERT_TO_GGUF.get(fused.group(2))
        if fused is None or stem is None:
            return None
        return f"blk.{fused.group(1)}.{stem}.weight", None, True
    suffix = match.group(2)
    expert = _EXPERT_INDEX.search(suffix)
    if expert is None:
        stem = _SUFFIX_TO_GGUF.get(suffix)
        index = None
    else:
        stem = _EXPERT_PROJ_TO_GGUF.get(suffix.rsplit(".", 1)[-1])
        index = int(expert.group(1))
    if stem is None:
        return None
    return f"blk.{match.group(1)}.{stem}.weight", index, False


def gguf_tensor_name(param_name: str) -> str | None:
    """Map an HF parameter name to its GGUF tensor name.

    A routed expert maps to the expert stack that holds it. All 128
    experts of a projection share one name (#159). A fused expert
    stack maps to the same name (#202).

    Args:
        param_name: The dotted parameter name the loaded model
            reports.

    Returns:
        The GGUF tensor name, or None when the table has no mapping —
        the caller treats that as uncovered.

    Examples:
        Map one routed expert to its expert stack:

        ```python
        from vramfit.adapters.outbound.scan.imatrix import gguf_tensor_name

        name = "backbone.layers.3.mixer.experts.57.down_proj.weight"
        assert gguf_tensor_name(name) == "blk.3.ffn_down_exps.weight"
        ```
    """
    resolved = _resolve_name(param_name)
    return None if resolved is None else resolved[0]


def _check_counts(path: Path, name: str, count: torch.Tensor) -> None:
    """Refuse a counts tensor that cannot be a chunk tally.

    A count divides the sums. An infinite count drives every column
    weight to zero, and a count that is negative or not a number
    lands in the zero-count branch and weighs every column 1. Both
    results pass `check_imatrix_weights`, so a 30-hour scan would
    price against a dead importance signal.

    Args:
        path: The imatrix file, named in the message.
        name: The GGUF tensor name, named in the message.
        count: The counts tensor as read.

    Raises:
        ValueError: If the tensor is empty, or holds a value that is
            negative or not finite.
    """
    if count.numel() == 0:
        raise ValueError(f"{path}: {name}.counts is empty")
    if not bool(torch.isfinite(count).all()):
        raise ValueError(f"{path}: {name}.counts holds a value that is not finite")
    if bool((count < 0).any()):
        raise ValueError(f"{path}: {name}.counts holds a negative value")


def load_imatrix(path: Path) -> dict[str, ImatrixEntry]:
    """Read a GGUF imatrix into one entry per GGUF tensor name.

    Args:
        path: The ``.gguf`` imatrix file ``llama-imatrix`` wrote.

    Returns:
        `ImatrixEntry` per GGUF tensor name, keeping one row per
        matrix. A dense tensor holds one row. An expert stack holds
        one row per expert. A column whose chunk count is zero
        weighs 1, per ``load_imatrix``.

    Raises:
        ValueError: If the file is not an imatrix, a tensor name
            carries neither known suffix, a sums tensor arrives
            without its counts twin (or the reverse), or the file
            holds no data at all. Every malformation here would
            otherwise shrink coverage silently. The refusal also
            covers a counts tensor that is empty, or that holds a
            value which is negative or not finite, and an
            ``in_sum2`` whose shape disagrees with its counts. A
            non-finite count would divide the sums to zeros or ones,
            and both pass `check_imatrix_weights`.

            The unknown-suffix refusal is deliberately stricter than
            the C loader, which skips unrecognized tensors. A suffix
            rename in a future imatrix format must fail loudly here.
            It must not price a 30-hour scan unassisted.
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
        _check_counts(path, name, count)
        if sum2.numel() % matrices:
            raise ValueError(
                f"{path}: {name}.in_sum2 has {sum2.numel()} entries, "
                f"not divisible by its {matrices} counts"
            )
        # The buffer reads as (matrices, columns), which is the order
        # llama-quantize slices per expert (llama-quant.cpp:1256-1262).
        # A transposed writer would pass the divisibility check above
        # and stripe every expert's row across all the others.
        if sum2.dim() > 1 and sum2.shape[0] != matrices:
            raise ValueError(
                f"{path}: {name}.in_sum2 reads as {tuple(sum2.shape)}, "
                f"not {matrices} rows of {sum2.numel() // matrices}"
            )
        if sum2.dim() == 1 and matrices != 1:
            raise ValueError(
                f"{path}: {name}.in_sum2 is one dimension, and its counts "
                f"claim {matrices} matrices"
            )
        per_matrix = sum2.reshape(matrices, -1)
        divisors = count.reshape(matrices, 1)
        entries[name] = ImatrixEntry(
            column_weights=torch.where(
                divisors > 0, per_matrix / divisors, torch.ones_like(per_matrix)
            ),
            counts=count.reshape(-1),
        )
    return entries


def _matrix_row(
    entry: ImatrixEntry, param_name: str, gguf_name: str, expert: int | None
) -> int:
    """Locate one parameter's row inside its imatrix entry.

    Args:
        entry: The entry `load_imatrix` read for `gguf_name`.
        param_name: The HF dotted parameter name.
        gguf_name: The GGUF tensor name `param_name` maps to.
        expert: The routed-expert index, or None.

    Returns:
        The row index. A routed expert reads its own index. Every
        other parameter reads row 0.

    Raises:
        ValueError: If the entry's matrix count does not fit the
            parameter. A routed expert needs an entry of more than
            one matrix, and an index inside it. Every other
            parameter needs an entry of exactly one matrix. Either
            mismatch means the imatrix and the checkpoint describe
            different models. Reading row 0 anyway would price
            against another expert's columns.
    """
    matrices = int(entry.counts.numel())
    if expert is None:
        if matrices != 1:
            raise ValueError(
                f"{gguf_name} holds {matrices} matrices, and {param_name} "
                "carries no expert index. The imatrix does not describe "
                "this checkpoint."
            )
        return 0
    # Guard one matrix separately. Expert 0 would otherwise index a
    # dense row and price against it, while experts 1 and up raise.
    if matrices == 1:
        raise ValueError(
            f"{param_name} is expert {expert}, and {gguf_name} holds one "
            "matrix. The imatrix comes from a model without an expert stack."
        )
    if expert >= matrices:
        raise ValueError(
            f"{param_name} is expert {expert}, and {gguf_name} holds "
            f"only {matrices} matrices"
        )
    return expert


def _zero_coverage_message(
    by_gguf_name: Mapping[str, ImatrixEntry], rows_by_param: Mapping[str, int]
) -> str:
    """Report why an imatrix covered no parameter.

    Four causes reach the same empty result, and the loop tallies
    each. Naming the file alone sends an operator to regenerate a
    correct matrix, which costs GPU hours and fails the same way. An
    unmapped decoder root lands in the first count (#423 nested its
    decoder and every name reported unmapped until the root joined
    the table). The counts point at the cause.
    The fused cause is a rule, not a defect: this reader prices the
    expert stacks unassisted (ADR-0026), and the q0 reader carries
    their assisted fit (ADR-0018, 2026-08-21 amendment).

    Args:
        by_gguf_name: Entries keyed by GGUF tensor name.
        rows_by_param: Row length per HF parameter name.

    Returns:
        The message, counting the parameters under each cause.
    """
    unmapped = fused = absent = misaligned = 0
    for name, rows in rows_by_param.items():
        resolved = _resolve_name(name)
        if resolved is None:
            unmapped += 1
        elif resolved[2]:
            fused += 1
        elif resolved[0] not in by_gguf_name:
            absent += 1
        elif rows % SUPER_BLOCK:
            misaligned += 1
    return (
        f"the imatrix covers none of the {len(rows_by_param)} parameters. "
        f"{unmapped} names have no GGUF mapping, so check the name table "
        f"covers this family. {fused} are fused expert stacks, which stay "
        f"unassisted by rule (ADR-0026). {absent} mapped to a tensor the "
        f"file does not hold, so check the file matches this model. "
        f"{misaligned} have rows that do not divide into "
        f"{SUPER_BLOCK}-element super-blocks, which no k-quant fit can "
        "price (ADR-0020)."
    )


def resolve_assisted_weights(
    by_gguf_name: Mapping[str, ImatrixEntry], rows_by_param: Mapping[str, int]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Match loaded imatrix entries to a set of parameters.

    The meter calls this after model load. `load_imatrix` runs
    first, so a malformed file refuses before the load burns
    minutes. A routed expert reads its own row of the expert stack,
    so all 128 experts of a projection resolve against one entry
    (#177). A fused expert stack reports uncovered by rule
    (ADR-0026, 2026-08-13 amendment). Under this reader the expert
    stacks price unassisted. The q0 reader carries their assisted
    fit (ADR-0018, 2026-08-21 amendment).

    Args:
        by_gguf_name: Entries keyed by GGUF tensor name, from
            `load_imatrix`.
        rows_by_param: Row length per HF parameter name.

    Returns:
        ``(covered, uncovered)`` — float32 column weights keyed by
        parameter name, and the names the imatrix does not cover, in
        input order. A parameter whose rows do not divide into
        super-blocks joins the uncovered set. It cannot price
        assisted (ADR-0020), and the fallback beats refusing a
        multi-day scan over one tensor. The run log and the console
        echo report it with the rest of the uncovered names.

    Raises:
        ValueError: If a covered tensor's weight length does not
            match the parameter's row length. A silent mismatch
            would price against the wrong columns. The refusal also
            covers an entry whose matrix count does not fit the
            parameter, two parameters that claim one imatrix row
            (#193), and a run where no parameter is covered at all.
            A scan on zero coverage would price every cell
            unassisted under the assisted label.
    """
    covered: dict[str, torch.Tensor] = {}
    uncovered: list[str] = []
    claimed: dict[tuple[str, int], str] = {}
    for name, rows in rows_by_param.items():
        resolved = _resolve_name(name)
        if resolved is None:
            uncovered.append(name)
            continue
        gguf_name, expert, fused = resolved
        # The rule fires before the entry lookup and the super-block
        # gate. Either would refuse with a wrong diagnosis. A fused
        # expert stack with 256-divisible rows would reach
        # _matrix_row and fail as a dense mismatch.
        if fused:
            uncovered.append(name)
            continue
        entry = by_gguf_name.get(gguf_name)
        if entry is None or rows % SUPER_BLOCK:
            uncovered.append(name)
            continue
        row = _matrix_row(entry, name, gguf_name, expert)
        claimant = claimed.setdefault((gguf_name, row), name)
        if claimant != name:
            raise ValueError(
                f"{name} and {claimant} both claim {gguf_name} row {row}. "
                "One of them would price against the wrong columns."
            )
        weight = entry.column_weights[row]
        if weight.numel() != rows:
            raise ValueError(
                f"imatrix weights for {name} ({gguf_name}) have "
                f"{weight.numel()} entries, the parameter rows have {rows}"
            )
        covered[name] = weight
    if rows_by_param and not covered:
        raise ValueError(_zero_coverage_message(by_gguf_name, rows_by_param))
    return covered, tuple(uncovered)


def _lround(value: float) -> int:
    """Round a stored count the C loader's way.

    torch.round breaks a tie to even. ``std::lround``
    (``imatrix-loader.cpp:158``) breaks it away from zero, and
    counts are never negative here.

    Args:
        value: The float32 count the file stores.

    Returns:
        The rounded count.
    """
    return math.floor(value + 0.5)


def resolve_imatrix_counts(
    by_gguf_name: Mapping[str, ImatrixEntry], shapes: Mapping[str, Sequence[int]]
) -> tuple[dict[str, int | tuple[int, ...]], tuple[str, ...]]:
    """Match each parameter to its imatrix count or count vector.

    The return shape separates the two quantities an imatrix counts
    (ADR-0026, 2026-08-13 amendment). A dense name keeps its scalar
    chunk tally, which is no routing frequency — folding it into an
    expert distribution would add a phantom expert (#193). An
    indexed 2D parameter reads its own scalar count by expert index
    (#187). A fused 3D parameter reads its whole entry as one count
    vector, keyed by the loaded parameter name. Element ``i`` is
    expert ``i``'s routing frequency: transformers stacks checkpoint
    experts in numeric order, the same order the imatrix rows keep.
    The read never constructs indexed names for a fused parameter.

    Routing frequency prices nothing. So this resolver applies no
    super-block gate, and an expert stack whose rows refuse a
    k-quant fit still reports its counts. ADR-0026 decisions 4 and 5
    read these counts. Decision 2 stays Proposed and no caller may
    weight damage by them until #178 reports.

    Args:
        by_gguf_name: Entries keyed by GGUF tensor name, from
            `load_imatrix`.
        shapes: Parameter shapes keyed by the names the loaded
            model reports.

    Returns:
        ``(covered, uncovered)`` — a count per parameter name, and
        the names the imatrix does not cover, both in input order.
        A dense name holds a scalar chunk count. A fused expert
        stack holds its count vector. An uncovered name stays out
        of the mapping rather than reading zero. Zero is a real
        count that ADR-0026 decision 5 reports. Each count rounds
        half away from zero, per `_lround`.

    Raises:
        ValueError: If the entry's matrix count does not fit the
            parameter. For a fused expert stack the shape must be
            3D and the entry's count length must equal the
            parameter's first dimension. A mismatch means the
            imatrix and the checkpoint describe different models.

    Examples:
        Read one fused expert stack's routing distribution:

        ```python
        entries = load_imatrix(Path("model.imatrix.gguf"))
        stack = "model.layers.3.mixer.experts.up_proj"
        counts, uncovered = resolve_imatrix_counts(entries, {stack: (128, 1856, 2688)})
        assert len(counts[stack]) == 128
        ```
    """
    covered: dict[str, int | tuple[int, ...]] = {}
    uncovered: list[str] = []
    for name, shape in shapes.items():
        resolved = _resolve_name(name)
        entry = None if resolved is None else by_gguf_name.get(resolved[0])
        if resolved is None or entry is None:
            uncovered.append(name)
            continue
        gguf_name, expert, fused = resolved
        if fused:
            # The name decides the fused fork, and the shape vouches
            # for it. Indexing dimension 0 blind would read a 2D
            # shape as a stack and refuse an empty one as IndexError.
            if len(shape) != _FUSED_STACK_DIMS:
                raise ValueError(
                    f"{name} names a fused expert stack, and its shape "
                    f"{tuple(shape)} has {len(shape)} dimensions, not "
                    f"{_FUSED_STACK_DIMS}"
                )
            experts = int(shape[0])
            matrices = int(entry.counts.numel())
            if matrices != experts:
                raise ValueError(
                    f"{name} holds {experts} experts on its first "
                    f"dimension, and {gguf_name} counts {matrices} "
                    "matrices. The imatrix does not describe this "
                    "checkpoint."
                )
            covered[name] = tuple(_lround(c) for c in entry.counts.tolist())
            continue
        count = entry.counts[_matrix_row(entry, name, gguf_name, expert)]
        covered[name] = _lround(float(count.item()))
    return covered, tuple(uncovered)


def expert_stack_count_vectors(
    counts: Mapping[str, int | tuple[int, ...]], members: Iterable[str]
) -> tuple[tuple[int, ...], ...] | None:
    """Select one group's expert-stack count vectors, all or nothing.

    The selection rule behind ADR-0026 decision 4, scoped by the
    2026-08-13 #201 amendment: the summary reduces expert-stack
    count vectors only, and it appears only when every expert-stack
    member resolved its full count vector. A scalar chunk tally
    never enters — the router, the shared experts, and every dense
    member stay out. An empty resolution leaves the summary absent
    and never refuses (the #198 amendment) — #194 owns separating a
    dense-only group from an unresolved one.

    Args:
        counts: Resolved counts per parameter name, from
            `resolve_imatrix_counts`.
        members: The group's member parameter names.

    Returns:
        One count vector per fused expert-stack member, or None when
        the group holds no expert stack or any of its expert stacks
        did not resolve.

    Examples:
        A covered one-stack group:

        ```python
        stack = "model.layers.3.mixer.experts.up_proj"
        vectors = expert_stack_count_vectors({stack: (3, 9)}, [stack])
        assert vectors == ((3, 9),)
        ```
    """
    stacks = [
        counts.get(name)
        for name in members
        if (resolved := _resolve_name(name)) is not None and resolved[2]
    ]
    vectors = tuple(vector for vector in stacks if isinstance(vector, tuple))
    if not stacks or len(vectors) != len(stacks):
        return None
    return vectors


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
