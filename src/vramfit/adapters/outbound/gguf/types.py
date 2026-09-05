"""Pure GGUF type mapping: nominal bits to K-quant types (ADR-0012).

The decision core of the GGUF backend, kept free of IO so the mapping
is testable and the verified fake can share it. Nominal precisions
map to K-quant types (the full llama.cpp capability set since
ADR-0013) and nominal 16 maps to `f16`, the passthrough a recipe
holds an unmeasured group at (ADR-0029 decision 4), layer groups
map to escaped `blk.<n>.` regex patterns
across the three naming families the scan produces, routed-expert
stack groups map to their fused `blk.<n>.ffn_<proj>_exps.` tensor
(#159, #161) through their own type table — k-quant super-blocks do
not divide the stack rows, so the rows carry block-32 and block-64
types (`q8_0`, `q5_1`, `q5_0`, `q4_0`, `q2_0`), and nominal 3 refuses
over the empty 2.25-4.25 bits-per-weight gap (ADR-0028) — layer-class
groups map
through the class table to `blk.<n>.<stem>.` patterns, where an
unquantizable class instead pins at the F16 passthrough and refuses
any lower width (the 2026-08-20 amendment), protected tensors map
through the same class table to per-tensor patterns under a free
prefix (ADR-0022, #365), excluded pairs map to the full GGUF
tensor names ``--exclude-weights`` deletes by substring
(ADR-0023),
and the embedding and `lm_head` groups map to the quantizer's
dedicated embedding and output flags. Every mapped group and
protected tensor hangs from one root, and that root sits in the
scan name table's `NAME_TABLE_ROOTS` (#208). The embedding name set
carries four names: `model.embed_tokens`, `backbone.embeddings`,
its reconciled form `model.embeddings`, and Gemma 4's nested
`model.language_model.embed_tokens` (#423). The backend's own runtime
name is the domain's `LLAMA_CPP` constant, so the table key and
the pack check cannot drift apart. A
recipe recorded for a foreign runtime, or anything the table cannot
map, raises `PackError` instead of guessing. The base ftype table
reaches tensors only: the label the packed file declares is the
modal type by bytes, which
[vramfit.adapters.outbound.gguf.file_type][] writes after the
quantizer runs (ADR-0012 decision 3 as amended 2026-09-04).

`all_overrides` composes the protection and group overrides in the
quantizer's priority order. One function owns that composition, so
the pack step and its pre-run match check cannot disagree on which
overrides exist (#303).

`output_group_type` and `output_tensor_type` split one question in
two. The second answers what value the output flag carries. The first
answers whether the recipe scanned a head at all, which is what
decides whether that flag must reach a tensor in the base GGUF (#306).
Only a scanned `lm_head` group makes it load-bearing — ADR-0012
decision 2 rules the tied fallback a no-op.

Examples:
    Map a recipe to quantizer inputs:

    ```python
    from vramfit.adapters.outbound.gguf import types

    base = types.base_type(recipe)
    embed = types.token_embedding_type(recipe)
    overrides = types.all_overrides(recipe, row_widths)
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: The subprocess driver
      that feeds these values to ``llama-quantize``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from vramfit.domain.errors import VramfitError
from vramfit.domain.model import Recipe
from vramfit.domain.pack import TypeOverride
from vramfit.domain.runtime import (
    LLAMA_CPP,
    rows_refuse_super_block,
    unquantizable_filter,
)
from vramfit.domain.scan import NAME_TABLE_ROOTS
from vramfit.domain.sizes import REFERENCE_BITS, reconcile_root

# The 16 row is the F16 passthrough (ADR-0029 decision 4). A recipe
# holds an unmeasured group at reference precision, and `f16` is what
# reference precision packs as.
GGML_TYPE_BY_BITS: Final[dict[int, str]] = {
    16: "f16",
    8: "q8_0",
    6: "q6_k",
    5: "q5_k",
    4: "q4_k",
    3: "q3_k",
    2: "q2_k",
}

# The expert-stack type table (ADR-0028). Every k-quant packs
# 256-element super-blocks, and the expert-stack rows (2688, 1856) do
# not divide by 256 — the quantizer would silently substitute. Every
# entry here has a block size that divides both row widths. Effective
# bits per weight: f16 at 16.00, q8_0 at 8.50, q5_1 at 6.00, q5_0 at
# 5.50, q4_0 at 4.50, q2_0 at 2.25. The 6 and 5 rows date from the
# 2026-09-04 amendment (#232), and both block 32. The f16 row is the
# ADR-0029 passthrough, and it has no block to divide. The table also
# reaches a layer-class group whose rows refuse the super-block — the
# Nemotron-H classes qualify at 2688 (the 2026-08-20 amendment).
EXPERT_STACK_TYPE_BY_BITS: Final[dict[int, str]] = {
    16: "f16",
    8: "q8_0",
    6: "q5_1",
    5: "q5_0",
    4: "q4_0",
    2: "q2_0",
}

# The quantizer's positional type argument speaks ftype names, not
# tensor-type names. Each entry is the pure-type ftype for the same
# nominal bits as GGML_TYPE_BY_BITS, and the two tables agree at
# every key. The expert-stack table above disagrees at 2, 4, 5, and 6
# on purpose (ADR-0028): k-quants do not divide the stack rows, and a
# dense tensor no override covers still takes this k-quant floor.
# The base ftype reaches tensors only. The label the file declares
# is the modal type by bytes, written after the quantizer runs
# (ADR-0012 decision 3 as amended 2026-09-04, #413, #414).
BASE_FTYPE_BY_BITS: Final[dict[int, str]] = {
    16: "F16",
    8: "Q8_0",
    6: "Q6_K",
    5: "Q5_K_S",
    4: "Q4_K_S",
    3: "Q3_K_S",
    2: "Q2_K",
}

# The one runtime this backend packs for. A recipe planned for
# another runtime must not silently become a GGUF (ADR-0013).
GGUF_RUNTIME: Final[str] = LLAMA_CPP

# The embedding group names the scan produces, across naming
# families. llama-family checkpoints say `model.embed_tokens`.
# Nemotron-H says `backbone.embeddings`, and ADR-0029's size source
# reconciles that root to `model.embeddings` (the 2026-08-20
# amendment). Gemma 4 wraps the decoder in a multimodal shell and
# says `model.language_model.embed_tokens` — the scan side maps the
# same name (#423). All four drive the one `--token-embedding-type`
# flag, so the backend needs the names, not a pattern.
EMBEDDING_GROUPS: Final[frozenset[str]] = frozenset(
    {
        "model.embed_tokens",
        "backbone.embeddings",
        "model.embeddings",
        "model.language_model.embed_tokens",
    }
)

OUTPUT_GROUP: Final[str] = "lm_head"

# A layer group, under the three naming families the scan produces
# (`domain.scan` names them the same way). The prefix is free, so
# `model.layers.4` and Nemotron-H's `backbone.layers.4` both yield 4
# (#160). GGUF numbers every layer `blk.<n>.`, whatever the
# checkpoint calls it.
_LAYER_GROUP: Final[re.Pattern[str]] = re.compile(r"^.+\.(?:layers|h|blocks)\.(\d+)$")

# A routed-expert stack group: a layer prefix, then `.experts.` with
# the expert index already collapsed by `group_key`, then the
# projection (#161). The dot before `experts` matters — it keeps
# `shared_experts` out of the fused-stack table. A shared expert is
# its own GGUF tensor, `ffn_up_shexp`, and maps as a layer class
# below (the 2026-08-20 amendment).
_EXPERT_STACK: Final[re.Pattern[str]] = re.compile(
    r"^.+\.(?:layers|h|blocks)\.(\d+)\.(?:.*\.)?experts\.([A-Za-z0-9_]+)$"
)

# A layer-class group: a layer prefix, then the class suffix the
# class table keys on. The suffix carries two or three dot-separated
# segments — `mixer.shared_experts.down_proj` carries three (the
# 2026-08-20 amendment). Expert stacks match too, so `tensor_overrides`
# tries the stack shape first.
_CLASS_GROUP: Final[re.Pattern[str]] = re.compile(
    r"^.+\.(?:layers|h|blocks)\.(\d+)\.(.+)$"
)

# The parameter-tree root a layer or expert-stack group hangs from.
# `blk.<n>.` addresses exactly one layer stack, so a recipe that
# names two of them cannot pack. The target carries `backbone` and
# `mtp`, and a multimodal checkpoint carries a vision tower that
# GGUF names `v.blk.<n>.` instead. A recipe under one such foreign
# root would map every group to `blk.<n>.` and match nothing, so
# `_claim_root` also holds the root to the scan name table's
# `NAME_TABLE_ROOTS` (#208).
_STACK_ROOT: Final[re.Pattern[str]] = re.compile(
    r"^(.+?)\.(?:layers|h|blocks)\.\d+(?:\.|$)"
)

# llama.cpp fuses one layer's routed experts into a single 3D tensor
# that carries one quantization type, so the pack addresses the
# stack and never one expert inside it (#159). HF projection name to
# fused GGUF tensor.
GGUF_EXPERT_STACK_BY_HF: Final[dict[str, str]] = {
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
    "gate_proj": "ffn_gate_exps",
}

# The fixed class table (ADR-0022, extended by the 2026-08-20
# ADR-0012 amendment): HF tensor suffix to GGUF tensor stem. The
# first seven rows are the quantized projections of a llama-family
# layer. The nine Nemotron-H rows follow, verified against
# `gguf-py/gguf/tensor_mapping.py` at the pinned instrument. The
# `mixer.gate` row exists for the name mapping alone — the class pins
# at the F16 passthrough through `UNQUANTIZABLE_CLASS_FILTERS`.
GGUF_SUFFIX_BY_HF: Final[dict[str, str]] = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "mixer.in_proj": "ssm_in",
    "mixer.out_proj": "ssm_out",
    "mixer.gate": "ffn_gate_inp",
    "mixer.shared_experts.up_proj": "ffn_up_shexp",
    "mixer.shared_experts.down_proj": "ffn_down_shexp",
    "mixer.q_proj": "attn_q",
    "mixer.k_proj": "attn_k",
    "mixer.v_proj": "attn_v",
    "mixer.o_proj": "attn_output",
}

# A protection target: a layer tensor under a free prefix, in the
# family shape `_CLASS_GROUP` holds (#160). ADR-0012's 2026-08-20
# amendment rules the free prefix — the tensor name maps under any
# root, and `_claim_root` holds a recipe to one supported root. The suffix carries no depth
# limit, so the three-segment shared-expert rows map too.
# `GGUF_SUFFIX_BY_HF` still decides what maps.
_LAYER_TENSOR: Final[re.Pattern[str]] = re.compile(
    r"^.+\.(?:layers|h|blocks)\.(\d+)\.(.+)\.weight$"
)


class PackError(VramfitError, RuntimeError):
    """The pack backend cannot map or apply a recipe.

    Raised for precisions outside the ADR-0012 table (as amended by
    ADR-0013), groups without a GGUF tensor mapping, recipes planned
    for another runtime, and toolchain failures. Inherits
    `VramfitError` per ADR-0011 and `RuntimeError` for the port
    contract.

    Examples:
        A 7-bit assignment has no mapping:

        ```python
        ggml_type_for(7)  # raises PackError
        ```
    """


def check_runtime(recipe: Recipe) -> None:
    """Reject a recipe planned for a runtime this backend cannot serve.

    A recipe whose runtime is None passes — it was planned without a
    runtime constraint, and the type tables still decide what it can
    map.

    Args:
        recipe: The recipe to pack.

    Raises:
        PackError: If the recipe records a runtime other than
            ``llama.cpp``.

    Examples:
        A vLLM recipe never packs to GGUF:

        ```python
        check_runtime(vllm_recipe)  # raises PackError
        ```
    """
    if recipe.runtime is not None and recipe.runtime != GGUF_RUNTIME:
        raise PackError(
            f'recipe targets runtime "{recipe.runtime}" — the GGUF backend '
            f"packs for {GGUF_RUNTIME} (ADR-0013)"
        )


def ggml_type_for(bits: int) -> str:
    """Map one nominal precision to its K-quant tensor type.

    Args:
        bits: Nominal precision from a recipe assignment.

    Returns:
        The GGUF tensor-type name, e.g. ``q4_k``.

    Raises:
        PackError: If the ADR-0012 table has no entry for ``bits``.

    Examples:
        The 4-bit entry of the table:

        ```python
        assert ggml_type_for(4) == "q4_k"
        ```
    """
    try:
        return GGML_TYPE_BY_BITS[bits]
    except KeyError:
        raise PackError(
            f"no GGUF type maps {bits}-bit — the ADR-0012 table covers "
            f"{sorted(GGML_TYPE_BY_BITS)}"
        ) from None


def expert_stack_type_for(bits: int, group: str, kind: str = "expert stack") -> str:
    """Map one ADR-0028-routed precision to its tensor type.

    Routed-expert stacks map here, and so does a layer-class group
    whose rows refuse the 256 super-block (the 2026-08-20 amendment).

    Args:
        bits: Nominal precision from a recipe assignment.
        group: The group, named in every refusal.
        kind: What the refusal calls the group — ``expert stack`` or
            ``layer-class group``.

    Returns:
        The GGUF tensor-type name, e.g. ``q2_0``.

    Raises:
        PackError: If ``bits`` is 3 — no GGUF type lands between 2.25
            and 4.25 bits per weight on the stack rows (ADR-0028
            decision 2). Also if the table has no entry for ``bits``.

    Examples:
        The 2-bit entry of the table:

        ```python
        assert expert_stack_type_for(2, "m.layers.0.experts.up_proj") == "q2_0"
        ```
    """
    if bits == 3:  # noqa: PLR2004 - the ADR-0028 decision 2 refusal is about exactly nominal 3
        raise PackError(
            f'{kind} "{group}" cannot pack at nominal 3 — no GGUF type '
            f"lands between 2.25 and 4.25 bits per weight on the stack rows "
            f"(ADR-0028). The neighboring table entries are 2 -> q2_0 "
            f"(2.25 bits/weight) and 4 -> q4_0 (4.50 bits/weight)"
        )
    try:
        return EXPERT_STACK_TYPE_BY_BITS[bits]
    except KeyError:
        raise PackError(
            f'{kind} "{group}" has no type for {bits}-bit — the '
            f"ADR-0028 table covers {sorted(EXPERT_STACK_TYPE_BY_BITS)}"
        ) from None


def base_type(recipe: Recipe) -> str:
    """Choose the quantizer's base ftype: the recipe's precision floor.

    Applied with ``--pure``, the base type is what any tensor no
    override covers gets — exactly, with the quantizer's heuristic
    mixing disabled (ADR-0012).

    Args:
        recipe: The recipe to pack.

    Returns:
        The ftype name, e.g. ``Q4_K_S``.

    Raises:
        PackError: If the lowest assigned precision has no table
            entry.

    Examples:
        A recipe mixing 8- and 4-bit floors at 4:

        ```python
        assert base_type(recipe) == "Q4_K_S"
        ```
    """
    floor = min(assignment.bits for assignment in recipe.assignments)
    try:
        return BASE_FTYPE_BY_BITS[floor]
    except KeyError:
        raise PackError(
            f"no GGUF base type maps {floor}-bit — the ADR-0012 table covers "
            f"{sorted(BASE_FTYPE_BY_BITS)}"
        ) from None


def token_embedding_type(recipe: Recipe) -> str | None:
    """Map the embedding group's assignment to the embedding flag.

    ``--token-embedding-type`` binds the embedding tensor before any
    pattern override, so the embedding group never becomes a pattern.
    When the model ties embeddings, this assignment also governs the
    output head (ADR-0012). The group carries one of the names in
    `EMBEDDING_GROUPS`, which differ by naming family.

    Args:
        recipe: The recipe to pack.

    Returns:
        The tensor-type name for the embedding, or None when the
        recipe has no embedding group.

    Raises:
        PackError: If the embedding assignment's precision has no
            table entry.

    Examples:
        An embedding kept at 8-bit:

        ```python
        assert token_embedding_type(recipe) == "q8_0"
        ```
    """
    for assignment in recipe.assignments:
        if assignment.group in EMBEDDING_GROUPS:
            return ggml_type_for(assignment.bits)
    return None


def output_group_type(recipe: Recipe) -> str | None:
    """Map a scanned ``lm_head`` group's own assignment to a type.

    The scan produces this group on models with an untied head, and
    the 2026-07-29 amendment gives it its own assignment. So a
    non-None result means the recipe asks the output head for a type
    of its own, rather than inheriting the embedding's.

    `output_tensor_type` reads it for the flag's value. The pack step
    reads it for a second reason: only a scanned head makes the flag
    load-bearing, so only then must the base GGUF carry
    ``output.weight`` (#306).

    Args:
        recipe: The recipe to pack.

    Returns:
        The tensor-type name the ``lm_head`` group assigns, or None
        when the recipe carries no such group.

    Raises:
        PackError: If the head assignment's precision has no table
            entry.

    Examples:
        A recipe scanned on a tied model carries no head group:

        ```python
        assert output_group_type(recipe) is None
        ```
    """
    for assignment in recipe.assignments:
        if assignment.group == OUTPUT_GROUP:
            return ggml_type_for(assignment.bits)
    return None


def output_tensor_type(recipe: Recipe) -> str | None:
    """Map the output head's assignment to the output flag.

    ``--output-tensor-type`` binds the output tensor before any
    pattern override. An ``lm_head`` group — scanned on models with
    an untied head — carries its own assignment, which
    `output_group_type` reads. Without one, the embedding assignment
    pins the head, so a tied model's single scanned group governs
    both tensors (ADR-0012).

    This collapses the two sources into the flag's value. A caller
    that needs to tell them apart reads `output_group_type` instead.

    Args:
        recipe: The recipe to pack.

    Returns:
        The tensor-type name for the output head, or None when the
        recipe has neither an ``lm_head`` nor an embedding group.

    Raises:
        PackError: If the governing assignment's precision has no
            table entry.

    Examples:
        An untied head held at 4-bit:

        ```python
        assert output_tensor_type(recipe) == "q4_k"
        ```
    """
    scanned = output_group_type(recipe)
    if scanned is not None:
        return scanned
    return token_embedding_type(recipe)


def gguf_tensor_name(tensor: str) -> str:
    r"""Map one HF layer tensor name to its GGUF tensor name.

    Args:
        tensor: HF tensor name, e.g.
            ``model.layers.4.self_attn.v_proj.weight``.

    Returns:
        The GGUF tensor name, e.g. ``blk.4.attn_v.weight``.

    Raises:
        PackError: If the tensor's class is unquantizable — the
            quantizer drops any override on such a tensor and exits
            0, so the pair would record a type the artifact does not
            carry (the 2026-08-20 amendment). That check runs first
            and needs no class-table row. Also if the name is not a
            ``.weight`` tensor under a ``.layers/.h/.blocks.`` layer
            family, or its suffix has no class-table row (ADR-0022).

    Examples:
        The G1 protection target:

        ```python
        assert (
            gguf_tensor_name("model.layers.4.self_attn.v_proj.weight")
            == "blk.4.attn_v.weight"
        )
        ```
    """
    # The filter check comes first, and reads the group form rather
    # than `_LAYER_TENSOR`. `mixer.conv1d` has no class-table row,
    # and its refusal must still name the upstream filter, not a
    # missing mapping.
    filter_name = unquantizable_filter(tensor.removesuffix(".weight"), LLAMA_CPP)
    if filter_name is not None:
        raise PackError(
            f'protected tensor "{tensor}" maps to a tensor llama-quantize '
            f'refuses to quantize, through the "{filter_name}" filter in '
            f"tensor_allows_quantization — the class packs at the F16 "
            f"passthrough and takes no per-tensor override (ADR-0012, "
            f"2026-08-20 amendment)"
        )
    match = _LAYER_TENSOR.match(tensor)
    if match is None:
        raise PackError(
            f'protected tensor "{tensor}" has no GGUF mapping — the name '
            f"is not a .weight tensor under a .layers/.h/.blocks. layer "
            f"family (ADR-0022)"
        )
    if match.group(2) not in GGUF_SUFFIX_BY_HF:
        raise PackError(
            f'protected tensor "{tensor}" has no GGUF mapping — the suffix '
            f'"{match.group(2)}" has no class-table row. The table holds '
            f"{sorted(GGUF_SUFFIX_BY_HF)} (ADR-0022)"
        )
    return f"blk.{match.group(1)}.{GGUF_SUFFIX_BY_HF[match.group(2)]}.weight"


def protection_overrides(recipe: Recipe) -> tuple[TypeOverride, ...]:
    r"""Translate resolved protection pairs into quantizer overrides.

    One override per protected tensor, from the recipe's resolved
    (tensor, precision) pairs — user glob input never reaches the
    quantizer (ADR-0022). Callers place these *before* the group
    overrides: the quantizer applies the first matching pattern, so
    order encodes priority.

    Args:
        recipe: The recipe to pack.

    Returns:
        Overrides in recipe order, empty for an unprotected recipe.

    Raises:
        PackError: If a protected tensor has no class-table mapping,
            or its precision has no type-table entry.

    Examples:
        A protected v_proj becomes an escaped tensor pattern:

        ```python
        assert protection_overrides(recipe) == (
            TypeOverride(r"blk\.4\.attn_v\.", "q5_k"),
        )
        ```
    """
    overrides: list[TypeOverride] = []
    for pair in recipe.protected_tensors:
        gguf_name = gguf_tensor_name(pair.tensor)
        prefix = gguf_name.removesuffix("weight")
        overrides.append(
            TypeOverride(
                pattern=re.escape(prefix),
                quant_type=ggml_type_for(pair.bits),
            )
        )
    return tuple(overrides)


def imatrix_exclusion_names(recipe: Recipe) -> tuple[str, ...]:
    """Name the GGUF tensors whose imatrix rows the pack must drop.

    One name per protected pair marked ``exclude_imatrix``
    (ADR-0023). The quantizer's ``--exclude-weights`` matches by
    substring, so the full tensor name — ``blk.<n>.<class>.weight``
    — matches exactly one imatrix row.

    Args:
        recipe: The recipe to pack.

    Returns:
        Full GGUF tensor names, in recipe order. Empty for a recipe
        without exclusions.

    Raises:
        PackError: If an excluded tensor has no class-table mapping.

    Examples:
        The G1c exclusion for layer 1:

        ```python
        assert imatrix_exclusion_names(recipe) == ("blk.1.attn_v.weight",)
        ```
    """
    return tuple(
        gguf_tensor_name(pair.tensor)
        for pair in recipe.protected_tensors
        if pair.exclude_imatrix
    )


def gguf_stack_prefix(group: str) -> str | None:
    """Map one routed-expert stack group to its GGUF tensor prefix.

    Args:
        group: Recipe group name, e.g.
            ``backbone.layers.3.mixer.experts.down_proj``.

    Returns:
        The GGUF tensor prefix, e.g. ``blk.3.ffn_down_exps.``, or
        None when the group is not a routed-expert stack.

    Raises:
        PackError: If the group is a routed-expert stack whose
            projection has no entry in the fused-stack table.

    Examples:
        The Nemotron 3.5 Lightning down projection:

        ```python
        group = "backbone.layers.3.mixer.experts.down_proj"
        assert gguf_stack_prefix(group) == "blk.3.ffn_down_exps."
        ```
    """
    match = _EXPERT_STACK.match(group)
    if match is None:
        return None
    suffix = GGUF_EXPERT_STACK_BY_HF.get(match.group(2))
    if suffix is None:
        raise PackError(
            f'expert stack "{group}" has no GGUF mapping — llama.cpp fuses '
            f"the projections {sorted(GGUF_EXPERT_STACK_BY_HF)} (#159)"
        )
    return f"blk.{match.group(1)}.{suffix}."


def _claim_root(name: str, roots: dict[str, str], kind: str = "group") -> None:
    """Hold every mapped group and protected tensor to one root.

    Args:
        name: Recipe group or protected tensor name.
        roots: Roots claimed so far, mapped to the labeled claimant
            of each. Updated in place.
        kind: What the refusal calls ``name`` — ``group`` or
            ``protected tensor`` (#367).

    Raises:
        PackError: If ``name`` hangs from a second root — the first
            matching override would land on the other root's tensor.
            The refusal names both claimants and both roots. Also if
            the one root is outside the scan name table's
            ``NAME_TABLE_ROOTS`` — every override would address
            ``blk.<n>.`` and match nothing (#208). The refusal names
            the root and the supported roots.
    """
    match = _STACK_ROOT.match(name)
    if match is None:
        return
    root = match.group(1)
    roots.setdefault(root, f'{kind} "{name}"')
    if len(roots) > 1:
        first = next(iter(roots))
        raise PackError(
            f'{roots[first]} under root "{first}" and {kind} "{name}" under '
            f'root "{root}" name two layer stacks — a GGUF pack numbers one '
            f'stack "blk.<n>." and the first matching override would land '
            f"on the other root's tensor (#183, #367)"
        )
    if root not in NAME_TABLE_ROOTS:
        raise PackError(
            f'{kind} "{name}" hangs from root "{root}", which the scan name '
            f"table does not support — the supported roots are "
            f"{', '.join(NAME_TABLE_ROOTS)}. A GGUF pack numbers the "
            f'decoder stack "blk.<n>.", so an override for another root '
            f"would match nothing (#208)"
        )


def _refuses_super_block(group: str, row_widths: Mapping[str, int]) -> bool:
    """Decide whether one group's measured rows take the ADR-0028 table.

    The pack reads the same widths the plan priced from, so the
    predicted type and the emitted type cannot disagree (issue #515).

    Args:
        group: Recipe group name, under either naming root.
        row_widths: Elements per row per group, from
            `vramfit.domain.sizes.discovered_group_rows`.

    Returns:
        True when the super-block does not divide the group's rows.

    Raises:
        PackError: If the group has no measured row width. A name
            cannot supply one, and a default would misprice the
            group silently.
        SizeSourceError: If the group hangs from a root the
            reconcile table does not carry. That refusal names the
            root, and the missing-width message would hide it.
    """
    width = row_widths.get(group)
    if width is None:
        # The recipe may spell a group under the checkpoint's root
        # while the widths key on the map's (ADR-0029 decision 7).
        width = row_widths.get(reconcile_root(group))
    if width is None:
        raise PackError(
            f'group "{group}" has no measured row width, and the 256 '
            f"super-block decision reads that width rather than a class "
            f"name (ADR-0028, issue #515). The checkpoint the pack reads "
            f"states it — check that --model names the checkpoint the "
            f"recipe was planned from (ADR-0029 decision 1)"
        )
    return rows_refuse_super_block(width)


def _class_override(
    group: str, bits: int, row_widths: Mapping[str, int]
) -> tuple[TypeOverride, ...] | None:
    r"""Map one layer-class group to its override, or hold it at F16.

    The class table keys on the tensor suffix under a free prefix,
    at two or three dot-separated segments (the 2026-08-20
    amendment). A class whose measured rows refuse the 256
    super-block routes through the ADR-0028 table, and the rest keep
    the ADR-0012 k-quant table. An unquantizable class emits no override — the
    quantizer refuses its tensors and holds them at the convert
    dtype, so the F16 passthrough is what packing already does.

    Args:
        group: Recipe group name, e.g.
            ``model.layers.3.mixer.in_proj``.
        bits: Nominal precision from the group's assignment.
        row_widths: Elements per row per group, from
            `vramfit.domain.sizes.discovered_group_rows`.

    Returns:
        A one-override tuple for a mapped class, an empty tuple for
        an unquantizable class held at the passthrough, or None when
        the group is not a class-table group.

    Raises:
        PackError: If an unquantizable class takes a width below the
            F16 passthrough — the refusal names the group and the
            upstream filter. Also if the precision has no entry in
            the class's type table.

    Examples:
        The Nemotron-H Mamba input projection at the passthrough:

        ```python
        assert _class_override(
            "model.layers.3.mixer.in_proj", 16, {"model.layers.3.mixer.in_proj": 2688}
        ) == (TypeOverride(r"blk\.3\.ssm_in\.", "f16"),)
        ```
    """
    match = _CLASS_GROUP.match(group)
    if match is None:
        return None
    filter_name = unquantizable_filter(group, LLAMA_CPP)
    if filter_name is not None:
        if bits != REFERENCE_BITS:
            raise PackError(
                f'group "{group}" maps to a tensor llama-quantize refuses to '
                f'quantize, through the "{filter_name}" filter in '
                f"tensor_allows_quantization — the class packs at the F16 "
                f"passthrough and never at {bits}-bit (ADR-0012, 2026-08-20 "
                f"amendment)"
            )
        return ()
    stem = GGUF_SUFFIX_BY_HF.get(match.group(2))
    if stem is None:
        return None
    quant = (
        expert_stack_type_for(bits, group, kind="layer-class group")
        if _refuses_super_block(group, row_widths)
        else ggml_type_for(bits)
    )
    return (TypeOverride(re.escape(f"blk.{match.group(1)}.{stem}."), quant),)


def tensor_overrides(
    recipe: Recipe, row_widths: Mapping[str, int]
) -> tuple[TypeOverride, ...]:
    r"""Translate recipe groups into quantizer tensor-type overrides.

    Three group shapes map. A layer group under any of the three
    naming families and any prefix (``model.layers.<n>``,
    ``backbone.layers.<n>``, Gemma 4's nested
    ``model.language_model.layers.<n>``) becomes the escaped
    pattern ``blk\.<n>\.``. A routed-expert
    stack group becomes the escaped pattern for its fused tensor,
    e.g. ``blk\.<n>\.ffn_up_exps\.`` (#159, #161). A layer-class
    group becomes the escaped pattern for its class-table stem, e.g.
    ``blk\.<n>\.ssm_in\.`` (the 2026-08-20 amendment). Escaping
    matters — an unescaped ``blk.1.`` would also match ``blk.11.``.
    The embedding and ``lm_head`` groups map to dedicated flags and
    are skipped here.

    A layer-class group of an unquantizable class emits no override.
    The quantizer refuses its tensors and holds them at the convert
    dtype, so the F16 passthrough is what packing already does. A
    lower width on such a group refuses, naming the upstream filter.

    Expert-stack and class overrides come first, ahead of the layer
    overrides. The quantizer applies the first matching pattern, and
    ``blk\.1\.`` also matches ``blk.1.ffn_up_exps.weight``. Callers
    place the protection overrides ahead of all three — a per-tensor
    pattern is the most specific (ADR-0022).

    Every mapped group and protected tensor must hang from one
    parameter-tree root. ``blk.<n>.`` addresses a single layer
    stack, so a recipe naming two of them maps both onto it. The
    first matching override would land on the other root's tensor.
    A protected tensor claims its root with the groups (#367). The
    one root must sit in the scan name table's ``NAME_TABLE_ROOTS``.
    A recipe under a foreign root, such as a vision tower GGUF names
    ``v.blk.<n>.``, would emit overrides that match nothing (#208).

    Args:
        recipe: The recipe to pack.
        row_widths: Elements per row per group, from
            `vramfit.domain.sizes.discovered_group_rows` over the
            checkpoint the recipe was planned from. The ADR-0028
            routing reads this width (issue #515).

    Returns:
        Expert-stack overrides in recipe order, then layer-class
        overrides in recipe order, then layer overrides in recipe
        order.

    Raises:
        PackError: If a group is not a layer group, a routed-expert
            stack, a layer-class group, the embedding, or the output
            head. Also if a routed-expert stack names a projection
            outside the fused-stack table, an ADR-0028-routed group
            names a precision outside that table (nominal 3 refuses
            over the empty 2.25-4.25 gap), an unquantizable class
            takes a width below the F16 passthrough, a group the
            routing reaches has no measured row width, the groups and
            protected tensors hang from two roots or from a root the
            scan name table does not support, or a precision has no
            table entry.

    Examples:
        The group ``model.layers.7`` at 4-bit becomes an escaped
        pattern:

        ```python
        assert TypeOverride(r"blk\.7\.", "q4_k") in tensor_overrides(recipe, {})
        ```
    """
    stacks: list[TypeOverride] = []
    classes: list[TypeOverride] = []
    layers: list[TypeOverride] = []
    roots: dict[str, str] = {}
    for pair in recipe.protected_tensors:
        _claim_root(pair.tensor, roots, kind="protected tensor")
    for assignment in recipe.assignments:
        if assignment.group in EMBEDDING_GROUPS or assignment.group == OUTPUT_GROUP:
            continue
        _claim_root(assignment.group, roots)
        prefix = gguf_stack_prefix(assignment.group)
        if prefix is not None:
            # A stack takes the ADR-0028 table only when its measured
            # rows refuse the super-block. A stack of 2048-wide rows
            # takes the k-quant table, like any other group (#515).
            bits = (
                expert_stack_type_for(assignment.bits, assignment.group)
                if _refuses_super_block(assignment.group, row_widths)
                else ggml_type_for(assignment.bits)
            )
            stacks.append(TypeOverride(re.escape(prefix), bits))
            continue
        mapped = _class_override(assignment.group, assignment.bits, row_widths)
        if mapped is not None:
            classes.extend(mapped)
            continue
        match = _LAYER_GROUP.match(assignment.group)
        if match is None:
            raise PackError(
                f'group "{assignment.group}" has no GGUF tensor mapping — the '
                "backend maps layer groups, routed-expert stacks, layer-class "
                "groups, the embedding, and the output head (ADR-0012, "
                "ADR-0022)"
            )
        bits = ggml_type_for(assignment.bits)
        layers.append(TypeOverride(rf"blk\.{match.group(1)}\.", bits))
    return tuple(stacks) + tuple(classes) + tuple(layers)


def all_overrides(
    recipe: Recipe, row_widths: Mapping[str, int]
) -> tuple[TypeOverride, ...]:
    """Compose every override the quantizer receives, in priority order.

    Protection overrides lead, because the quantizer applies the
    first matching pattern and a per-tensor pattern is the most
    specific of the three (ADR-0022). The group overrides follow in
    `tensor_overrides`' own order.

    One function owns the composition, so the pack step and the
    pre-run match check never drift apart on which overrides exist
    (#190, #303).

    Args:
        recipe: The recipe to pack.
        row_widths: Elements per row per group, from
            `vramfit.domain.sizes.discovered_group_rows`.

    Returns:
        Protection overrides, then expert-stack and layer overrides.

    Raises:
        PackError: If any group or precision has no mapping, or a
            group the ADR-0028 routing reaches has no measured row
            width. See `protection_overrides` and `tensor_overrides`.

    Examples:
        The pack step and its pre-run check read one list:

        ```python
        overrides = all_overrides(recipe, row_widths)
        ```
    """
    return protection_overrides(recipe) + tensor_overrides(recipe, row_widths)
