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
not divide the stack rows, and nominal 3 refuses over the empty
2.25-4.25 bits-per-weight gap (ADR-0028) — protected tensors map
through the fixed HF-to-GGUF class table to
per-tensor patterns (ADR-0022), excluded pairs map to the full GGUF
tensor names ``--exclude-weights`` deletes by substring (ADR-0023),
and the embedding and `lm_head` groups map to the quantizer's
dedicated embedding and output flags. The backend's own runtime
name is the domain's `LLAMA_CPP` constant, so the table key and
the pack check cannot drift apart. A
recipe recorded for a foreign runtime, or anything the table cannot
map, raises `PackError` instead of guessing.

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
    overrides = types.all_overrides(recipe)
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: The subprocess driver
      that feeds these values to ``llama-quantize``.
"""

from __future__ import annotations

import re
from typing import Final

from vramfit.domain.errors import VramfitError
from vramfit.domain.model import Recipe
from vramfit.domain.pack import TypeOverride
from vramfit.domain.runtime import LLAMA_CPP

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
# bits per weight: f16 at 16.00, q8_0 at 8.50, q4_0 at 4.50, q2_0 at
# 2.25. The f16 row is the ADR-0029 passthrough, and it has no block
# to divide.
EXPERT_STACK_TYPE_BY_BITS: Final[dict[int, str]] = {
    16: "f16",
    8: "q8_0",
    4: "q4_0",
    2: "q2_0",
}

# The quantizer's positional type argument speaks ftype names, not
# tensor-type names. Each entry is the pure-type ftype for the same
# nominal bits as GGML_TYPE_BY_BITS.
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
# Nemotron-H says `backbone.embeddings`. Both drive the one
# `--token-embedding-type` flag, so the backend needs the names, not
# a pattern.
EMBEDDING_GROUPS: Final[frozenset[str]] = frozenset(
    {"model.embed_tokens", "backbone.embeddings"}
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
# projection (#161). The dot before `experts` matters — it refuses
# `shared_experts`, which GGUF names `ffn_up_shexp` and this table
# does not carry (#183).
_EXPERT_STACK: Final[re.Pattern[str]] = re.compile(
    r"^.+\.(?:layers|h|blocks)\.(\d+)\.(?:.*\.)?experts\.([A-Za-z0-9_]+)$"
)

# The parameter-tree root a layer or expert-stack group hangs from.
# `blk.<n>.` addresses exactly one layer stack, so a recipe that
# names two of them cannot pack. The target carries `backbone` and
# `mtp`, and a multimodal checkpoint carries a vision tower that
# GGUF names `v.blk.<n>.` instead.
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

# The fixed class table (ADR-0022): HF tensor suffix to GGUF tensor
# suffix, for the seven quantized projections of a llama-family layer.
GGUF_SUFFIX_BY_HF: Final[dict[str, str]] = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}

_LAYER_TENSOR: Final[re.Pattern[str]] = re.compile(
    r"^model\.layers\.(\d+)\.([a-z_]+\.[a-z_]+)\.weight$"
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


def expert_stack_type_for(bits: int, group: str) -> str:
    """Map one expert-stack precision to its ADR-0028 tensor type.

    Args:
        bits: Nominal precision from a recipe assignment.
        group: The expert-stack group, named in every refusal.

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
            f'expert stack "{group}" cannot pack at nominal 3 — no GGUF type '
            f"lands between 2.25 and 4.25 bits per weight on the stack rows "
            f"(ADR-0028). The neighboring table entries are 2 -> q2_0 "
            f"(2.25 bits/weight) and 4 -> q4_0 (4.50 bits/weight)"
        )
    try:
        return EXPERT_STACK_TYPE_BY_BITS[bits]
    except KeyError:
        raise PackError(
            f'expert stack "{group}" has no type for {bits}-bit — the '
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
        PackError: If the name is not a layer tensor, or its suffix
            has no class-table entry (ADR-0022).

    Examples:
        The G1 protection target:

        ```python
        assert (
            gguf_tensor_name("model.layers.4.self_attn.v_proj.weight")
            == "blk.4.attn_v.weight"
        )
        ```
    """
    match = _LAYER_TENSOR.match(tensor)
    if match is None or match.group(2) not in GGUF_SUFFIX_BY_HF:
        raise PackError(
            f'protected tensor "{tensor}" has no GGUF mapping — the class '
            f"table covers layer tensors {sorted(GGUF_SUFFIX_BY_HF)} (ADR-0022)"
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


def _claim_root(group: str, roots: dict[str, str]) -> None:
    """Hold every mapped group to one parameter-tree root.

    Args:
        group: Recipe group name.
        roots: Roots claimed so far, mapped to the group that
            claimed each. Updated in place.

    Raises:
        PackError: If ``group`` hangs from a second root.
    """
    match = _STACK_ROOT.match(group)
    if match is None:
        return
    root = match.group(1)
    roots.setdefault(root, group)
    if len(roots) > 1:
        first = next(iter(roots))
        raise PackError(
            f'groups "{roots[first]}" and "{group}" name two layer stacks — '
            f'a GGUF pack numbers one stack "blk.<n>." and would silently '
            f"drop the other (#183)"
        )


def tensor_overrides(recipe: Recipe) -> tuple[TypeOverride, ...]:
    r"""Translate recipe groups into quantizer tensor-type overrides.

    Two group shapes map. A layer group under any of the three
    naming families — ``model.layers.<n>``, ``backbone.layers.<n>``
    — becomes the escaped pattern ``blk\.<n>\.``. A routed-expert
    stack group becomes the escaped pattern for its fused tensor,
    e.g. ``blk\.<n>\.ffn_up_exps\.`` (#159, #161). Escaping matters
    — an unescaped ``blk.1.`` would also match ``blk.11.``. The
    embedding and ``lm_head`` groups map to dedicated flags and are
    skipped here.

    Expert-stack overrides come first, ahead of the layer overrides.
    The quantizer applies the first matching pattern, and
    ``blk\.1\.`` also matches ``blk.1.ffn_up_exps.weight``. Callers
    place the protection overrides ahead of both — a per-tensor
    pattern is the most specific of the three (ADR-0022).

    Every mapped group must hang from one parameter-tree root.
    ``blk.<n>.`` addresses a single layer stack, so a recipe naming
    two of them would map both onto it and silently drop one.

    Args:
        recipe: The recipe to pack.

    Returns:
        Expert-stack overrides in recipe order, then layer overrides
        in recipe order.

    Raises:
        PackError: If a group is not a layer group, a routed-expert
            stack, the embedding, or the output head. Also if a
            routed-expert stack names a projection outside the
            fused-stack table or a precision outside the ADR-0028
            stack table (nominal 3 refuses over the empty 2.25-4.25
            gap), if the groups hang from two roots, or if a
            precision has no table entry.

    Examples:
        The group ``model.layers.7`` at 4-bit becomes an escaped
        pattern:

        ```python
        assert TypeOverride(r"blk\.7\.", "q4_k") in tensor_overrides(recipe)
        ```
    """
    stacks: list[TypeOverride] = []
    layers: list[TypeOverride] = []
    roots: dict[str, str] = {}
    for assignment in recipe.assignments:
        if assignment.group in EMBEDDING_GROUPS or assignment.group == OUTPUT_GROUP:
            continue
        _claim_root(assignment.group, roots)
        prefix = gguf_stack_prefix(assignment.group)
        if prefix is not None:
            bits = expert_stack_type_for(assignment.bits, assignment.group)
            stacks.append(TypeOverride(re.escape(prefix), bits))
            continue
        match = _LAYER_GROUP.match(assignment.group)
        if match is None:
            raise PackError(
                f'group "{assignment.group}" has no GGUF tensor mapping — the '
                "backend maps layer groups, routed-expert stacks, the "
                "embedding, and the output head (ADR-0012, ADR-0022)"
            )
        bits = ggml_type_for(assignment.bits)
        layers.append(TypeOverride(rf"blk\.{match.group(1)}\.", bits))
    return tuple(stacks) + tuple(layers)


def all_overrides(recipe: Recipe) -> tuple[TypeOverride, ...]:
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

    Returns:
        Protection overrides, then expert-stack and layer overrides.

    Raises:
        PackError: If any group or precision has no mapping. See
            `protection_overrides` and `tensor_overrides`.

    Examples:
        The pack step and its pre-run check read one list:

        ```python
        overrides = all_overrides(recipe)
        ```
    """
    return protection_overrides(recipe) + tensor_overrides(recipe)
