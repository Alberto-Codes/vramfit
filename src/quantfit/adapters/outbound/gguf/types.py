"""Pure GGUF type mapping: nominal bits to K-quant types (ADR-0012).

The decision core of the GGUF backend, kept free of IO so the mapping
is testable and the verified fake can share it. Nominal precisions
map to K-quant types (the full llama.cpp capability set since
ADR-0013), layer groups map to escaped `blk.<n>.` regex patterns,
protected tensors map through the fixed HF-to-GGUF class table to
per-tensor patterns (ADR-0022), excluded pairs map to the full GGUF
tensor names ``--exclude-weights`` deletes by substring (ADR-0023),
and the embedding and `lm_head` groups map to the quantizer's
dedicated embedding and output flags. The backend's own runtime
name is the domain's `LLAMA_CPP` constant, so the table key and
the pack check cannot drift apart. A
recipe recorded for a foreign runtime, or anything the table cannot
map, raises `PackError` instead of guessing.

Examples:
    Map a recipe to quantizer inputs:

    ```python
    from quantfit.adapters.outbound.gguf import types

    base = types.base_type(recipe)
    embed = types.token_embedding_type(recipe)
    overrides = types.tensor_overrides(recipe)
    ```

See Also:
    - [quantfit.adapters.outbound.gguf.pack][]: The subprocess driver
      that feeds these values to ``llama-quantize``.
"""

from __future__ import annotations

import re
from typing import Final

from quantfit.domain.errors import QuantfitError
from quantfit.domain.model import Recipe
from quantfit.domain.pack import TypeOverride
from quantfit.domain.runtime import LLAMA_CPP

GGML_TYPE_BY_BITS: Final[dict[int, str]] = {
    8: "q8_0",
    6: "q6_k",
    5: "q5_k",
    4: "q4_k",
    3: "q3_k",
    2: "q2_k",
}

# The quantizer's positional type argument speaks ftype names, not
# tensor-type names. Each entry is the pure-type ftype for the same
# nominal bits as GGML_TYPE_BY_BITS.
BASE_FTYPE_BY_BITS: Final[dict[int, str]] = {
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

EMBEDDING_GROUP: Final[str] = "model.embed_tokens"

OUTPUT_GROUP: Final[str] = "lm_head"

_LAYER_GROUP: Final[re.Pattern[str]] = re.compile(r"^model\.layers\.(\d+)$")

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


class PackError(QuantfitError, RuntimeError):
    """The pack backend cannot map or apply a recipe.

    Raised for precisions outside the ADR-0012 table (as amended by
    ADR-0013), groups without a GGUF tensor mapping, recipes planned
    for another runtime, and toolchain failures. Inherits
    `QuantfitError` per ADR-0011 and `RuntimeError` for the port
    contract.

    Examples:
        A 16-bit assignment has no mapping:

        ```python
        ggml_type_for(16)  # raises PackError
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
    output head (ADR-0012).

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
        if assignment.group == EMBEDDING_GROUP:
            return ggml_type_for(assignment.bits)
    return None


def output_tensor_type(recipe: Recipe) -> str | None:
    """Map the output head's assignment to the output flag.

    ``--output-tensor-type`` binds the output tensor before any
    pattern override. An ``lm_head`` group — scanned on models with
    an untied head — carries its own assignment. Without one, the
    embedding assignment pins the head, so a tied model's single
    scanned group governs both tensors (ADR-0012).

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
    for assignment in recipe.assignments:
        if assignment.group == OUTPUT_GROUP:
            return ggml_type_for(assignment.bits)
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


def tensor_overrides(recipe: Recipe) -> tuple[TypeOverride, ...]:
    r"""Translate layer groups into quantizer tensor-type overrides.

    One override per layer group: ``model.layers.<n>`` becomes the
    escaped pattern ``blk\.<n>\.``. Escaping matters — an unescaped
    ``blk.1.`` would also match ``blk.11.``. The embedding and
    ``lm_head`` groups map to dedicated flags and are skipped here.

    Args:
        recipe: The recipe to pack.

    Returns:
        Overrides in recipe order. The quantizer applies the first
        match, and the patterns are mutually exclusive.

    Raises:
        PackError: If a group is not a layer group, the embedding,
            or the output head, or its precision has no table entry.

    Examples:
        The group ``model.layers.7`` at 4-bit becomes an escaped
        pattern:

        ```python
        assert TypeOverride(r"blk\.7\.", "q4_k") in tensor_overrides(recipe)
        ```
    """
    overrides: list[TypeOverride] = []
    for assignment in recipe.assignments:
        if assignment.group in (EMBEDDING_GROUP, OUTPUT_GROUP):
            continue
        match = _LAYER_GROUP.match(assignment.group)
        if match is None:
            raise PackError(
                f'group "{assignment.group}" has no GGUF tensor mapping — the '
                "v1 backend maps layer groups, the embedding, and the output "
                "head (ADR-0012)"
            )
        overrides.append(
            TypeOverride(
                pattern=rf"blk\.{match.group(1)}\.",
                quant_type=ggml_type_for(assignment.bits),
            )
        )
    return tuple(overrides)
