"""Pure GGUF type mapping: nominal bits to K-quant types (ADR-0012).

The decision core of the GGUF backend, kept free of IO so the mapping
is testable and the verified fake can share it. Nominal precisions
map to K-quant types, layer groups map to escaped `blk.<n>.` regex
patterns, and the embedding group maps to the quantizer's dedicated
embedding flag. Anything the table cannot map raises `PackError`
instead of guessing.

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

GGML_TYPE_BY_BITS: Final[dict[int, str]] = {
    8: "q8_0",
    4: "q4_k",
    3: "q3_k",
    2: "q2_k",
}

# The quantizer's positional type argument speaks ftype names, not
# tensor-type names. Each entry is the pure-type ftype for the same
# nominal bits as GGML_TYPE_BY_BITS.
BASE_FTYPE_BY_BITS: Final[dict[int, str]] = {
    8: "Q8_0",
    4: "Q4_K_S",
    3: "Q3_K_S",
    2: "Q2_K",
}

EMBEDDING_GROUP: Final[str] = "model.embed_tokens"

_LAYER_GROUP: Final[re.Pattern[str]] = re.compile(r"^model\.layers\.(\d+)$")


class PackError(QuantfitError, RuntimeError):
    """The pack backend cannot map or apply a recipe.

    Raised for precisions outside the ADR-0012 table, groups without
    a GGUF tensor mapping, and toolchain failures. Inherits
    `QuantfitError` per ADR-0011 and `RuntimeError` for the port
    contract.

    Examples:
        A 6-bit assignment has no mapping today:

        ```python
        ggml_type_for(6)  # raises PackError
        ```
    """


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


def tensor_overrides(recipe: Recipe) -> tuple[TypeOverride, ...]:
    r"""Translate layer groups into quantizer tensor-type overrides.

    One override per layer group: ``model.layers.<n>`` becomes the
    escaped pattern ``blk\.<n>\.``. Escaping matters — an unescaped
    ``blk.1.`` would also match ``blk.11.``. The embedding group maps
    to a dedicated flag and is skipped here.

    Args:
        recipe: The recipe to pack.

    Returns:
        Overrides in recipe order. The quantizer applies the first
        match, and the patterns are mutually exclusive.

    Raises:
        PackError: If a group is neither a layer group nor the
            embedding, or its precision has no table entry.

    Examples:
        Layer 7 at 4-bit becomes an escaped pattern:

        ```python
        assert overrides[7].pattern == r"blk\.7\."
        ```
    """
    overrides: list[TypeOverride] = []
    for assignment in recipe.assignments:
        if assignment.group == EMBEDDING_GROUP:
            continue
        match = _LAYER_GROUP.match(assignment.group)
        if match is None:
            raise PackError(
                f'group "{assignment.group}" has no GGUF tensor mapping — the '
                "v1 backend maps layer groups and the embedding (ADR-0012)"
            )
        overrides.append(
            TypeOverride(
                pattern=rf"blk\.{match.group(1)}\.",
                ggml_type=ggml_type_for(assignment.bits),
            )
        )
    return tuple(overrides)
