"""Pure pack-step accounting: the result record and the budget re-check.

The pack step hands a recipe to a runtime's quantizer and gets a file
back. The domain owns what survives that exchange without IO: the
record of what was driven (`PackResult`) and the arithmetic that
re-checks real bytes against the planned budget (ADR-0012). Type
tables and subprocess details live in
[quantfit.adapters.outbound.gguf][].

Examples:
    Re-check a packed file against its recipe's budget:

    ```python
    from quantfit.domain.pack import weight_budget_margin

    margin = weight_budget_margin(recipe, packed_bytes=2_000_000_000)
    fits = margin >= 0
    ```

See Also:
    - [quantfit.ports.outbound][]: `RecipePacker`, which returns
      `PackResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantfit.domain.model import Recipe


@dataclass(frozen=True, slots=True)
class TypeOverride:
    r"""One per-tensor type override driven into the runtime's quantizer.

    Attributes:
        pattern (str): Regex the quantizer matches against tensor
            names. The first matching override wins.
        ggml_type (str): Quantization type name for matching tensors.

    Examples:
        Layer 7 of a recipe packed at 4-bit:

        ```python
        from quantfit.domain.pack import TypeOverride

        override = TypeOverride(pattern=r"blk\.7\.", ggml_type="q4_k")
        ```
    """

    pattern: str
    ggml_type: str

    def __post_init__(self) -> None:
        """Reject empty override halves.

        Raises:
            ValueError: If ``pattern`` or ``ggml_type`` is empty.
        """
        if not self.pattern:
            raise ValueError("pattern must not be empty")
        if not self.ggml_type:
            raise ValueError("ggml_type must not be empty")


@dataclass(frozen=True, slots=True)
class PackResult:
    """The pack step's accounting record for one packed model.

    Attributes:
        packed_bytes (int): Real size of the packed model file.
        base_type (str): Quantizer base type for tensors no override
            covers.
        token_embedding_type (str | None): Type forced on the
            embedding tensor, or None when the recipe has no
            embedding group.
        overrides (tuple[TypeOverride, ...]): Ordered per-tensor
            overrides, in recipe order.

    Examples:
        Inspect the real size of a packed model:

        ```python
        print(result.packed_bytes)
        ```
    """

    packed_bytes: int
    base_type: str
    token_embedding_type: str | None
    overrides: tuple[TypeOverride, ...]

    def __post_init__(self) -> None:
        """Enforce the result invariants.

        Raises:
            ValueError: If ``packed_bytes`` is not positive or
                ``base_type`` is empty.
        """
        if self.packed_bytes <= 0:
            raise ValueError("packed_bytes must be positive")
        if not self.base_type:
            raise ValueError("base_type must not be empty")


def weight_budget_margin(recipe: Recipe, packed_bytes: int) -> int:
    """Compute the weight-budget margin of a packed model.

    Nominal-bit predictions undershoot real GGUF sizes (ADR-0012), so
    the pack step re-checks the file it wrote against the budget the
    recipe was solved for.

    Args:
        recipe: The recipe the packed model applies.
        packed_bytes: Real size of the packed model file.

    Returns:
        ``weight_budget_bytes - packed_bytes``. Non-negative means
        the packed model fits.

    Raises:
        ValueError: If ``packed_bytes`` is not positive.

    Examples:
        A packed model 100 bytes under budget:

        ```python
        margin = weight_budget_margin(recipe, recipe.plan.weight_budget_bytes - 100)
        assert margin == 100
        ```
    """
    if packed_bytes <= 0:
        raise ValueError("packed_bytes must be positive")
    return recipe.plan.weight_budget_bytes - packed_bytes
