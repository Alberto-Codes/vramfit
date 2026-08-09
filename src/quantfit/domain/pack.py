"""Pure pack-step accounting: the result record and the budget re-check.

The pack step hands a recipe to a runtime's quantizer and gets a file
back. The domain owns what survives that exchange without IO: the
record of what was driven (`PackResult`), the arithmetic that
re-checks real bytes against the planned budget (ADR-0012), the
smoke-test verdict against the perplexity ceiling (ADR-0017), and
the reconstruction-check verdict on protected packs — the stripped
reference recipe and the collapsed-tensor judgment (ADR-0022). Type
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

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from quantfit.domain.model import Recipe


@dataclass(frozen=True, slots=True)
class TypeOverride:
    r"""One per-tensor type override driven into the runtime's quantizer.

    Attributes:
        pattern (str): Regex the quantizer matches against tensor
            names. The first matching override wins.
        quant_type (str): Quantization type name for matching tensors.
            An opaque token to the domain — the backend owns the
            vocabulary (ADR-0012 for GGUF).

    Examples:
        Layer 7 of a recipe packed at 4-bit:

        ```python
        from quantfit.domain.pack import TypeOverride

        override = TypeOverride(pattern=r"blk\.7\.", quant_type="q4_k")
        ```
    """

    pattern: str
    quant_type: str

    def __post_init__(self) -> None:
        """Reject empty override halves.

        Raises:
            ValueError: If ``pattern`` or ``quant_type`` is empty.
        """
        if not self.pattern:
            raise ValueError("pattern must not be empty")
        if not self.quant_type:
            raise ValueError("quant_type must not be empty")


@dataclass(frozen=True, slots=True)
class PackResult:
    """The pack step's accounting record for one packed model.

    Attributes:
        packed_bytes (int): Real size of the packed model file.
        base_type (str): Quantizer base type for tensors no override
            covers.
        token_embedding_type (str | None): Type forced on the
            embedding tensor. None when the recipe has no embedding
            group.
        output_tensor_type (str | None): Type forced on the output
            head — the ``lm_head`` group's own assignment when the
            scan measured one, the embedding assignment otherwise
            (ADR-0012). None when the recipe has neither group.
        overrides (tuple[TypeOverride, ...]): Ordered per-tensor
            overrides, in recipe order. Patterns are unique — the
            quantizer applies the first match, so a duplicate would
            silently shadow its successor.
        imatrix_path (str | None): Importance matrix file driven into
            the quantizer (ADR-0016). None when the pack ran without
            one.
        imatrix_uncovered (tuple[str, ...]): Tensors the importance
            matrix did not cover — the quantizer only warns and
            quantizes them unassisted (ADR-0016). Excludes the
            intentional misses in ``imatrix_excluded``. Empty
            without an imatrix.
        imatrix_excluded (tuple[str, ...]): Tensors whose imatrix
            rows the pack dropped on the recipe's instruction
            (ADR-0023). Empty without an imatrix — an exclusion
            without a matrix is a no-op.

    Examples:
        Inspect the real size of a packed model:

        ```python
        print(result.packed_bytes)
        ```
    """

    packed_bytes: int
    base_type: str
    token_embedding_type: str | None
    output_tensor_type: str | None
    overrides: tuple[TypeOverride, ...]
    imatrix_path: str | None = None
    imatrix_uncovered: tuple[str, ...] = ()
    imatrix_excluded: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the result invariants.

        Raises:
            ValueError: If ``packed_bytes`` is not positive,
                ``base_type`` is empty, ``token_embedding_type``,
                ``output_tensor_type``, or ``imatrix_path`` is empty,
                ``imatrix_uncovered`` or ``imatrix_excluded`` is set
                without an ``imatrix_path``, or two overrides share a
                pattern.
        """
        if self.packed_bytes <= 0:
            raise ValueError("packed_bytes must be positive")
        if not self.base_type:
            raise ValueError("base_type must not be empty")
        if self.token_embedding_type is not None and not self.token_embedding_type:
            raise ValueError("token_embedding_type must not be empty")
        if self.output_tensor_type is not None and not self.output_tensor_type:
            raise ValueError("output_tensor_type must not be empty")
        if self.imatrix_path is not None and not self.imatrix_path:
            raise ValueError("imatrix_path must not be empty")
        if self.imatrix_uncovered and self.imatrix_path is None:
            raise ValueError("imatrix_uncovered requires an imatrix_path")
        if self.imatrix_excluded and self.imatrix_path is None:
            raise ValueError("imatrix_excluded requires an imatrix_path")
        patterns = [override.pattern for override in self.overrides]
        if len(set(patterns)) != len(patterns):
            raise ValueError("override patterns must be unique")


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


def without_protections(recipe: Recipe) -> Recipe:
    """Strip a recipe's protections for the reconstruction reference.

    The reconstruction check compares each protected tensor against
    the same pack at its unprotected type (ADR-0022). The reference
    pack applies this recipe: identical in every way except the
    protections.

    Args:
        recipe: The protected recipe.

    Returns:
        The recipe with empty protections and imatrix exclusions,
        equal otherwise. The exclusions ride the protections
        (ADR-0023), so the reference measures the assisted fit at
        the unprotected types.

    Examples:
        The reference recipe packs no protected tensors:

        ```python
        assert without_protections(recipe).protected_tensors == ()
        ```
    """
    return replace(
        recipe,
        plan=replace(recipe.plan, protections={}, imatrix_exclusions=()),
        protected_tensors=(),
    )


def collapsed_tensors(
    protected_rmse: Mapping[str, float],
    reference_rmse: Mapping[str, float],
) -> tuple[str, ...]:
    """Judge the reconstruction check: name the collapsed tensors.

    A protected tensor must reconstruct closer to the f16 base than
    it does at its unprotected type (ADR-0022). A tensor whose
    protected error is not strictly smaller is collapsed — including
    a NaN measurement, which must never pass a gate.

    Args:
        protected_rmse: Reconstruction error per tensor, measured on
            the protected pack.
        reference_rmse: Reconstruction error for the same tensors,
            measured on the unprotected reference pack.

    Returns:
        The collapsed tensor names, sorted.

    Raises:
        ValueError: If the two measurements cover different tensors —
            a missing reference would silently pass its tensor.

    Examples:
        The G1 collapse signature (5.1x worse at the higher type):

        ```python
        assert collapsed_tensors({"t": 0.0241}, {"t": 0.0048}) == ("t",)
        ```
    """
    if set(protected_rmse) != set(reference_rmse):
        raise ValueError(
            "protected and reference measurements must cover the same tensors"
        )
    return tuple(
        sorted(
            name
            for name, rmse in protected_rmse.items()
            if not (math.isfinite(rmse) and rmse < reference_rmse[name])
        )
    )


def smoke_passed(perplexity: float, threshold: float) -> bool:
    """Judge one smoke-test measurement against the ceiling.

    The smoke test proves a packed model emits language (ADR-0017).
    Destroyed artifacts measure perplexity near 10^6 and working ones
    below 100, so the verdict is a plain ceiling. A non-finite
    measurement fails — NaN must never pass a gate. Perplexity is
    mathematically at least 1, so a lower value signals a broken
    tool and fails too.

    Args:
        perplexity: The measured perplexity over the smoke chunks.
        threshold: The ceiling a passing measurement stays under.

    Returns:
        True when ``perplexity`` is finite, at least 1, and below
        ``threshold``.

    Raises:
        ValueError: If ``threshold`` is not positive and finite — an
            infinite ceiling would disable the gate.

    Examples:
        A destroyed artifact fails the default ceiling:

        ```python
        assert smoke_passed(1_020_627.9, threshold=1000.0) is False
        ```
    """
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")
    return math.isfinite(perplexity) and 1.0 <= perplexity < threshold
