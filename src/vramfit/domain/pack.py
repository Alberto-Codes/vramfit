"""Pure pack-step accounting: the result record and the budget re-check.

The pack step hands a recipe to a runtime's quantizer and gets a file
back. The domain owns what survives that exchange without IO: the
record of what was driven (`PackResult`), the arithmetic that
re-checks real bytes against the planned budget and against the
recipe's own prediction (ADR-0012), the
smoke-test verdict against the perplexity ceiling (ADR-0017), the
zero-count judgment on the imatrix counts the pack reads
(`zero_count_experts`, ADR-0026 decision 5), the modal-type rule
that names a mixed-precision file (`modal_type`, ADR-0012 decision 3
as amended 2026-09-04), and
the reconstruction-check verdict on protected packs — the stripped
reference recipe and the collapsed-tensor judgment (ADR-0022), and
the pre-encoding stage's own record: the tensors vramfit's assisted
``Q2_0`` encoder fitted, the encoder revision, and the measured cost
(`PreEncodeCost`, ADR-0032 decision 3). Type tables and subprocess
details live in [vramfit.adapters.outbound.gguf][].

Examples:
    Re-check a packed file against its recipe's budget:

    ```python
    from vramfit.domain.pack import fits_weight_budget, weight_budget_margin

    margin = weight_budget_margin(recipe, packed_bytes=2_000_000_000)
    fits = fits_weight_budget(margin)
    ```

See Also:
    - [vramfit.ports.outbound][]: `RecipePacker`, which returns
      `PackResult`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from vramfit.domain.model import Recipe

# The predicted-bytes tolerance (ADR-0012 decision 4, amended
# 2026-09-04). A delta past this fraction of the prediction warns.
PREDICTED_BYTES_TOLERANCE: Final[float] = 0.01


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
        from vramfit.domain.pack import TypeOverride

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
class PreEncodeCost:
    """What the pre-encoding stage measured about itself (ADR-0032).

    ADR-0032's open questions leave the preprocessor's full-file cost
    unmeasured, so every pack that pre-encodes records its own.

    Attributes:
        peak_rss_bytes (int): The encoder program's peak resident
            set.
        mixed_gguf_bytes (int): The temporary mixed GGUF's size.
        payload_bytes (int): The pre-encoded payloads' total size.

    Examples:
        The run log carries the three figures on ``model_packed``:

        ```python
        cost = PreEncodeCost(
            peak_rss_bytes=2**30, mixed_gguf_bytes=2**36, payload_bytes=2**30
        )
        ```
    """

    peak_rss_bytes: int
    mixed_gguf_bytes: int
    payload_bytes: int

    def __post_init__(self) -> None:
        """Reject a negative measurement.

        Raises:
            ValueError: If any figure is negative.
        """
        if min(self.peak_rss_bytes, self.mixed_gguf_bytes, self.payload_bytes) < 0:
            raise ValueError("a pre-encode cost must not be negative")


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
            overrides: protections, then expert stacks, then layer
            groups, each in recipe order. Order carries priority.
            The quantizer applies the first match, so a broader
            pattern placed first would shadow a narrower one.
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
        imatrix_zero_count_experts (tuple[tuple[str, int], ...]):
            ``(stack, expert)`` pairs the importance matrix counts
            zero times (ADR-0026 decision 5), sorted. The quantizer
            fills such an expert's row with ones and prints no
            warning, so only this record names the case. A report,
            never a gate. Separate from ``imatrix_uncovered``, which
            names whole tensors. Empty without an imatrix.
        file_type (str | None): The ftype name the packed file
            declares in ``general.file_type``: the type covering the
            most bytes, written after the quantizer ran (ADR-0012
            decision 3 as amended 2026-09-04). The quantizer's own
            stamp is ``base_type``, which on a mixed pack can name a
            type the file does not hold (#413). None when the packer
            wrote no label. The token is the runtime's own ftype
            name, so the backend owns the vocabulary.
        floored_layers (tuple[str, ...]): Layers the base model
            carries that no override reached, in layer order. They
            took the ``base_type`` floor, which ADR-0012 decision 3
            makes the designed outcome — so this is a report, never a
            gate (#307). It names the gap between the recipe and the
            file, which the packed size otherwise carries with no
            signal. The tokens are the runtime's own layer names, so
            the backend owns the vocabulary.
        pre_encoded (tuple[str, ...]): Tensors the preprocessor
            encoded with vramfit's assisted ``Q2_0`` encoder before
            the quantizer ran, in file order (ADR-0032 decision 1).
            Each packs assisted (decision 3). Empty when the recipe
            took the stock path.
        q2_0_encoder (str | None): The encoder revision that produced
            the pre-encoded tensors, recorded with the result
            (ADR-0032 decision 3). None when nothing was pre-encoded.
        pre_encode_cost (PreEncodeCost | None): What the stage
            measured about itself. None when nothing was pre-encoded.

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
    imatrix_zero_count_experts: tuple[tuple[str, int], ...] = ()
    floored_layers: tuple[str, ...] = ()
    file_type: str | None = None
    pre_encoded: tuple[str, ...] = ()
    q2_0_encoder: str | None = None
    pre_encode_cost: PreEncodeCost | None = None

    def __post_init__(self) -> None:
        """Enforce the result invariants.

        Raises:
            ValueError: If ``packed_bytes`` is not positive,
                ``base_type`` is empty, ``token_embedding_type``,
                ``output_tensor_type``, ``imatrix_path``, or
                ``file_type`` is empty,
                ``imatrix_uncovered``, ``imatrix_excluded``, or
                ``imatrix_zero_count_experts`` is set without an
                ``imatrix_path``, a zero-count pair names an empty
                stack or a negative expert index, a floored layer is
                empty, two overrides share a pattern, a pre-encoded
                tensor is empty or arrives without an ``imatrix_path``,
                or ``pre_encoded``, ``q2_0_encoder``, and
                ``pre_encode_cost`` are not all set or all unset.
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
        if self.file_type is not None and not self.file_type:
            raise ValueError("file_type must not be empty")
        if self.imatrix_uncovered and self.imatrix_path is None:
            raise ValueError("imatrix_uncovered requires an imatrix_path")
        if self.imatrix_excluded and self.imatrix_path is None:
            raise ValueError("imatrix_excluded requires an imatrix_path")
        if self.imatrix_zero_count_experts and self.imatrix_path is None:
            raise ValueError("imatrix_zero_count_experts requires an imatrix_path")
        _check_zero_count_pairs(self.imatrix_zero_count_experts)
        if any(not layer for layer in self.floored_layers):
            raise ValueError("a floored layer must not be empty")
        patterns = [override.pattern for override in self.overrides]
        if len(set(patterns)) != len(patterns):
            raise ValueError("override patterns must be unique")
        _check_pre_encoded(self)


def _check_pre_encoded(result: PackResult) -> None:
    """Refuse a pre-encoding record that is half-stated.

    Args:
        result: The record under construction.

    Raises:
        ValueError: If a pre-encoded name is empty, the record names
            pre-encoded tensors without an imatrix, or the three
            pre-encoding fields disagree on whether the stage ran.
    """
    if any(not name for name in result.pre_encoded):
        raise ValueError("a pre-encoded tensor name must not be empty")
    ran = bool(result.pre_encoded)
    if ran and result.imatrix_path is None:
        raise ValueError("pre_encoded requires an imatrix_path")
    if (result.q2_0_encoder is not None) != ran or (
        result.pre_encode_cost is not None
    ) != ran:
        raise ValueError(
            "pre_encoded, q2_0_encoder, and pre_encode_cost record one stage "
            "together — set all three or none"
        )
    if result.q2_0_encoder is not None and not result.q2_0_encoder:
        raise ValueError("q2_0_encoder must not be empty")


def _check_zero_count_pairs(pairs: tuple[tuple[str, int], ...]) -> None:
    """Refuse a malformed zero-count report entry.

    Args:
        pairs: The ``(stack, expert)`` pairs a `PackResult` records.

    Raises:
        ValueError: If a pair names an empty stack or a negative
            expert index — neither can come from a real read.
    """
    for stack, expert in pairs:
        if not stack:
            raise ValueError("a zero-count pair's stack must not be empty")
        if expert < 0:
            raise ValueError("a zero-count pair's expert index must not be negative")


def zero_count_experts(
    stack_counts: Mapping[str, Sequence[int]],
) -> tuple[tuple[str, int], ...]:
    """Name every expert an importance matrix counts zero times.

    The judgment behind ADR-0026 decision 5: the quantizer fills a
    zero-count expert's row with ones and prints no warning, so the
    pack path reads the counts itself and reports. The counts arrive
    rounded (`ImatrixCountSource`), so zero here is the same zero
    the C loader tests.

    Args:
        stack_counts: One rounded count vector per expert-stack
            entry, keyed by GGUF tensor name. Element ``i`` is
            expert ``i``'s tally.

    Returns:
        ``(stack, expert)`` pairs, sorted by stack name then expert
        index. Empty when every expert holds a nonzero count — the
        healthy case.

    Examples:
        One starved expert inside a covered stack:

        ```python
        from vramfit.domain.pack import zero_count_experts

        pairs = zero_count_experts({"blk.3.ffn_up_exps.weight": (7, 0, 12)})
        assert pairs == (("blk.3.ffn_up_exps.weight", 1),)
        ```
    """
    return tuple(
        sorted(
            (stack, expert)
            for stack, counts in stack_counts.items()
            for expert, count in enumerate(counts)
            if count == 0
        )
    )


def modal_type(bytes_by_type: Mapping[str, int]) -> str:
    """Name the type that covers the most bytes in a packed file.

    The rule behind ADR-0012 decision 3 as amended 2026-09-04. One
    file-type field cannot name a mixed-precision recipe, so the
    field declares the type most of the file is. On the published
    30B pack that is ``Q4_0`` at 74.3 % of the bytes, where the
    quantizer had stamped ``Q2_K`` over no ``Q2_K`` tensor (#413).

    Args:
        bytes_by_type: Bytes each type covers, keyed by type name.
            Every count must be positive.

    Returns:
        The type name with the largest byte count. A tie goes to the
        name that sorts first, so the answer never depends on the
        table's order.

    Raises:
        ValueError: If the table is empty, a name is empty, or a
            count is not positive — none can come from a real read.

    Examples:
        The 30B pack's composition, in tenths of a percent:

        ```python
        from vramfit.domain.pack import modal_type

        assert modal_type({"Q4_0": 743, "Q8_0": 138, "Q2_0": 117}) == "Q4_0"
        ```
    """
    if not bytes_by_type:
        raise ValueError("bytes_by_type must not be empty")
    for name, count in bytes_by_type.items():
        if not name:
            raise ValueError("a type name must not be empty")
        if count <= 0:
            raise ValueError(f"bytes for {name} must be positive")
    return max(sorted(bytes_by_type), key=lambda name: bytes_by_type[name])


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


def fits_weight_budget(margin: int) -> bool:
    """Judge a weight-budget margin.

    The one definition of what "fits the weight budget" means.
    `vramfit pack` gates on it, the refinement pass judges each packed
    arm by it, and
    `vramfit.domain.refinement_record.ArmRecord` reads it to validate
    a recorded arm, so the three cannot disagree about which file a
    stage may keep.

    Args:
        margin: A margin from `weight_budget_margin`.

    Returns:
        True when the packed file fits the budget its recipe was
        solved for.

    Examples:
        A margin of zero fits:

        ```python
        from vramfit.domain.pack import fits_weight_budget

        assert fits_weight_budget(0)
        assert not fits_weight_budget(-1)
        ```
    """
    return margin >= 0


def predicted_bytes_delta(predicted_total_bytes: int, packed_bytes: int) -> int:
    """Compute how far the packed file lands from the recipe's prediction.

    The budget margin alone cannot see a wrong prediction: publication
    #2 fit its budget on two cancelling pricing errors and reported
    only its margin (#409). The pack step prints this delta beside the
    margin, so the size model's miss is on record (ADR-0012 decision
    4, amended 2026-09-04).

    Args:
        predicted_total_bytes: The recipe's ``plan.predicted_total_bytes``.
        packed_bytes: Real size of the packed model file.

    Returns:
        ``packed_bytes - predicted_total_bytes``. Positive means the
        file outgrew the prediction.

    Raises:
        ValueError: If either count is not positive.

    Examples:
        A packed file 100 bytes over its prediction:

        ```python
        assert predicted_bytes_delta(2_000, 2_100) == 100
        ```
    """
    if predicted_total_bytes <= 0:
        raise ValueError("predicted_total_bytes must be positive")
    if packed_bytes <= 0:
        raise ValueError("packed_bytes must be positive")
    return packed_bytes - predicted_total_bytes


def predicted_bytes_within_tolerance(
    predicted_total_bytes: int, packed_bytes: int
) -> bool:
    """Judge the prediction delta against the predicted-bytes tolerance.

    ``PREDICTED_BYTES_TOLERANCE`` is a fraction of the prediction,
    applied in both directions. The judgment never refuses a pack: the
    weight budget stays the only size gate (ADR-0012 decision 4).

    Args:
        predicted_total_bytes: The recipe's ``plan.predicted_total_bytes``.
        packed_bytes: Real size of the packed model file.

    Returns:
        True when ``|delta| <= PREDICTED_BYTES_TOLERANCE *
        predicted_total_bytes``.

    Raises:
        ValueError: If either count is not positive.

    Examples:
        One percent over a 10,000-byte prediction sits on the edge:

        ```python
        assert predicted_bytes_within_tolerance(10_000, 10_100) is True
        assert predicted_bytes_within_tolerance(10_000, 10_101) is False
        ```
    """
    delta = predicted_bytes_delta(predicted_total_bytes, packed_bytes)
    return abs(delta) <= PREDICTED_BYTES_TOLERANCE * predicted_total_bytes


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
