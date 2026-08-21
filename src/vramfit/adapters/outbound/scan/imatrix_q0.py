"""Imatrix reader for the `q0` method family (ADR-0018, `q0-imx`).

Resolves per-parameter imatrix weights for the assisted ``Q4_0``
fit, keyed on the names the loaded model reports. One reader serves
one method family (ADR-0018, 2026-08-21 amendment, decision 2):
[vramfit.adapters.outbound.scan.imatrix][]'s
``resolve_assisted_weights`` keeps its fused-stack refusal and its
super-block gate, unchanged, for ``kquant``. This reader accepts a
fused expert stack and resolves its whole entry — one weight row per
expert, in imatrix row order, exactly as ``llama-quant.cpp`` slices
the matrix per expert (``imatrix + i03 * ne0``, sums over counts —
the mechanics #381's harness proved).

Each resolved entry vouches against the parameter's shape, on
ADR-0026's pattern. A parameter without coverage takes the
unassisted path — the C behavior for a NULL imatrix row (ADR-0020
decision 2). A row length that does not divide into ``QK4_0`` blocks
also reports uncovered: the assisted fit cannot align its column
weights, and the fallback beats refusing a multi-day scan over one
tensor.

Examples:
    Resolve weights for the parameters a meter discovered:

    ```python
    covered, uncovered = resolve_q0_assisted_weights(
        load_imatrix(path), {n: tuple(p.shape) for n, p in params}
    )
    ```

See Also:
    - [vramfit.adapters.outbound.scan.q0_assisted][]: Consumes the
      resolved weights.
    - [vramfit.adapters.outbound.scan.imatrix][]: Loads the artifact
      and owns the name table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from vramfit.adapters.outbound.scan.imatrix import (
    ImatrixEntry,
    _matrix_row,
    _resolve_name,
)
from vramfit.adapters.outbound.scan.q0_ref import QK4_0

# A fused expert stack shapes as (experts, rows, columns). Mirrors
# the loader-side constant — the reader vouches the same layout.
_FUSED_STACK_DIMS = 3


def _fused_weights(
    entry: ImatrixEntry, name: str, gguf_name: str, shape: Sequence[int]
) -> torch.Tensor:
    """Vouch and resolve a fused expert stack's whole entry.

    Args:
        entry: The entry `load_imatrix` read for ``gguf_name``.
        name: The HF dotted parameter name.
        gguf_name: The GGUF tensor name ``name`` maps to.
        shape: The loaded parameter's shape.

    Returns:
        The entry's column weights, one row per expert.

    Raises:
        ValueError: If the parameter is not 3-D, or the entry's
            matrix count does not equal the parameter's expert
            count. Either mismatch means the imatrix and the
            checkpoint describe different models. Resolving anyway
            would fit experts against other experts' columns.
    """
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
            f"{name} holds {experts} experts on its first dimension, and "
            f"{gguf_name} counts {matrices} matrices. The imatrix does "
            "not describe this checkpoint."
        )
    return entry.column_weights


def _zero_coverage_message(
    by_gguf_name: Mapping[str, ImatrixEntry], shapes: Mapping[str, Sequence[int]]
) -> str:
    """Report why an imatrix covered no parameter.

    Three causes reach the same empty result, and naming the file
    alone sends an operator to regenerate a correct matrix. The
    counts point at the cause.

    Args:
        by_gguf_name: Entries keyed by GGUF tensor name.
        shapes: Parameter shapes keyed by HF parameter name.

    Returns:
        The message, counting the parameters under each cause.
    """
    unmapped = 0
    absent = 0
    misaligned = 0
    for name, shape in shapes.items():
        resolved = _resolve_name(name)
        if resolved is None:
            unmapped += 1
        elif resolved[0] not in by_gguf_name:
            absent += 1
        elif int(shape[-1]) % QK4_0:
            misaligned += 1
    return (
        f"the imatrix covers none of the {len(shapes)} parameters. "
        f"{unmapped} names have no GGUF mapping, so check the name table "
        f"covers this family. {absent} mapped to a tensor the file does "
        f"not hold, so check the file matches this model. {misaligned} "
        f"have rows that do not divide into {QK4_0}-element Q4_0 blocks, "
        "which the assisted fit cannot align (ADR-0018)."
    )


def resolve_q0_assisted_weights(
    by_gguf_name: Mapping[str, ImatrixEntry], shapes: Mapping[str, Sequence[int]]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Match loaded imatrix entries to a set of parameters.

    The meter calls this after model load, when the parameter shapes
    exist. A routed expert reads its own row of the expert stack
    (#177). A fused expert stack resolves its whole entry — a 2-D
    weight tensor, one row per expert in imatrix order, the order
    transformers stacks checkpoint experts (#202).

    Args:
        by_gguf_name: Entries keyed by GGUF tensor name, from
            ``load_imatrix``.
        shapes: Parameter shapes keyed by the names the loaded model
            reports.

    Returns:
        ``(covered, uncovered)`` — float32 weights keyed by
        parameter name (1-D for a dense or indexed parameter, 2-D
        for a fused expert stack), and the names the imatrix does
        not cover, in input order. A parameter whose rows do not
        divide into ``QK4_0`` blocks joins the uncovered set and
        prices unassisted.

    Raises:
        ValueError: If a covered tensor's weight length does not
            match the parameter's row length, an entry's matrix
            count does not fit the parameter, two parameters claim
            one imatrix row (#193), or no parameter is covered at
            all. A scan on zero coverage would price every cell
            unassisted under the assisted label.
    """
    covered: dict[str, torch.Tensor] = {}
    uncovered: list[str] = []
    claimed: dict[tuple[str, int], str] = {}

    def claim(gguf_name: str, row: int, name: str) -> None:
        """Record one row's claimant, refusing a second (#193).

        Args:
            gguf_name: The claimed entry's GGUF tensor name.
            row: The claimed matrix row.
            name: The claiming parameter.

        Raises:
            ValueError: If another parameter already claimed the row.
        """
        claimant = claimed.setdefault((gguf_name, row), name)
        if claimant != name:
            raise ValueError(
                f"{name} and {claimant} both claim {gguf_name} row {row}. "
                "One of them would price against the wrong columns."
            )

    for name, shape in shapes.items():
        rows = int(shape[-1])
        resolved = _resolve_name(name)
        if resolved is None:
            uncovered.append(name)
            continue
        gguf_name, expert, fused = resolved
        entry = by_gguf_name.get(gguf_name)
        if entry is None or rows % QK4_0:
            uncovered.append(name)
            continue
        if fused:
            weight = _fused_weights(entry, name, gguf_name, shape)
            for row in range(int(entry.counts.numel())):
                claim(gguf_name, row, name)
        else:
            row = _matrix_row(entry, name, gguf_name, expert)
            claim(gguf_name, row, name)
            weight = entry.column_weights[row]
        if int(weight.shape[-1]) != rows:
            raise ValueError(
                f"imatrix weights for {name} ({gguf_name}) have "
                f"{int(weight.shape[-1])} columns, the parameter rows "
                f"have {rows}"
            )
        covered[name] = weight
    if shapes and not covered:
        raise ValueError(_zero_coverage_message(by_gguf_name, shapes))
    return covered, tuple(uncovered)


def check_q0_imatrix_weights(
    weights: Mapping[str, torch.Tensor], shapes: Mapping[str, Sequence[int]]
) -> None:
    """Refuse q0 imatrix weights that cannot match the model.

    The meter runs this at construction, over any weight source —
    resolved from a file or passed directly. A typoed name or a
    wrong-shape tensor would price cells against the wrong columns,
    silently, hours in.

    Args:
        weights: Weights keyed by HF parameter name — 1-D column
            weights, or 2-D per-expert weights for a 3-D parameter.
        shapes: Parameter shapes keyed by discovered parameter name.

    Raises:
        ValueError: If a weighted name is not a discovered
            parameter, a weight tensor fits neither layout, its
            column count does not match the parameter's row length,
            the rows do not divide into ``QK4_0`` blocks, or a
            weight is negative or non-finite.
    """
    for name, columns in weights.items():
        shape = shapes.get(name)
        if shape is None:
            raise ValueError(f'imatrix weights name unknown parameter "{name}"')
        rows = int(shape[-1])
        two_d = columns.dim() == 2  # noqa: PLR2004 - the per-expert layout
        if two_d and (
            len(shape) != _FUSED_STACK_DIMS or int(columns.shape[0]) != int(shape[0])
        ):
            raise ValueError(
                f"imatrix weights for {name} pair expert rows with a 3-D "
                f"expert stack — got weights {tuple(columns.shape)} against "
                f"a parameter of shape {tuple(shape)}"
            )
        if columns.dim() not in (1, 2) or int(columns.shape[-1]) != rows:
            raise ValueError(
                f"imatrix weights for {name} must be 1-D with {rows} "
                f"entries, or 2-D with {rows} columns, got shape "
                f"{tuple(columns.shape)}"
            )
        if rows % QK4_0:
            raise ValueError(
                f"{name} has rows of {rows}, not divisible into "
                f"{QK4_0}-element Q4_0 blocks — the assisted fit cannot "
                "align its column weights (ADR-0018)"
            )
        if not bool(torch.isfinite(columns).all()) or bool((columns < 0).any()):
            raise ValueError(
                f"imatrix weights for {name} must be finite and non-negative"
            )
