"""Exclusion matching: hold a recipe's exclusions against the imatrix.

The check refuses an ``--exclude-weights`` name the importance matrix
carries no row for (#309). It is the sibling of
[vramfit.adapters.outbound.gguf.override_match][], one flag over.

``tools/quantize/quantize.cpp`` erases a row with
``it->first.find(name)`` over the loaded matrix, a plain substring
search. A name no entry contains erases nothing. The loop counts no
erasures and prints nothing, and the quantizer exits 0. Verified
against llama.cpp at commit ``3653e6d6d`` (b10326, the pinned
instrument) and at ``e9fa0781f``, which carry the same loop.

Two things then go wrong, and ADR-0023 names both.

- **The tensor keeps the fit the recipe asked to drop.** Decision 1
  buys the exclusion to swap a collapsed tensor's assisted fit for the
  unweighted one. A row that survives keeps the collapse the recipe
  exists to remedy.
- **The record states an exclusion that never applied.** Decision 4
  files each deleted row under ``imatrix_excluded`` and keeps
  ``imatrix_uncovered`` an honest record of *unintentional* gaps. An
  unapplied exclusion enters the first field and discounts the tensor
  out of the second, so the tensor reads as neither a gap nor an
  applied exclusion.

**Why this refuses rather than reports.** ADR-0023 addresses two
neighbouring cases and answers both by refusing or by inertness.
Decision 1 refuses a glob that matches no protected tensor, and its
2026-08-09 (#59) amendment refuses an exclusion whose every pair
drops as a no-op. Decision 4 makes an exclusion inert without a
matrix, which `LlamaCppPacker.pack` honors by emitting no flag. The
record addresses no third case, so an exclusion that reaches a real
matrix and erases nothing is a malformed input rather than a ruled
outcome. #307 is the contrasting case: ADR-0012 decision 3 supplies
the floored layer's mechanism, so that one reports and #320 owns the
refusal question.

The comparison is the tool's own. It searches the exclusion name
inside each entry name, with no case folding — ``quantize.cpp``
lower-cases a ``--tensor-type`` pattern before it compiles, and it
does no such thing to an exclusion.

The names the pack emits are full GGUF tensor names, which decision 4
requires and which reach exactly one row. A partial name matches many
rows, and this check passes it. Over-deletion is ADR-0023's own
subject and no concern of this module.

gguf-py rides the scan extra and the pack extra includes it, so the
read defers to first use. ``vramfit pack --help`` keeps working on a
base install (ADR-0005), and a matrix-less pack reads nothing.

Examples:
    Hold a recipe's exclusions against the matrix the quantizer loads:

    ```python
    check_exclusion_match(("blk.1.attn_v.weight",), Path("m.imatrix.gguf"))
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: the caller, which runs
      this before it builds the quantizer command.
    - [vramfit.adapters.outbound.gguf.imatrix_counts][]:
      `imatrix_entry_names`, the read this holds them against.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from vramfit.adapters.outbound.gguf.imatrix_counts import imatrix_entry_names
from vramfit.adapters.outbound.gguf.types import PackError


def unmatched_exclusions(
    excluded: Sequence[str], names: Iterable[str]
) -> tuple[str, ...]:
    """Name the exclusions that no entry name contains.

    The comparison searches each exclusion inside each entry name,
    which is what ``llama-quantize`` does with the same strings.
    Repeated exclusions report once, in first-seen order.

    Args:
        excluded: The full GGUF tensor names the pack would drive
            into ``--exclude-weights``.
        names: The imatrix's entry names.

    Returns:
        The unmatched exclusions, without repeats, in recipe order.

    Examples:
        A tensor the matrix never priced reports its name:

        ```python
        names = ("blk.0.attn_v.weight",)
        assert unmatched_exclusions(("blk.1.attn_v.weight",), names) == (
            "blk.1.attn_v.weight",
        )
        ```
    """
    entry_names = tuple(names)
    unmatched: list[str] = []
    for name in excluded:
        if name in unmatched:
            continue
        if not any(name in entry for entry in entry_names):
            unmatched.append(name)
    return tuple(unmatched)


def check_exclusion_match(excluded: Sequence[str], imatrix: Path) -> None:
    """Hold a recipe's exclusions against the imatrix, refusing a no-op.

    The refusal runs before the quantizer, so a recipe naming a row
    the matrix does not carry costs no quantize run and writes no
    file.

    Args:
        excluded: The full GGUF tensor names the pack would drive
            into ``--exclude-weights``. An empty sequence skips the
            read, because a recipe excluding nothing reaches no row
            by design.
        imatrix: The importance matrix the quantizer loads.

    Raises:
        PackError: If any exclusion reaches no entry (#309), if
            gguf-py is missing, or if the reader refuses the matrix.

    Examples:
        A recipe excluding a tensor the matrix never priced refuses
        here rather than packing:

        ```python
        check_exclusion_match(names, Path("model.imatrix.gguf"))
        ```
    """
    if not excluded:
        return
    unmatched = unmatched_exclusions(excluded, imatrix_entry_names(imatrix))
    if not unmatched:
        return
    details = ", ".join(f'"{name}"' for name in unmatched)
    raise PackError(
        f"the imatrix {imatrix} carries no row for {len(unmatched)} of "
        f"{len(excluded)} recipe exclusions: {details}. The quantizer "
        f"erases no row for such a name and exits 0. The tensor then keeps "
        f"the assisted fit the recipe asked to drop, and the record states "
        f"an exclusion that never applied (ADR-0023). Check the recipe's "
        f"protected tensors against the imatrix's entry names"
    )
