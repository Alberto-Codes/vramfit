"""The ``vramfit pack`` command's imatrix reporting.

Two console reports, split from
[vramfit.adapters.inbound.cli_pack][] so each module stays under the
file-size cap. `_warn_imatrix_provenance` runs before the pack and
compares the ``--imatrix`` value against the recipe's record
(ADR-0020, ADR-0023). `_report_imatrix_effects` runs after it and
echoes what the matrix reached: the exclusions the recipe
instructed (ADR-0023), the tensors it did not cover (ADR-0016), and
the routed experts it covers at a count of zero (ADR-0026 decision
5). The three are separate cases. An exclusion is intentional, an
uncovered tensor is a whole-tensor gap, and a zero-count expert
sits inside a stack the matrix does cover. The zero-count line
names the first `_ECHO_LIMIT` experts and counts the rest, because
a bad corpus can starve thousands of the MoE target's 2,944 cells.
The run log carries every pair.

Every report warns and none refuses. A pack that ignores its own
provenance still produces a file, and the operator decides what
that file is worth.

Examples:
    Report one pack's imatrix effects:

    ```python
    from vramfit.adapters.inbound.cli_pack_imatrix import _report_imatrix_effects

    _report_imatrix_effects(result)
    ```

See Also:
    - [vramfit.adapters.inbound.cli_pack][]: The command that calls
      both reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import typer

from vramfit.domain.model import Recipe
from vramfit.domain.pack import PackResult

# How many zero-count experts the console names before it counts the
# rest. The run log carries every one.
_ECHO_LIMIT: Final[int] = 10


def _warn_imatrix_provenance(recipe: Recipe, imatrix: Path | None) -> None:
    """Warn when the pack's imatrix cannot honor the recipe's record.

    An assisted-priced recipe is only comparable to a pack that
    consumes the same imatrix file (ADR-0020), and a recipe's
    imatrix exclusions change nothing without a matrix to exclude
    from (ADR-0023). Warnings, not refusals — packing itself works
    either way.

    Args:
        recipe: The loaded recipe.
        imatrix: The ``--imatrix`` value, or None.
    """
    if recipe.imatrix is not None:
        if imatrix is None:
            typer.echo(
                "warning: the recipe was priced with imatrix "
                f'"{recipe.imatrix}" but --imatrix is absent — the pack '
                "will not match the map's frame (ADR-0020)",
                err=True,
            )
        elif imatrix.resolve() != Path(recipe.imatrix).resolve():
            typer.echo(
                f'warning: --imatrix "{imatrix}" differs from the recipe\'s '
                f'recorded imatrix "{recipe.imatrix}" — the pack will not '
                "match the map's frame (ADR-0020)",
                err=True,
            )
    excluded_pairs = [p for p in recipe.protected_tensors if p.exclude_imatrix]
    if excluded_pairs and imatrix is None:
        typer.echo(
            f"warning: the recipe marks {len(excluded_pairs)} imatrix "
            "exclusions but --imatrix is absent — without a matrix the "
            "exclusions change nothing (ADR-0023)",
            err=True,
        )


def _report_imatrix_effects(result: PackResult) -> None:
    """Echo what the imatrix did and did not reach.

    Three separate records, one line each. The zero-count line takes
    a prefix of the experts and counts the remainder.

    Args:
        result: The pack step's accounting record.
    """
    if result.imatrix_excluded:
        names = ", ".join(result.imatrix_excluded)
        typer.echo(
            f"imatrix exclusions applied: {names} — these tensors "
            "quantized with the unweighted fit (ADR-0023)"
        )
    if result.imatrix_uncovered:
        names = ", ".join(result.imatrix_uncovered)
        typer.echo(
            f"warning: the importance matrix did not cover: {names} — "
            "these tensors quantized unassisted (token_embd is expected)",
            err=True,
        )
    starved = result.imatrix_zero_count_experts
    if starved:
        # A stack the matrix covers, an expert inside it the router
        # never fired. The quantizer emits no warning for this, so
        # this line is the only report (ADR-0026 decision 5).
        # A bad corpus can starve thousands of the target's 2,944
        # cells, so the console takes a prefix and the run log keeps
        # every pair.
        experts = ", ".join(f"{e.stack}[{e.expert}]" for e in starved[:_ECHO_LIMIT])
        rest = len(starved) - _ECHO_LIMIT
        typer.echo(
            f"warning: the importance matrix fired no token through "
            f"{len(starved)} experts: {experts}"
            + (f", and {rest} more" if rest > 0 else "")
            + " — these experts quantized unassisted inside a covered "
            "stack (ADR-0026). The run log names them all.",
            err=True,
        )
