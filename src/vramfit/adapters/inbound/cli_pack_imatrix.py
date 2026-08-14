"""The imatrix stages of ``vramfit pack``: provenance, counts, echoes.

Split from [vramfit.adapters.inbound.cli_pack][] to keep that module
under the size cap. This module owns what ``--imatrix`` adds to the
pack flow: the provenance warnings against the recipe's record
(ADR-0020, ADR-0023), the count read that finds a zero-count expert
(ADR-0026 decision 5, the 2026-08-13 #198 amendment), and the
console echoes of what the matrix did and did not reach (ADR-0016).

Examples:
    The pack command drives the count read between its stages:

    ```python
    zero_counts = _read_zero_count_experts(run_log, imatrix, base_path)
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.imatrix_counts][]: The reader
      this module wires.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vramfit.adapters.inbound.cli_pack_smoke import _halt
from vramfit.adapters.inbound.run_log import SafeRunLog
from vramfit.adapters.outbound.gguf.imatrix_counts import GgufImatrixCounts
from vramfit.domain.model import Recipe
from vramfit.domain.pack import PackResult, zero_count_experts
from vramfit.ports.outbound import ImatrixCountSource


def _build_count_source(imatrix: Path, base_gguf: Path) -> ImatrixCountSource:
    """Wire the gguf-py count reader for one ``--imatrix`` pack run.

    Unit tests monkeypatch this seam with the verified fake, keeping
    the command's orchestration testable without gguf-py (ADR-0009).

    Args:
        imatrix: The importance matrix the pack consumes.
        base_gguf: The f16 base GGUF whose shapes vouch for it.

    Returns:
        The wired source.
    """
    return GgufImatrixCounts(imatrix=imatrix, base_gguf=base_gguf)


def _read_zero_count_experts(
    run_log: SafeRunLog, imatrix: Path | None, base_gguf: Path
) -> tuple[tuple[str, int], ...]:
    """Read the imatrix counts and judge them, refusing an unvouchable file.

    Runs after convert and before quantize (ADR-0026 decision 5, the
    #198 amendment): an empty report is what a healthy matrix
    returns, so a file the reader cannot vouch for refuses here — in
    seconds, before the quantizer runs for minutes. The report
    itself never stops the pack.

    Args:
        run_log: The pack run's log.
        imatrix: The ``--imatrix`` value, or None for a matrix-less
            pack, which reads nothing.
        base_gguf: The f16 base GGUF the convert stage ensured.

    Returns:
        The ``(stack, expert)`` pairs the matrix counts zero times,
        sorted. Empty without an imatrix.

    Raises:
        typer.Exit: With code 1 when the reader refuses the file or
            gguf-py is missing.
    """
    if imatrix is None:
        return ()
    try:
        counts = _build_count_source(imatrix, base_gguf).expert_stack_counts()
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "imatrix_counts", exc) from exc
    return zero_count_experts(counts)


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

    The coverage lines state counts only (#191's shape): a joined
    list buries the split it reports. On the MoE target the
    zero-count report enumerates 5,888 stack-expert pairs. The
    ``model_packed`` run-log event names every uncovered tensor and
    every zero-count pair. The exclusions line keeps its names: the
    operator chose that set, and it is small by design (ADR-0023).

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
        typer.echo(
            "warning: the importance matrix did not cover "
            f"{len(result.imatrix_uncovered)} tensors — they quantized "
            "unassisted, and the run log names them (token_embd is "
            "expected)",
            err=True,
        )
    if result.imatrix_zero_count_experts:
        typer.echo(
            "warning: the importance matrix counts zero samples for "
            f"{len(result.imatrix_zero_count_experts)} stack-expert "
            "pairs — the quantizer fits them unassisted, and the run "
            "log names each pair (ADR-0026)",
            err=True,
        )
