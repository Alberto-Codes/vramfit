"""Reconstruction-check stage for the ``quantfit pack`` command.

The mandatory guard on protected imatrix packs (ADR-0022): fit
collapse is invisible to the smoke test, so every protected tensor
must prove it reconstructs closer to the f16 base than it does at
its unprotected type. The reference is a second pack of the same
recipe with its protections stripped — the same assisted fit at the
unprotected types, which a cheap unweighted re-quantize cannot
reproduce. The stage refuses and names the collapsed tensors; it
never repacks on its own, so the packed file stays recipe-driven
(ADR-0012 decision 3).

Examples:
    The pack command drives the stage like this:

    ```python
    if recipe.protected_tensors and imatrix is not None:
        _run_reconstruction(run_log, recipe, reference_packer, ...)
    ```

See Also:
    - [quantfit.adapters.inbound.cli_pack][]: The command that
      drives this stage.
    - [quantfit.adapters.outbound.gguf.reconstruction][]: The
        measurement adapter `_build_checker` wires.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import typer

from quantfit.adapters.inbound.cli_pack_smoke import _halt
from quantfit.adapters.inbound.run_log import SafeRunLog
from quantfit.adapters.outbound.gguf.reconstruction import GgufReconstructionChecker
from quantfit.adapters.outbound.gguf.types import gguf_tensor_name
from quantfit.domain.model import Recipe
from quantfit.domain.pack import collapsed_tensors, without_protections
from quantfit.ports.outbound import RecipePacker, ReconstructionChecker


def _reconstruction_stage(
    run_log: SafeRunLog,
    recipe: Recipe,
    imatrix: Path | None,
    out: Path,
    base_path: Path,
    reference_packer_for: Callable[[Path], RecipePacker],
) -> None:
    """Decide whether the reconstruction gate applies, and run it.

    The check is mandatory on protected imatrix packs (ADR-0022). A
    protected pack without an imatrix skips it with a note — every
    known fit collapse involved a promotion under one — and an
    unprotected pack passes through silently.

    Args:
        run_log: Sink for the stage's events.
        recipe: The recipe that was just packed.
        imatrix: The importance matrix the pack used, or None.
        out: The packed model.
        base_path: The f16 base GGUF.
        reference_packer_for: Builds a packer writing the reference
            file at the given path.

    Raises:
        typer.Exit: With code 1 when the gate runs and fails.
    """
    if not recipe.protected_tensors:
        return
    if imatrix is None:
        typer.echo(
            "reconstruction check skipped: the pack ran without an imatrix "
            "— every known fit collapse involved a promotion under one "
            "(ADR-0022)"
        )
        return
    reference_path = out.with_name(f"{out.stem}-reconstruction-ref.gguf")
    _run_reconstruction(
        run_log,
        recipe,
        reference_packer_for(reference_path),
        out,
        base_path,
        reference_path,
    )


def _build_checker(packed: Path, base: Path) -> ReconstructionChecker:
    """Wire the gguf-py reconstruction adapter for one packed file.

    Unit tests monkeypatch this seam with the verified fake
    (ADR-0009).

    Args:
        packed: The packed model to measure.
        base: The f16 base GGUF.

    Returns:
        The wired checker.
    """
    return GgufReconstructionChecker(packed=packed, base=base)


def _run_reconstruction(
    run_log: SafeRunLog,
    recipe: Recipe,
    reference_packer: RecipePacker,
    out: Path,
    base_path: Path,
    reference_path: Path,
) -> None:
    """Gate a protected imatrix pack on the reconstruction check.

    Packs the unprotected reference, measures every protected tensor
    against the f16 base in both files, deletes the reference file,
    and halts when any tensor is collapsed — the revision is the
    user's: exclude the named tensors from the protection and
    re-plan (ADR-0022).

    Args:
        run_log: Sink for the ``reconstruction_checked`` event.
        recipe: The protected recipe that was just packed.
        reference_packer: Packer wired to write ``reference_path``.
        out: The packed protected model.
        base_path: The f16 base GGUF.
        reference_path: Destination for the reference pack, deleted
            after measurement.

    Raises:
        typer.Exit: With code 1 when the reference pack fails, a
            measurement fails, or a protected tensor is collapsed
            (the packed file is kept).
    """
    hf_by_gguf = {
        gguf_tensor_name(p.tensor): p.tensor for p in recipe.protected_tensors
    }
    names = tuple(hf_by_gguf)
    typer.echo(
        f"reconstruction check: packing the unprotected reference -> "
        f"{reference_path} (ADR-0022)"
    )
    started = time.monotonic()
    try:
        reference_packer.pack(without_protections(recipe))
        protected_rmse = dict(_build_checker(out, base_path).rmse(names))
        reference_rmse = dict(_build_checker(reference_path, base_path).rmse(names))
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "reconstruction", exc) from exc
    finally:
        reference_path.unlink(missing_ok=True)
    collapsed = collapsed_tensors(protected_rmse, reference_rmse)
    run_log.emit(
        "reconstruction_checked",
        {
            "tensors": {
                name: {
                    "protected_rmse": protected_rmse[name],
                    "reference_rmse": reference_rmse[name],
                    "collapsed": name in collapsed,
                }
                for name in names
            },
            "reference": str(reference_path),
            "seconds": round(time.monotonic() - started, 3),
            "passed": not collapsed,
        },
    )
    for name in names:
        verdict = "COLLAPSED" if name in collapsed else "ok"
        typer.echo(
            f"reconstruction {name}: protected rmse {protected_rmse[name]:.6f}, "
            f"unprotected rmse {reference_rmse[name]:.6f} — {verdict}"
        )
    if collapsed:
        failed = ", ".join(hf_by_gguf[name] for name in collapsed)
        error = RuntimeError(
            f"fit collapse on {failed} — the protection makes these tensors "
            f"reconstruct worse than their unprotected type. Exclude them "
            f"from --protect and re-plan (ADR-0022). The file is kept at {out}"
        )
        raise _halt(run_log, "reconstruction", error)
    typer.echo("reconstruction check passed — no fit collapse")
