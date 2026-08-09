"""Reconstruction-check stage for the ``quantfit pack`` command.

The mandatory guard on protected imatrix packs (ADR-0022): fit
collapse is invisible to the smoke test, so every protected tensor
must prove it reconstructs closer to the f16 base than it does at
its unprotected type. The reference is a second pack of the same
recipe with its protections stripped — the same assisted fit at the
unprotected types, which a cheap unweighted re-quantize cannot
reproduce. The stage refuses and names the collapsed tensors, suggesting the
``--exclude-imatrix`` flags for the ones whose exclusion remedy is
still unused (ADR-0023); it
never repacks on its own, so the packed file stays recipe-driven
(ADR-0012 decision 3). The mapping pre-flight lives here too, and
the run-log event guards non-finite measurements the sink would
reject (ADR-0011).

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

import math
import time
from collections.abc import Callable
from pathlib import Path

import typer

from quantfit.adapters.inbound.cli_pack_smoke import _halt
from quantfit.adapters.inbound.run_log import SafeRunLog
from quantfit.adapters.outbound.gguf.reconstruction import GgufReconstructionChecker
from quantfit.adapters.outbound.gguf.types import (
    PackError,
    ggml_type_for,
    gguf_tensor_name,
)
from quantfit.domain.model import Recipe
from quantfit.domain.pack import collapsed_tensors, without_protections
from quantfit.ports.outbound import RecipePacker, ReconstructionChecker


def _check_protected_mappable(recipe: Recipe) -> None:
    """Reject an unpackable protection before any tool runs.

    The class table and the type table judge every resolved pair
    here, so a protection the backend cannot drive fails in
    milliseconds — not after the convert stage writes a full-size
    base GGUF (ADR-0022). An unconstrained plan accepts any positive
    floor, so this is where a 7-bit floor meets the ADR-0012 table.

    Args:
        recipe: The recipe about to pack.

    Raises:
        typer.Exit: With code 1 when a protected tensor has no GGUF
            mapping or its precision has no type-table entry.
    """
    for pair in recipe.protected_tensors:
        try:
            gguf_tensor_name(pair.tensor)
            ggml_type_for(pair.bits)
        except PackError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc


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
    unprotected pack passes through silently. A recipe that records
    protections but resolved zero pairs also skips with a note: every
    floor was a per-tensor no-op at plan time (issue #59).

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
        if recipe.plan.protections:
            typer.echo(
                "reconstruction check skipped: the recipe records "
                "protections but resolved no pairs — every floor was a "
                "per-tensor no-op at plan time (issue #59)"
            )
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
    user's, and `_collapse_error` picks the remedies the refusal
    offers. The event guards
    non-finite measurements — the sink rejects NaN, and the halt
    record must survive (ADR-0011).

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
        collapsed = collapsed_tensors(protected_rmse, reference_rmse)
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "reconstruction", exc) from exc
    finally:
        reference_path.unlink(missing_ok=True)

    def finite(value: float) -> float | None:
        """Guard one measurement for the run-log sink.

        The sink rejects NaN and infinity (ADR-0011), and a collapsed
        measurement can be exactly that. The text copy in the event
        keeps the real value on record.

        Args:
            value: The measured error.

        Returns:
            The value, or None when non-finite.
        """
        return value if math.isfinite(value) else None

    run_log.emit(
        "reconstruction_checked",
        {
            "tensors": {
                name: {
                    "protected_rmse": finite(protected_rmse[name]),
                    "protected_rmse_text": str(protected_rmse[name]),
                    "reference_rmse": finite(reference_rmse[name]),
                    "reference_rmse_text": str(reference_rmse[name]),
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
        raise _halt(
            run_log,
            "reconstruction",
            _collapse_error(recipe, collapsed, hf_by_gguf, out),
        )
    typer.echo("reconstruction check passed — no fit collapse")


def _collapse_error(
    recipe: Recipe,
    collapsed: tuple[str, ...],
    hf_by_gguf: dict[str, str],
    out: Path,
) -> RuntimeError:
    """Build the refusal, suggesting only remedies not yet applied.

    A collapsed tensor whose pair already carries ``exclude_imatrix``
    has exhausted the ADR-0023 remedy — suggesting the same flag
    again would send the user in a circle. Such a tensor gets the
    ADR-0022 remedy only: drop its protection.

    Args:
        recipe: The protected recipe that was just packed.
        collapsed: Collapsed GGUF tensor names.
        hf_by_gguf: GGUF-to-HF name mapping for the protected pairs.
        out: The packed model, kept for inspection.

    Returns:
        The refusal, naming every collapsed tensor.
    """
    already = {p.tensor for p in recipe.protected_tensors if p.exclude_imatrix}
    failed = ", ".join(hf_by_gguf[name] for name in collapsed)
    fresh = [hf_by_gguf[name] for name in collapsed if hf_by_gguf[name] not in already]
    exhausted = [hf_by_gguf[name] for name in collapsed if hf_by_gguf[name] in already]
    remedies = []
    if fresh:
        flags = " ".join(f'--exclude-imatrix "{name}"' for name in fresh)
        remedies.append(
            f"Re-plan with {flags} to keep those promotions on the "
            f"unweighted fit (ADR-0023), or exclude the tensors from "
            f"--protect (ADR-0022)."
        )
    if exhausted:
        names = ", ".join(exhausted)
        remedies.append(
            f"The unweighted fit already failed for {names} — exclude "
            f"them from --protect and re-plan (ADR-0022)."
        )
    return RuntimeError(
        f"fit collapse on {failed} — the protection makes these tensors "
        f"reconstruct worse than their unprotected type. "
        + " ".join(remedies)
        + f" The file is kept at {out}"
    )
