"""The ``vramfit pack`` command: apply a recipe through the GGUF backend.

The composition root for the pack step (ADR-0010, ADR-0012). It
validates the toolchain paths and the recipe's protection mapping
up front — an unpackable protection fails in milliseconds, not
after the convert stage — loads the recipe, wires the
`RecipePacker` port to the llama.cpp adapter, and drives the two
stages — convert, then quantize (imatrix-assisted when ``--imatrix``
is given, ADR-0016, with the recipe's imatrix exclusions applied,
ADR-0023) — emitting one run-log event per stage. Between the
stages an ``--imatrix`` pack reads the matrix's counts and reports
every zero-count expert (ADR-0026 decision 5) — the imatrix
warnings, the count read, and the echoes live in
[vramfit.adapters.inbound.cli_pack_imatrix][]. A type-fallback
warning in the quantizer's output halts the quantize stage with the
file kept, and the ``pack_halted`` event carries stage
``type_fallback`` and every rewrite (ADR-0028). The ``model_packed``
event and one ``warning:`` line name every layer the base GGUF
numbers that no override reached. Those layers pack at the recipe's
floor and the quantizer reports none of them (#307). After
packing it re-checks the real bytes against the recipe's weight
budget, gates a protected
imatrix pack on the reconstruction check
(ADR-0022 — the stage lives in
[vramfit.adapters.inbound.cli_pack_check][]), ships the
``--mmproj`` projector sidecar beside the artifact both checks
accepted (ADR-0030 — the stage lives in
[vramfit.adapters.inbound.cli_pack_sidecar][]), then proves the
artifact emits language when ``--smoke-text``
is given (ADR-0017 — the smoke stage lives in
[vramfit.adapters.inbound.cli_pack_smoke][]). A failed check exits
1 and keeps the file for inspection.

Examples:
    Pack a recipe with a local llama.cpp checkout:

    ```console
    $ vramfit pack recipe.json --llama-cpp ~/llama.cpp --out packed.gguf
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: The adapter this
      command wires.
    - [vramfit.domain.pack][]: The budget re-check.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from vramfit.adapters.inbound.cli_pack_check import (
    _check_protected_mappable,
    _reconstruction_stage,
)
from vramfit.adapters.inbound.cli_pack_imatrix import (
    _read_zero_count_experts,
    _report_imatrix_effects,
    _warn_imatrix_provenance,
)
from vramfit.adapters.inbound.cli_pack_sidecar import (
    _ship_sidecar_stage,
    check_sidecar_collisions,
)
from vramfit.adapters.inbound.cli_pack_smoke import (
    _check_inputs,
    _halt,
    _halt_type_fallback,
    _run_smoke,
)
from vramfit.adapters.inbound.run_log import SafeRunLog
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker, TypeFallbackError
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.recipe_json import load_recipe
from vramfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile
from vramfit.domain.budget import format_size
from vramfit.domain.model import Recipe
from vramfit.domain.pack import PackResult, weight_budget_margin
from vramfit.ports.outbound import RecipePacker


def _build_packer(
    model_dir: Path,
    base_gguf: Path,
    out: Path,
    llama_cpp: Path,
    python_bin: Path,
    threads: int,
    imatrix: Path | None,
) -> RecipePacker:
    """Wire the llama.cpp adapter for one pack run.

    Unit tests monkeypatch this seam with the verified fake, keeping
    the command's orchestration testable without the toolchain
    (ADR-0009).

    Args:
        model_dir: Hugging Face checkpoint directory.
        base_gguf: Full-precision base GGUF path.
        out: Packed model destination.
        llama_cpp: llama.cpp checkout with built tools.
        python_bin: Interpreter for the convert script.
        threads: Quantizer thread count.
        imatrix: Importance matrix file, or None (ADR-0016).

    Returns:
        The wired packer.
    """
    return LlamaCppPacker(
        model_dir=model_dir,
        base_gguf=base_gguf,
        out_path=out,
        convert_script=llama_cpp / "convert_hf_to_gguf.py",
        quantize_bin=llama_cpp / "build" / "bin" / "llama-quantize",
        python_bin=python_bin,
        threads=threads,
        imatrix=imatrix,
    )


def _load_recipe(path: Path) -> Recipe:
    """Load the recipe artifact, halting cleanly when it is invalid.

    Args:
        path: The recipe file.

    Returns:
        The validated recipe.

    Raises:
        typer.Exit: With code 1 when the file is missing or invalid.
    """
    try:
        return load_recipe(path)
    except (OSError, ArtifactError) as exc:
        typer.echo(f"error: {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _report_floored_layers(result: PackResult) -> None:
    """Echo the layers the recipe never addressed.

    The quantizer prints nothing for them and exits 0. So only this
    line and the ``model_packed`` event name the case (#307).

    **The packed file grows, and the line says so.** A layer reaches
    no override only when the recipe holds no assignment for it.
    ``plan.predicted_total_bytes`` sums the assignment sizes, so it
    never counted that layer. The quantizer still writes the tensors
    at the floor. The margin line two statements later can therefore
    read ``OVER`` for exactly this reason.

    The count leads and the names follow. A base GGUF numbering one
    unpriced block yields one name. A wrong-variant base can yield
    dozens, so the run log carries the list as data.

    Args:
        result: The pack step's accounting record.
    """
    if not result.floored_layers:
        return
    count = len(result.floored_layers)
    # A base GGUF numbering one unpriced block is the narrow case, so
    # the singular is not an edge.
    noun = "layer" if count == 1 else "layers"
    names = ", ".join(result.floored_layers)
    typer.echo(
        f"warning: the base GGUF carries {count} {noun} no override "
        f"reached: {names}. They packed at the {result.base_type} "
        "floor. The recipe priced none of them, so the file exceeds "
        "plan.predicted_total_bytes by their cost (#307)",
        err=True,
    )


def _report_pack_effects(result: PackResult) -> None:
    """Echo everything the packed file carries that the recipe does not.

    Both reports name a gap the quantizer leaves unreported on a zero
    exit. The floored layers come first, because they explain a size
    the imatrix lines do not.

    Args:
        result: The pack step's accounting record.
    """
    _report_floored_layers(result)
    _report_imatrix_effects(result)


def _size_check_stage(
    run_log: SafeRunLog, recipe: Recipe, packed_bytes: int, out: Path
) -> None:
    """Re-check the packed file's real bytes against the budget.

    Nominal-bit predictions undershoot GGUF's effective bits, so the
    recipe's promise is re-proven on the artifact (ADR-0014).

    Args:
        run_log: The pack run's event log.
        recipe: The recipe the pack applied.
        packed_bytes: Real size of the packed model file.
        out: The packed model path, for the report.

    Raises:
        typer.Exit: With code 1 when the packed bytes exceed the
            weight budget (via ``_halt``); the file is kept.
    """
    margin = weight_budget_margin(recipe, packed_bytes)
    fits = margin >= 0
    run_log.emit(
        "size_checked",
        {
            "packed_bytes": packed_bytes,
            "weight_budget_bytes": recipe.plan.weight_budget_bytes,
            "margin_bytes": margin,
            "fits": fits,
        },
    )
    typer.echo(
        f"packed {len(recipe.assignments)} groups -> {out} "
        f"({format_size(packed_bytes)}), weight budget "
        f"{format_size(recipe.plan.weight_budget_bytes)}, margin "
        f"{format_size(abs(margin))} {'under' if fits else 'OVER'}"
    )
    if not fits:
        error = RuntimeError(
            f"packed model exceeds the weight budget by {format_size(-margin)} "
            f"— the file is kept at {out}"
        )
        raise _halt(run_log, "size_check", error)


def pack(
    recipe_path: Annotated[
        Path, typer.Argument(metavar="RECIPE", help="Recipe produced by vramfit plan.")
    ],
    llama_cpp: Annotated[
        Path,
        typer.Option(
            help="llama.cpp checkout with convert_hf_to_gguf.py and built tools."
        ),
    ],
    model: Annotated[
        Path | None,
        typer.Option(
            help="Model checkpoint directory. Default: the recipe's model_id."
        ),
    ] = None,
    out: Annotated[Path, typer.Option(help="Packed model path.")] = Path("packed.gguf"),
    base_gguf: Annotated[
        Path | None,
        typer.Option(
            help="f16 base GGUF path, reused when present. Default: beside --out."
        ),
    ] = None,
    python_bin: Annotated[
        Path | None,
        typer.Option(
            help="Interpreter for the convert script — the pack extra "
            "provisions its dependencies. Default: this one."
        ),
    ] = None,
    threads: Annotated[
        int,
        typer.Option(min=1, help="Thread count for the quantizer and the smoke test."),
    ] = 8,
    imatrix: Annotated[
        Path | None,
        typer.Option(
            help="Importance matrix for the quantizer (ADR-0016). "
            "Generate with llama-imatrix against the base GGUF."
        ),
    ] = None,
    mmproj: Annotated[
        Path | None,
        typer.Option(
            help="Vendor mmproj to ship beside --out as the projector "
            "sidecar (ADR-0030). The copy is byte-identical."
        ),
    ] = None,
    smoke_text: Annotated[
        Path | None,
        typer.Option(
            help="Text for the post-pack smoke test (ADR-0017). "
            "Without it the packed model stays unproven."
        ),
    ] = None,
    smoke_chunks: Annotated[
        int, typer.Option(min=1, help="Smoke-test chunk count.")
    ] = 2,
    smoke_threshold: Annotated[
        float,
        typer.Option(
            min=0.0, help="Perplexity ceiling a passing smoke test stays under."
        ),
    ] = 1000.0,
    runlog: Annotated[
        Path | None,
        typer.Option(help="Run-log path (JSONL). Default: <stem>.runlog.jsonl."),
    ] = None,
) -> None:
    """Pack a recipe into a GGUF model llama.cpp can serve.

    Converts the checkpoint to an f16 base GGUF once (reusing an
    existing file), then drives ``llama-quantize`` with one type
    override per layer group. An ``lm_head`` group drives the output
    head directly. Without one the embedding assignment pins an
    untied head (ADR-0012). The ``--python-bin`` interpreter
    runs the convert script — the ``pack`` extra provisions its
    dependencies. ``--imatrix`` hands the quantizer an importance
    matrix (ADR-0016). The command then reads that matrix's
    ``.counts`` tensors against the base GGUF and reports every
    expert the matrix counts zero times — the quantizer fits such
    an expert unassisted and prints no warning (ADR-0026 decision
    5). A matrix the reader cannot vouch for halts before the
    quantizer runs, and the report lands in the result only beside
    its matrix path. The read needs gguf-py, which the pack extra
    provisions. A recipe priced on an assisted map records
    its imatrix — the command warns when ``--imatrix`` is absent or
    names a different file, because the pack would not match the
    map's frame (ADR-0020). A recipe with imatrix exclusions packs
    the marked tensors on the unweighted fit, and the command warns
    when no matrix makes the exclusions inert (ADR-0023). An
    expert-stack group maps through its own type table (ADR-0028):
    8 to Q8_0, 4 to Q4_0, 2 to Q2_0. Nominal 3 refuses there — no
    GGUF type lands between 2.25 and 4.25 bits per weight on the
    stack rows. The quantizer's output is scanned for the
    type-fallback warning pair, and a match halts with the file
    kept — a rewritten type breaks the recipe the artifact claims
    to carry (ADR-0028). A layer the base GGUF numbers that no
    override reached packs at the recipe's floor (decision 3). The
    quantizer prints nothing, so the command warns and the run log
    names each one (#307). Such a layer carries no assignment, so it
    adds bytes the recipe never priced and the size re-check below
    grows more likely to refuse. #320 carries whether the case should
    refuse outright. The
    command re-checks the packed file's real
    bytes against the recipe's weight budget — nominal-bit
    predictions undershoot GGUF's effective bits. A protected
    recipe's override composition is checked before any stage runs,
    catching an unmappable pair and a protection under a second
    root (#367). The check judges nothing else. A protected recipe
    packed with ``--imatrix`` must then pass the reconstruction
    check — a collapsed tensor halts with its name, and the revision
    is the user's (ADR-0022). ``--mmproj`` ships the vendor mmproj
    beside ``--out`` as the unquantized projector sidecar (ADR-0030
    decision 2). The stage runs after the size check and the
    reconstruction gate pass. The copy is byte-identical, and
    SHA-256 of source and copy proves it. The ``sidecar_shipped``
    event carries the digest. A copy that would overwrite a
    run-owned file refuses before any tool runs. The sidecar never
    enters the weight budget — the vision line is a serving
    measurement, not the file size (ADR-0030 decision 3). With
    ``--smoke-text`` it then proves the packed model emits language:
    ``--smoke-chunks`` perplexity chunks under the
    ``--smoke-threshold`` ceiling (ADR-0017). A run-log write
    failure warns once, naming the file, and disables the log
    (ADR-0011).

    Raises:
        typer.BadParameter: If the llama.cpp checkout misses a needed
            tool, ``--imatrix``, ``--mmproj``, or ``--smoke-text`` is
            not a file, ``--mmproj`` is empty or its sidecar copy
            would overwrite a run-owned file,
            ``--smoke-threshold`` is not positive, or the ``--out``
            or ``--runlog`` directory does not exist.
        typer.Exit: With code 1 when the recipe is invalid, the model
            directory does not exist, a toolchain stage fails, the
            packed model exceeds the weight budget, the
            reconstruction check finds a collapsed tensor, the
            sidecar copy fails or does not match its source, or the
            smoke test fails (the file is kept in each case).

    Examples:
        Command line usage:

        ```console
        $ vramfit pack recipe.json --llama-cpp ~/llama.cpp
        ```
    """
    _check_inputs(llama_cpp, out, imatrix, mmproj, smoke_text, smoke_threshold)

    recipe = _load_recipe(recipe_path)
    # A protected recipe's override composition must fail here, in
    # milliseconds — not after the convert stage writes a full-size
    # base GGUF (ADR-0022, #367).
    _check_protected_mappable(recipe)
    _warn_imatrix_provenance(recipe, imatrix)
    model_dir = model if model is not None else Path(recipe.model_id)
    if not model_dir.is_dir():
        typer.echo(
            f'error: model directory "{model_dir}" does not exist — the '
            "recipe's model_id is not a local path, pass --model",
            err=True,
        )
        raise typer.Exit(code=1)
    base_path = (
        base_gguf
        if base_gguf is not None
        else out.with_name(f"{model_dir.name}-f16.gguf")
    )

    runlog_path = (
        runlog if runlog is not None else out.with_name(out.stem + ".runlog.jsonl")
    )
    if not runlog_path.parent.is_dir():
        raise typer.BadParameter(
            f"--runlog: directory {runlog_path.parent} does not exist"
        )
    check_sidecar_collisions(mmproj, out, base_path, runlog_path)
    run_log = SafeRunLog(JsonlRunLogFile(runlog_path), path=runlog_path)
    run_log.emit(
        "pack_started",
        {
            "recipe": str(recipe_path),
            "model": str(model_dir),
            "out": str(out),
            "base_gguf": str(base_path),
            "groups": len(recipe.assignments),
        },
    )

    packer = _build_packer(
        model_dir,
        base_path,
        out,
        llama_cpp,
        python_bin if python_bin is not None else Path(sys.executable),
        threads,
        imatrix,
    )

    reused = base_path.exists()
    typer.echo(
        f"reusing base GGUF {base_path}"
        if reused
        else f"converting {model_dir} -> {base_path} (minutes at 3B scale)"
    )
    started = time.monotonic()
    try:
        base_bytes = packer.convert()
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "convert", exc) from exc
    run_log.emit(
        "gguf_converted",
        {
            "path": str(base_path),
            "bytes": base_bytes,
            "seconds": round(time.monotonic() - started, 3),
            "reused": reused,
        },
    )

    # The count read runs between the stages: it needs the base GGUF
    # the convert stage ensured, and an unvouchable matrix must
    # refuse before the quantizer runs for minutes (ADR-0026).
    zero_counts = _read_zero_count_experts(run_log, imatrix, base_path)

    started = time.monotonic()
    try:
        result = packer.pack(recipe)
    except TypeFallbackError as exc:
        raise _halt_type_fallback(run_log, exc) from exc
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "quantize", exc) from exc
    # The result's own imatrix_path gates the merge: PackResult
    # refuses the field without a matrix, and an unguarded replace
    # would trade the run log's clean halt for a traceback.
    if zero_counts and result.imatrix_path is not None:
        result = replace(result, imatrix_zero_count_experts=zero_counts)
    run_log.emit(
        "model_packed",
        {
            "out": str(out),
            "packed_bytes": result.packed_bytes,
            "seconds": round(time.monotonic() - started, 3),
            "base_type": result.base_type,
            "token_embedding_type": result.token_embedding_type,
            "output_tensor_type": result.output_tensor_type,
            "overrides": len(result.overrides),
            "imatrix": result.imatrix_path,
            "imatrix_uncovered": list(result.imatrix_uncovered),
            "imatrix_excluded": list(result.imatrix_excluded),
            "imatrix_zero_count_experts": [
                [stack, expert] for stack, expert in result.imatrix_zero_count_experts
            ],
            "floored_layers": list(result.floored_layers),
        },
    )
    _report_pack_effects(result)

    _size_check_stage(run_log, recipe, result.packed_bytes, out)

    # The mandatory guard on protected imatrix packs (ADR-0022): fit
    # collapse is invisible to the smoke test, so the gate runs first.
    _reconstruction_stage(
        run_log,
        recipe,
        imatrix,
        out,
        base_path,
        lambda reference_path: _build_packer(
            model_dir,
            base_path,
            reference_path,
            llama_cpp,
            python_bin if python_bin is not None else Path(sys.executable),
            threads,
            imatrix,
        ),
    )

    _ship_sidecar_stage(run_log, mmproj, out)

    if smoke_text is None:
        typer.echo(
            "warning: packed model is unproven — pass --smoke-text to run "
            "the smoke test (ADR-0017)",
            err=True,
        )
    else:
        _run_smoke(
            run_log, llama_cpp, out, smoke_text, smoke_chunks, smoke_threshold, threads
        )
    run_log.emit(
        "pack_finished",
        {
            "out": str(out),
            "packed_bytes": result.packed_bytes,
            "smoked": smoke_text is not None,
        },
    )
