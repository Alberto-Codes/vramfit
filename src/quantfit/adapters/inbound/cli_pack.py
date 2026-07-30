"""The ``quantfit pack`` command: apply a recipe through the GGUF backend.

The composition root for the pack step (ADR-0010, ADR-0012). It
validates the toolchain paths up front, loads the recipe, wires the
`RecipePacker` port to the llama.cpp adapter, and drives the two
stages — convert, then quantize (imatrix-assisted when ``--imatrix``
is given, ADR-0016) — emitting one run-log event per stage. After
packing it re-checks the real bytes against the recipe's weight
budget, then proves the artifact emits language when ``--smoke-text``
is given (ADR-0017, the `SmokeTester` port). A failed check exits 1
and keeps the file for inspection.

Examples:
    Pack a recipe with a local llama.cpp checkout:

    ```console
    $ quantfit pack recipe.json --llama-cpp ~/llama.cpp --out packed.gguf
    ```

See Also:
    - [quantfit.adapters.outbound.gguf.pack][]: The adapter this
      command wires.
    - [quantfit.domain.pack][]: The budget re-check.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from quantfit.adapters.inbound.run_log import SafeRunLog
from quantfit.adapters.outbound.gguf.pack import LlamaCppPacker
from quantfit.adapters.outbound.gguf.smoke import LlamaCppSmokeTester
from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.recipe_json import load_recipe
from quantfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile
from quantfit.domain.budget import format_size
from quantfit.domain.model import Recipe
from quantfit.domain.pack import smoke_passed, weight_budget_margin
from quantfit.ports.outbound import RecipePacker, SmokeTester


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


def _build_smoke_tester(
    llama_cpp: Path,
    out: Path,
    smoke_text: Path,
    chunks: int,
    threads: int,
) -> SmokeTester:
    """Wire the llama.cpp smoke adapter for one packed model.

    Unit tests monkeypatch this seam with the verified fake
    (ADR-0009).

    Args:
        llama_cpp: llama.cpp checkout with built tools.
        out: The packed model to prove.
        smoke_text: Text the smoke chunks run over.
        chunks: Chunk count.
        threads: Tool thread count.

    Returns:
        The wired smoke tester.
    """
    return LlamaCppSmokeTester(
        perplexity_bin=llama_cpp / "build" / "bin" / "llama-perplexity",
        model_path=out,
        text_path=smoke_text,
        chunks=chunks,
        threads=threads,
    )


def _check_inputs(
    llama_cpp: Path,
    out: Path,
    imatrix: Path | None,
    smoke_text: Path | None,
    smoke_threshold: float,
) -> None:
    """Reject unusable inputs before any tool runs.

    Args:
        llama_cpp: llama.cpp checkout with built tools.
        out: Packed model destination.
        imatrix: Importance matrix file, or None (ADR-0016).
        smoke_text: Smoke-test text file, or None (ADR-0017).
        smoke_threshold: Perplexity ceiling for the smoke test.

    Raises:
        typer.BadParameter: If the checkout misses a needed tool, a
            given file does not exist, the threshold is not positive,
            or the ``--out`` directory does not exist.
    """
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not convert_script.is_file() or not quantize_bin.is_file():
        raise typer.BadParameter(
            f"--llama-cpp: {llama_cpp} misses convert_hf_to_gguf.py or "
            "build/bin/llama-quantize — build the tools first"
        )
    if imatrix is not None and not imatrix.is_file():
        raise typer.BadParameter(f"--imatrix: {imatrix} is not a file")
    if smoke_text is not None:
        perplexity_bin = llama_cpp / "build" / "bin" / "llama-perplexity"
        if not smoke_text.is_file():
            raise typer.BadParameter(f"--smoke-text: {smoke_text} is not a file")
        if not perplexity_bin.is_file():
            raise typer.BadParameter(
                f"--smoke-text: {llama_cpp} misses build/bin/llama-perplexity "
                "— build the tools first"
            )
    if smoke_threshold <= 0:
        raise typer.BadParameter("--smoke-threshold: must be positive")
    if not out.parent.is_dir():
        raise typer.BadParameter(f"--out: directory {out.parent} does not exist")


def _run_smoke(
    run_log: SafeRunLog,
    llama_cpp: Path,
    out: Path,
    smoke_text: Path,
    smoke_chunks: int,
    smoke_threshold: float,
    threads: int,
) -> None:
    """Prove the packed model emits language, or halt (ADR-0017).

    Args:
        run_log: Sink for the ``smoke_tested`` event.
        llama_cpp: llama.cpp checkout with built tools.
        out: The packed model to prove.
        smoke_text: Text the smoke chunks run over.
        smoke_chunks: Chunk count.
        smoke_threshold: Perplexity ceiling.
        threads: Tool thread count.

    Raises:
        typer.Exit: With code 1 when the tool fails or the measured
            perplexity misses the ceiling (the file is kept).
    """
    tester = _build_smoke_tester(llama_cpp, out, smoke_text, smoke_chunks, threads)
    try:
        perplexity = tester.smoke()
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "smoke", exc) from exc
    passed = smoke_passed(perplexity, smoke_threshold)
    run_log.emit(
        "smoke_tested",
        {
            # The run-log sink rejects NaN and infinity (ADR-0011),
            # and a destroyed artifact can measure exactly that.
            "perplexity": perplexity if math.isfinite(perplexity) else None,
            "threshold": smoke_threshold,
            "chunks": smoke_chunks,
            "passed": passed,
        },
    )
    typer.echo(
        f"smoke test: perplexity {perplexity:.4f} over {smoke_chunks} "
        f"chunks, ceiling {smoke_threshold:g} — "
        f"{'passed' if passed else 'FAILED'}"
    )
    if not passed:
        error = RuntimeError(
            f"packed model fails the smoke test (perplexity {perplexity:.4f} "
            f"against ceiling {smoke_threshold:g}) — the file is kept at {out}"
        )
        raise _halt(run_log, "smoke", error)


def _halt(run_log: SafeRunLog, stage: str, exc: Exception) -> typer.Exit:
    """Report one failed stage on both channels.

    Args:
        run_log: Sink for the ``pack_halted`` event.
        stage: The stage that failed.
        exc: The failure.

    Returns:
        The exit to raise, code 1.
    """
    typer.echo(f"error: {exc}", err=True)
    run_log.emit("pack_halted", {"stage": stage, "error": str(exc)})
    return typer.Exit(code=1)


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


def pack(
    recipe_path: Annotated[
        Path, typer.Argument(metavar="RECIPE", help="Recipe produced by quantfit plan.")
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
    threads: Annotated[int, typer.Option(min=1, help="Quantizer thread count.")] = 8,
    imatrix: Annotated[
        Path | None,
        typer.Option(
            help="Importance matrix for the quantizer (ADR-0016). "
            "Generate with llama-imatrix against the base GGUF."
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
    matrix (ADR-0016). The command re-checks the packed file's real
    bytes against the recipe's weight budget — nominal-bit
    predictions undershoot GGUF's effective bits. With
    ``--smoke-text`` it then proves the packed model emits language:
    a few perplexity chunks under the ``--smoke-threshold`` ceiling
    (ADR-0017). A run-log write failure warns once, naming the file,
    and disables the log (ADR-0011).

    Raises:
        typer.BadParameter: If the llama.cpp checkout misses a needed
            tool, ``--imatrix`` or ``--smoke-text`` is not a file,
            ``--smoke-threshold`` is not positive, or the ``--out``
            or ``--runlog`` directory does not exist.
        typer.Exit: With code 1 when the recipe is invalid, the model
            directory does not exist, a toolchain stage fails, the
            packed model exceeds the weight budget, or the smoke test
            fails (the file is kept).

    Examples:
        Command line usage:

        ```console
        $ quantfit pack recipe.json --llama-cpp ~/llama.cpp
        ```
    """
    _check_inputs(llama_cpp, out, imatrix, smoke_text, smoke_threshold)

    recipe = _load_recipe(recipe_path)
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

    started = time.monotonic()
    try:
        result = packer.pack(recipe)
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "quantize", exc) from exc
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
        },
    )

    margin = weight_budget_margin(recipe, result.packed_bytes)
    fits = margin >= 0
    run_log.emit(
        "size_checked",
        {
            "packed_bytes": result.packed_bytes,
            "weight_budget_bytes": recipe.plan.weight_budget_bytes,
            "margin_bytes": margin,
            "fits": fits,
        },
    )
    typer.echo(
        f"packed {len(recipe.assignments)} groups -> {out} "
        f"({format_size(result.packed_bytes)}), weight budget "
        f"{format_size(recipe.plan.weight_budget_bytes)}, margin "
        f"{format_size(abs(margin))} {'under' if fits else 'OVER'}"
    )
    if not fits:
        error = RuntimeError(
            f"packed model exceeds the weight budget by {format_size(-margin)} "
            f"— the file is kept at {out}"
        )
        raise _halt(run_log, "size_check", error)

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
        "pack_finished", {"out": str(out), "packed_bytes": result.packed_bytes}
    )
