"""Smoke-stage helpers for the ``vramfit pack`` command.

The pack composition root splits here: input pre-flight, the halt
path shared by every stage, the type-fallback halt with its
rewrite payload (ADR-0028), and the smoke stage itself (ADR-0017).
`_run_smoke` wires the `SmokeTester` port to the llama.cpp adapter,
emits the ``smoke_tested`` event — with the non-finite guard the
run-log sink demands (ADR-0011) — and halts with the file kept when
the measured perplexity misses the ceiling.

Examples:
    The pack command drives the smoke stage like this:

    ```python
    _check_inputs(llama_cpp, out, imatrix, smoke_text, smoke_threshold)
    _run_smoke(run_log, llama_cpp, out, smoke_text, chunks, ceiling, threads)
    ```

See Also:
    - [vramfit.adapters.inbound.cli_pack][]: The command that
      drives these helpers.
    - [vramfit.adapters.outbound.gguf.smoke][]: The adapter
      `_build_smoke_tester` wires.
"""

from __future__ import annotations

import math
from pathlib import Path

import typer

from vramfit.adapters.inbound.run_log import SafeRunLog
from vramfit.adapters.outbound.gguf.pack import TypeFallbackError
from vramfit.adapters.outbound.gguf.smoke import LlamaCppSmokeTester
from vramfit.domain.pack import smoke_passed
from vramfit.ports.outbound import SmokeTester


def _halt_type_fallback(run_log: SafeRunLog, exc: TypeFallbackError) -> typer.Exit:
    """Report the type-fallback halt with its rewrites (ADR-0028).

    A rewritten type breaks the recipe the artifact claims to carry,
    so the pack's fallback scan halts where its imatrix-miss scan
    records and continues. The event carries every rewritten tensor
    and the substituted types (ADR-0028 decision 3).

    Args:
        run_log: Sink for the ``pack_halted`` event.
        exc: The failure, carrying the parsed rewrites.

    Returns:
        The exit to raise, code 1. The packed file is kept.
    """
    typer.echo(f"error: {exc}", err=True)
    run_log.emit(
        "pack_halted",
        {
            "stage": "type_fallback",
            "error": str(exc),
            "rewritten": [
                {
                    "tensor": tensor,
                    "requested_type": requested,
                    "substituted_type": substituted,
                }
                for tensor, requested, substituted in exc.rewritten
            ],
        },
    )
    return typer.Exit(code=1)


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


def _check_inputs(
    llama_cpp: Path,
    out: Path,
    imatrix: Path | None,
    smoke_text: Path | None,
    smoke_threshold: float,
    mmproj: Path | None = None,
) -> None:
    """Reject unusable inputs before any tool runs.

    Args:
        llama_cpp: llama.cpp checkout with built tools.
        out: Packed model destination.
        imatrix: Importance matrix file, or None (ADR-0016).
        smoke_text: Smoke-test text file, or None (ADR-0017).
        smoke_threshold: Perplexity ceiling for the smoke test.
        mmproj: Vendor mmproj to ship as the projector sidecar, or
            None (ADR-0030).

    Raises:
        typer.BadParameter: If the checkout misses a needed tool, a
            given file does not exist, the threshold is not positive,
            the ``--out`` directory does not exist, or ``--mmproj``
            carries the ``--out`` file name.
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
    if mmproj is not None:
        if not mmproj.is_file():
            raise typer.BadParameter(f"--mmproj: {mmproj} is not a file")
        if mmproj.name == out.name:
            # Fail here, in milliseconds — not after the quantize
            # stage, when the sidecar copy would overwrite the
            # decoder GGUF it ships beside.
            raise typer.BadParameter(
                f"--mmproj: {mmproj.name} carries the --out file name — "
                "the sidecar copy would overwrite the decoder GGUF"
            )
    if smoke_text is not None:
        perplexity_bin = llama_cpp / "build" / "bin" / "llama-perplexity"
        if not smoke_text.is_file():
            raise typer.BadParameter(f"--smoke-text: {smoke_text} is not a file")
        if not perplexity_bin.is_file():
            raise typer.BadParameter(
                f"--smoke-text: {llama_cpp} misses build/bin/llama-perplexity "
                "— build the tools first"
            )
    if smoke_threshold <= 0 or not math.isfinite(smoke_threshold):
        raise typer.BadParameter("--smoke-threshold: must be positive and finite")
    if not out.parent.is_dir():
        raise typer.BadParameter(f"--out: directory {out.parent} does not exist")


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
            # and a destroyed artifact can measure exactly that. The
            # text field keeps the real value on record.
            "perplexity": perplexity if math.isfinite(perplexity) else None,
            "perplexity_text": str(perplexity),
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
