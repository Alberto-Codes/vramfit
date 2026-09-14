"""The ``vramfit refine`` command: search a recipe's neighbourhood.

The composition root for the refinement pass (ADR-0031). It loads the
recipe and the map that priced it, measures the checkpoint's row
widths the way ``pack`` does, wires the `RecipePacker` port to the
llama.cpp adapter once per arm and the `RuntimeDivergenceMeter` port
to ``llama-perplexity``, runs the pass
([vramfit.adapters.inbound.refine_loop][]), and writes the search
record beside the recipe.

Every arm is packed and measured. Nothing is ranked by the map,
because the map does not order the neighbourhood it prices
(ADR-0031). A recipe with no legal swap declines before the first
pack, so a target the protocol cannot reach costs nothing.

The caller states the evidence bar. The command carries no default
for it (ADR-0031 decision 7).

The command writes a sidecar and never a refined recipe. Promoting a
winning arm to an artifact needs a tier-3 slice and a serve test, and
those are not this command's business.

Examples:
    Search the published recipe's neighbourhood:

    ```console
    $ vramfit refine recipe.json --map map.json --bar 7.8 ...
    ```

See Also:
    - [vramfit.adapters.inbound.refine_loop][]: The measure loop.
    - [vramfit.domain.refinement][]: The neighbour generator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from vramfit.adapters.inbound.cli_pack_check import _resolve_row_widths
from vramfit.adapters.inbound.refine_loop import run_pass
from vramfit.adapters.inbound.run_log import SafeRunLog
from vramfit.adapters.outbound.calibration_digest import content_identity
from vramfit.adapters.outbound.gguf.divergence import LlamaCppDivergenceMeter
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.recipe_json import load_recipe
from vramfit.adapters.outbound.refinement_sidecar_json import (
    JsonRefinementSidecarFile,
)
from vramfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile
from vramfit.adapters.outbound.sensitivity_map_json import load_sensitivity_map
from vramfit.domain.evals import CorpusReference
from vramfit.domain.refinement_record import MeasurementFrame, RefinementSidecar
from vramfit.ports.outbound import RefinementSidecarSink


def _halt(message: str) -> None:
    """Report a failure and exit 1.

    Args:
        message: The operator-facing line.

    Raises:
        Exit: Always, with code 1.
    """
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _report_outcome(sidecar: RefinementSidecar) -> None:
    """Print what the pass found.

    Args:
        sidecar: The pass's search record.
    """
    if sidecar.declined is not None:
        typer.echo(f"declined: {sidecar.declined}")
        return
    control = sidecar.control
    if control is None:
        return
    typer.echo(
        f"control: {control.mean:.6f} mean divergence over {control.chunks} chunks"
    )
    for arm in sorted(sidecar.arms, key=lambda a: a.mean):
        typer.echo(
            f"  {arm.arm}: {arm.mean:.6f} "
            f"({arm.delta:+.6f}, {arm.sigma:+.1f} sigma, "
            f"{arm.better_chunks}/{arm.chunks} better)"
        )
    if sidecar.winner is None:
        typer.echo(f"no arm cleared {sidecar.bar} sigma — the recipe stands")
        return
    won = sidecar.winning_arm()
    if won is not None:
        typer.echo(
            f"winner: {sidecar.winner} at {won.mean:.6f}, "
            f"{won.sigma:+.1f} sigma against the control"
        )


def refine(
    recipe_path: Annotated[
        Path, typer.Argument(metavar="RECIPE", help="Recipe produced by vramfit plan.")
    ],
    map_path: Annotated[
        Path, typer.Option("--map", help="Sensitivity map that priced the recipe.")
    ],
    llama_cpp: Annotated[
        Path,
        typer.Option(
            help="llama.cpp checkout with convert_hf_to_gguf.py and built tools."
        ),
    ],
    base_logits: Annotated[
        Path,
        typer.Option(
            help="Logits stored from the reference build, which every arm "
            "measures against. Write them with llama-perplexity "
            "--kl-divergence-base over the f16 base GGUF."
        ),
    ],
    eval_text: Annotated[
        Path, typer.Option(help="Evaluation text the chunks run over.")
    ],
    runtime_build: Annotated[
        str,
        typer.Option(help="Runtime binary build identity recorded in the sidecar."),
    ],
    hardware: Annotated[
        str, typer.Option(help="Card the pass runs on, recorded in the sidecar.")
    ],
    bar: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Evidence bar in sigma an arm must clear to win. The "
            "project's artifact precedent is 7.8. The command has no "
            "default (ADR-0031 decision 7).",
        ),
    ],
    model: Annotated[
        Path | None,
        typer.Option(
            help="Model checkpoint directory. Default: the recipe's model_id."
        ),
    ] = None,
    base_gguf: Annotated[
        Path | None,
        typer.Option(help="f16 base GGUF path, reused when present."),
    ] = None,
    imatrix: Annotated[
        Path | None,
        typer.Option(help="Importance matrix for the quantizer (ADR-0016)."),
    ] = None,
    out_dir: Annotated[
        Path, typer.Option(help="Directory the arm packs are written in.")
    ] = Path("arms"),
    out: Annotated[
        Path | None,
        typer.Option(help="Sidecar path. Default: beside the recipe."),
    ] = None,
    limit: Annotated[
        int, typer.Option(min=1, help="Most arms to pack and measure.")
    ] = 15,
    threads: Annotated[int, typer.Option(min=1, help="Tool thread count.")] = 8,
    keep_packs: Annotated[
        bool,
        typer.Option(help="Keep each arm's packed file instead of deleting it."),
    ] = False,
    python_bin: Annotated[
        Path | None,
        typer.Option(help="Interpreter for the convert script. Default: this one."),
    ] = None,
    runlog: Annotated[
        Path | None,
        typer.Option(help="Run-log path. Default: beside the sidecar."),
    ] = None,
) -> None:
    """Search a recipe's equal-byte neighbourhood in the runtime frame.

    Packs and measures every arm it selects, because the sensitivity
    map does not order the neighbourhood it prices (ADR-0031). Writes
    one sidecar recording every arm, the winner, the stated bar, and
    the frame.

    Args:
        recipe_path: The recipe to refine.
        map_path: The map that priced it.
        llama_cpp: llama.cpp checkout with built tools.
        base_logits: Reference logits every arm measures against.
        eval_text: Evaluation text.
        runtime_build: Runtime build identity for the record.
        hardware: Card identity for the record.
        bar: Evidence bar in sigma, stated by the caller.
        model: Checkpoint directory, or None to use the model_id.
        base_gguf: f16 base GGUF path, or None to place it beside the
            arms.
        imatrix: Importance matrix, or None.
        out_dir: Directory the arm packs go in.
        out: Sidecar path, or None to place it beside the recipe.
        limit: Most arms to measure.
        threads: Tool thread count.
        keep_packs: Keep each packed arm.
        python_bin: Convert-script interpreter, or None for this one.
        runlog: Run-log path, or None to place it beside the sidecar.

    Raises:
        Exit: With code 1 when an input refuses or the toolchain
            fails.
        OSError: If the evaluation text cannot be read.
    """
    try:
        recipe = load_recipe(recipe_path)
        map_ = load_sensitivity_map(map_path)
    except ArtifactError as error:
        _halt(str(error))
        return
    model_dir = model if model is not None else Path(recipe.model_id)
    if not model_dir.is_dir():
        _halt(
            f'model directory "{model_dir}" does not exist — the recipe\'s '
            "model_id is not a local path, pass --model"
        )
    for label, path in (("--base-logits", base_logits), ("--eval-text", eval_text)):
        if not path.is_file():
            _halt(f"{label}: {path} does not exist")
    corpus_sha256, corpus_bytes = content_identity(eval_text)
    if corpus_bytes == 0:
        _halt(f"--eval-text: {eval_text} holds no bytes")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = (
        out if out is not None else recipe_path.with_suffix(".refinement.json")
    )
    runlog_path = (
        runlog
        if runlog is not None
        else sidecar_path.with_name(sidecar_path.stem + ".runlog.jsonl")
    )
    run_log = SafeRunLog(JsonlRunLogFile(runlog_path), path=runlog_path)
    try:
        row_widths = _resolve_row_widths(recipe, model_dir)
    except PackError as error:
        _halt(str(error))
        return
    base_path = (
        base_gguf if base_gguf is not None else out_dir / f"{model_dir.name}-f16.gguf"
    )

    def packer_for(packed: str) -> LlamaCppPacker:
        """Wire the llama.cpp adapter for one arm.

        Args:
            packed: Where this arm's packed file goes.

        Returns:
            The wired packer.
        """
        return LlamaCppPacker(
            model_dir=model_dir,
            base_gguf=base_path,
            out_path=Path(packed),
            convert_script=llama_cpp / "convert_hf_to_gguf.py",
            quantize_bin=llama_cpp / "build" / "bin" / "llama-quantize",
            python_bin=python_bin if python_bin is not None else Path(sys.executable),
            threads=threads,
            imatrix=imatrix,
            row_widths=row_widths,
        )

    # The frame names the evaluation corpus by content, never by path
    # alone (ADR-0031 decision 5). This process reads and hashes those
    # exact bytes as it runs, which is what "measured" marks.
    frame = MeasurementFrame(
        runtime_build=runtime_build,
        hardware=hardware,
        corpus=CorpusReference(
            file=str(eval_text),
            sha256=corpus_sha256,
            size_bytes=corpus_bytes,
            provenance="measured",
        ),
        reference=str(base_logits),
    )
    meter = LlamaCppDivergenceMeter(
        perplexity_bin=llama_cpp / "build" / "bin" / "llama-perplexity",
        text_path=eval_text,
        base_logits=base_logits,
        threads=threads,
    )
    try:
        sidecar = run_pass(
            recipe,
            map_,
            packer_for,
            meter,
            frame,
            bar=bar,
            limit=limit,
            out_dir=out_dir,
            keep_packs=keep_packs,
            report=run_log.emit,
        )
    except PackError as error:
        _halt(str(error))
        return
    sink: RefinementSidecarSink = JsonRefinementSidecarFile(sidecar_path)
    sink.save(sidecar)
    _report_outcome(sidecar)
    typer.echo(f"wrote {sidecar_path}")
