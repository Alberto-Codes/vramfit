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

Every refusal leaves one ``error:`` line and exit 1. The domain
error root covers them: a recipe whose pins or protections do not
resolve against this map, a measurement too short to pair, and a
toolchain failure all halt the same way. Every path the pass needs
is checked before the convert stage: the three llama.cpp tools, the
importance matrix, and the destinations the sidecar and the run log
are written to. A missing path costs no card time, and a finished
pass is never discarded at its last step.

`LlamaCppTools` names the three tools once. The pre-flight checks
those paths and the wiring runs them, so the two cannot disagree
about which file they mean.

The command reports what it measured and never a verdict on the
recipe. The arms are a sample of the neighbourhood whenever the arm
budget was smaller, and the sidecar records both counts.

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

import os
import sys
from dataclasses import dataclass
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
from vramfit.domain.errors import VramfitError
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


@dataclass(frozen=True, slots=True)
class LlamaCppTools:
    """The three tools one pass runs, named once.

    The pre-flight checks these paths and the wiring runs them, so
    neither can mean a different file from the other. A second
    spelling would let the check pass while the run dies after the
    convert, which is the cost this record exists to prevent.

    Attributes:
        convert_script (Path): ``convert_hf_to_gguf.py``, which
            writes the f16 base.
        quantize_bin (Path): ``llama-quantize``, which packs each
            arm.
        perplexity_bin (Path): ``llama-perplexity``, which measures
            each arm's divergence.

    Examples:
        Resolve a checkout's tools:

        ```python
        tools = LlamaCppTools.under(Path("~/llama.cpp"))
        ```
    """

    convert_script: Path
    quantize_bin: Path
    perplexity_bin: Path

    @classmethod
    def under(cls, llama_cpp: Path) -> LlamaCppTools:
        """Name the tools a checkout carries.

        Args:
            llama_cpp: The llama.cpp checkout.

        Returns:
            The three paths, built once for both the pre-flight and
            the wiring.
        """
        built = llama_cpp / "build" / "bin"
        return cls(
            convert_script=llama_cpp / "convert_hf_to_gguf.py",
            quantize_bin=built / "llama-quantize",
            perplexity_bin=built / "llama-perplexity",
        )


def _check_toolchain(tools: LlamaCppTools) -> None:
    """Refuse a checkout missing a tool the pass runs.

    The pass drives three tools and reaches the last one only after
    the convert stage and the control pack. On the rented card this
    stage is priced against, that is about 20 minutes and a
    full-size f16 base GGUF spent to learn a path does not exist, so
    every tool is checked before any of them runs.

    Args:
        tools: The checkout's tools, as the wiring will run them.

    Raises:
        Exit: With code 1 when a tool is missing or cannot execute.
    """
    if not tools.convert_script.is_file():
        _halt(
            f"--llama-cpp: {tools.convert_script} does not exist — "
            "build the tools first"
        )
    for binary in (tools.quantize_bin, tools.perplexity_bin):
        if not binary.is_file():
            _halt(f"--llama-cpp: {binary} does not exist — build the tools first")
        if not os.access(binary, os.X_OK):
            _halt(f"--llama-cpp: {binary} is not executable")


def _check_destination(label: str, path: Path) -> None:
    """Refuse a destination the pass could not write when it finishes.

    The sidecar is the pass's only artifact and it is written last,
    after every pack and every measurement, so every reason the write
    could fail is checked before the first tool runs. The write
    replaces a temporary file onto ``path``, which refuses a path
    that names a directory as surely as one whose parent is missing.
    The command offers both ``--out-dir`` and ``--out``, so an
    ``--out`` naming an existing directory is the reachable mistake.

    Args:
        label: The option that named the path, for the message.
        path: The file the pass will write.

    Raises:
        Exit: With code 1 when the path names a directory, or its
            parent is missing or refuses a write.
    """
    if path.is_dir():
        _halt(f"{label}: {path} is a directory, so the pass cannot write it")
    parent = path.parent
    if not parent.is_dir():
        _halt(f"{label}: directory {parent} does not exist")
    if not os.access(parent, os.W_OK):
        _halt(f"{label}: directory {parent} is not writable")


def _sample_phrase(sidecar: RefinementSidecar) -> str:
    """Word how much of the neighbourhood the pass measured.

    Args:
        sidecar: The pass's search record.

    Returns:
        The arm count, naming the neighbourhood it was drawn from
        when the pass measured only part of it.
    """
    measured = len(sidecar.arms)
    if measured == sidecar.neighbourhood_moves:
        return f"{measured} evaluated"
    return f"{measured} evaluated of a neighbourhood of {sidecar.neighbourhood_moves}"


def _report_outcome(sidecar: RefinementSidecar) -> None:
    """Print what the pass measured.

    Every line states a measurement. None states a verdict on the
    recipe: the arms are a sample of the neighbourhood whenever the
    caller's budget was smaller, and no sample supports a conclusion
    about the arms it never measured.

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
        typer.echo(
            f"no arm among the {_sample_phrase(sidecar)} cleared {sidecar.bar} sigma"
        )
        return
    won = sidecar.winning_arm()
    if won is not None:
        typer.echo(
            f"winner: {sidecar.winner} at {won.mean:.6f}, "
            f"{won.sigma:+.1f} sigma against the control, "
            f"among the {_sample_phrase(sidecar)}"
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

    The measured row widths reach the pass, which reprices every
    candidate group through the predictor the plan step used. The
    frame and the sidecar write sit inside the guarded region, so an
    empty ``--runtime-build`` and a refused write each leave one
    ``error:`` line rather than a traceback. Every path the pass
    needs is checked before the first tool runs: the tools, the
    matrix, both destinations, and the arm directory.

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
        Exit: With code 1 when an input refuses, a measurement
            cannot be paired, or the toolchain fails.
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
    inputs = (("--base-logits", base_logits), ("--eval-text", eval_text))
    for label, path in inputs if imatrix is None else (*inputs, ("--imatrix", imatrix)):
        if not path.is_file():
            _halt(f"{label}: {path} does not exist")
    tools = LlamaCppTools.under(llama_cpp)
    _check_toolchain(tools)
    sidecar_path = (
        out if out is not None else recipe_path.with_suffix(".refinement.json")
    )
    runlog_path = (
        runlog
        if runlog is not None
        else sidecar_path.with_name(sidecar_path.stem + ".runlog.jsonl")
    )
    _check_destination("--out", sidecar_path)
    _check_destination("--runlog", runlog_path)
    try:
        corpus_sha256, corpus_bytes = content_identity(eval_text)
    except OSError as error:
        _halt(f"--eval-text: {error}")
        return
    if corpus_bytes == 0:
        _halt(f"--eval-text: {eval_text} holds no bytes")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _halt(f"--out-dir: {error}")
        return
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

        Reads the tools the pre-flight checked, so the packer runs
        the files that were verified to exist.

        Args:
            packed: Where this arm's packed file goes.

        Returns:
            The wired packer.
        """
        return LlamaCppPacker(
            model_dir=model_dir,
            base_gguf=base_path,
            out_path=Path(packed),
            convert_script=tools.convert_script,
            quantize_bin=tools.quantize_bin,
            python_bin=python_bin if python_bin is not None else Path(sys.executable),
            threads=threads,
            imatrix=imatrix,
            row_widths=row_widths,
        )

    meter = LlamaCppDivergenceMeter(
        perplexity_bin=tools.perplexity_bin,
        text_path=eval_text,
        base_logits=base_logits,
        threads=threads,
    )
    sink: RefinementSidecarSink = JsonRefinementSidecarFile(sidecar_path)
    try:
        # The frame names the evaluation corpus by content, never by
        # path alone (ADR-0031 decision 5). This process reads and
        # hashes those exact bytes as it runs, which "measured" marks.
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
        sidecar = run_pass(
            recipe,
            map_,
            packer_for,
            meter,
            frame,
            bar=bar,
            limit=limit,
            out_dir=out_dir,
            row_widths=row_widths,
            keep_packs=keep_packs,
            report=run_log.emit,
        )
        sink.save(sidecar)
    except (VramfitError, OSError) as error:
        _halt(str(error))
        return
    _report_outcome(sidecar)
    typer.echo(f"wrote {sidecar_path}")
