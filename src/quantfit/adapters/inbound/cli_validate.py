"""The ``quantfit validate`` command: the whole-recipe validation pass.

The composition root for the validation pass (ADR-0006). It loads the
recipe, builds the same torch-backed meter the scan uses (including
the scan's within-group method and imatrix selection, ADR-0018 and
ADR-0020), perturbs
every group to its assigned precision in one pass, and compares the
measured whole-recipe damage against the recipe's summed marginal
damages — the direct test of the additivity assumption behind
marginal scanning. The frame resolves from the recipe's recorded
method token, and a contradicting flag is refused (ADR-0019). The
comparison logic is pure and lives in
[quantfit.domain.validation][]. Every failure halts with a clean
``error:`` line. Failures after the run log opens also emit a
``validation_halted`` event (ADR-0011).

Examples:
    Validate a recipe against its model:

    ```console
    $ quantfit validate recipe.json --calibration calib.txt
    ```

See Also:
    - [quantfit.adapters.outbound.scan.meter][]: The meter this
      command builds.
    - [quantfit.domain.validation][]: The pure comparison.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Literal

import typer

from quantfit.adapters.inbound.cli_options import check_imatrix, parse_gpu_memory
from quantfit.adapters.inbound.cli_scan import _build_meter
from quantfit.adapters.inbound.run_log import SafeRunLog, rss_hwm_gb
from quantfit.adapters.inbound.scan_events import start_run
from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.recipe_json import load_recipe
from quantfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile
from quantfit.domain.model import Recipe
from quantfit.domain.scan import (
    KQUANT_IMX_METHOD,
    KQUANT_METHOD,
    KQUANT_PRECISIONS,
    SCAN_METHOD,
)
from quantfit.domain.validation import validation_result
from quantfit.ports.outbound import DamageMeter

# Method token -> the meter method that measures it. The assisted
# token measures through kquant with the imatrix (ADR-0020).
_TOKEN_TO_METHOD: dict[str, Literal["rtn", "kquant"]] = {
    SCAN_METHOD: "rtn",
    KQUANT_METHOD: "kquant",
    KQUANT_IMX_METHOD: "kquant",
}


def _resolve_within_group(
    text: str | None, imatrix: Path | None, recipe: Recipe
) -> tuple[Literal["rtn", "kquant"], str]:
    """Resolve the pass's method against the recipe's provenance.

    The pass only checks additivity when its frame matches the map
    that priced the recipe (ADR-0019). A recipe that records its
    map's method is the source of truth: the command refuses a
    mismatched flag instead of measuring numbers that compare
    nothing. A recipe without the record leaves the pairing to the
    caller, with a warning.

    Args:
        text: The ``--within-group`` value, or None to follow the
            recipe's recorded method (falling back to ``rtn``).
        imatrix: The ``--imatrix`` path, or None for unassisted.
        recipe: The loaded recipe whose assignments the pass measures.

    Returns:
        The validated method name and the token the pass measures
        under.

    Raises:
        typer.BadParameter: If the method is unknown, ``kquant``
            meets assignments outside its port coverage (ADR-0018),
            ``--imatrix`` arrives without the kquant method or is
            not a file, the recipe records a token this version
            does not know, or the resolved frame contradicts the
            recipe's recorded method (ADR-0019).
    """
    recorded = recipe.within_group
    if recorded is not None and recorded not in _TOKEN_TO_METHOD:
        raise typer.BadParameter(
            f'--within-group: the recipe records method "{recorded}", which '
            "this version does not know — upgrade quantfit"
        )
    if text is None:
        method = _TOKEN_TO_METHOD[recorded] if recorded is not None else "rtn"
    elif text in ("rtn", "kquant"):
        method = text
    else:
        raise typer.BadParameter(
            f'--within-group: expected "rtn" or "kquant", got "{text}"'
        )
    check_imatrix(imatrix, method)
    if method == "kquant":
        uncovered = sorted(
            {a.bits for a in recipe.assignments if a.bits not in KQUANT_PRECISIONS}
        )
        if uncovered:
            raise typer.BadParameter(
                f"--within-group kquant covers precisions "
                f"{sorted(KQUANT_PRECISIONS, reverse=True)} (ADR-0018) — "
                f"the recipe assigns {uncovered}"
            )
    if method == "rtn":
        token = SCAN_METHOD
    else:
        token = KQUANT_METHOD if imatrix is None else KQUANT_IMX_METHOD
    _check_provenance(recorded, token, imatrix)
    return method, token


def _check_provenance(recorded: str | None, token: str, imatrix: Path | None) -> None:
    """Refuse a measurement frame that contradicts the recipe's record.

    Args:
        recorded: The recipe's recorded method token, or None.
        token: The token the resolved frame would measure under.
        imatrix: The ``--imatrix`` path, or None for unassisted.

    Raises:
        typer.BadParameter: If the recipe records an assisted map
            and no imatrix was given, or the resolved token differs
            from the record (ADR-0019).
    """
    if recorded == KQUANT_IMX_METHOD and imatrix is None:
        raise typer.BadParameter(
            "--imatrix: the recipe was priced on an assisted map — pass "
            "the map's imatrix file (ADR-0020)"
        )
    if recorded is not None and token != recorded:
        raise typer.BadParameter(
            f'--within-group: the recipe was priced on a "{recorded}" map '
            f'and this pass would measure "{token}" — the pass must match '
            "the map's method (ADR-0019)"
        )
    if recorded is None:
        typer.echo(
            "warning: the recipe does not record its map's method — "
            "match --within-group and --imatrix to the map that priced it",
            err=True,
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


def _check_groups(meter: DamageMeter, recipe: Recipe, run_log: SafeRunLog) -> None:
    """Refuse a recipe whose groups differ from the model's.

    A mismatch means the recipe was planned for a different model or
    grouping — measuring it would compare unrelated numbers. The halt
    event carries the stage and the mismatch detail, with no cell
    count — the validation pass has no scan grid.

    Args:
        meter: The loaded meter, holding the discovered groups.
        recipe: The recipe under validation.
        run_log: Sink for the ``validation_halted`` event.

    Raises:
        typer.Exit: With code 1 on any mismatch. The message names the
            first differing groups on each side.
    """
    discovered = {spec.name for spec in meter.groups()}
    assigned = {a.group for a in recipe.assignments}
    missing = sorted(discovered - assigned)
    unexpected = sorted(assigned - discovered)
    if not missing and not unexpected:
        return
    parts = []
    if unexpected:
        parts.append(f'the recipe assigns unknown groups (first: "{unexpected[0]}")')
    if missing:
        parts.append(f'the recipe misses discovered groups (first: "{missing[0]}")')
    detail = " and ".join(parts)
    typer.echo(
        f"error: recipe groups do not match the model's groups — {detail}. "
        "Check the model path and --group-by against the scan",
        err=True,
    )
    run_log.emit(
        "validation_halted",
        {"stage": "group_match", "error": detail, "rss_hwm_gb": rss_hwm_gb()},
    )
    raise typer.Exit(code=1)


def validate(
    recipe_path: Annotated[
        Path, typer.Argument(metavar="RECIPE", help="Recipe produced by quantfit plan.")
    ],
    calibration: Annotated[Path, typer.Option(help="Calibration text file (UTF-8).")],
    model: Annotated[
        str | None,
        typer.Option(
            help="Model id or checkpoint path. Default: the recipe's model_id."
        ),
    ] = None,
    max_tokens: Annotated[
        int, typer.Option(min=2, help="Calibration token budget.")
    ] = 131072,
    group_by: Annotated[
        str, typer.Option(help="Grouping granularity: layer or tensor.")
    ] = "layer",
    device: Annotated[
        str, typer.Option(help="Device map: auto, cpu, or cuda.")
    ] = "auto",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow model repos with custom code.")
    ] = False,
    gpu_memory: Annotated[
        str | None,
        typer.Option(
            help="Byte cap on GPU 0 model shards, e.g. 17GiB. Requires "
            "--device auto. Leaves workspace for activations and "
            "quantization."
        ),
    ] = None,
    within_group: Annotated[
        str | None,
        typer.Option(
            help="Within-group method: rtn, or kquant for the "
            "K-quant-faithful port (ADR-0018). Default: the method "
            "the recipe records, or rtn for recipes without the "
            "record."
        ),
    ] = None,
    imatrix: Annotated[
        Path | None,
        typer.Option(
            help="GGUF imatrix for assisted K-quant measurement "
            "(ADR-0020). Required when the recipe was priced on an "
            "assisted map — use the map's imatrix file."
        ),
    ] = None,
    runlog: Annotated[
        Path | None,
        typer.Option(
            help="Run-log path (JSONL). Default: <recipe stem>.validation.runlog.jsonl."
        ),
    ] = None,
) -> None:
    """Measure whole-recipe damage and compare it against the prediction.

    The validation pass (ADR-0006). The command quantizes every group
    to its assigned precision in one pass. The pass uses the scan's
    own quantization, selected with ``--within-group`` and
    ``--imatrix`` — the pass only checks additivity when its frame
    matches the map that priced the recipe (ADR-0019). A recipe
    that records its map's method resolves the frame by itself, and
    the command refuses flags that contradict the record. The
    command then reports the
    measured damage next to the recipe's summed marginal damages. The gap is the
    additivity assumption leaking. Use the scan's calibration file
    and ``--max-tokens`` — damage values are only comparable within
    one calibration set. The command reports the numbers and does not
    gate on them: the invalidation threshold is an open question in
    ADR-0006 until measured gaps exist. With offloaded groups the
    pass restores originals from the model's safetensors shards, so
    the model must be a local safetensors directory (ADR-0015).

    Raises:
        typer.BadParameter: If ``--group-by``, ``--within-group``, or
            ``--gpu-memory`` is malformed, ``--gpu-memory`` is given
            without ``--device auto``, ``--within-group kquant``
            meets recipe assignments the port does not cover,
            ``--imatrix`` arrives without the kquant method or is
            not a file, the resolved frame contradicts the recipe's
            recorded method (ADR-0019), or the ``--runlog``
            directory does not exist.
        typer.Exit: With code 1 when the recipe is invalid, the scan
            extra is missing, the model or calibration cannot load,
            the recipe's groups do not match the model's, or the
            measurement fails.

    Examples:
        Command line usage:

        ```console
        $ quantfit validate recipe.json --calibration calib.txt --max-tokens 32768
        ```
    """
    if group_by not in ("layer", "tensor"):
        raise typer.BadParameter(
            f'--group-by: expected "layer" or "tensor", got "{group_by}"'
        )
    gpu_memory_bytes = parse_gpu_memory(gpu_memory, device)

    recipe = _load_recipe(recipe_path)
    parsed_within_group, method_token = _resolve_within_group(
        within_group, imatrix, recipe
    )
    model_id = model if model is not None else recipe.model_id
    if model is not None and model != recipe.model_id:
        typer.echo(
            f'warning: --model "{model}" differs from the recipe\'s model_id '
            f'"{recipe.model_id}" — the comparison assumes the scanned model',
            err=True,
        )

    runlog_path = (
        runlog
        if runlog is not None
        else recipe_path.with_name(recipe_path.stem + ".validation.runlog.jsonl")
    )
    if not runlog_path.parent.is_dir():
        raise typer.BadParameter(
            f"--runlog: directory {runlog_path.parent} does not exist"
        )
    run_log = SafeRunLog(JsonlRunLogFile(runlog_path), path=runlog_path)

    meter = start_run(
        run_log,
        {
            "recipe": str(recipe_path),
            "model": model_id,
            "groups": len(recipe.assignments),
            "group_by": group_by,
            "max_tokens": max_tokens,
            "device": device,
            "gpu_memory_bytes": gpu_memory_bytes,
            "within_group": method_token,
            "imatrix": None if imatrix is None else str(imatrix),
        },
        lambda: _build_meter(
            model_id,
            calibration,
            max_tokens=max_tokens,
            group_by=group_by,
            device=device,
            trust_remote_code=trust_remote_code,
            gpu_memory=gpu_memory_bytes,
            within_group=parsed_within_group,
            imatrix=imatrix,
        ),
        prefix="validation",
    )
    _check_groups(meter, recipe, run_log)

    assignments = {a.group: a.bits for a in recipe.assignments}
    started = time.monotonic()
    try:
        measured = meter.measure_recipe(assignments)
        result = validation_result(recipe, measured)
    except (RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"error: validation halted: {exc}", err=True)
        run_log.emit(
            "validation_halted",
            {"stage": "measure", "error": str(exc), "rss_hwm_gb": rss_hwm_gb()},
        )
        raise typer.Exit(code=1) from exc
    tokens = meter.calibration_tokens()
    run_log.emit(
        "validation_finished",
        {
            "predicted_damage": result.predicted_damage,
            "measured_damage": result.measured_damage,
            "gap": result.gap,
            "ratio": result.ratio,
            "groups": len(recipe.assignments),
            "calibration_tokens": tokens,
            "seconds": round(time.monotonic() - started, 3),
            "rss_hwm_gb": rss_hwm_gb(),
        },
    )
    typer.echo(f"validated {len(recipe.assignments)} groups over {tokens} tokens")
    typer.echo(f"summed marginal damage (predicted)  {result.predicted_damage:.6f}")
    typer.echo(f"whole-recipe damage (measured)      {result.measured_damage:.6f}")
    if result.ratio is None:
        typer.echo(f"gap {result.gap:+.6f}")
    else:
        percent = (result.ratio - 1.0) * 100.0
        typer.echo(f"gap {result.gap:+.6f} ({percent:+.1f} % of predicted)")
