"""The ``vramfit validate`` command: the whole-recipe validation pass.

The composition root for the validation pass (ADR-0006). It loads the
recipe, builds the same torch-backed meter the scan uses (including
the scan's within-group method and imatrix selection, ADR-0018 and
ADR-0020 — a recipe priced on a ``q0-ref`` map validates in that
frame too), perturbs
every group to its assigned precision in one pass, and compares the
measured whole-recipe damage against the recipe's summed marginal
damages — the direct test of the additivity assumption behind
marginal scanning. The frame resolves from the recipe's recorded
method token, a contradicting flag is refused (ADR-0019), and an
``--imatrix`` that differs from the recipe's recorded file draws a
warning — a different file contaminates the comparison (ADR-0020).
The comparison logic is pure and lives in
[vramfit.domain.validation][]. Every failure halts with a clean
``error:`` line. Failures after the run log opens also emit a
``validation_halted`` event (ADR-0011).

Examples:
    Validate a recipe against its model:

    ```console
    $ vramfit validate recipe.json --calibration calib.txt
    ```

See Also:
    - [vramfit.adapters.outbound.scan.meter][]: The meter this
      command builds.
    - [vramfit.domain.validation][]: The pure comparison.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Literal

import typer

from vramfit.adapters.inbound.cli_options import (
    check_imatrix,
    echo_imatrix_coverage,
    parse_gpu_memory,
)
from vramfit.adapters.inbound.cli_scan import _build_meter
from vramfit.adapters.inbound.run_log import SafeRunLog, rss_hwm_gb
from vramfit.adapters.inbound.scan_events import start_run
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.recipe_json import load_recipe
from vramfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile
from vramfit.domain.model import Recipe
from vramfit.domain.scan import (
    KQUANT_IMX_METHOD,
    KQUANT_METHOD,
    KQUANT_PRECISIONS,
    Q0_REF_METHOD,
    Q0_REF_PRECISIONS,
    SCAN_METHOD,
)
from vramfit.domain.validation import validation_result
from vramfit.ports.outbound import DamageMeter

# Method token -> the meter method that measures it. The assisted
# token measures through kquant with the imatrix (ADR-0020).
_TOKEN_TO_METHOD: dict[str, Literal["rtn", "kquant", "q0"]] = {
    SCAN_METHOD: "rtn",
    KQUANT_METHOD: "kquant",
    KQUANT_IMX_METHOD: "kquant",
    Q0_REF_METHOD: "q0",
}
# Each method's ported precision coverage (ADR-0018). RTN covers
# every precision, so it is absent.
_METHOD_COVERAGE: dict[str, tuple[int, ...]] = {
    "kquant": KQUANT_PRECISIONS,
    "q0": Q0_REF_PRECISIONS,
}


def _resolve_within_group(
    text: str | None, imatrix: Path | None, recipe: Recipe
) -> tuple[Literal["rtn", "kquant", "q0"], str]:
    """Resolve the pass's method against the recipe's provenance.

    The pass only checks additivity when its frame matches the map
    that priced the recipe (ADR-0019). A recipe that records its
    map's method is the source of truth: the command refuses a
    mismatched flag instead of measuring numbers that compare
    nothing, and the record conflict refuses before any flag-level
    pairing rule — its message names the real cause. A recipe
    without the record leaves the pairing to the caller, with a
    warning.

    Args:
        text: The ``--within-group`` value, or None to follow the
            recipe's recorded method (falling back to ``rtn``).
        imatrix: The ``--imatrix`` path, or None for unassisted.
        recipe: The loaded recipe whose assignments the pass measures.

    Returns:
        The validated method name and the token the pass measures
        under.

    Raises:
        typer.BadParameter: If the method is unknown, ``kquant`` or
            ``q0`` meets assignments outside its port coverage
            (ADR-0018), ``--imatrix`` arrives without the kquant
            method or is not a file, the recipe records a token this
            version does not know, or the resolved frame contradicts
            the recipe's recorded method (ADR-0019).
    """
    recorded = recipe.within_group
    if recorded is not None and recorded not in _TOKEN_TO_METHOD:
        raise typer.BadParameter(
            f'--within-group: the recipe records method "{recorded}", which '
            "this version does not know — upgrade vramfit"
        )
    if text is None:
        method = _TOKEN_TO_METHOD[recorded] if recorded is not None else "rtn"
    elif text in ("rtn", "kquant", "q0"):
        method = text
    else:
        raise typer.BadParameter(
            f'--within-group: expected "rtn", "kquant", or "q0", got "{text}"'
        )
    # Record conflicts refuse first — their messages name the real
    # cause. A flag-level message here ("--imatrix requires kquant")
    # would send the user into a second failure.
    _check_provenance(recorded, text, method, imatrix)
    check_imatrix(imatrix, method)
    covered = _METHOD_COVERAGE.get(method)
    if covered is not None:
        uncovered = sorted(
            {a.bits for a in recipe.assignments if a.bits not in covered}
        )
        if uncovered:
            raise typer.BadParameter(
                f"--within-group {method} covers precisions "
                f"{sorted(covered, reverse=True)} (ADR-0018) — "
                f"the recipe assigns {uncovered}"
            )
    if method == "rtn":
        token = SCAN_METHOD
    elif method == "q0":
        token = Q0_REF_METHOD
    else:
        token = KQUANT_METHOD if imatrix is None else KQUANT_IMX_METHOD
    return method, token


def _check_provenance(
    recorded: str | None,
    text: str | None,
    method: str,
    imatrix: Path | None,
) -> None:
    """Refuse a measurement frame that contradicts the recipe's record.

    Each refusal names the recorded provenance — the actual
    conflict — never a flag the user did not pass.

    Args:
        recorded: The recipe's recorded method token, or None.
        text: The explicit ``--within-group`` value, or None.
        method: The resolved method name.
        imatrix: The ``--imatrix`` path, or None for unassisted.

    Raises:
        typer.BadParameter: If an explicit method flag contradicts
            the record, ``--imatrix`` meets a recipe priced on an
            unassisted map, or the recipe records an assisted map
            and no imatrix was given (ADR-0019, ADR-0020).
    """
    if recorded is None:
        typer.echo(
            "warning: the recipe does not record its map's method — "
            "match --within-group and --imatrix to the map that priced it",
            err=True,
        )
        return
    if text is not None and _TOKEN_TO_METHOD[recorded] != method:
        raise typer.BadParameter(
            f"--within-group {text}: the recipe was priced on a "
            f'"{recorded}" map — the pass must match the map\'s method '
            "(ADR-0019)"
        )
    if imatrix is not None and recorded != KQUANT_IMX_METHOD:
        raise typer.BadParameter(
            f'--imatrix: the recipe was priced on a "{recorded}" map, not '
            "an assisted one — the pass must match the map's method "
            "(ADR-0019)"
        )
    if recorded == KQUANT_IMX_METHOD and imatrix is None:
        raise typer.BadParameter(
            "--imatrix: the recipe was priced on an assisted map — pass "
            "the map's imatrix file (ADR-0020)"
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
        Path, typer.Argument(metavar="RECIPE", help="Recipe produced by vramfit plan.")
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
        str, typer.Option(help="Grouping granularity: layer, tensor, or stack.")
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
            help="Within-group method: rtn, kquant for the "
            "K-quant-faithful port, or q0 for the block "
            "quantizers Q2_0/Q4_0/Q8_0 (ADR-0018). Default: the "
            "method the recipe records, or rtn for recipes without "
            "the record."
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
    the command refuses flags that contradict the record. An
    ``--imatrix`` that differs from the recipe's recorded file
    draws a warning, and the command echoes the imatrix coverage
    split like the scan does. The
    command then reports the
    measured damage next to the recipe's summed marginal damages. The gap is the
    additivity assumption leaking. Use the scan's calibration file
    and ``--max-tokens`` — damage values are only comparable within
    one calibration set. The command reports the numbers and does not
    gate on them: the invalidation threshold is an open question in
    ADR-0006 until measured gaps exist. With offloaded groups the
    pass restores originals from the model's safetensors shards, so
    the model must be a local safetensors directory (ADR-0015).
    Pass the ``--group-by`` the scan used. A recipe priced on a
    ``stack``-keyed map names groups the other granularities never
    produce, so a mismatch surfaces as an unknown group.

    Raises:
        typer.BadParameter: If ``--group-by``, ``--within-group``, or
            ``--gpu-memory`` is malformed — ``--within-group`` takes
            ``rtn``, ``kquant``, or ``q0`` — ``--gpu-memory`` is given
            without ``--device auto``, ``--within-group kquant`` or
            ``q0`` meets recipe assignments the port does not
            cover,
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
        $ vramfit validate recipe.json --calibration calib.txt --max-tokens 32768
        ```
    """
    if group_by not in ("layer", "tensor", "stack"):
        raise typer.BadParameter(
            f'--group-by: expected "layer", "tensor", or "stack", got "{group_by}"'
        )
    gpu_memory_bytes = parse_gpu_memory(gpu_memory, device)

    recipe = _load_recipe(recipe_path)
    parsed_within_group, method_token = _resolve_within_group(
        within_group, imatrix, recipe
    )
    if imatrix is not None:
        # Provenance compares by path string — resolve so relative
        # spellings cannot split or mix identities.
        imatrix = imatrix.resolve()
        if recipe.imatrix is not None and imatrix != Path(recipe.imatrix).resolve():
            typer.echo(
                f'warning: --imatrix "{imatrix}" differs from the map\'s '
                f'recorded imatrix "{recipe.imatrix}" — a different file '
                "contaminates the additivity comparison (ADR-0020)",
                err=True,
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
            # Null means the frame was the caller's guess, not a
            # provenance-verified one — the gap reads differently.
            "recipe_within_group": recipe.within_group,
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
    echo_imatrix_coverage(meter)
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
