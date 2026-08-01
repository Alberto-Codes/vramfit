"""The ``quantfit scan`` command: measure damage, checkpoint, emit a map.

The scan loop lives here because the inbound adapter is the
composition root: it validates every option up front (sizes parse
with the project grammar, and ``--within-group kquant`` must pair
with precisions the port covers, ADR-0018), builds the torch-backed
meter (lazily, so the base install never imports torch), drives the
`DamageMeter` and `ScanCheckpointStore` ports cell by cell, and hands
the finished measurements to the pure assembly logic in
[quantfit.domain.scan][].
Every failure — a missing extra, a bad destination, an unstable
measurement, a checkpoint write — halts with a clean ``error:`` line.
The checkpoint keeps every finished cell. The run log records the
halt with its stage (ADR-0011).

Examples:
    Scan a local checkpoint at the ADR-0010 candidate set:

    ```console
    $ quantfit scan ./model --calibration calib.txt --out sensitivity.json
    ```

See Also:
    - [quantfit.adapters.outbound.scan.meter][]: The torch meter this
      command builds.
    - [quantfit.domain.scan][]: Work planning and map assembly.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

from quantfit.adapters.inbound.scan_events import (
    SafeRunLog,
    measure_cells,
    rss_hwm_gb,
    start_run,
)
from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile
from quantfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from quantfit.adapters.outbound.sensitivity_map_json import JsonSensitivityMapFile
from quantfit.domain.budget import parse_size
from quantfit.domain.errors import QuantfitError
from quantfit.domain.model import ScanMeta
from quantfit.domain.scan import (
    KQUANT_METHOD,
    KQUANT_PRECISIONS,
    SCAN_METHOD,
    assemble_map,
    plan_measurements,
    scan_fingerprint,
)
from quantfit.ports.outbound import (
    DamageMeter,
    ScanCheckpointStore,
    SensitivityMapSink,
)

INSTALL_HINT = (
    'the scan extra is not installed — install it with: uv pip install "quantfit[scan]"'
)


class ScanExtraMissingError(QuantfitError, RuntimeError):
    """The scan adapter package itself failed to import.

    Distinguishes "torch is not installed" from every other
    ImportError a model load can raise (missing tokenizer backends,
    broken CUDA builds) — those must surface as themselves. Inherits
    `QuantfitError` per ADR-0011.

    Examples:
        The scan command maps it to the install hint:

        ```python
        raise ScanExtraMissingError(INSTALL_HINT)
        ```
    """


def _build_meter(
    model: str,
    calibration: Path,
    max_tokens: int,
    group_by: Literal["layer", "tensor"],
    device: str,
    trust_remote_code: bool,
    gpu_memory: int | None,
    within_group: Literal["rtn", "kquant"] = "rtn",
) -> DamageMeter:
    """Build the torch-backed meter, importing torch only now.

    A missing module maps to `ScanExtraMissingError`. Every other
    import failure surfaces unchanged. Unit tests monkeypatch this
    seam with the verified fake, keeping the command's orchestration
    testable without a GPU (ADR-0009).

    Args:
        model: Hugging Face model id or local checkpoint path.
        calibration: UTF-8 calibration text file.
        max_tokens: Upper bound on calibration tokens.
        group_by: Grouping granularity.
        device: transformers ``device_map`` value.
        trust_remote_code: Allow repos with custom modeling code.
        gpu_memory: Byte cap on GPU 0 model shards under ``auto``
            sharding.
        within_group: Within-group method (ADR-0018).

    Returns:
        The loaded meter.

    Raises:
        ScanExtraMissingError: If the scan extra is not installed.
        ValueError: If the calibration file yields too few tokens.
        OSError: If the model or calibration file cannot be read.
    """
    try:
        from quantfit.adapters.outbound.scan.meter import (  # noqa: PLC0415 - lazy: keeps the base CLI torch-free (ADR-0005)
            TorchDamageMeter,
        )
    except ModuleNotFoundError as exc:
        # Only a missing module means the extra is absent. Any other
        # ImportError (broken CUDA libs, an adapter bug) surfaces
        # as itself through the caller's generic handler.
        raise ScanExtraMissingError(INSTALL_HINT) from exc

    return TorchDamageMeter(
        model,
        calibration,
        max_tokens=max_tokens,
        group_by=group_by,
        device=device,
        trust_remote_code=trust_remote_code,
        max_gpu_memory=gpu_memory,
        within_group=within_group,
    )


def _open_run_log(out: Path, runlog: Path | None) -> SafeRunLog:
    """Resolve, validate, and wrap the run-log destination.

    Args:
        out: The sensitivity-map path the default derives from.
        runlog: An explicit run-log path, or None for the default.

    Returns:
        The policy-wrapped sink, which names the resolved path in its
        disable warning.

    Raises:
        typer.BadParameter: If the run-log directory does not exist —
            rejected before the model load burns an hour.
    """
    path = runlog if runlog is not None else out.with_name(out.stem + ".runlog.jsonl")
    if not path.parent.is_dir():
        raise typer.BadParameter(f"--runlog: directory {path.parent} does not exist")
    return SafeRunLog(JsonlRunLogFile(path), path=path)


def _parse_precisions(text: str) -> tuple[int, ...]:
    """Parse the ``--precisions`` CSV into a validated tuple.

    Args:
        text: Comma-separated bit widths, e.g. ``"8,4,3,2"``.

    Returns:
        The parsed precisions.

    Raises:
        typer.BadParameter: If a value is not an integer, any value is
            below 2, or the list is empty or not strictly descending.
    """
    try:
        precisions = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise typer.BadParameter(
            f'--precisions: expected comma-separated integers, got "{text}"'
        ) from None
    if any(bits < 2 for bits in precisions):  # noqa: PLR2004 - scan floor (ADR-0010)
        raise typer.BadParameter(
            "--precisions: the scan floors at 2 bits — remove values below 2"
        )
    try:
        ScanMeta(
            metric="probe",
            calibration="probe",
            calibration_tokens=1,
            precisions=precisions,
            group_by="layer",
            started_at="probe",
        )
    except ValueError as exc:
        raise typer.BadParameter(f"--precisions: {exc}") from exc
    return precisions


def _parse_within_group(
    text: str, precisions: tuple[int, ...]
) -> Literal["rtn", "kquant"]:
    """Validate the ``--within-group`` choice against the precisions.

    Args:
        text: The flag value.
        precisions: The parsed candidate precisions.

    Returns:
        The validated method name.

    Raises:
        typer.BadParameter: If the method is unknown, or ``kquant``
            is combined with precisions outside its port coverage
            (ADR-0018) — rejected before the model load burns an hour.
    """
    if text not in ("rtn", "kquant"):
        raise typer.BadParameter(
            f'--within-group: expected "rtn" or "kquant", got "{text}"'
        )
    if text == "kquant":
        uncovered = [p for p in precisions if p not in KQUANT_PRECISIONS]
        if uncovered:
            raise typer.BadParameter(
                f"--within-group kquant covers precisions "
                f"{sorted(KQUANT_PRECISIONS, reverse=True)} (ADR-0018) — "
                f"remove {uncovered} from --precisions"
            )
    return text


def scan(
    model: Annotated[
        str, typer.Argument(help="Hugging Face model id or local checkpoint path.")
    ],
    calibration: Annotated[Path, typer.Option(help="Calibration text file (UTF-8).")],
    out: Annotated[Path, typer.Option(help="Output sensitivity map path.")] = Path(
        "sensitivity.json"
    ),
    precisions: Annotated[
        str, typer.Option(help="Candidate precisions, descending CSV.")
    ] = "8,4,3,2",
    group_by: Annotated[
        str, typer.Option(help="Grouping granularity: layer or tensor.")
    ] = "layer",
    max_tokens: Annotated[
        int, typer.Option(min=2, help="Calibration token budget.")
    ] = 131072,
    device: Annotated[
        str, typer.Option(help="Device map: auto, cpu, or cuda.")
    ] = "auto",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow model repos with custom code.")
    ] = False,
    resume: Annotated[
        bool, typer.Option(help="Resume from the checkpoint file if present.")
    ] = True,
    gpu_memory: Annotated[
        str | None,
        typer.Option(
            help="Byte cap on GPU 0 model shards, e.g. 17GiB. Requires "
            "--device auto. Leaves workspace for activations and "
            "quantization."
        ),
    ] = None,
    within_group: Annotated[
        str,
        typer.Option(
            help="Within-group method: rtn, or kquant for the "
            "K-quant-faithful port (ADR-0018, 2/3-bit only)."
        ),
    ] = "rtn",
    runlog: Annotated[
        Path | None,
        typer.Option(help="Run-log path (JSONL). Default: <stem>.runlog.jsonl."),
    ] = None,
) -> None:
    """Measure per-group quantization damage and write a sensitivity map.

    Every finished cell lands in a checkpoint file next to ``--out``
    (``<stem>.checkpoint.json``), so a crashed scan resumes instead
    of restarting. ``--no-resume`` discards any existing checkpoint.
    ``--gpu-memory`` caps the shards that ``auto`` sharding places on
    GPU 0 (parsed with the project size grammar, validated up front),
    keeping workspace free for activations and logits. Every run
    appends machine-readable events to ``--runlog`` (default beside
    ``--out``), the ADR-0011 run log. Every halt event carries the
    same fields, with ``cells_kept`` null when the stage cannot know
    the count. Groups offloaded to host RAM under the cap measure
    through accelerate's weights map (ADR-0015). The meter refuses
    weights offloaded beyond host RAM — see the how-to.

    Raises:
        typer.BadParameter: If ``--precisions``, ``--group-by``,
            ``--within-group``, or ``--gpu-memory`` is malformed,
            ``--gpu-memory`` is given without ``--device auto``,
            ``--within-group kquant`` is combined with precisions the
            port does not cover, or the ``--out`` or ``--runlog``
            directory does not exist.
        typer.Exit: With code 1 when the scan extra is missing, the
            model or calibration cannot load, the checkpoint belongs to
            a different scan, a measurement fails, a checkpoint write
            fails, or the map cannot be written.

    Examples:
        Command line usage:

        ```console
        $ quantfit scan ./model --calibration calib.txt --max-tokens 4096
        ```
    """
    parsed_precisions = _parse_precisions(precisions)
    if group_by not in ("layer", "tensor"):
        raise typer.BadParameter(
            f'--group-by: expected "layer" or "tensor", got "{group_by}"'
        )
    parsed_within_group = _parse_within_group(within_group, parsed_precisions)
    # Reject an unwritable destination now — not after the model loads
    # and the first calibration pass has burned an hour.
    if not out.parent.is_dir():
        raise typer.BadParameter(f"--out: directory {out.parent} does not exist")
    # Parse the cap with the project grammar — accelerate reads "17gb"
    # as gigabits, an 8x smaller cap than this CLI means by it.
    gpu_memory_bytes: int | None = None
    if gpu_memory is not None:
        if device != "auto":
            raise typer.BadParameter(
                f'--gpu-memory requires --device auto, got --device "{device}"'
            )
        try:
            gpu_memory_bytes = parse_size(gpu_memory)
        except ValueError as exc:
            raise typer.BadParameter(f"--gpu-memory: {exc}") from exc

    run_log = _open_run_log(out, runlog)
    meter = start_run(
        run_log,
        {
            "model": model,
            "precisions": list(parsed_precisions),
            "group_by": group_by,
            "max_tokens": max_tokens,
            "device": device,
            "gpu_memory_bytes": gpu_memory_bytes,
            "within_group": within_group,
        },
        lambda: _build_meter(
            model,
            calibration,
            max_tokens=max_tokens,
            group_by=group_by,
            device=device,
            trust_remote_code=trust_remote_code,
            gpu_memory=gpu_memory_bytes,
            within_group=parsed_within_group,
        ),
    )

    meta = ScanMeta(
        metric="kl_divergence",
        calibration=str(calibration),
        calibration_tokens=meter.calibration_tokens(),
        precisions=parsed_precisions,
        group_by=group_by,
        started_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        within_group=SCAN_METHOD if parsed_within_group == "rtn" else KQUANT_METHOD,
    )
    fingerprint = scan_fingerprint(model, meta)
    checkpoint_path = out.with_name(out.stem + ".checkpoint.json")
    if not resume and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError as exc:
            typer.echo(f"error: {checkpoint_path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"discarded checkpoint {checkpoint_path}")
    store: ScanCheckpointStore = JsonScanCheckpointFile(checkpoint_path)

    specs = meter.groups()
    try:
        done = store.load(fingerprint)
        todo = plan_measurements(
            specs, parsed_precisions, [(m.group, m.bits) for m in done]
        )
    except (ArtifactError, ValueError, OSError) as exc:
        typer.echo(
            f"error: {checkpoint_path}: {exc} — pass --no-resume to discard it",
            err=True,
        )
        run_log.emit(
            "scan_halted",
            {
                "stage": "checkpoint_load",
                "error": str(exc),
                "cells_kept": None,
                "rss_hwm_gb": rss_hwm_gb(),
            },
        )
        raise typer.Exit(code=1) from exc

    if done:
        typer.echo(f"resuming: {len(done)} of {len(done) + len(todo)} cells done")
        run_log.emit("resume_loaded", {"cells": len(done), "remaining": len(todo)})
    measurements = measure_cells(meter, store, fingerprint, list(done), todo, run_log)

    sink: SensitivityMapSink = JsonSensitivityMapFile(out)
    try:
        map_ = assemble_map(model, meta, specs, measurements)
        sink.save(map_)
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {out}: {exc}", err=True)
        run_log.emit(
            "scan_halted",
            {
                "stage": "map_write",
                "error": str(exc),
                "cells_kept": len(measurements),
                "rss_hwm_gb": rss_hwm_gb(),
            },
        )
        raise typer.Exit(code=1) from exc
    run_log.emit(
        "scan_finished",
        {
            "out": str(out),
            "groups": len(specs),
            "cells": len(measurements),
            "rss_hwm_gb": rss_hwm_gb(),
        },
    )
    typer.echo(
        f"scanned {len(specs)} groups x {len(parsed_precisions)} precisions "
        f"over {meta.calibration_tokens} tokens -> {out}"
    )
