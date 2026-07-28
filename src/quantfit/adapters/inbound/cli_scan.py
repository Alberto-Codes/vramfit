"""The ``quantfit scan`` command: measure damage, checkpoint, emit a map.

The scan loop lives here because the inbound adapter is the
composition root: it builds the torch-backed meter (lazily, so the
base install never imports torch), drives the `DamageMeter` and
`ScanCheckpointStore` ports cell by cell, and hands the finished
measurements to the pure assembly logic in [quantfit.domain.scan][].

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

from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from quantfit.adapters.outbound.sensitivity_map_json import JsonSensitivityMapFile
from quantfit.domain.model import ScanMeta
from quantfit.domain.scan import (
    Measurement,
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


def _build_meter(
    model: str,
    calibration: Path,
    max_tokens: int,
    group_by: Literal["layer", "tensor"],
    device: str,
    trust_remote_code: bool,
) -> DamageMeter:
    """Build the torch-backed meter, importing torch only now.

    Unit tests monkeypatch this seam with the verified fake, keeping
    the command's orchestration testable without a GPU (ADR-0009).

    Args:
        model: Hugging Face model id or local checkpoint path.
        calibration: UTF-8 calibration text file.
        max_tokens: Upper bound on calibration tokens.
        group_by: Grouping granularity.
        device: transformers ``device_map`` value.
        trust_remote_code: Allow repos with custom modeling code.

    Returns:
        The loaded meter.

    Raises:
        ImportError: If the scan extra is not installed.
        ValueError: If the calibration file yields too few tokens.
        OSError: If the model or calibration file cannot be read.
    """
    from quantfit.adapters.outbound.scan.meter import (  # noqa: PLC0415 - lazy: keeps the base CLI torch-free (ADR-0005)
        TorchDamageMeter,
    )

    return TorchDamageMeter(
        model,
        calibration,
        max_tokens=max_tokens,
        group_by=group_by,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def _parse_precisions(text: str) -> tuple[int, ...]:
    """Parse the ``--precisions`` CSV into a validated tuple.

    Args:
        text: Comma-separated bit widths, e.g. ``"8,4,3,2"``.

    Returns:
        The parsed precisions.

    Raises:
        typer.BadParameter: If a value is not an integer, or the list
            is empty, non-positive, or not strictly descending.
    """
    try:
        precisions = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise typer.BadParameter(
            f'--precisions: expected comma-separated integers, got "{text}"'
        ) from None
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
) -> None:
    """Measure per-group quantization damage and write a sensitivity map.

    Every finished cell lands in a checkpoint file next to ``--out``
    (``<out stem>.checkpoint.json``), so a crashed scan resumes instead
    of restarting. ``--no-resume`` discards any existing checkpoint.

    Raises:
        typer.BadParameter: If ``--precisions`` or ``--group-by`` is
            malformed.
        typer.Exit: With code 1 when the scan extra is missing, the
            model or calibration cannot load, the checkpoint belongs to
            a different scan, or a measurement fails.

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
    grouping: Literal["layer", "tensor"] = group_by

    try:
        meter = _build_meter(
            model, calibration, max_tokens, grouping, device, trust_remote_code
        )
    except ImportError as exc:
        typer.echo(f"error: {INSTALL_HINT}", err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    started_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = ScanMeta(
        metric="kl_divergence",
        calibration=str(calibration),
        calibration_tokens=meter.calibration_tokens(),
        precisions=parsed_precisions,
        group_by=grouping,
        started_at=started_at,
    )
    fingerprint = scan_fingerprint(model, meta)
    checkpoint_path = out.with_name(out.stem + ".checkpoint.json")
    if not resume:
        checkpoint_path.unlink(missing_ok=True)
    store: ScanCheckpointStore = JsonScanCheckpointFile(checkpoint_path)

    specs = meter.groups()
    try:
        done = store.load(fingerprint)
        todo = plan_measurements(
            specs, parsed_precisions, [(m.group, m.bits) for m in done]
        )
    except (ArtifactError, ValueError) as exc:
        typer.echo(
            f"error: {checkpoint_path}: {exc} — pass --no-resume to discard it",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    if done:
        typer.echo(f"resuming: {len(done)} of {len(done) + len(todo)} cells done")
    measurements = list(done)
    total = len(done) + len(todo)
    for i, (group, bits) in enumerate(todo, start=len(done) + 1):
        try:
            damage = meter.measure(group, bits)
        except (RuntimeError, ValueError, OSError) as exc:
            typer.echo(
                f"error: measuring {group} at {bits}-bit failed: {exc} "
                f"(checkpoint keeps {i - 1} cells)",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        measurement = Measurement(group=group, bits=bits, damage=damage)
        store.append(fingerprint, measurement)
        measurements.append(measurement)
        typer.echo(f"[{i}/{total}] {group} @ {bits}-bit damage {damage:.6f}")

    map_ = assemble_map(model, meta, specs, measurements)
    sink: SensitivityMapSink = JsonSensitivityMapFile(out)
    try:
        sink.save(map_)
    except OSError as exc:
        typer.echo(f"error: {out}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"scanned {len(specs)} groups x {len(parsed_precisions)} precisions "
        f"over {meta.calibration_tokens} tokens -> {out}"
    )
