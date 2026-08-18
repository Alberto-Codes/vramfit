"""The ``vramfit scan`` command: measure damage, checkpoint, emit a map.

The scan loop lives here because the inbound adapter is the
composition root: it validates every option up front (sizes parse
with the project grammar, ``--within-group kquant`` and ``q0``
must each pair with precisions their port covers, ADR-0018, and
``--imatrix`` pairs only with the kquant method, ADR-0020), builds
the
torch-backed
meter (lazily, so the base install never imports torch), drives the
`DamageMeter` and `ScanCheckpointStore` ports cell by cell, and hands
the finished measurements to the pure assembly logic in
[vramfit.domain.scan][]. An assisted scan records the
``kquant-imx`` token and the resolved imatrix path in the map, the
fingerprint, and the run log — relative spellings must not split
or mix checkpoint identities.
``--groups`` restricts a run to named groups, so a caller that wants
46 of 210 groups pays for 46 (#282). Only the list shape checks here.
`resolve_grid` matches each name against the loaded meter's groups,
after the model loads and before any cell measures.
Every failure — a missing extra, a bad destination, an unstable
measurement, a checkpoint write — halts with a clean ``error:`` line.
The checkpoint keeps every finished cell. The run log records the
halt with its stage (ADR-0011).

Examples:
    Scan a local checkpoint at the ADR-0010 candidate set:

    ```console
    $ vramfit scan ./model --calibration calib.txt --out sensitivity.json
    ```

See Also:
    - [vramfit.adapters.outbound.scan.meter][]: The torch meter this
      command builds.
    - [vramfit.domain.scan][]: Work planning and map assembly.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

from vramfit.adapters.inbound.cli_options import (
    check_imatrix,
    echo_imatrix_coverage,
    parse_gpu_memory,
)
from vramfit.adapters.inbound.scan_events import (
    SafeRunLog,
    measure_cells,
    resolve_grid,
    rss_hwm_gb,
    start_run,
)
from vramfit.adapters.outbound.run_log_jsonl import JsonlRunLogFile
from vramfit.adapters.outbound.scan_checkpoint_json import JsonScanCheckpointFile
from vramfit.adapters.outbound.sensitivity_map_json import JsonSensitivityMapFile
from vramfit.domain.errors import VramfitError
from vramfit.domain.model import ScanMeta
from vramfit.domain.scan import (
    KQUANT_IMX_METHOD,
    KQUANT_METHOD,
    KQUANT_PRECISIONS,
    Q0_REF_METHOD,
    Q0_REF_PRECISIONS,
    SCAN_METHOD,
    assemble_map,
    scan_fingerprint,
)
from vramfit.ports.outbound import (
    DamageMeter,
    ScanCheckpointStore,
    SensitivityMapSink,
)

INSTALL_HINT = (
    'the scan extra is not installed — install it with: uv pip install "vramfit[scan]"'
)


class ScanExtraMissingError(VramfitError, RuntimeError):
    """The scan adapter package itself failed to import.

    Distinguishes "torch is not installed" from every other
    ImportError a model load can raise (missing tokenizer backends,
    broken CUDA builds) — those must surface as themselves. Inherits
    `VramfitError` per ADR-0011.

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
    group_by: Literal["layer", "tensor", "stack"],
    device: str,
    trust_remote_code: bool,
    gpu_memory: int | None,
    within_group: Literal["rtn", "kquant", "q0"] = "rtn",
    imatrix: Path | None = None,
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
        group_by: Grouping granularity — ``layer``, ``tensor``, or
            ``stack``.
        device: transformers ``device_map`` value.
        trust_remote_code: Allow repos with custom modeling code.
        gpu_memory: Byte cap on GPU 0 model shards under ``auto``
            sharding.
        within_group: Within-group method (ADR-0018) — ``rtn``,
            ``kquant``, or ``q0``.
        imatrix: GGUF imatrix file for assisted pricing (ADR-0020),
            or None for an unassisted meter.

    Returns:
        The loaded meter.

    Raises:
        ScanExtraMissingError: If the scan extra is not installed.
        ValueError: If the calibration file yields too few tokens, or
            the imatrix is malformed or covers no parameter.
        OSError: If the model, calibration, or imatrix file cannot
            be read.
    """
    try:
        from vramfit.adapters.outbound.scan.meter import (  # noqa: PLC0415 - lazy: keeps the base CLI torch-free (ADR-0005)
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
        imatrix_path=imatrix,
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


def _parse_groups(text: str | None) -> tuple[str, ...]:
    """Parse the ``--groups`` CSV into a validated tuple.

    Only the list shape is checkable here. Whether each name matches a
    discovered group needs the loaded meter, so `select_groups` decides
    that after the model loads and before any cell measures.

    Args:
        text: Comma-separated group names, or None for every group.

    Returns:
        The requested names, or an empty tuple for every group.

    Raises:
        typer.BadParameter: If an entry is blank or a name repeats.
            Each is a typo, and a typo must not survive the model load.
    """
    if text is None:
        return ()
    names = tuple(part.strip() for part in text.split(","))
    if any(not name for name in names):
        raise typer.BadParameter(
            f'--groups: an entry is empty, got "{text}" — '
            "omit the flag to scan every group"
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        listed = ", ".join(f'"{name}"' for name in repeated)
        raise typer.BadParameter(f"--groups: {listed} repeats")
    return names


def _parse_within_group(
    text: str, precisions: tuple[int, ...], imatrix: Path | None
) -> tuple[Literal["rtn", "kquant", "q0"], str]:
    """Validate the ``--within-group`` choice against the precisions.

    Args:
        text: The flag value.
        precisions: The parsed candidate precisions.
        imatrix: The ``--imatrix`` path, or None for unassisted.

    Returns:
        The validated method name and its fingerprint token — the
        token is the vocabulary run logs and maps share (ADR-0018).
        An imatrix turns the kquant token into the assisted one
        (ADR-0020).

    Raises:
        typer.BadParameter: If the method is unknown, ``kquant`` or
            ``q0`` is combined with precisions outside its port
            coverage (ADR-0018), ``--imatrix`` arrives without the
            kquant method, or the imatrix file does not exist — each
            rejected before the model load burns an hour.
    """
    if text not in ("rtn", "kquant", "q0"):
        raise typer.BadParameter(
            f'--within-group: expected "rtn", "kquant", or "q0", got "{text}"'
        )
    check_imatrix(imatrix, text)
    covered = {"kquant": KQUANT_PRECISIONS, "q0": Q0_REF_PRECISIONS}.get(text)
    if covered is not None:
        uncovered = [p for p in precisions if p not in covered]
        if uncovered:
            raise typer.BadParameter(
                f"--within-group {text} covers precisions "
                f"{sorted(covered, reverse=True)} (ADR-0018) — "
                f"remove {uncovered} from --precisions"
            )
    if text == "rtn":
        return text, SCAN_METHOD
    if text == "q0":
        return text, Q0_REF_METHOD
    return text, KQUANT_METHOD if imatrix is None else KQUANT_IMX_METHOD


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
        str, typer.Option(help="Grouping granularity: layer, tensor, or stack.")
    ] = "layer",
    groups: Annotated[
        str | None,
        typer.Option(
            help="Restrict the run to these group names (CSV). Names "
            "must be keys --group-by produces. Default: every "
            "discovered group."
        ),
    ] = None,
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
            help="Within-group method: rtn, kquant for the "
            "K-quant-faithful port (ADR-0018, precisions 8/4/3/2), "
            "or gguf for the block quantizers Q2_0/Q4_0/Q8_0 "
            "(precisions 8/4/2)."
        ),
    ] = "rtn",
    imatrix: Annotated[
        Path | None,
        typer.Option(
            help="GGUF imatrix for assisted K-quant pricing "
            "(ADR-0020). Requires --within-group kquant. Use the "
            "file the pack step will consume."
        ),
    ] = None,
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
    ``--within-group`` selects the quantization the meter applies
    inside a perturbed group (ADR-0018): ``rtn`` is the v1 default,
    ``kquant`` prices cells with the ported K-quant reference
    quantizers, and ``q0`` prices them with the ported block
    quantizers ``Q2_0``, ``Q4_0``, and ``Q8_0``. Each pairs only
    with precisions its port covers. ``q0`` reaches the rows no
    K-quant tiles — ``llama-quantize`` substitutes another type for a
    256-element super-block on rows of 2688 or 1856, and ``kquant``
    now refuses such a cell instead of pricing a frame the pack
    cannot apply.
    ``--imatrix`` adds the pack's importance matrix to the kquant
    fit (assisted pricing, ADR-0020) — the map then records the
    resolved imatrix path beside the method, and the run log
    records how many parameters the imatrix covers. The map, the
    fingerprint, and the run log all record the method as its token
    (``rtn-block32``, ``kquant-ref``, ``kquant-imx``, or
    ``q0-ref``).
    ``--group-by`` sets the map key. ``stack`` keys on the unit a
    pack addresses (#161): it collapses a mixture-of-experts layer's
    routed experts into one group per projection, and keeps every
    other weight separate. On a dense model it matches ``tensor``.
    ``--groups`` restricts the run to named groups, so a caller that
    wants 46 of 210 groups pays for 46. The names must be keys
    ``--group-by`` produces. A name matching no discovered group halts
    the run, after the model loads and before any cell measures. The
    map then carries the selected groups alone.
    The selection stays out of the fingerprint. So a narrow run and a
    wide run share one checkpoint. A narrowed run reuses the wide run's
    cells for the groups it selects. The cells in deselected groups
    stay in the file, and the run reports how many it ignored. A
    checkpoint from a different scan still halts the run.

    Raises:
        typer.BadParameter: If ``--precisions``, ``--group-by``,
            ``--groups``,
            ``--within-group``, or ``--gpu-memory`` is malformed,
            ``--gpu-memory`` is given without ``--device auto``,
            ``--within-group kquant`` or ``q0`` is combined with
            precisions its port does not cover, ``--imatrix`` is
            given without
            ``--within-group kquant`` or is not a file, or the
            ``--out`` or ``--runlog`` directory does not exist.
        typer.Exit: With code 1 when the scan extra is missing, the
            model or calibration cannot load, a ``--groups`` name
            matches no discovered group, the checkpoint belongs to
            a different scan, a measurement fails, a checkpoint write
            fails, or the map cannot be written.

    Examples:
        Command line usage:

        ```console
        $ vramfit scan ./model --calibration calib.txt --max-tokens 4096
        ```
    """
    parsed_precisions = _parse_precisions(precisions)
    parsed_groups = _parse_groups(groups)
    if group_by not in ("layer", "tensor", "stack"):
        raise typer.BadParameter(
            f'--group-by: expected "layer", "tensor", or "stack", got "{group_by}"'
        )
    parsed_within_group, method_token = _parse_within_group(
        within_group, parsed_precisions, imatrix
    )
    if imatrix is not None:
        # The fingerprint and the map key on the path string —
        # resolve so relative spellings cannot split or mix
        # checkpoint identities (ADR-0020).
        imatrix = imatrix.resolve()
    # Reject an unwritable destination now — not after the model loads
    # and the first calibration pass has burned an hour.
    if not out.parent.is_dir():
        raise typer.BadParameter(f"--out: directory {out.parent} does not exist")
    gpu_memory_bytes = parse_gpu_memory(gpu_memory, device)

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
            "within_group": method_token,
            "imatrix": None if imatrix is None else str(imatrix),
            "groups": list(parsed_groups) or None,
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
            imatrix=imatrix,
        ),
    )
    echo_imatrix_coverage(meter)

    meta = ScanMeta(
        metric="kl_divergence",
        calibration=str(calibration),
        calibration_tokens=meter.calibration_tokens(),
        precisions=parsed_precisions,
        group_by=group_by,
        started_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        within_group=method_token,
        imatrix=None if imatrix is None else str(imatrix),
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

    specs, done, todo = resolve_grid(
        meter,
        store,
        run_log,
        fingerprint=fingerprint,
        precisions=parsed_precisions,
        groups=parsed_groups,
        checkpoint_path=checkpoint_path,
    )
    measurements = measure_cells(meter, store, fingerprint, done, todo, run_log)

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
