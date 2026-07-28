"""Scan-command event wiring: the run loop and its timings.

Holds the scan command's event machinery so `quantfit.adapters.inbound.cli_scan`
stays under the size cap. The shared run-log policy wrapper lives in
[quantfit.adapters.inbound.run_log][] — this module drives it through
the scan grid, stamping each cell with its timing and memory
high-water mark (ADR-0011). The validate command reuses `start_run`
with its own event prefix.

Examples:
    Measure the remaining cells of a scan:

    ```python
    measurements = measure_cells(meter, store, fp, done, todo, run_log)
    ```

See Also:
    - [quantfit.adapters.inbound.run_log][]: `SafeRunLog`, the policy
      wrapper these helpers emit through.
    - [quantfit.adapters.inbound.cli_scan][]: The command that drives it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import typer

from quantfit.adapters.inbound.run_log import SafeRunLog, rss_hwm_gb
from quantfit.domain.scan import Measurement
from quantfit.ports.outbound import DamageMeter, ScanCheckpointStore


def start_run(
    run_log: SafeRunLog,
    payload: Mapping[str, object],
    build: Callable[[], DamageMeter],
    prefix: str = "scan",
) -> DamageMeter:
    """Emit the opening events and build the meter.

    Args:
        run_log: Sink for the started, ``meter_built``, and halt
            events.
        payload: The ``<prefix>_started`` fields.
        build: Zero-argument meter builder — a closure keeps the real
            call statically checked.
        prefix: Command name for the ``<prefix>_started`` and
            ``<prefix>_halted`` events — ``scan`` or ``validation``.

    Returns:
        The loaded meter.

    Raises:
        typer.Exit: With code 1 when the meter cannot be built — the
            command echoes the failure and logs ``<prefix>_halted``.
            Scan halts carry ``cells_kept`` null (the stage cannot
            know the count); validation halts have no scan grid and
            omit it.
    """
    run_log.emit(f"{prefix}_started", payload)
    build_started = time.monotonic()
    # Only a failed adapter import means "extra not installed" —
    # construction errors (missing tokenizer backend, CUDA out of
    # memory, bad repo) must surface as themselves.
    try:
        meter = build()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        fields: dict[str, object] = {
            "stage": "meter_build",
            "error": str(exc),
            "rss_hwm_gb": rss_hwm_gb(),
        }
        # Every scan halt carries the same fields (ADR-0011); the
        # validation pass has no scan grid, so no cell count.
        if prefix == "scan":
            fields["cells_kept"] = None
        run_log.emit(f"{prefix}_halted", fields)
        raise typer.Exit(code=1) from exc
    run_log.emit(
        "meter_built",
        {
            "seconds": round(time.monotonic() - build_started, 3),
            "calibration_tokens": meter.calibration_tokens(),
            "groups": len(meter.groups()),
            "rss_hwm_gb": rss_hwm_gb(),
        },
    )
    return meter


def measure_cells(
    meter: DamageMeter,
    store: ScanCheckpointStore,
    fingerprint: str,
    done: list[Measurement],
    todo: tuple[tuple[str, int], ...],
    run_log: SafeRunLog,
) -> list[Measurement]:
    """Measure every remaining cell, checkpointing and logging each one.

    Args:
        meter: The damage meter to drive.
        store: The checkpoint store recording each finished cell.
        fingerprint: The scan's identity string.
        done: Measurements already checkpointed.
        todo: Remaining (group, bits) cells, in measurement order.
        run_log: Sink for ``cell_measured`` and ``scan_halted`` events.

    Returns:
        All measurements, checkpointed ones first.

    Raises:
        typer.Exit: With code 1 when a measurement or checkpoint write
            fails — the message reports how many cells the checkpoint
            keeps.
    """
    measurements = list(done)
    total = len(done) + len(todo)
    for i, (group, bits) in enumerate(todo, start=len(done) + 1):
        started = time.monotonic()
        try:
            damage = meter.measure(group, bits)
            measurement = Measurement(group=group, bits=bits, damage=damage)
            store.append(fingerprint, measurement)
        except (RuntimeError, ValueError, OSError) as exc:
            typer.echo(
                f"error: scan halted at {group} {bits}-bit: {exc} "
                f"(checkpoint keeps {i - 1} cells)",
                err=True,
            )
            run_log.emit(
                "scan_halted",
                {
                    "stage": "measure",
                    "group": group,
                    "bits": bits,
                    "error": str(exc),
                    "cells_kept": i - 1,
                    "rss_hwm_gb": rss_hwm_gb(),
                },
            )
            raise typer.Exit(code=1) from exc
        measurements.append(measurement)
        run_log.emit(
            "cell_measured",
            {
                "group": group,
                "bits": bits,
                "damage": damage,
                "seconds": round(time.monotonic() - started, 3),
                "cell": i,
                "of": total,
                "rss_hwm_gb": rss_hwm_gb(),
            },
        )
        typer.echo(f"[{i}/{total}] {group} @ {bits}-bit damage {damage:.6f}")
    return measurements
