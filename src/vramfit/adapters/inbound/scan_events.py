"""Scan-command event wiring: the run loop and its timings.

Holds the scan command's event machinery so `vramfit.adapters.inbound.cli_scan`
stays under the size cap. The shared run-log policy wrapper lives in
[vramfit.adapters.inbound.run_log][] — this module drives it through
the scan grid, stamping each cell with its timing and memory
high-water mark (ADR-0011). ``meter_built`` also records the
imatrix coverage split for assisted meters (ADR-0020). The
validate command reuses `start_run` with its own event prefix.

Examples:
    Measure the remaining cells of a scan:

    ```python
    measurements = measure_cells(meter, store, fp, done, todo, run_log)
    ```

See Also:
    - [vramfit.adapters.inbound.run_log][]: `SafeRunLog`, the policy
      wrapper these helpers emit through.
    - [vramfit.adapters.inbound.cli_scan][]: The command that drives it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import typer

from vramfit.adapters.inbound.run_log import SafeRunLog, rss_hwm_gb
from vramfit.domain.scan import Measurement
from vramfit.ports.outbound import DamageMeter, ScanCheckpointStore


def start_run(
    run_log: SafeRunLog,
    payload: Mapping[str, object],
    build: Callable[[], DamageMeter],
    prefix: str = "scan",
) -> DamageMeter:
    """Emit the opening events and build the meter.

    ``meter_built`` records build seconds, calibration tokens, the
    group count, ``offloaded_groups`` — how many groups measure
    through the weights map (ADR-0015), null for meters without the
    notion — and the imatrix coverage split (ADR-0020):
    ``imatrix_covered`` counts the parameters that price assisted,
    ``imatrix_uncovered`` names the ones that fall back, and both
    are null for unassisted meters. An imatrix in the payload with
    no split on the meter draws a console warning — the coverage
    contract rides on attribute names, and a silent mismatch would
    read as "unassisted".

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
            # Adapter detail the port does not carry: the torch meter
            # counts groups measured through the weights map
            # (ADR-0015). Meters without the notion report null —
            # distinct from a real zero.
            "offloaded_groups": getattr(meter, "offloaded_group_count", None),
            # The assisted-pricing coverage split (ADR-0020) — null
            # for unassisted meters and meters without the notion.
            "imatrix_covered": getattr(meter, "imatrix_covered_count", None),
            "imatrix_uncovered": _imatrix_uncovered(meter),
            "rss_hwm_gb": rss_hwm_gb(),
        },
    )
    # Cross-check the coverage contract: an imatrix in the payload
    # with no split on the meter means the meter is not reporting
    # assisted pricing — a renamed attribute or a stand-in meter
    # would otherwise pass silently as "unassisted".
    if (
        payload.get("imatrix") is not None
        and getattr(meter, "imatrix_covered_count", None) is None
    ):
        typer.echo(
            "warning: an imatrix was given but the meter reports no "
            "coverage split — verify assisted pricing is active",
            err=True,
        )
    return meter


def _imatrix_uncovered(meter: DamageMeter) -> list[str] | None:
    """Read the meter's uncovered-parameter names for the run log.

    Args:
        meter: The built meter.

    Returns:
        The names as a JSON-ready list, or None for unassisted
        meters and meters without the notion.
    """
    uncovered = getattr(meter, "imatrix_uncovered", None)
    return None if uncovered is None else list(uncovered)


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
