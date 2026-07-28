"""Run-log wiring for the scan command: safe sink, run loop, timings.

Holds the scan command's event machinery so `quantfit.adapters.inbound.cli_scan`
stays under the size cap. `SafeRunLog` enforces the ADR-0011 failure
policy: a run-log write failure warns once on the human channel and
disables further events — measurement work outlives its telemetry, and
a halt event can never displace the error it reports. Every event
carries a ``run_id``, so reruns and resumes stay separable in one file.

Examples:
    Wrap a sink for one run:

    ```python
    safe = SafeRunLog(JsonlRunLogFile(path))
    safe.emit("scan_started", {"model": "m"})
    ```

See Also:
    - [quantfit.adapters.outbound.run_log_jsonl][]: The sink this wraps.
    - [quantfit.adapters.inbound.cli_scan][]: The command that drives it.
"""

from __future__ import annotations

import resource
import time
import uuid
from collections.abc import Callable, Mapping

import typer

from quantfit.domain.scan import Measurement
from quantfit.ports.outbound import DamageMeter, RunLogSink, ScanCheckpointStore


def rss_hwm_gb() -> float:
    """Report the process resident-set high-water mark in GB.

    Returns:
        ``ru_maxrss`` converted from KiB (the Linux unit) to GB,
        rounded to two decimals.
    """
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 976_562.5, 2)


class SafeRunLog:
    """`RunLogSink` wrapper implementing the run-log failure policy.

    The first failed write echoes one ``warning:`` line and disables
    the run log for the rest of the process. Nothing is silent, no
    measurement dies for its telemetry, and an emit inside an error
    handler cannot displace the error it reports (ADR-0011).

    Attributes:
        run_id (str): Twelve hex characters stamped on every event.

    Examples:
        A dead sink swallows nothing silently:

        ```python
        safe = SafeRunLog(sink)
        safe.emit("scan_started", {})  # warns once if the sink fails
        ```
    """

    def __init__(self, sink: RunLogSink) -> None:
        """Wrap a sink and mint the run identity.

        Args:
            sink: The real sink to protect.
        """
        self._sink = sink
        self._dead = False
        self.run_id = uuid.uuid4().hex[:12]

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        """Record one event, warning once and disabling on failure.

        Args:
            event: Past-tense event name.
            fields: JSON-representable payload. ``run_id`` is added.
        """
        if self._dead:
            return
        try:
            self._sink.emit(event, {"run_id": self.run_id, **fields})
        except (OSError, TypeError, ValueError) as exc:
            self._dead = True
            typer.echo(f"warning: run log disabled: {exc}", err=True)


def start_run(
    run_log: SafeRunLog,
    payload: Mapping[str, object],
    build: Callable[[], DamageMeter],
) -> DamageMeter:
    """Emit the opening events and build the meter.

    Args:
        run_log: Sink for ``scan_started``/``meter_built``/halt events.
        payload: The ``scan_started`` fields.
        build: Zero-argument meter builder — a closure keeps the real
            call statically checked.

    Returns:
        The loaded meter.

    Raises:
        typer.Exit: With code 1 when the meter cannot be built — the
            command echoes the failure and logs ``scan_halted`` with
            ``cells_kept`` null (the stage cannot know the count).
    """
    run_log.emit("scan_started", payload)
    build_started = time.monotonic()
    # Only a failed adapter import means "extra not installed" —
    # construction errors (missing tokenizer backend, CUDA out of
    # memory, bad repo) must surface as themselves.
    try:
        meter = build()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        run_log.emit(
            "scan_halted",
            {
                "stage": "meter_build",
                "error": str(exc),
                "cells_kept": None,
                "rss_hwm_gb": rss_hwm_gb(),
            },
        )
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
