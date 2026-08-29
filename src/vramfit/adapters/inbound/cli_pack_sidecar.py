"""Sidecar stage for the ``vramfit pack`` command.

Ships the vendor mmproj beside the packed decoder GGUF as the
unquantized projector sidecar, byte-identical (ADR-0030 decision
2). The stage runs after the size check and the reconstruction
gate pass — the sidecar completes an artifact both checks accept.
The copy and its hash proof live
in the outbound adapter; this stage wires them to the run log and
the human channel, and `check_sidecar_collisions` refuses an
mmproj whose copy would overwrite a file the run owns. The sidecar
never enters the weight budget: the
vision line is a serving measurement, not the file size (ADR-0030
decision 3).

Examples:
    The pack command drives the stage like this:

    ```python
    check_sidecar_collisions(mmproj, out, base_path, runlog_path)
    _ship_sidecar_stage(run_log, mmproj, out)
    ```

See Also:
    - [vramfit.adapters.inbound.cli_pack][]: The command that
      drives this stage.
    - [vramfit.adapters.outbound.gguf.sidecar][]: The copy and the
      hash proof.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vramfit.adapters.inbound.cli_pack_check import reconstruction_reference_path
from vramfit.adapters.inbound.cli_pack_smoke import _halt
from vramfit.adapters.inbound.run_log import SafeRunLog
from vramfit.adapters.outbound.gguf.sidecar import ship_sidecar
from vramfit.domain.budget import format_size


def check_sidecar_collisions(
    mmproj: Path | None, out: Path, base_path: Path, runlog_path: Path
) -> None:
    """Refuse an mmproj whose sidecar copy would overwrite a run file.

    The copy lands at the vendor file name beside ``--out``. Four
    run-owned files live in that directory: the decoder GGUF, the
    base GGUF, the run log, and the reconstruction reference. A
    matching mmproj name would overwrite one of them after minutes
    of toolchain work — refuse before any tool runs.

    Args:
        mmproj: The vendor mmproj, or None to check nothing.
        out: Packed model destination.
        base_path: The f16 base GGUF the convert stage owns.
        runlog_path: The run-log file.

    Raises:
        typer.BadParameter: If the sidecar copy's path equals a
            run-owned path.
    """
    if mmproj is None:
        return
    destination = out.with_name(mmproj.name)
    owned = {
        out: "packed decoder GGUF (--out)",
        base_path: "f16 base GGUF",
        runlog_path: "run log",
        reconstruction_reference_path(out): "reconstruction reference",
    }
    label = owned.get(destination)
    if label is not None:
        raise typer.BadParameter(
            f"--mmproj: the sidecar copy {destination} would overwrite the {label}"
        )


def _ship_sidecar_stage(run_log: SafeRunLog, mmproj: Path | None, out: Path) -> None:
    """Ship the projector sidecar beside the packed artifact.

    The ``sidecar_shipped`` event carries the digest the
    publication step reuses (ADR-0030 consequences).

    Args:
        run_log: The pack run's event log.
        mmproj: The vendor mmproj, or None to ship nothing.
        out: The packed decoder GGUF the sidecar ships beside.

    Raises:
        typer.Exit: With code 1 when the copy fails or does not
            match the source (via ``_halt``).
    """
    if mmproj is None:
        return
    try:
        shipped = ship_sidecar(mmproj, beside=out)
    except (RuntimeError, ValueError, OSError) as exc:
        raise _halt(run_log, "sidecar", exc) from exc
    run_log.emit(
        "sidecar_shipped",
        {
            "mmproj": str(mmproj),
            "path": str(shipped.path),
            "bytes": shipped.n_bytes,
            "sha256": shipped.sha256,
        },
    )
    typer.echo(
        f"shipped projector sidecar {shipped.path} "
        f"({format_size(shipped.n_bytes)}), byte-identical, "
        f"sha256 {shipped.sha256}"
    )
