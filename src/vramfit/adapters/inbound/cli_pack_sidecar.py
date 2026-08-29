"""Sidecar stage for the ``vramfit pack`` command.

Ships the vendor mmproj beside the packed decoder GGUF as the
unquantized projector sidecar, byte-identical (ADR-0030 decision
2). The stage runs after the size check passes — the sidecar
completes an artifact that fits. The copy and its hash proof live
in the outbound adapter; this stage wires them to the run log and
the human channel. The sidecar never enters the weight budget: the
vision line is a serving measurement, not the file size (ADR-0030
decision 3).

Examples:
    The pack command drives the stage like this:

    ```python
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

from vramfit.adapters.inbound.cli_pack_smoke import _halt
from vramfit.adapters.inbound.run_log import SafeRunLog
from vramfit.adapters.outbound.gguf.sidecar import ship_sidecar
from vramfit.domain.budget import format_size


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
