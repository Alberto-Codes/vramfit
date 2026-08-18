"""Option checks the scan and validate commands share.

The two commands build the same meter, so their overlapping options
follow one rule each: the ``--gpu-memory`` cap parses with the
project size grammar and requires ``--device auto``, and
``--imatrix`` pairs only with the kquant within-group method
(ADR-0020) — RTN has no weighted C counterpart, and the ``gguf``
method's ``gguf-imx`` token is reserved but unbuilt (ADR-0018).
Both reject before any model load burns an hour. The
imatrix coverage echo lives here too — the scan and the validation
pass report the split identically.

Examples:
    Parse a cap the way both commands do:

    ```python
    from vramfit.adapters.inbound.cli_options import parse_gpu_memory

    assert parse_gpu_memory("1GiB", "auto") == 2**30
    ```

See Also:
    - [vramfit.adapters.inbound.cli_scan][]: The scan command.
    - [vramfit.adapters.inbound.cli_validate][]: The validate command.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vramfit.domain.budget import parse_size
from vramfit.ports.outbound import DamageMeter


def check_imatrix(imatrix: Path | None, method: str) -> None:
    """Refuse an ``--imatrix`` that cannot pair with the method.

    Args:
        imatrix: The ``--imatrix`` path, or None for unassisted.
        method: The resolved within-group method name.

    Raises:
        typer.BadParameter: If the imatrix arrives without the
            kquant method (ADR-0020), or the file does not exist.
            The ``gguf`` method refuses it too. Its ``gguf-imx``
            token is reserved for a real assisted path —
            ``quantize_row_q4_0_impl`` fits with imatrix weights —
            and nothing builds it yet (ADR-0018).
    """
    if imatrix is None:
        return
    if method != "kquant":
        raise typer.BadParameter(
            "--imatrix requires --within-group kquant (ADR-0020) — "
            "RTN has no weighted C counterpart, and gguf-imx is "
            "reserved but unbuilt (ADR-0018)"
        )
    if not imatrix.is_file():
        raise typer.BadParameter(f"--imatrix: {imatrix} is not a file")


def echo_imatrix_coverage(meter: DamageMeter) -> None:
    """Report the assisted-pricing coverage split on the console.

    Uncovered parameters price unassisted under the assisted label
    (ADR-0020) — the operator must see the split, in the scan and
    in the validation pass alike. The console states the counts
    only. A model with fused expert stacks leaves 181 of 210
    parameters uncovered (#191), and a joined list of names buries
    the split it reports. The run log's ``meter_built`` event names
    every uncovered parameter.
    Silent for unassisted meters and meters without the notion.

    Args:
        meter: The built meter.
    """
    covered = getattr(meter, "imatrix_covered_count", None)
    if covered is None:
        return
    uncovered: tuple[str, ...] = getattr(meter, "imatrix_uncovered", None) or ()
    detail = (
        f" ({len(uncovered)} uncovered — the run log names them)" if uncovered else ""
    )
    typer.echo(
        f"imatrix covers {covered} of {covered + len(uncovered)} parameters{detail}"
    )


def parse_gpu_memory(gpu_memory: str | None, device: str) -> int | None:
    """Parse the ``--gpu-memory`` option against the device choice.

    Parsed with the project size grammar — accelerate reads ``17gb``
    as gigabits, an 8x smaller cap than this CLI means by it.

    Args:
        gpu_memory: The raw option value, or None for no cap.
        device: The ``--device`` value the cap requires to be auto.

    Returns:
        The cap in bytes, or None when no cap was given.

    Raises:
        typer.BadParameter: If the size is malformed, or a cap is
            given without ``--device auto``.
    """
    if gpu_memory is None:
        return None
    if device != "auto":
        raise typer.BadParameter(
            f'--gpu-memory requires --device auto, got --device "{device}"'
        )
    try:
        return parse_size(gpu_memory)
    except ValueError as exc:
        raise typer.BadParameter(f"--gpu-memory: {exc}") from exc
