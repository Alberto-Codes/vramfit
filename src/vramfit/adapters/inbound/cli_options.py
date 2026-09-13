"""Option checks the scan and validate commands share.

The two commands build the same meter, so their overlapping options
follow one rule each: the ``--gpu-memory`` cap parses with the
project size grammar and requires ``--device auto``, and
``--imatrix`` pairs with the kquant or q0 within-group method
(ADR-0018, ADR-0020) — RTN has no weighted C counterpart. The
scan's ``--within-group`` parser lives here beside that pairing
rule, which it calls.
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
from typing import Literal

import typer

from vramfit.domain.budget import parse_size
from vramfit.domain.scan import (
    KQUANT_IMX_METHOD,
    KQUANT_METHOD,
    KQUANT_PRECISIONS,
    Q0_IMX_METHOD,
    Q0_REF_METHOD,
    Q0_REF_PRECISIONS,
    SCAN_METHOD,
)
from vramfit.ports.outbound import DamageMeter


def check_imatrix(imatrix: Path | None, method: str) -> None:
    """Refuse an ``--imatrix`` that cannot pair with the method.

    Args:
        imatrix: The ``--imatrix`` path, or None for unassisted.
        method: The resolved within-group method name.

    Raises:
        typer.BadParameter: If the imatrix arrives with the ``rtn``
            method (ADR-0018, ADR-0020), or the file does not
            exist. ``kquant`` fits its covered tensors with the
            weights, and ``q0`` fits nominal 4 through
            ``quantize_row_q4_0_impl``.
    """
    if imatrix is None:
        return
    if method not in ("kquant", "q0"):
        raise typer.BadParameter(
            "--imatrix requires --within-group kquant or q0 "
            "(ADR-0018, ADR-0020) — RTN has no weighted C counterpart"
        )
    if not imatrix.is_file():
        raise typer.BadParameter(f"--imatrix: {imatrix} is not a file")


def echo_imatrix_coverage(meter: DamageMeter) -> None:
    """Report the assisted-pricing coverage split on the console.

    Uncovered parameters price unassisted under the assisted label
    (ADR-0020) — the operator must see the split, in the scan and
    in the validation pass alike. The console states the counts
    only. Native Nemotron 3.5 Lightning discovery reports 164
    parameters (#571), and a joined list of names buries the split.
    The run log's ``meter_built`` event names every uncovered
    parameter.
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


def parse_within_group(
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
        An imatrix turns the kquant or q0 token into its assisted
        one (ADR-0018, ADR-0020).

    Raises:
        typer.BadParameter: If the method is unknown, ``kquant`` or
            ``q0`` is combined with precisions outside its port
            coverage (ADR-0018), ``--imatrix`` arrives with the rtn
            method, or the imatrix file does not exist — each
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
        return text, Q0_REF_METHOD if imatrix is None else Q0_IMX_METHOD
    return text, KQUANT_METHOD if imatrix is None else KQUANT_IMX_METHOD
