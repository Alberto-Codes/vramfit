"""Wire `plan`'s independent size source (ADR-0029).

The plan command reads its input map's groups and, since ADR-0029,
the checkpoint's own tensor sizes. This module holds that wiring:
build the safetensors source, aggregate its tensors into groups, and
report what the two inputs cover. It stays out of
[vramfit.adapters.inbound.cli][] so that module keeps its size.

`plan` runs without ``--checkpoint``, and then the map defines the
model as it did before ADR-0029. That reading is silent no longer:
the command reports which groups it prices either way, on the same
channel the runtime filter reports its own narrowing. The solver
refuses that reading on a map whose family holds a class the scan
skips, because only a size source prices it (#409).

A checkpoint that carries none of the map's groups refuses. ADR-0029
leaves a map-source disagreement unruled, and this is not one — a
total miss is the wrong directory. Continuing would price both views
of the model and roughly double it.

One disagreement is ruled (ADR-0029 open question 2, 2026-09-04). A
map scanned before the discovery skip (#204) under ``layer``
granularity folds an unquantizable-class tensor into its layer group,
and the source holds that tensor by its own name. The plan prices it
twice, and the command warns naming the map, the group, and the
tensors. The prices stand: the direction is conservative, and the
remedy is a re-scan.

Examples:
    Read the checkpoint the map was scanned from:

    ```python
    from pathlib import Path

    from vramfit.adapters.inbound.cli_plan_sizes import discovered_groups

    groups = discovered_groups(Path("/models/nemotron-30b"), map_, map_path)
    ```

See Also:
    - [vramfit.domain.sizes][]: The grouping arithmetic.
    - [vramfit.adapters.outbound.safetensors_sizes][]: The source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import typer

from vramfit.adapters.outbound.safetensors_sizes import SafetensorsSizes
from vramfit.domain.errors import VramfitError
from vramfit.domain.model import SensitivityMap
from vramfit.domain.sizes import (
    discovered_group_bytes,
    discovered_group_rows,
    held_class_overlaps,
    uncovered_groups,
)
from vramfit.ports.outbound import TensorSizeSource


@dataclass(frozen=True, slots=True)
class CheckpointGroups:
    """What one checkpoint read tells the plan about its groups.

    Attributes:
        bytes (Mapping[str, int] | None): Bytes at reference precision
            per discovered group, or None when the caller passed no
            ``--checkpoint``.
        rows (Mapping[str, int] | None): Elements per row per group
            the 256 super-block decision reaches (issue #515), or
            None when the caller passed no ``--checkpoint``. The
            solver's refusal tells the two causes apart from it.

    Examples:
        ```python
        from vramfit.adapters.inbound.cli_plan_sizes import CheckpointGroups

        groups = CheckpointGroups(bytes=None, rows=None)
        ```
    """

    bytes: Mapping[str, int] | None
    rows: Mapping[str, int] | None


def discovered_groups(
    checkpoint: Path | None, map_: SensitivityMap, map_path: Path
) -> CheckpointGroups:
    """Read the checkpoint's group sizes and row widths, and report coverage.

    One read serves both. The solver prices from the bytes and routes
    the 256 super-block decision from the row widths (issue #515), so
    two reads could disagree about one checkpoint.

    Args:
        checkpoint: The checkpoint directory to read, or None when the
            caller passed no ``--checkpoint``.
        map_: The loaded sensitivity map, whose ``scan.group_by``
            fixes the grouping granularity.
        map_path: Where the map was read from, named in the double
            count warning.

    Returns:
        The group bytes and row widths. Both are None when no
        checkpoint was given.

    Raises:
        typer.Exit: With code 1 when the checkpoint cannot be read or
            priced, and when no map group appears in it. The source's
            own message carries the reason for the first.
    """
    if checkpoint is None:
        # The runtime filter reports its narrowing on this channel for
        # the same reason: an unstated narrowing reads as a whole
        # model. Nothing here is anomalous, so it is not a warning.
        typer.echo(
            f"no --checkpoint: this plan prices the {len(map_.groups)} groups "
            f"the map carries and reads no other size source (ADR-0029)"
        )
        return CheckpointGroups(bytes=None, rows=None)

    source: TensorSizeSource = SafetensorsSizes(checkpoint)
    try:
        sizes = source.tensor_sizes()
        groups = discovered_group_bytes(sizes, map_.scan.group_by)
        rows = discovered_group_rows(sizes, map_.scan.group_by)
    except VramfitError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"error: {checkpoint}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    covered = [g.name for g in map_.groups]
    held = uncovered_groups(groups, covered)
    typer.echo(
        f"checkpoint holds {len(groups)} groups: "
        f"{len(groups) - len(held)} measured by the map, "
        f"{len(held)} held at reference precision"
    )
    missing = [name for name in covered if name not in groups]
    if len(missing) == len(covered):
        # Not a disagreement between two views of one model, which
        # ADR-0029 leaves unruled. Nothing matched, so this is the
        # wrong checkpoint. Continuing would price the map's groups
        # and the checkpoint's groups together and roughly double the
        # model.
        typer.echo(
            f"error: {checkpoint}: none of the map's {len(covered)} groups "
            f'appear in this checkpoint, starting with "{covered[0]}" — '
            f"this is not the checkpoint the scan measured",
            err=True,
        )
        raise typer.Exit(code=1)
    if missing:
        typer.echo(
            f"warning: the checkpoint does not carry {len(missing)} of the "
            f'map\'s groups, the first being "{missing[0]}" — is this the '
            f"checkpoint the scan measured?",
            err=True,
        )
    for group, tensors in held_class_overlaps(groups, map_):
        typer.echo(
            f'warning: {map_path}: group "{group}" folds {len(tensors)} tensors '
            f"of a class the quantizer refuses ({', '.join(tensors)}), and the "
            f"checkpoint holds each by its own name. The plan prices them "
            f"twice: inside the group at its assigned width, and held at the "
            f"convert dtype. The map predates the discovery skip (#204). "
            f"Re-scan to remove the double count (ADR-0029)",
            err=True,
        )
    return CheckpointGroups(bytes=groups, rows=rows)
