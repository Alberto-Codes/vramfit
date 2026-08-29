"""Options the ``budget`` and ``capacity`` commands share.

Both commands resolve the attention geometry from exactly one source
— ``--model-config`` through the `ModelShapeSource` port, or the
manual ``--attn-layers --kv-heads --head-dim`` triple — and validate
the same ``--kv-dtype`` set. Both resolve ``--vision-line`` the same
way: the card's vision claim licenses the subtraction (ADR-0030
decision 3). ``plan`` shares the size-option rule.
Each rule lives here once.

Examples:
    Resolve a manual shape the way both commands do:

    ```python
    from vramfit.adapters.inbound.cli_shape import resolve_shape

    shape = resolve_shape(None, attn_layers=49, kv_heads=8, head_dim=128)
    ```

See Also:
    - [vramfit.adapters.inbound.cli][]: The ``budget`` command.
    - [vramfit.adapters.inbound.cli_capacity][]: The ``capacity``
      command.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vramfit.adapters.outbound.hf_config import HfConfigFile, config_claims_vision
from vramfit.domain.budget import (
    KV_DTYPE_BYTES,
    ModelShape,
    format_size,
    kv_growth_bytes_per_token,
    kv_window_pool_bytes,
    parse_size,
)
from vramfit.ports.outbound import ModelShapeSource


def parse_size_option(value: str, option: str) -> int:
    """Parse a size CLI option, converting errors to usage errors.

    Args:
        value: The raw option value, e.g. ``"24GiB"``.
        option: The option name, for the error message.

    Returns:
        The size in bytes.

    Raises:
        typer.BadParameter: If the value is not a recognizable size.
    """
    try:
        return parse_size(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{option}: {exc}") from exc


def check_kv_dtype(kv_dtype: str) -> None:
    """Refuse a ``--kv-dtype`` outside the KV dtype table.

    Args:
        kv_dtype: The raw option value.

    Raises:
        typer.BadParameter: If the dtype is not in `KV_DTYPE_BYTES`.
    """
    if kv_dtype not in KV_DTYPE_BYTES:
        raise typer.BadParameter(
            f"--kv-dtype: unknown dtype {kv_dtype!r} — "
            f"choose from {sorted(KV_DTYPE_BYTES)}"
        )


def resolve_shape(
    model_config: Path | None,
    attn_layers: int | None,
    kv_heads: int | None,
    head_dim: int | None,
) -> ModelShape:
    """Resolve the attention shape from exactly one source.

    Args:
        model_config: The ``--model-config`` path, or None.
        attn_layers: The ``--attn-layers`` value, or None.
        kv_heads: The ``--kv-heads`` value, or None.
        head_dim: The ``--head-dim`` value, or None.

    Returns:
        The resolved shape — uniform for the manual triple.

    Raises:
        typer.BadParameter: If both or neither source is given.
        typer.Exit: With code 1 when the config cannot be read or
            parsed.
    """
    manual = (attn_layers, kv_heads, head_dim)
    if model_config is not None and any(v is not None for v in manual):
        raise typer.BadParameter(
            "give either --model-config or the manual shape options, not both"
        )
    if model_config is not None:
        shape_source: ModelShapeSource = HfConfigFile(model_config)
        try:
            return shape_source.load()
        except (OSError, ValueError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    if attn_layers is None or kv_heads is None or head_dim is None:
        raise typer.BadParameter(
            "give --model-config, or all of --attn-layers, --kv-heads, --head-dim"
        )
    return ModelShape.uniform(
        attn_layers=attn_layers, kv_heads=kv_heads, head_dim=head_dim
    )


def resolve_vision_line(
    model_config: Path | None, vision_line: str | None
) -> tuple[int, str | None]:
    """Resolve the vision line the ledger subtracts (ADR-0030).

    The model card's claim licenses the subtraction: the ledger
    subtracts the measured vision line only when the card claims
    vision. A card that claims no vision subtracts nothing and
    states the absence (decision 3) — a supplied ``--vision-line``
    does not apply there, and the note says so. The manual shape
    triple carries no card, so it admits no ``--vision-line``.

    ADR-0030 leaves open whether the budget warns or refuses on a
    vision-claiming card with no measured line. This resolver
    subtracts nothing there and states the gap, deciding neither.

    Args:
        model_config: The ``--model-config`` path, or None.
        vision_line: The ``--vision-line`` size string, or None.

    Returns:
        The bytes to subtract, and a note detail for the ledger's
        ``vision`` label — None when the card claims vision and the
        line subtracts, or when no card exists.

    Raises:
        typer.BadParameter: If ``--vision-line`` is malformed, or
            arrives with the manual shape triple.
        typer.Exit: With code 1 when the config cannot be read.
    """
    # Reject a malformed size before any IO, as the sibling size
    # options do.
    line_bytes = (
        None if vision_line is None else parse_size_option(vision_line, "--vision-line")
    )
    if model_config is None:
        if line_bytes is not None:
            raise typer.BadParameter(
                "--vision-line: needs --model-config — the card's vision "
                "claim licenses the subtraction (ADR-0030)"
            )
        return 0, None
    try:
        claims = config_claims_vision(model_config)
    except (OSError, ValueError) as exc:
        # `resolve_shape` already read this file through the same
        # loader, so this fires only when the file changes between
        # the two reads.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not claims:
        if line_bytes is not None:
            return 0, (
                "none claimed — nothing subtracted, --vision-line does "
                "not apply (ADR-0030)"
            )
        return 0, "none claimed — nothing subtracted"
    if line_bytes is None:
        return 0, "claimed — no --vision-line supplied, nothing subtracted"
    return line_bytes, None


def kv_detail(shape: ModelShape, kv_dtype: str) -> str:
    """Render the KV growth and window-pool note both commands print.

    Args:
        shape: The resolved attention shape.
        kv_dtype: The validated KV dtype.

    Returns:
        A note like ``"KV grows 81920 bytes/token, fp16, + 1.17 GiB
        window pool per sequence"`` — the pool clause only when the
        shape has sliding layers (#421).
    """
    per_token = kv_growth_bytes_per_token(shape, kv_dtype)
    pool = kv_window_pool_bytes(shape, kv_dtype)
    detail = f"KV grows {per_token} bytes/token, {kv_dtype}"
    if pool:
        detail += f", + {format_size(pool)} window pool per sequence"
    return detail
