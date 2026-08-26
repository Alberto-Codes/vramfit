"""Options the ``budget`` and ``capacity`` commands share.

Both commands resolve the attention geometry from exactly one source
— ``--model-config`` through the `ModelShapeSource` port, or the
manual ``--attn-layers --kv-heads --head-dim`` triple — and validate
the same ``--kv-dtype`` set. ``plan`` shares the size-option rule.
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

from vramfit.adapters.outbound.hf_config import HfConfigFile
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


def kv_detail(shape: ModelShape, kv_dtype: str) -> str:
    """Render the KV growth and window-pool note both commands print.

    Args:
        shape: The resolved attention shape.
        kv_dtype: The validated KV dtype.

    Returns:
        A note like ``"KV grows 40960 bytes/token, fp16, + 800.00 MiB
        window pool per sequence"`` — the pool clause only when the
        shape has sliding layers (#421).
    """
    per_token = kv_growth_bytes_per_token(shape, kv_dtype)
    pool = kv_window_pool_bytes(shape, kv_dtype)
    detail = f"KV grows {per_token} bytes/token, {kv_dtype}"
    if pool:
        detail += f", + {format_size(pool)} window pool per sequence"
    return detail
