"""Typer CLI entry point for quantfit.

Exposes the ``quantfit`` console script. ``version``, ``budget``, and
``plan`` are implemented; ``scan`` is a stub until the scan pipeline lands.

Examples:
    Show the installed version:

    ```console
    $ quantfit version
    quantfit 0.1.0
    ```

See Also:
    - [quantfit.budget][]: The math behind the ``budget`` command.
    - [quantfit.solver][]: The solver behind the ``plan`` command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantfit import __version__
from quantfit.artifacts import ArtifactError, SensitivityMap
from quantfit.budget import (
    KV_DTYPE_BYTES,
    Budget,
    ModelShape,
    format_size,
    kv_bytes_per_token,
    kv_cache_bytes,
    parse_size,
)
from quantfit.solver import (
    DEFAULT_FORMAT_OVERHEAD,
    InfeasibleBudgetError,
    PinError,
    solve,
)

app = typer.Typer(
    name="quantfit",
    help="Selective per-layer quantization to fit large models on a single GPU.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the installed quantfit version.

    Examples:
        Command line usage:

        ```console
        $ quantfit version
        quantfit 0.1.0
        ```
    """
    typer.echo(f"quantfit {__version__}")


@app.command()
def scan() -> None:
    """Measure per-layer quantization sensitivity (not yet implemented).

    Raises:
        typer.Exit: Always, with exit code 1, until the scan pipeline lands.

    Examples:
        Command line usage:

        ```console
        $ quantfit scan
        scan is not implemented yet -- see the roadmap in the README.
        ```
    """
    typer.echo("scan is not implemented yet -- see the roadmap in the README.")
    raise typer.Exit(code=1)


def _parse_size_option(value: str, option: str) -> int:
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


@app.command()
def budget(
    vram: Annotated[str, typer.Option(help="Total VRAM, e.g. 24GiB.")] = "24GiB",
    context: Annotated[
        int, typer.Option(min=1, help="Context length in tokens.")
    ] = 16384,
    kv_dtype: Annotated[
        str, typer.Option(help="KV-cache dtype: fp16, bf16, or fp8.")
    ] = "fp16",
    sequences: Annotated[int, typer.Option(min=1, help="Concurrent sequences.")] = 1,
    overhead: Annotated[
        str, typer.Option(help="Runtime overhead reservation.")
    ] = "2GiB",
    model_config: Annotated[
        Path | None,
        typer.Option(help="Model config.json to derive the attention shape from."),
    ] = None,
    attn_layers: Annotated[
        int | None,
        typer.Option(min=1, help="Attention layer count (manual shape)."),
    ] = None,
    kv_heads: Annotated[
        int | None,
        typer.Option(min=1, help="KV heads per layer (manual shape)."),
    ] = None,
    head_dim: Annotated[
        int | None,
        typer.Option(min=1, help="Head dimension (manual shape)."),
    ] = None,
) -> None:
    """Print the VRAM budget breakdown for a model and serving shape.

    The attention shape comes from exactly one source: ``--model-config``,
    or the manual triple ``--attn-layers --kv-heads --head-dim``.

    Raises:
        typer.BadParameter: If both or neither shape source is given, a
            size/dtype option is malformed, or an integer option is not
            positive.
        typer.Exit: With code 1 when the weight budget is not positive.

    Examples:
        Budget for the north-star target from its config:

        ```console
        $ quantfit budget --model-config config.json --vram 24GiB --kv-dtype fp8
        ```
    """
    manual = (attn_layers, kv_heads, head_dim)
    if model_config is not None and any(v is not None for v in manual):
        raise typer.BadParameter(
            "give either --model-config or the manual shape options, not both"
        )
    if model_config is None and any(v is None for v in manual):
        raise typer.BadParameter(
            "give --model-config, or all of --attn-layers, --kv-heads, --head-dim"
        )
    if kv_dtype not in KV_DTYPE_BYTES:
        raise typer.BadParameter(
            f"--kv-dtype: unknown dtype {kv_dtype!r}; "
            f"choose from {sorted(KV_DTYPE_BYTES)}"
        )
    if model_config is not None:
        try:
            shape = ModelShape.from_config_json(model_config)
        except (OSError, ValueError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    else:
        assert attn_layers is not None and kv_heads is not None  # typer-checked
        assert head_dim is not None
        shape = ModelShape.uniform(
            attn_layers=attn_layers, kv_heads=kv_heads, head_dim=head_dim
        )

    ledger = Budget(
        vram_total_bytes=_parse_size_option(vram, "--vram"),
        kv_cache_bytes=kv_cache_bytes(shape, context, kv_dtype, sequences),
        runtime_overhead_bytes=_parse_size_option(overhead, "--overhead"),
    )
    per_token = kv_bytes_per_token(shape, kv_dtype)
    typer.echo(
        f"attention layers      {len(shape.kv_heads_per_layer)}"
        f"  (KV {per_token} bytes/token, {kv_dtype})"
    )
    typer.echo(f"VRAM total            {format_size(ledger.vram_total_bytes)}")
    typer.echo(
        f"- KV cache            {format_size(ledger.kv_cache_bytes)}"
        f"  ({context} tokens x {sequences} seq)"
    )
    typer.echo(f"- runtime overhead    {format_size(ledger.runtime_overhead_bytes)}")
    typer.echo(f"= weight budget       {format_size(ledger.weight_budget_bytes)}")
    if ledger.weight_budget_bytes <= 0:
        typer.echo("error: nothing left for weights", err=True)
        raise typer.Exit(code=1)


@app.command()
def plan(
    sensitivity_map: Annotated[
        Path, typer.Argument(help="Sensitivity map produced by quantfit scan.")
    ],
    vram: Annotated[str, typer.Option(help="Hard VRAM ceiling, e.g. 24GiB.")],
    kv_headroom: Annotated[
        str, typer.Option(help="Reserved for KV cache and runtime.")
    ] = "4GiB",
    pin: Annotated[
        list[str] | None,
        typer.Option(help='Pin groups to a precision: "glob=bits". Repeatable.'),
    ] = None,
    out: Annotated[Path, typer.Option(help="Output recipe path.")] = Path(
        "recipe.json"
    ),
    format_overhead: Annotated[
        float, typer.Option(help="Quantization-format overhead fraction.")
    ] = DEFAULT_FORMAT_OVERHEAD,
) -> None:
    """Solve a sensitivity map into a recipe under a VRAM budget.

    Raises:
        typer.BadParameter: If a ``--pin`` is not of the form
            ``pattern=bits`` or a size option is malformed.
        typer.Exit: With code 1 when the map is invalid, the budget is
            infeasible (the gap is reported), or nothing is left for
            weights.

    Examples:
        Plan with the first layer pinned at 8-bit:

        ```console
        $ quantfit plan sensitivity.json --vram 24GiB --pin "model.layers.0.*=8"
        ```
    """
    pins: dict[str, int] = {}
    for raw in pin or []:
        pattern, sep, bits_text = raw.partition("=")
        if not sep or not pattern or not bits_text.lstrip("-").isdigit():
            raise typer.BadParameter(f'--pin {raw!r}: expected the form "pattern=bits"')
        pins[pattern] = int(bits_text)

    vram_bytes = _parse_size_option(vram, "--vram")
    headroom_bytes = _parse_size_option(kv_headroom, "--kv-headroom")
    weight_budget = vram_bytes - headroom_bytes
    if weight_budget <= 0:
        typer.echo("error: --kv-headroom leaves nothing for weights", err=True)
        raise typer.Exit(code=1)

    try:
        map_ = SensitivityMap.load(sensitivity_map)
    except (OSError, ArtifactError) as exc:
        typer.echo(f"error: {sensitivity_map}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        recipe = solve(
            map_,
            weight_budget_bytes=weight_budget,
            vram_budget_bytes=vram_bytes,
            kv_headroom_bytes=headroom_bytes,
            pins=pins,
            format_overhead=format_overhead,
        )
    except PinError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except InfeasibleBudgetError as exc:
        typer.echo(
            f"error: no recipe fits the {format_size(weight_budget)} weight "
            f"budget; minimum achievable is {format_size(exc.minimum_bytes)} "
            f"({format_size(exc.gap_bytes)} over)",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    recipe.save(out)
    typer.echo(
        f"planned {len(recipe.assignments)} groups: "
        f"{format_size(recipe.plan.predicted_total_bytes)} of "
        f"{format_size(weight_budget)} weight budget, "
        f"predicted damage {recipe.plan.predicted_damage:.4f}, "
        f"{len(recipe.plan.trace)} downgrades -> {out}"
    )
