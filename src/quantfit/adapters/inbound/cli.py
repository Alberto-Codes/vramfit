"""Typer CLI: the inbound (driving) adapter and composition root.

Exposes the ``quantfit`` console script. ``version``, ``budget``,
``plan``, ``scan``, and ``pack`` are implemented — the scan and pack
command bodies live in [quantfit.adapters.inbound.cli_scan][] and
[quantfit.adapters.inbound.cli_pack][] to keep this module under the
size cap. The CLI wires outbound adapters to the pure domain, typing
them against the ports so the seams stay explicit. Every IO boundary —
artifact and config reads, checkpoint and artifact writes, and model
loading — converts failures to a clean ``error:`` line and a nonzero
exit. Domain failures surface through one catch of the
`QuantfitError` root (ADR-0011), whose messages print verbatim.
Malformed options — including a NaN or infinite overhead — are
usage errors, rejected before any work starts. ``plan`` records
its ``--runtime`` in the recipe and reports what the runtime
filter drops (ADR-0013).

Examples:
    Show the installed version:

    ```console
    $ quantfit version
    quantfit 0.1.0
    ```

See Also:
    - [quantfit.domain.budget][]: The math behind the ``budget`` command.
    - [quantfit.domain.solver][]: The solver behind the ``plan`` command.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer

from quantfit import __version__
from quantfit.adapters.inbound import cli_pack, cli_scan
from quantfit.adapters.outbound.hf_config import HfConfigFile
from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.adapters.outbound.recipe_json import JsonRecipeFile
from quantfit.adapters.outbound.sensitivity_map_json import JsonSensitivityMapFile
from quantfit.domain.budget import (
    DEFAULT_RUNTIME_OVERHEAD_BYTES,
    KV_DTYPE_BYTES,
    Budget,
    ModelShape,
    format_size,
    kv_bytes_per_token,
    kv_cache_bytes,
    parse_size,
)
from quantfit.domain.errors import QuantfitError
from quantfit.domain.runtime import LLAMA_CPP, RUNTIME_CAPABILITIES
from quantfit.domain.solver import DEFAULT_FORMAT_OVERHEAD, solve
from quantfit.ports.outbound import (
    ModelShapeSource,
    RecipeSink,
    SensitivityMapSource,
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


app.command(name="scan")(cli_scan.scan)
app.command(name="pack")(cli_pack.pack)


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
    ] = format_size(DEFAULT_RUNTIME_OVERHEAD_BYTES),
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

    The attention shape comes from exactly one source: ``--model-config``
    (read through the `ModelShapeSource` port), or the manual triple
    ``--attn-layers --kv-heads --head-dim``. The ``--overhead`` default
    derives from ``quantfit.domain.budget.DEFAULT_RUNTIME_OVERHEAD_BYTES``.

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
    if kv_dtype not in KV_DTYPE_BYTES:
        raise typer.BadParameter(
            f"--kv-dtype: unknown dtype {kv_dtype!r} — "
            f"choose from {sorted(KV_DTYPE_BYTES)}"
        )
    if model_config is not None:
        shape_source: ModelShapeSource = HfConfigFile(model_config)
        try:
            shape = shape_source.load()
        except (OSError, ValueError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    else:
        if attn_layers is None or kv_heads is None or head_dim is None:
            raise typer.BadParameter(
                "give --model-config, or all of --attn-layers, --kv-heads, --head-dim"
            )
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
        float,
        typer.Option(min=0.0, help="Quantization-format overhead fraction."),
    ] = DEFAULT_FORMAT_OVERHEAD,
    runtime: Annotated[
        str,
        typer.Option(help="Target runtime the recipe is planned for."),
    ] = LLAMA_CPP,
) -> None:
    """Solve a sensitivity map into a recipe under a VRAM budget.

    The candidate precisions come from the map, filtered to what
    ``--runtime`` (default llama.cpp) can serve (ADR-0013) — the
    recipe records the runtime for the pack step, and the command
    reports any scanned precisions the runtime dropped. Solver
    rejections (bad pins, an infeasible budget, a runtime serving
    nothing) surface through one catch of the `QuantfitError` root.
    The solver's own messages carry the details, including the
    infeasibility gap.

    Raises:
        typer.BadParameter: If a ``--pin`` is not of the form
            ``pattern=bits`` with positive bits, a size option is
            malformed, ``--format-overhead`` is negative, NaN, or
            infinite, or ``--runtime`` is not in the capability
            table.
        typer.Exit: With code 1 when the map is invalid, the recipe
            cannot be written, the solver rejects the plan, or nothing
            is left for weights.

    Examples:
        Plan with the first layer pinned at 8-bit:

        ```console
        $ quantfit plan sensitivity.json --vram 24GiB --pin "model.layers.0.*=8"
        ```
    """
    # typer's min=0.0 lets NaN and inf through (nan < 0.0 is False),
    # and the solver's own guard is a plain ValueError — reject both
    # here as the usage error they are.
    if not math.isfinite(format_overhead):
        raise typer.BadParameter(
            f"--format-overhead: must be finite, got {format_overhead}"
        )
    if runtime not in RUNTIME_CAPABILITIES:
        raise typer.BadParameter(
            f"--runtime: unknown runtime {runtime!r} — "
            f"choose from {sorted(RUNTIME_CAPABILITIES)}"
        )
    pins: dict[str, int] = {}
    for raw in pin or []:
        pattern, sep, bits_text = raw.partition("=")
        try:
            bits = int(bits_text)
        except ValueError:
            bits = 0
        if not sep or not pattern or bits <= 0:
            raise typer.BadParameter(
                f'--pin {raw!r}: expected the form "pattern=bits" with positive bits'
            )
        # A repeated pattern must keep its *last* position so later pins
        # override earlier ones, as documented.
        pins.pop(pattern, None)
        pins[pattern] = bits

    vram_bytes = _parse_size_option(vram, "--vram")
    headroom_bytes = _parse_size_option(kv_headroom, "--kv-headroom")
    weight_budget = vram_bytes - headroom_bytes
    if weight_budget <= 0:
        typer.echo("error: --kv-headroom leaves nothing for weights", err=True)
        raise typer.Exit(code=1)

    map_source: SensitivityMapSource = JsonSensitivityMapFile(sensitivity_map)
    try:
        map_ = map_source.load()
    except (OSError, ArtifactError) as exc:
        typer.echo(f"error: {sensitivity_map}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # The runtime filter is a silent narrowing otherwise — say what
    # the target runtime removed from the scanned candidate set.
    capability = RUNTIME_CAPABILITIES[runtime]
    dropped = [p for p in map_.scan.precisions if p not in capability]
    if dropped:
        kept = [p for p in map_.scan.precisions if p in capability]
        typer.echo(
            f"runtime {runtime} serves {kept} of the scanned "
            f"{list(map_.scan.precisions)} — candidates {dropped} dropped"
        )

    try:
        recipe = solve(
            map_,
            weight_budget_bytes=weight_budget,
            vram_budget_bytes=vram_bytes,
            kv_headroom_bytes=headroom_bytes,
            pins=pins,
            format_overhead=format_overhead,
            runtime=runtime,
        )
    except QuantfitError as exc:
        # One honest catch for the root (ADR-0011): the solver's
        # errors carry their own user-facing messages.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    sink: RecipeSink = JsonRecipeFile(out)
    try:
        sink.save(recipe)
    except OSError as exc:
        typer.echo(f"error: {out}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"planned {len(recipe.assignments)} groups for {runtime}: "
        f"{format_size(recipe.plan.predicted_total_bytes)} of "
        f"{format_size(weight_budget)} weight budget, "
        f"predicted damage {recipe.plan.predicted_damage:.4f}, "
        f"{len(recipe.plan.trace)} downgrades -> {out}"
    )
