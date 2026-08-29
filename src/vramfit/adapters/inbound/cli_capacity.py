"""The ``vramfit capacity`` command: the budget ledger run in reverse.

Reads a packed recipe and subtracts three terms from the card: the
recipe's predicted weight bytes, the runtime overhead, and the
measured vision line when the card claims vision (ADR-0030
decision 3). The command then reports what the remaining KV
headroom buys (#422): the largest context, the sequence count at a
fixed ``--context``, and an image capacity at the measured
``--tokens-per-image`` cost the caller supplies (ADR-0030 decision
4). The solvers live in
[vramfit.domain.capacity][] and search `kv_cache_bytes` itself, so
the readout stays exact on a mixed sliding/global stack.

Examples:
    Read the capacity a recipe buys on its target card:

    ```console
    $ vramfit capacity recipe.json --model-config config.json
    ```

See Also:
    - [vramfit.domain.capacity][]: The inverse solvers.
    - [vramfit.adapters.inbound.cli][]: The forward ``budget``
      command and the composition root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vramfit.adapters.inbound.cli_shape import (
    check_kv_dtype,
    kv_detail,
    parse_size_option,
    resolve_shape,
    resolve_vision_line,
)
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.adapters.outbound.recipe_json import load_recipe
from vramfit.domain.budget import DEFAULT_RUNTIME_OVERHEAD_BYTES, format_size
from vramfit.domain.capacity import (
    image_capacity,
    max_context_tokens,
    max_sequences,
)


def capacity(
    recipe: Annotated[Path, typer.Argument(help="Recipe produced by vramfit plan.")],
    vram: Annotated[
        str | None,
        typer.Option(
            help="Total VRAM, e.g. 24GiB. Default: the recipe's recorded VRAM budget."
        ),
    ] = None,
    context: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Fixed context in tokens — adds the sequence-capacity line.",
        ),
    ] = None,
    kv_dtype: Annotated[
        str, typer.Option(help="KV-cache dtype: fp16, bf16, or fp8.")
    ] = "fp16",
    sequences: Annotated[
        int, typer.Option(min=1, help="Concurrent sequences for the context line.")
    ] = 1,
    tokens_per_image: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Measured image token cost — adds the image-capacity "
            "line. The caller supplies the measurement, and it wins "
            "over the config's claim (ADR-0030 decision 4).",
        ),
    ] = None,
    overhead: Annotated[
        str, typer.Option(help="Runtime overhead reservation.")
    ] = format_size(DEFAULT_RUNTIME_OVERHEAD_BYTES),
    vision_line: Annotated[
        str | None,
        typer.Option(
            help="Measured vision line to subtract when the model card "
            "claims vision, e.g. 1600MiB (ADR-0030). Measure it at the "
            "serve ladder, never from the mmproj file size."
        ),
    ] = None,
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
    """Print the capacity readout for a packed recipe.

    The KV headroom is the card minus the recipe's predicted weight
    bytes, minus ``--overhead``, minus the ``--vision-line`` the
    card's claim licenses. The attention shape comes from
    exactly one source: ``--model-config`` or the manual triple, as
    in ``vramfit budget``. ``--vram`` defaults to the VRAM budget
    the recipe records. The context line solves at ``--sequences``.
    A capacity line prints ``unbounded`` when the KV cache stops
    growing inside the headroom — the reading is then not
    KV-limited. The image line converts the context capacity at the
    measured ``--tokens-per-image`` cost the caller supplies
    (ADR-0030 decision 4). The context and image lines both read per
    the ``--sequences`` split and say so. The headroom subtracts
    ``--vision-line`` only when the card claims vision, and a card
    that claims no vision draws a stated absence instead (ADR-0030
    decision 3).

    Raises:
        typer.BadParameter: If both or neither shape source is
            given, a size/dtype option is malformed, an integer
            option is not positive, or ``--vision-line`` arrives
            with the manual shape triple.
        typer.Exit: With code 1 when the recipe or config cannot be
            read, or the KV headroom is not positive.

    Examples:
        Capacity for a packed recipe on its recorded card:

        ```console
        $ vramfit capacity recipe.json --model-config config.json
        ```
    """
    check_kv_dtype(kv_dtype)
    shape = resolve_shape(model_config, attn_layers, kv_heads, head_dim)
    vision_bytes, vision_note = resolve_vision_line(model_config, vision_line)
    # Reject malformed options before any IO, as budget and plan do.
    vram_option = None if vram is None else parse_size_option(vram, "--vram")
    overhead_bytes = parse_size_option(overhead, "--overhead")
    try:
        recipe_ = load_recipe(recipe)
    except (OSError, ArtifactError) as exc:
        typer.echo(f"error: {recipe}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    vram_bytes = recipe_.plan.vram_budget_bytes if vram_option is None else vram_option
    weight_bytes = recipe_.plan.predicted_total_bytes
    headroom = vram_bytes - weight_bytes - overhead_bytes - vision_bytes

    typer.echo(
        f"attention layers      {len(shape.kv_layers)}  ({kv_detail(shape, kv_dtype)})"
    )
    typer.echo(f"VRAM total            {format_size(vram_bytes)}")
    typer.echo(f"- weights (recipe)    {format_size(weight_bytes)}")
    typer.echo(f"- runtime overhead    {format_size(overhead_bytes)}")
    if vision_note is not None:
        typer.echo(f"vision                {vision_note}")
    elif vision_line is not None:
        # A supplied, licensed line always prints, zero included —
        # the ledger states every vision position (ADR-0030).
        typer.echo(
            f"- vision line         {format_size(vision_bytes)}  (measured, ADR-0030)"
        )
    typer.echo(f"= KV headroom         {format_size(headroom)}")
    if headroom <= 0:
        typer.echo("error: the recipe leaves nothing for KV cache", err=True)
        raise typer.Exit(code=1)

    seq_note = f"{sequences} sequence" + ("s" if sequences != 1 else "")
    tokens = max_context_tokens(shape, headroom, kv_dtype, sequences)
    if tokens is None:
        typer.echo(f"max context           unbounded  ({seq_note})")
    else:
        typer.echo(f"max context           {tokens} tokens  ({seq_note})")

    if context is not None:
        count = max_sequences(shape, headroom, context, kv_dtype)
        # None needs a shape that allocates no KV at all, which no
        # admitted config produces — rendered defensively.
        rendered = "unbounded" if count is None else str(count)
        typer.echo(f"max sequences         {rendered}  (at {context} tokens)")

    if tokens_per_image is not None:
        if tokens is None:
            typer.echo(
                f"image capacity        unbounded"
                f"  ({tokens_per_image} tokens per image, {seq_note})"
            )
        else:
            images = image_capacity(tokens, tokens_per_image)
            typer.echo(
                f"image capacity        {images} images"
                f"  ({tokens_per_image} tokens per image, {seq_note})"
            )
