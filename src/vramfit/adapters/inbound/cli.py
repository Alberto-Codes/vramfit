"""Typer CLI: the inbound (driving) adapter and composition root.

Exposes the ``vramfit`` console script. ``version``, ``budget``,
``plan``, ``scan``, ``pack``, ``validate``, and ``capacity`` are
implemented — the scan, pack, validate, and capacity command bodies
live in [vramfit.adapters.inbound.cli_scan][],
[vramfit.adapters.inbound.cli_pack][],
[vramfit.adapters.inbound.cli_validate][], and
[vramfit.adapters.inbound.cli_capacity][] to keep this module under
the size cap. ``budget`` reports KV growth per token, plus the
window pool on a mixed sliding/global stack (#421). ``capacity``
runs the same ledger in reverse from a packed recipe (#422).
The CLI wires outbound adapters to the pure domain, typing
them against the ports so the seams stay explicit. Every IO boundary —
artifact and config reads, checkpoint and artifact writes, and model
loading — converts failures to a clean ``error:`` line and a nonzero
exit. Domain failures surface through one catch of the
`VramfitError` root (ADR-0011), whose messages print verbatim.
Malformed options — including a NaN or infinite overhead — are
usage errors, rejected before any work starts. ``plan`` records
its ``--runtime`` in the recipe and reports what the runtime
filter drops (ADR-0013), its ``--protect`` rules resolve to
per-tensor floors with a warning for a dead rule and for each
dropped no-op pair (ADR-0022, issue #59), and its
``--exclude-imatrix`` globs mark protected tensors with a warning
when a glob overreaches the protected set (ADR-0023). Its
``--checkpoint`` reads the model's safetensors headers for a size
source independent of the map, so a partial map no longer defines
the model (ADR-0029) — that wiring lives in
[vramfit.adapters.inbound.cli_plan_sizes][].
``--format-overhead`` defaults per size
model (ADR-0014): the residual when the runtime has an
effective-bits table, the scalar otherwise. An artifact field a
reader does not know draws a ``warning:`` line as well: the app
callback installs the reporter the outbound readers call (#261,
ADR-0013). The protection warnings live in
[vramfit.adapters.inbound.cli_protection_warnings][].

Examples:
    Show the installed version:

    ```console
    $ vramfit version
    vramfit 0.1.0
    ```

See Also:
    - [vramfit.domain.budget][]: The math behind the ``budget`` command.
    - [vramfit.domain.solver][]: The solver behind the ``plan`` command.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer

from vramfit import __version__
from vramfit.adapters.inbound import cli_capacity, cli_pack, cli_scan, cli_validate
from vramfit.adapters.inbound.cli_plan_sizes import discovered_bytes
from vramfit.adapters.inbound.cli_protection_warnings import warn_protection_gaps
from vramfit.adapters.inbound.cli_shape import (
    check_kv_dtype,
    kv_detail,
    parse_size_option,
    resolve_shape,
)
from vramfit.adapters.outbound.json_common import (
    ArtifactError,
    set_unknown_field_reporter,
)
from vramfit.adapters.outbound.recipe_json import JsonRecipeFile
from vramfit.adapters.outbound.sensitivity_map_json import JsonSensitivityMapFile
from vramfit.domain.budget import (
    DEFAULT_RUNTIME_OVERHEAD_BYTES,
    Budget,
    format_size,
    kv_cache_bytes,
)
from vramfit.domain.errors import VramfitError
from vramfit.domain.runtime import LLAMA_CPP, RUNTIME_CAPABILITIES
from vramfit.domain.solver import (
    DEFAULT_FORMAT_OVERHEAD,
    DEFAULT_RESIDUAL_OVERHEAD,
    solve,
)
from vramfit.ports.outbound import (
    RecipeSink,
    SensitivityMapSource,
)

app = typer.Typer(
    name="vramfit",
    help="Selective per-layer quantization to fit large models on a single GPU.",
    no_args_is_help=True,
)


def _echo_unknown_artifact_field(message: str) -> None:
    """Print one unknown-field report on the human channel.

    The line matches the run-log failure line (ADR-0011 decision 2).
    The JSON path the report carries names the field. The stdlib
    rendering would prefix a placeholder origin instead.

    Args:
        message: The report text, already carrying its JSON path.
    """
    typer.echo(f"warning: {message}", err=True)


@app.callback()
def main() -> None:
    """Wire the process-wide reporting every command shares.

    The artifact readers report a field they do not know through the
    stdlib `warnings` module (#261, ADR-0013). This runs before any
    command and routes those reports to the human channel. It keeps no
    token, because the process exits with the command.
    """
    set_unknown_field_reporter(_echo_unknown_artifact_field)


@app.command()
def version() -> None:
    """Print the installed vramfit version.

    Examples:
        Command line usage:

        ```console
        $ vramfit version
        vramfit 0.1.0
        ```
    """
    typer.echo(f"vramfit {__version__}")


app.command(name="scan")(cli_scan.scan)
app.command(name="pack")(cli_pack.pack)
app.command(name="validate")(cli_validate.validate)
app.command(name="capacity")(cli_capacity.capacity)


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
    ``--attn-layers --kv-heads --head-dim`` — the shared resolution
    lives in [vramfit.adapters.inbound.cli_shape][], which the
    ``capacity`` command uses too. The ``--overhead`` default
    derives from ``vramfit.domain.budget.DEFAULT_RUNTIME_OVERHEAD_BYTES``.
    The first output line reports KV growth per context token, plus
    the saturated per-sequence window pool when the shape has sliding
    layers (#421). The KV-cache line sums both terms at ``--context``
    and ``--sequences``.

    Raises:
        typer.BadParameter: If both or neither shape source is given, a
            size/dtype option is malformed, or an integer option is not
            positive.
        typer.Exit: With code 1 when the weight budget is not positive.

    Examples:
        Budget for the north-star target from its config:

        ```console
        $ vramfit budget --model-config config.json --vram 24GiB --kv-dtype fp8
        ```
    """
    check_kv_dtype(kv_dtype)
    shape = resolve_shape(model_config, attn_layers, kv_heads, head_dim)

    ledger = Budget(
        vram_total_bytes=parse_size_option(vram, "--vram"),
        kv_cache_bytes=kv_cache_bytes(shape, context, kv_dtype, sequences),
        runtime_overhead_bytes=parse_size_option(overhead, "--overhead"),
    )
    detail = kv_detail(shape, kv_dtype)
    typer.echo(f"attention layers      {len(shape.kv_layers)}  ({detail})")
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


def _parse_rules(raw_rules: list[str] | None, option: str) -> dict[str, int]:
    """Parse repeatable ``pattern=bits`` options into an ordered mapping.

    Serves ``--pin`` and ``--protect``, which share the rule form. A
    repeated pattern keeps its *last* position, so later rules
    override earlier ones, as documented.

    Args:
        raw_rules: The raw option values, e.g. ``["g*=8"]``.
        option: The option name, for the error message.

    Returns:
        Ordered mapping of pattern to bits.

    Raises:
        typer.BadParameter: If a rule is not ``pattern=bits`` with
            positive bits.
    """
    rules: dict[str, int] = {}
    for raw in raw_rules or []:
        pattern, sep, bits_text = raw.partition("=")
        try:
            bits = int(bits_text)
        except ValueError:
            bits = 0
        if not sep or not pattern or bits <= 0:
            raise typer.BadParameter(
                f'{option} {raw!r}: expected the form "pattern=bits" with positive bits'
            )
        rules.pop(pattern, None)
        rules[pattern] = bits
    return rules


@app.command()
def plan(
    sensitivity_map: Annotated[
        Path, typer.Argument(help="Sensitivity map produced by vramfit scan.")
    ],
    vram: Annotated[str, typer.Option(help="Hard VRAM ceiling, e.g. 24GiB.")],
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            help="Checkpoint the map was scanned from. Its safetensors "
            "headers price every group, so a map covering part of the "
            "model no longer defines it (ADR-0029)."
        ),
    ] = None,
    kv_headroom: Annotated[
        str, typer.Option(help="Reserved for KV cache and runtime.")
    ] = "4GiB",
    pin: Annotated[
        list[str] | None,
        typer.Option(help='Pin groups to a precision: "glob=bits". Repeatable.'),
    ] = None,
    protect: Annotated[
        list[str] | None,
        typer.Option(
            help="Hold tensors at a precision floor inside their groups: "
            '"glob=bits". Repeatable (ADR-0022).'
        ),
    ] = None,
    exclude_imatrix: Annotated[
        list[str] | None,
        typer.Option(
            help="Quantize matched protected tensors without their imatrix "
            'rows: "glob". Repeatable (ADR-0023). The fit-collapse remedy '
            "that keeps the promotion."
        ),
    ] = None,
    out: Annotated[Path, typer.Option(help="Output recipe path.")] = Path(
        "recipe.json"
    ),
    format_overhead: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Overhead fraction on top of the size model. Default: "
            f"{DEFAULT_RESIDUAL_OVERHEAD} when the runtime has an "
            f"effective-bits table, {DEFAULT_FORMAT_OVERHEAD} otherwise.",
        ),
    ] = None,
    runtime: Annotated[
        str,
        typer.Option(help="Target runtime the recipe is planned for."),
    ] = LLAMA_CPP,
) -> None:
    """Solve a sensitivity map into a recipe under a VRAM budget.

    The candidate precisions come from the map, filtered to what
    ``--runtime`` (default llama.cpp) can serve (ADR-0013) — the
    recipe records the runtime for the pack step, and the command
    reports any scanned precisions the runtime dropped. Sizes are
    predicted at the runtime's effective bits when it has a table
    (ADR-0014), and an omitted ``--format-overhead`` resolves to
    the size model's default — the recipe records the resolved
    value. Solver rejections (bad pins, an infeasible budget, a
    runtime serving nothing) surface through one catch of the
    `VramfitError` root. The solver's own messages carry the
    details, including the infeasibility gap.

    A ``--protect`` rule holds the matched tensors at a precision
    floor inside their groups (ADR-0022), priced by size only. A
    pair reaches the recipe only where the floor exceeds the
    group's assignment. A rule or a matched tensor that changes
    nothing draws a warning, never silence — the recipe drops a
    no-op pair, which would otherwise falsely fail the
    reconstruction check (issue #59).

    ``--checkpoint`` reads the model's safetensors headers for a size
    source independent of the map (ADR-0029). A group the checkpoint
    holds and the map does not measure holds at reference precision,
    and the recipe assigns it there. Without the option the map
    defines the model, and the command says so.

    A map field the reader does not know draws a warning too, and the
    plan continues (#261). The warning names the JSON path and states
    that a save drops the field.

    An ``--exclude-imatrix`` glob marks matched protected tensors to
    quantize without their imatrix rows (ADR-0023) — the remedy when
    the reconstruction check names a collapsed tensor. The pattern
    must land inside the protected set: the solver refuses a miss,
    and a glob that also matches unprotected tensors draws a
    warning naming what it did not cover.

    Raises:
        typer.BadParameter: If a ``--pin`` or ``--protect`` is not of
            the form ``pattern=bits`` with positive bits, a size
            option is malformed (the shared size rule lives in
            [vramfit.adapters.inbound.cli_shape][]),
            ``--format-overhead`` is negative, NaN, or infinite, or
            ``--runtime`` is not in the capability table.
        typer.Exit: With code 1 when the map is invalid, the recipe
            cannot be written, the solver rejects the plan, or nothing
            is left for weights.

    Examples:
        Plan with the first layer pinned at 8-bit:

        ```console
        $ vramfit plan sensitivity.json --vram 24GiB --pin "model.layers.0.*=8"
        ```
    """
    # typer's min=0.0 lets NaN and inf through (nan < 0.0 is False),
    # and the solver's own guard is a plain ValueError — reject both
    # here as the usage error they are.
    if format_overhead is not None and not math.isfinite(format_overhead):
        raise typer.BadParameter(
            f"--format-overhead: must be finite, got {format_overhead}"
        )
    if runtime not in RUNTIME_CAPABILITIES:
        raise typer.BadParameter(
            f"--runtime: unknown runtime {runtime!r} — "
            f"choose from {sorted(RUNTIME_CAPABILITIES)}"
        )
    pins = _parse_rules(pin, "--pin")
    protections = _parse_rules(protect, "--protect")
    exclusions = tuple(exclude_imatrix or [])

    vram_bytes = parse_size_option(vram, "--vram")
    headroom_bytes = parse_size_option(kv_headroom, "--kv-headroom")
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

    sizes = discovered_bytes(checkpoint, map_)

    try:
        recipe = solve(
            map_,
            weight_budget_bytes=weight_budget,
            vram_budget_bytes=vram_bytes,
            kv_headroom_bytes=headroom_bytes,
            pins=pins,
            protections=protections,
            imatrix_exclusions=exclusions,
            format_overhead=format_overhead,
            runtime=runtime,
            discovered_bytes=sizes,
        )
    except VramfitError as exc:
        # One honest catch for the root (ADR-0011): the solver's
        # errors carry their own user-facing messages.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if protections:
        state = {a.group: a.bits for a in recipe.assignments}
        warn_protection_gaps(protections, exclusions, map_, state, runtime)

    sink: RecipeSink = JsonRecipeFile(out)
    try:
        sink.save(recipe)
    except OSError as exc:
        typer.echo(f"error: {out}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    protected_note = (
        f", {len(recipe.protected_tensors)} protected tensors"
        if recipe.protected_tensors
        else ""
    )
    excluded_count = sum(1 for p in recipe.protected_tensors if p.exclude_imatrix)
    if excluded_count:
        protected_note += f" ({excluded_count} imatrix-excluded)"
    typer.echo(
        f"planned {len(recipe.assignments)} groups for {runtime}: "
        f"{format_size(recipe.plan.predicted_total_bytes)} of "
        f"{format_size(weight_budget)} weight budget, "
        f"predicted damage {recipe.plan.predicted_damage:.4f}, "
        f"{len(recipe.plan.trace)} downgrades{protected_note} -> {out}"
    )
