"""Pre-flight refusals for the ``vramfit refine`` command.

Everything `refine` refuses before it reads a whole file. The command
keeps the wiring and the reporting; the checks live here, so the
composition root stays readable and a new refusal has an obvious
home.

These refusals are remembered, not enforced: nothing in the code
stops a later one from landing below the content-identity reads.
`_check_recipe_resolves` enumerates the domain refusals a pass makes
before its first pack, so the next reader checks a stated set rather
than re-deriving one from the call graph.

Inside the pass the ordering is structural instead.
[vramfit.adapters.inbound.refine_loop][] packs and judges in one
call and measures in another, so a budget refusal cannot reach the
meter — there is no ordering to remember there.

Examples:
    Refuse a checkout missing a tool:

    ```python
    from pathlib import Path

    import typer

    from vramfit.adapters.inbound.cli_refine_preflight import _check_toolchain
    from vramfit.adapters.inbound.llama_cpp_layout import LlamaCppTools

    try:
        _check_toolchain(LlamaCppTools.under(Path("~/llama.cpp")))
    except typer.Exit:
        print("the checkout misses a tool the pass runs")
    ```

See Also:
    - [vramfit.adapters.inbound.cli_refine][]: The caller.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import typer

from vramfit.adapters.inbound.llama_cpp_layout import LlamaCppTools
from vramfit.domain.errors import VramfitError
from vramfit.domain.model import Recipe, SensitivityMap
from vramfit.domain.refinement import neighbours


def _halt(message: str) -> None:
    """Report a failure and exit 1.

    Args:
        message: The operator-facing line.

    Raises:
        Exit: Always, with code 1.
    """
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _check_toolchain(tools: LlamaCppTools) -> None:
    """Refuse a checkout missing a tool the pass runs.

    The pass drives three tools and reaches the last one only after
    the convert stage and the control pack. On the rented card this
    stage is priced against, that is about 20 minutes and a
    full-size f16 base GGUF spent to learn a path does not exist, so
    every tool is checked before any of them runs.

    Args:
        tools: The checkout's tools, as the wiring will run them.

    Raises:
        Exit: With code 1 when a tool is missing or cannot execute.
    """
    if not tools.convert_script.is_file():
        _halt(
            f"--llama-cpp: {tools.convert_script} does not exist — "
            "build the tools first"
        )
    for binary in (tools.quantize_bin, tools.perplexity_bin):
        if not binary.is_file():
            _halt(f"--llama-cpp: {binary} does not exist — build the tools first")
        if not os.access(binary, os.X_OK):
            _halt(f"--llama-cpp: {binary} is not executable")


def _check_destination(label: str, path: Path) -> None:
    """Refuse a destination the pass could not write when it finishes.

    The sidecar is the pass's only artifact and it is written last,
    after every pack and every measurement, so every reason the write
    could fail is checked before the first tool runs. The write
    replaces a temporary file onto ``path``, which refuses a path
    that names a directory as surely as one whose parent is missing.
    The command offers both ``--out-dir`` and ``--out``, so an
    ``--out`` naming an existing directory is the reachable mistake.

    Args:
        label: The option that named the path, for the message.
        path: The file the pass will write.

    Raises:
        Exit: With code 1 when the path names a directory, or its
            parent is missing or refuses a write.
    """
    if path.is_dir():
        _halt(f"{label}: {path} is a directory, so the pass cannot write it")
    parent = path.parent
    if not parent.is_dir():
        _halt(f"{label}: directory {parent} does not exist")
    if not os.access(parent, os.W_OK):
        _halt(f"{label}: directory {parent} is not writable")


def _check_recipe_resolves(
    recipe: Recipe, map_: SensitivityMap, row_widths: Mapping[str, int]
) -> None:
    """Refuse a recipe this map cannot resolve, before any file is read.

    `neighbours` is the whole set of domain refusals a pass makes
    before its first pack, so running it here is not an approximation
    of that set — it is that set. It refuses through three paths:

    - protection resolution, through
      `vramfit.domain.protection.expand_protections`, which raises
      `ProtectionError` for a pattern that matches no tensor, matches
      a single-tensor group, or hits a group with no
      ``tensor_bytes``.
    - group pricing, through
      `vramfit.domain.sizes.measured_width`, which raises
      `SizeSourceError` for a group rooted outside
      ``CHECKPOINT_ROOTS``. `_resolve_row_widths` does not cover it,
      because `consults_row_width` filters the set it checks.
    - pin resolution, through
      `vramfit.domain.pins.pinned_group_names`, which refuses
      nothing — an unresolvable pin declines rather than halting.

    The result is discarded. `run_pass` enumerates again, and the
    enumeration is pure, so the second call answers the same.

    Args:
        recipe: The recipe to refine.
        map_: The map that priced it.
        row_widths: Elements per row per group.

    Raises:
        Exit: With code 1 when the recipe does not resolve.
    """
    try:
        neighbours(recipe, map_, row_widths)
    except VramfitError as error:
        _halt(str(error))


def _check_frame_labels(runtime_build: str, hardware: str) -> None:
    """Refuse a frame label the operator left empty.

    `MeasurementFrame` refuses these too, and keeps doing so for every
    other caller. This is the cheap copy: the frame is built after the
    content-identity reads, so without it an unset shell variable
    costs a full read of the reference logits before refusing.

    Args:
        runtime_build: The runtime binary's build identity.
        hardware: The card the pass runs on.

    Raises:
        Exit: With code 1 when either label is empty.
    """
    for label, value in (
        ("--runtime-build", runtime_build),
        ("--hardware", hardware),
    ):
        if not value:
            _halt(f"{label}: must not be empty — it names the measurement frame")


def _check_input_files(
    base_logits: Path, eval_text: Path, imatrix: Path | None
) -> None:
    """Refuse an input file the pass would only miss on the card.

    Args:
        base_logits: Reference logits every arm measures against.
        eval_text: The evaluation corpus.
        imatrix: The importance matrix, or None for an unassisted
            pass.

    Raises:
        Exit: With code 1 when a named file does not exist.
    """
    named = (("--base-logits", base_logits), ("--eval-text", eval_text))
    if imatrix is not None:
        named = (*named, ("--imatrix", imatrix))
    for label, path in named:
        if not path.is_file():
            _halt(f"{label}: {path} does not exist")


def _make_arm_dir(out_dir: Path) -> None:
    """Create the directory the arm packs go in.

    It is created before the destinations are checked, so an ``--out``
    inside it resolves against a parent that exists and an ``--out``
    naming it refuses as the directory it now is.

    Args:
        out_dir: The arm directory.

    Raises:
        Exit: With code 1 when the directory cannot be created.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _halt(f"--out-dir: {error}")
