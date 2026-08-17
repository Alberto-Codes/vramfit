"""Override matching: refuse a pattern no base-GGUF tensor carries.

``llama-quantize`` compiles each ``--tensor-type`` pattern once and
searches it against every tensor name. A pattern that matches nothing
changes no type. The tool then exits 0 and reports nothing, so the
artifact ships at the base type while the recipe claims a mix. That is
the failure ADR-0028 refuses one level over, where the quantizer at
least warns.

No upstream warning covers this one. Verified against llama.cpp
``src/llama-quant.cpp`` at commit ``3653e6d6d`` (b10326, the pinned
instrument) and at ``e9fa0781f``: the match loop runs per tensor, it
never records which patterns went unused, and the file holds no
unused-pattern report. The sibling ``--override-tensor`` runtime flag
reports nothing either. So the pack step reads the base GGUF's tensor
names itself and holds every override against them, before the
quantizer runs for minutes.

The check replicates the tool's matching and not its priority
resolution. ``tools/quantize/quantize.cpp:332`` lower-cases a pattern
before it compiles, and ``src/llama-quant.cpp:694`` searches the name
rather than anchoring it. A pattern that matches a tensor an earlier
override also claims still passes here. The first matching pattern
wins upstream, and re-deriving that order would make this module a
second source of truth for it (#190).

gguf-py rides the scan extra and the pack extra includes it, so the
import defers to the first read. ``vramfit pack --help`` keeps working
on a base install (ADR-0005).

Examples:
    Hold a recipe's overrides against the file the quantizer reads:

    ```python
    check_overrides_match(overrides, Path("model-f16.gguf"))
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: the caller, which runs
      this before it builds the quantizer command.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import TypeOverride


def _load_gguf() -> Any:
    """Import gguf-py on first use, naming the extra when absent.

    Returns:
        The imported ``gguf`` module.

    Raises:
        PackError: If gguf-py is not installed.
    """
    try:
        import gguf  # noqa: PLC0415 - lazy: base install has no gguf-py (ADR-0005)
    except ImportError as exc:
        raise PackError(
            "holding a recipe's overrides against the base GGUF needs "
            "gguf-py — install the pack extra: uv sync --extra pack "
            "(ADR-0012)"
        ) from exc
    return gguf


def base_tensor_names(base_gguf: Path) -> tuple[str, ...]:
    """Read the base GGUF's tensor names in file order.

    The read is a memory-mapped header read. It never touches tensor
    data, so it costs the same on a 61 GiB base file as on a fixture.

    Args:
        base_gguf: The full-precision base GGUF the quantizer reads.

    Returns:
        Every tensor name the file declares, in file order.

    Raises:
        PackError: If gguf-py is missing or the file is not a GGUF.
        OSError: If the file cannot be read.

    Examples:
        Read the names the overrides must match:

        ```python
        names = base_tensor_names(Path("model-f16.gguf"))
        assert "blk.0.attn_v.weight" in names
        ```
    """
    gguf = _load_gguf()
    try:
        reader = gguf.GGUFReader(str(base_gguf))
    except ValueError as exc:
        raise PackError(f"base GGUF {base_gguf} is not a GGUF: {exc}") from exc
    return tuple(tensor.name for tensor in reader.tensors)


def unmatched_patterns(
    overrides: Sequence[TypeOverride], names: Iterable[str]
) -> tuple[str, ...]:
    r"""Name the override patterns that match none of ``names``.

    The comparison lower-cases each pattern and searches it, which is
    what ``llama-quantize`` does with the same string. Repeated
    patterns report once, in first-seen order.

    Args:
        overrides: The overrides the pack would drive into the
            quantizer.
        names: The base GGUF's tensor names.

    Returns:
        The unmatched patterns, without repeats, in override order.

    Raises:
        PackError: If a pattern does not compile. Every pattern this
            backend builds comes from `re.escape`, so a failure here
            means a caller supplied its own.

    Examples:
        A layer the base GGUF does not carry reports its pattern:

        ```python
        overrides = (TypeOverride(r"blk\.99\.", "q4_k"),)
        assert unmatched_patterns(overrides, ["blk.0.attn_v.weight"]) == (r"blk\.99\.",)
        ```
    """
    tensor_names = tuple(names)
    unmatched: list[str] = []
    for override in overrides:
        if override.pattern in unmatched:
            continue
        try:
            pattern = re.compile(override.pattern.lower())
        except re.error as exc:
            raise PackError(
                f'override pattern "{override.pattern}" does not compile: {exc}'
            ) from exc
        if not any(pattern.search(name) for name in tensor_names):
            unmatched.append(override.pattern)
    return tuple(unmatched)


def check_overrides_match(overrides: Sequence[TypeOverride], base_gguf: Path) -> None:
    """Refuse a pack whose overrides do not reach the base GGUF.

    The refusal runs before the quantizer, so a recipe naming the
    wrong tensor tree costs no quantize run and writes no file.

    Args:
        overrides: The overrides the pack would drive into the
            quantizer. An empty sequence passes.
        base_gguf: The base GGUF the quantizer reads.

    Raises:
        PackError: If any override matches no tensor, if gguf-py is
            missing, or if the base file is not a GGUF.
        OSError: If the base file cannot be read.

    Examples:
        A recipe under a foreign root refuses here rather than
        packing as a no-op:

        ```python
        check_overrides_match(overrides, Path("model-f16.gguf"))
        ```
    """
    if not overrides:
        return
    unmatched = unmatched_patterns(overrides, base_tensor_names(base_gguf))
    if not unmatched:
        return
    details = ", ".join(f'"{pattern}"' for pattern in unmatched)
    raise PackError(
        f"the base GGUF {base_gguf} carries no tensor for {len(unmatched)} "
        f"of {len(overrides)} override patterns: {details}. The quantizer "
        f"applies no such override and exits 0, so the packed file would "
        f"drop that part of the recipe without a report (#303). Check the "
        f"recipe's group names against the base GGUF's tensor names"
    )
