r"""Override matching: refuse a pattern no base-GGUF tensor carries.

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

The check compares one pattern against one name the way the tool
does. ``tools/quantize/quantize.cpp:332`` lower-cases a pattern
before it compiles, and ``src/llama-quant.cpp:694`` searches the name
rather than anchoring it.

It is deliberately a superset of the tool's match set, so it never
refuses a pack the tool would honour. Three upstream filters it does
not model, each of which lets a pattern pass here and still apply
nothing:

- The first matching pattern wins upstream. A pattern shadowed by an
  earlier override passes here. Re-deriving that order would make
  this module a second source of truth for it (#190).
- ``src/llama-quant.cpp:675`` skips a tensor `tensor_allows_quantization`
  rejects — under two dimensions, a name not ending in ``weight``, a
  norm, ``ffn_gate_inp.weight``, ``ssm_conv1d``, and more.
- ``src/llama-quant.cpp:678-683`` returns early for the embedding and
  the output head whenever their dedicated flags are set, which
  `LlamaCppPacker.pack` sets whenever they resolve.

So a pattern whose only match is a norm, the embedding, or the output
head passes this check and changes no type. #305 carries that
residual, and the flags themselves are unchecked (#306).

Every pattern the pack step builds is a ``blk\.<n>\.`` or
``blk\.<n>\.<class>\.`` prefix over real GGUF tensor classes, so
none of the three is reachable through this backend today.

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
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import TypeOverride

# What gguf-py raises on a file it cannot read. `ValueError` covers a
# bad magic, an unsupported version, a duplicate tensor name, and a
# name that is not UTF-8 (`UnicodeDecodeError` subclasses it).
# `IndexError` comes from an empty numpy slice on a truncated header,
# and `OSError` from the memory map. Anything outside this tuple is a
# defect in the reader rather than a defect in the file.
_READER_FAILURES: Final[tuple[type[Exception], ...]] = (
    ValueError,
    IndexError,
    struct.error,
    OSError,
)


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

    Every reader failure lands under `PackError`, because ADR-0011
    puts every deliberate failure under the `VramfitError` root.
    `GGUFReader` reads each header field as ``self._get(...)[0]``, so
    a short file raises `IndexError` from an empty numpy slice rather
    than a parse error. Measured over a 448-byte 4-tensor file: 125
    of 447 truncation lengths raise `IndexError`. An interrupted
    convert leaves exactly such a file, and `LlamaCppPacker.convert`
    reuses any existing base GGUF, so the next pack reads it.

    Args:
        base_gguf: The full-precision base GGUF the quantizer reads.

    Returns:
        Every tensor name the file declares, in file order. A GGUF
        holding no tensors returns an empty tuple, which then refuses
        every override rather than passing them.

    Raises:
        PackError: If gguf-py is missing, or the reader cannot read
            the file. The cause carries what the reader reported.

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
        return tuple(tensor.name for tensor in reader.tensors)
    except _READER_FAILURES as exc:
        raise PackError(
            f"cannot read the base GGUF {base_gguf}: "
            f"{type(exc).__name__}: {exc}. The pack step holds the "
            f"recipe's overrides against this file's tensor names "
            f"(ADR-0012). A partial file from an interrupted convert "
            f"reads this way — delete it and convert again"
        ) from exc


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
        PackError: If a pattern does not compile. A layer pattern is
            built by f-string over `_LAYER_GROUP`'s ``(\d+)``
            capture, and the other two producers escape a prefix
            drawn from the fixed GGUF class tables. So every pattern
            this backend builds is a literal today, and a failure
            here means a caller supplied its own pattern or that
            capture widened.

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
        A recipe naming a layer the base GGUF does not carry refuses
        here rather than packing:

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
