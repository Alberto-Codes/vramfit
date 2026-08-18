r"""Override matching: hold a recipe against the base GGUF's tensor names.

Three checks share one memory-mapped header read. The first refuses an
override pattern no tensor carries (#303). The second refuses a
dedicated flag whose target tensor the file lacks (#306). The third
names the layers the file carries that no override reaches (#307).

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
residual.

Every pattern the pack step builds is a ``blk\.<n>\.`` or
``blk\.<n>\.<class>\.`` prefix over real GGUF tensor classes, so
none of the three is reachable through this backend today.

`unmatched_flags` closes the third filter from the other side. The two
dedicated flags carry no pattern. Each binds an exact tensor name, by
`strcmp` and not by search. The embedding flag accepts either of two
names and the output flag accepts one. The base GGUF may carry none of
a flag's targets. That flag then binds nothing, applies nothing,
prints nothing, and the quantizer exits 0. The tensor takes the
``--pure`` floor while `PackResult` records the type the recipe asked
for (#306).

**Only a flag the recipe made load-bearing refuses.** Decision 2 of
ADR-0012 gives a scanned ``lm_head`` group its own
``--output-tensor-type`` assignment, and its 2026-08-16 (#307)
amendment reads that clause as the record's answer to one unscanned
unit silently taking the floor: prevent it. A base GGUF with no
``output.weight`` defeats exactly that, so this refuses.

**The tied fallback is exempt, because decision 2 already answers
it.** Without an ``lm_head`` group the embedding assignment drives the
output flag, and the decision states that on a model that ties
embeddings "the flag never applies". That is a ruled outcome and not a
malformed input, so refusing it would refuse a pack the record
sanctions. The recorded type also stays true there — a tied model's
head *is* the embedding tensor, which took ``--token-embedding-type``
at that same type. `output_group_type` is what separates the two
cases, and `LlamaCppPacker.pack` reads it for this check alone.

The comparison is the tool's own, and it folds no case.
``quantize.cpp:332`` lower-cases a ``--tensor-type`` pattern, and
these two flags reach `tensor_get_category` instead. That delegates to
two helpers which compare with `std::strcmp`
(``src/llama-quant.cpp:101-108``). So folding case here would pass a
file the quantizer never binds.

`floored_layers` runs the same comparison the other way. A layer the
base GGUF numbers that no override reaches takes the ``--pure`` base
ftype. ADR-0012 decision 3 supplies the mechanism, and its 2026-08-16
#307 amendment records why this reports rather than refuses. It is
the ADR-0026 decision 5 shape: the quantizer prints nothing, and the
pack path names the case rather than flattening it silently.

**The superset bias flips direction here.** The refusal above is a
superset of the tool's match set, so it never refuses a pack the tool
would honour. That same superset makes `covered` a superset of
truly-covered, so this report is a subset of truly-floored. It
under-reports, and under-reporting is the silence #307 exists to
break. Of the three upstream filters, only `tensor_allows_quantization`
can hide a layer here, and it needs a pattern whose every match is a
tensor the quantizer skips. `GGUF_SUFFIX_BY_HF` carries no such class,
so #305's residual bounds this too.

The unit is the layer index and not the tensor. A recipe of
expert-stack groups reaches one tensor class per layer on purpose, so
a per-tensor report would name every attention and dense tensor in the
model. A layer no pattern touches at all is the case the recipe did
not address.

gguf-py rides the scan extra and the pack extra includes it, so the
import defers to the first read. ``vramfit pack --help`` keeps working
on a base install (ADR-0005).

Examples:
    Hold a recipe's overrides against the file the quantizer reads:

    ```python
    floored = check_base_coverage(overrides, Path("model-f16.gguf"))
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

# The GGUF layer-stack prefix. Anchored, so a vision tower's
# `v.blk.<n>.` does not read as a decoder layer — #236 owns that root
# question and this report must not pre-empt it.
_BLK_LAYER: Final[re.Pattern[str]] = re.compile(r"^blk\.(\d+)\.")

# The exact tensor names each dedicated flag binds. `--token-embedding-type`
# takes either name and `--output-tensor-type` takes the one
# (`tensor_name_match_token_embd` and `tensor_name_match_output_weight`,
# llama.cpp src/llama-quant.cpp:101-108). Both flags bind through the
# tensor's category, so a file carrying no such tensor gives the flag
# nothing to bind and the early return at :678-683 never fires.
# Verified at commit `3653e6d6d` (b10326, the pinned instrument) and at
# `e9fa0781f`, which carry the same two matchers.
EMBEDDING_FLAG: Final[str] = "--token-embedding-type"
EMBEDDING_TARGETS: Final[tuple[str, ...]] = (
    "token_embd.weight",
    "per_layer_token_embd.weight",
)
OUTPUT_FLAG: Final[str] = "--output-tensor-type"
OUTPUT_TARGETS: Final[tuple[str, ...]] = ("output.weight",)


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


def _compiled(override: TypeOverride) -> re.Pattern[str]:
    r"""Compile one override the way ``llama-quantize`` compiles it.

    ``tools/quantize/quantize.cpp:332`` lower-cases a pattern before
    it compiles, and ``src/llama-quant.cpp:694`` searches the name
    rather than anchoring it.

    Args:
        override: The override to compile.

    Returns:
        The compiled pattern.

    Raises:
        PackError: If the pattern does not compile. A layer pattern is
            built by f-string over `_LAYER_GROUP`'s ``(\d+)`` capture,
            and the other two producers escape a prefix drawn from the
            fixed GGUF class tables. So every pattern this backend
            builds is a literal today, and a failure here means a
            caller supplied its own pattern or that capture widened.
    """
    try:
        return re.compile(override.pattern.lower())
    except re.error as exc:
        raise PackError(
            f'override pattern "{override.pattern}" does not compile: {exc}'
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
        PackError: If a pattern does not compile. See `_compiled`.

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
        pattern = _compiled(override)
        if not any(pattern.search(name) for name in tensor_names):
            unmatched.append(override.pattern)
    return tuple(unmatched)


def unmatched_flags(
    names: Iterable[str], *, embedding: bool, output: bool
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Name the dedicated flags whose target tensor the file lacks.

    Each flag binds an exact tensor name, so the comparison is
    membership and not a search. The embedding flag accepts either of
    two names, so a file carrying one of them satisfies it. The
    comparison folds no case, which is what ``llama-quantize`` does
    with the same strings.

    Args:
        names: The base GGUF's tensor names.
        embedding: Whether the pack emits ``--token-embedding-type``.
        output: Whether the pack emits ``--output-tensor-type`` for a
            scanned ``lm_head`` group. False for the tied fallback,
            which ADR-0012 decision 2 already rules a no-op.

    Returns:
        One ``(flag, targets)`` pair per unmatched flag, embedding
        first. Empty when every emitted flag has a target in the
        file.

    Examples:
        A base GGUF with no untied head cannot honour the flag:

        ```python
        names = ("token_embd.weight", "blk.0.attn_v.weight")
        assert unmatched_flags(names, embedding=True, output=True) == (
            ("--output-tensor-type", ("output.weight",)),
        )
        ```
    """
    carried = frozenset(names)
    unmatched: list[tuple[str, tuple[str, ...]]] = []
    if embedding and carried.isdisjoint(EMBEDDING_TARGETS):
        unmatched.append((EMBEDDING_FLAG, EMBEDDING_TARGETS))
    if output and carried.isdisjoint(OUTPUT_TARGETS):
        unmatched.append((OUTPUT_FLAG, OUTPUT_TARGETS))
    return tuple(unmatched)


def floored_layers(
    overrides: Sequence[TypeOverride], names: Iterable[str]
) -> tuple[str, ...]:
    r"""Name the base GGUF's layers that no override reaches.

    A layer is covered when at least one override pattern matches at
    least one tensor under it. A layer no pattern reaches takes the
    ``--pure`` base ftype (ADR-0012 decision 3), so this reports
    rather than refusing.

    The comparison lower-cases each pattern and searches it, the way
    `unmatched_patterns` does and the way ``llama-quantize`` does. It
    inherits that superset, so it under-reports rather than
    over-reports. The module docstring carries the reasoning.

    Args:
        overrides: The overrides the pack would drive into the
            quantizer. An empty sequence floors every layer, and this
            names them all. `check_base_coverage` never reaches that
            case, because it returns before the file read.
        names: The base GGUF's tensor names.

    Returns:
        The floored layer prefixes, e.g. ``blk.52.``, sorted by
        layer index. Empty when every layer the file numbers is
        reached, and empty for a file that numbers no layer.

    Raises:
        PackError: If a pattern does not compile. See `_compiled`.

    Examples:
        A recipe scanned over fewer layers than the file carries:

        ```python
        overrides = (TypeOverride(r"blk\.0\.", "q4_k"),)
        names = ("blk.0.attn_v.weight", "blk.1.attn_v.weight")
        assert floored_layers(overrides, names) == ("blk.1.",)
        ```
    """
    patterns = [_compiled(override) for override in overrides]
    covered: set[int] = set()
    present: set[int] = set()
    for name in names:
        match = _BLK_LAYER.match(name)
        if match is None:
            continue
        index = int(match.group(1))
        present.add(index)
        if index not in covered and any(p.search(name) for p in patterns):
            covered.add(index)
    return tuple(f"blk.{index}." for index in sorted(present - covered))


def check_base_coverage(
    overrides: Sequence[TypeOverride],
    base_gguf: Path,
    *,
    embedding_flag: bool = False,
    output_flag: bool = False,
) -> tuple[str, ...]:
    """Hold a recipe's overrides and flags against the base GGUF.

    One header read serves all three checks. Both refusals run before
    the quantizer, so a recipe naming the wrong tensor tree costs no
    quantize run and writes no file.

    The override refusal reports first. A recipe that fails both
    names the patterns, which is the coarser mismatch and the likelier
    cause.

    Args:
        overrides: The overrides the pack would drive into the
            quantizer. An empty sequence reports no floored layer,
            because a recipe driving no override packs everything at
            the floor on purpose.
        base_gguf: The base GGUF the quantizer reads.
        embedding_flag: Whether the pack emits
            ``--token-embedding-type``.
        output_flag: Whether the pack emits ``--output-tensor-type``
            for a scanned ``lm_head`` group. False for the tied
            fallback, which ADR-0012 decision 2 rules a no-op.

    Returns:
        The layer prefixes the file carries that no override reaches
        (#307). Empty for an empty override sequence, which reports
        no layer rather than reporting every layer. Such a recipe
        floors the whole model and #307 does not cover it.

    Raises:
        PackError: If any override matches no tensor (#303), if a
            dedicated flag has no target tensor (#306), if gguf-py is
            missing, or if the reader cannot read the base file.
            `base_tensor_names` wraps every reader failure, so no
            `OSError` reaches the caller.

    Examples:
        A recipe naming a layer the base GGUF does not carry refuses
        here rather than packing:

        ```python
        floored = check_base_coverage(overrides, Path("model-f16.gguf"))
        ```
    """
    if not overrides and not (embedding_flag or output_flag):
        return ()
    names = base_tensor_names(base_gguf)
    unmatched = unmatched_patterns(overrides, names)
    if unmatched:
        details = ", ".join(f'"{pattern}"' for pattern in unmatched)
        raise PackError(
            f"the base GGUF {base_gguf} carries no tensor for {len(unmatched)} "
            f"of {len(overrides)} override patterns: {details}. The quantizer "
            f"applies no such override and exits 0, so the packed file would "
            f"drop that part of the recipe without a report (#303). Check the "
            f"recipe's group names against the base GGUF's tensor names"
        )
    unreached = unmatched_flags(names, embedding=embedding_flag, output=output_flag)
    if unreached:
        details = ", ".join(
            f"{flag} (needs " + " or ".join(f'"{name}"' for name in targets) + ")"
            for flag, targets in unreached
        )
        raise PackError(
            f"the base GGUF {base_gguf} carries no target tensor for "
            f"{len(unreached)} dedicated flag"
            f"{'' if len(unreached) == 1 else 's'}: {details}. The quantizer "
            f"binds each flag by exact tensor name and exits 0 when the file "
            f"carries none, so that tensor would take the --pure floor while "
            f"the record states the recipe's type (#306). Check the recipe's "
            f"embedding and lm_head groups against the base GGUF"
        )
    if not overrides:
        return ()
    return floored_layers(overrides, names)
