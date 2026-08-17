"""gguf-py imatrix count reader: the pack step's zero-count source.

Implements the `ImatrixCountSource` port (ADR-0026 decision 5, the
2026-08-13 #198 amendment). ``llama-quantize`` fills a zero-count
expert's row with ones and prints no warning, because the stack
itself is present — the stdout scan behind ``imatrix_uncovered``
(ADR-0016) can never see the case. So the pack step reads the
matrix's ``.counts`` tensors itself, about 24 KB on the published
matrix.

The base GGUF vouches for the entries. ``vramfit pack`` requires the
base file before it runs, so the reader takes its tensor index — a
memory-mapped header read. An entry whose base tensor is 3D is an
expert stack, and element ``i`` of its counts is expert ``i``'s
tally. Every other matched entry is dense at a count length of 1.
An entry naming no base tensor skips silently — the strong
wrong-model signal runs the other way, as base tensors the matrix
lacks, and ``imatrix_uncovered`` already names those.

An empty report is what a healthy matrix returns, so a silent read
failure and a clean bill of health look alike. The reader separates
them by refusing, as `PackError`, on a closed list: not an imatrix,
no counts, an unknown tensor suffix, a sums tensor without its
counts twin, a count that is negative or not finite, or a count
length that contradicts the base tensor. gguf-py rides the scan
extra (the pack extra includes it), so the import is deferred to the
first read: the base install keeps ``vramfit pack --help`` and a
matrix-less pack working (ADR-0005).

`imatrix_entry_names` serves the exclusion check
([vramfit.adapters.outbound.gguf.exclusion_match][]) through the same
`_counts_by_entry`. One definition of a malformed imatrix serves both
callers, rather than a second and looser scan (the #198 amendment).
The list's sixth case needs the base GGUF, so it stays with
`expert_stack_counts` and that function's own docstring records it.

Examples:
    Read the counts the pack's quantizer will consume:

    ```python
    source = GgufImatrixCounts(
        imatrix=Path("model.imatrix.gguf"),
        base_gguf=Path("model-f16.gguf"),
    )
    stacks = source.expert_stack_counts()
    ```

See Also:
    - [vramfit.domain.pack][]: `zero_count_experts`, the verdict on
      these counts.
    - [vramfit.adapters.outbound.gguf.exclusion_match][]: the other
      caller of `imatrix_entry_names`.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.gguf.types import PackError

# A fused expert stack shapes as three dimensions in the base GGUF —
# ``ne`` of (columns, rows, experts). Any other rank is dense.
_STACK_DIMS = 3

# What gguf-py raises on a file it cannot read, measured by
# `override_match` over a truncated header and carried here for the
# same reason. `ValueError` covers a bad magic and an unsupported
# version, `IndexError` an empty numpy slice on a short file, and
# `OSError` the memory map. An interrupted ``llama-imatrix`` run
# leaves such a file, and both readers open it at a pack boundary
# that promises `PackError` (ADR-0011).
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
            "reading imatrix counts needs gguf-py — install the pack "
            "extra: uv sync --extra pack (ADR-0026)"
        ) from exc
    return gguf


def _open_reader(gguf_module: Any, path: Path, role: str) -> Any:
    """Open one GGUF file, translating every read failure to `PackError`.

    A truncated header raises `IndexError` from an empty numpy slice
    rather than a parse error, and the memory map raises `OSError`.
    Both reach a pack boundary that promises `PackError` (ADR-0011),
    so the tuple matches `override_match`'s rather than `ValueError`
    alone.

    Args:
        gguf_module: The imported ``gguf`` module.
        path: The file to open.
        role: The file's role in the message — ``imatrix`` or
            ``base GGUF``.

    Returns:
        An open ``GGUFReader``.

    Raises:
        PackError: If the file is not a GGUF, keeping the wording
            ADR-0026's refusal list records, or if the reader fails
            any other way. The cause carries what it reported.
    """
    try:
        return gguf_module.GGUFReader(str(path))
    except ValueError as exc:
        raise PackError(f"{role} {path} is not a GGUF: {exc}") from exc
    except _READER_FAILURES as exc:
        raise PackError(
            f"cannot read the {role} {path}: {type(exc).__name__}: {exc}. "
            f"A partial file from an interrupted run reads this way — "
            f"delete it and produce it again"
        ) from exc


def _counts_by_entry(reader: Any, path: Path) -> dict[str, list[float]]:
    """Collect each imatrix entry's raw counts, refusing malformation.

    Args:
        reader: An open ``GGUFReader`` over the imatrix.
        path: The imatrix file, named in messages.

    Returns:
        The stored counts per entry name, in file order.

    Raises:
        PackError: If the file's ``general.type`` is not ``imatrix``,
            a tensor ends in neither ``.in_sum2`` nor ``.counts``, a
            sums tensor lacks its counts twin, the file holds no
            counts at all, or a count is negative or not finite. The
            unknown-suffix refusal is stricter than the C loader,
            which skips unrecognized tensors — a future suffix must
            fail here rather than pass a health report that read
            nothing (the #198 amendment).
    """
    general_type = reader.fields.get("general.type")
    if general_type is None or general_type.contents() != "imatrix":
        raise PackError(f"{path} is not an imatrix GGUF (general.type mismatch)")
    counts: dict[str, list[float]] = {}
    sums: list[str] = []
    for tensor in reader.tensors:
        if tensor.name.endswith(".in_sum2"):
            sums.append(tensor.name.removesuffix(".in_sum2"))
        elif tensor.name.endswith(".counts"):
            values = [float(v) for v in tensor.data.reshape(-1)]
            name = tensor.name.removesuffix(".counts")
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise PackError(
                    f"{path}: {name}.counts holds a value that is "
                    "negative or not finite"
                )
            counts[name] = values
        else:
            raise PackError(
                f"{path}: unexpected tensor {tensor.name} — an imatrix "
                "holds only .in_sum2/.counts pairs"
            )
    if not counts:
        raise PackError(f"{path}: the imatrix holds no .counts tensors")
    orphans = sorted(set(sums) - set(counts))
    if orphans:
        raise PackError(f"{path}: {orphans[0]}.in_sum2 has no counts twin")
    return counts


def imatrix_entry_names(imatrix: Path) -> tuple[str, ...]:
    """Read the matrix's entry names, as ``--exclude-weights`` sees them.

    ``common/imatrix-loader.cpp`` keys each loaded entry by its tensor
    name with ``.in_sum2`` or ``.counts`` removed, which is what
    `_counts_by_entry` returns. So these keys are the names the
    quantizer matches an exclusion against.

    The read runs the file-level half of the #198 amendment's closed
    refusal list, through the same `_counts_by_entry` the count read
    uses. A second, lenient reader here would accept a file that read
    refuses.

    **It runs five of the list's six cases.** The sixth compares a
    matched entry's count length against its base tensor's matrix
    count, and it needs the base GGUF that `expert_stack_counts`
    opens. An exclusion check reads the matrix alone, so that case
    stays with the caller that has both files.

    **The pair rule is one-directional here and symmetric in C.**
    `common_imatrix_load` refuses a mismatched pair either way, and
    `_counts_by_entry` refuses only an ``.in_sum2`` with no
    ``.counts`` twin. So a ``.counts`` tensor with no sums twin yields
    a name here and makes the C loader exit 1. That is a loud
    downstream failure rather than a silent one, and #325 carries it.

    Args:
        imatrix: The importance matrix the pack consumes.

    Returns:
        Every entry name the matrix declares, in file order.

    Raises:
        PackError: If gguf-py is missing, the reader cannot open the
            file, or the matrix fails one of the five file-level
            cases on the closed refusal list.
        OSError: If the memory map fails while reading tensor data,
            which `_open_reader` cannot reach. `check_exclusion_match`
            translates it at the pack boundary.

    Examples:
        Read the names an exclusion must reach:

        ```python
        names = imatrix_entry_names(Path("model.imatrix.gguf"))
        assert "blk.1.attn_v.weight" in names
        ```
    """
    gguf = _load_gguf()
    reader = _open_reader(gguf, imatrix, "imatrix")
    return tuple(_counts_by_entry(reader, imatrix))


@dataclass(frozen=True, slots=True)
class GgufImatrixCounts:
    """`ImatrixCountSource` adapter reading GGUF files with gguf-py.

    Attributes:
        imatrix (Path): The importance matrix the pack consumes.
        base_gguf (Path): The f16 base GGUF whose tensor shapes vouch
            for the matrix's entries.

    Examples:
        The pack command wires one source per ``--imatrix`` run:

        ```python
        source: ImatrixCountSource = GgufImatrixCounts(imatrix, base_gguf)
        ```
    """

    imatrix: Path
    base_gguf: Path

    def expert_stack_counts(self) -> dict[str, tuple[int, ...]]:
        """Read the matrix's expert-stack count vectors.

        Returns:
            One count vector per expert-stack entry, keyed by GGUF
            tensor name. Element ``i`` is expert ``i``'s tally,
            rounded half up — ``std::lround`` on the non-negative
            counts an imatrix holds, so pack and scan agree on which
            counts are zero. Dense entries validate and stay out. An
            entry naming no base tensor skips silently.

        Raises:
            PackError: If gguf-py is missing, either file is not a
                GGUF, the matrix fails the closed refusal list (the
                #198 amendment), or a matched entry's count length
                differs from its base tensor's matrix count —
                ``ne[2]`` for a 3D tensor, 1 otherwise.
            OSError: If a file cannot be read.
        """
        gguf = _load_gguf()
        base = _open_reader(gguf, self.base_gguf, "base GGUF")
        shapes = {tensor.name: tuple(tensor.shape) for tensor in base.tensors}
        reader = _open_reader(gguf, self.imatrix, "imatrix")
        stacks: dict[str, tuple[int, ...]] = {}
        for name, values in _counts_by_entry(reader, self.imatrix).items():
            shape = shapes.get(name)
            if shape is None:
                continue
            is_stack = len(shape) == _STACK_DIMS
            expected = int(shape[_STACK_DIMS - 1]) if is_stack else 1
            if len(values) != expected:
                raise PackError(
                    f"{self.imatrix}: {name} holds {len(values)} counts, and "
                    f"its base tensor expects {expected} — the imatrix does "
                    "not describe this base GGUF"
                )
            if is_stack:
                # Round half up before the zero test, matching the C
                # loader (imatrix-loader.cpp:158) on counts checked
                # non-negative above.
                stacks[name] = tuple(math.floor(v + 0.5) for v in values)
        return stacks
