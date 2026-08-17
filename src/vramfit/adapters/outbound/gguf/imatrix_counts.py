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
([vramfit.adapters.outbound.gguf.exclusion_match][]) from the same
read. One definition of a readable imatrix serves both callers, so
neither can accept a file the other refuses (the #198 amendment).

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vramfit.adapters.outbound.gguf.types import PackError

# A fused expert stack shapes as three dimensions in the base GGUF —
# ``ne`` of (columns, rows, experts). Any other rank is dense.
_STACK_DIMS = 3


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
    """Open one GGUF file, translating a parse failure to `PackError`.

    Args:
        gguf_module: The imported ``gguf`` module.
        path: The file to open.
        role: The file's role in the message — ``imatrix`` or
            ``base GGUF``.

    Returns:
        An open ``GGUFReader``.

    Raises:
        PackError: If the file is not a GGUF.
        OSError: If the file cannot be read.
    """
    try:
        return gguf_module.GGUFReader(str(path))
    except ValueError as exc:
        raise PackError(f"{role} {path} is not a GGUF: {exc}") from exc


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
    name with ``.in_sum2`` or ``.counts`` removed, and it refuses a
    sums tensor without its counts twin. `_counts_by_entry` applies
    the same two rules, so its keys are the names the quantizer
    matches an exclusion against.

    The read runs the #198 amendment's closed refusal list, the way
    the count read does. A second, lenient reader here would accept a
    file the count read refuses.

    Args:
        imatrix: The importance matrix the pack consumes.

    Returns:
        Every entry name the matrix declares, in file order.

    Raises:
        PackError: If gguf-py is missing, the file is not a GGUF, or
            the matrix fails the closed refusal list.
        OSError: If the file cannot be read.

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
