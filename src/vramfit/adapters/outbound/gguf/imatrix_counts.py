"""gguf-py imatrix reader: the experts a matrix covers at count zero.

Implements the `ImatrixCountSource` port for the pack path
(ADR-0026 decision 5). ``llama-imatrix`` writes one ``.counts``
tensor per matrix entry, holding the chunk tally per matrix. A
dense tensor holds one tally. A fused expert stack holds one per
routed expert, and a zero there fills that expert's row with ones —
the unassisted fit. The quantizer emits no warning for it, because
the stack itself is present. This reader is the only thing that
finds it.

The read costs about 24 KB on the published matrix for the MoE
target: 185 entries hold 6,027 float32 counts between them. Only
the ``.counts`` tensors are touched, never the ``in_sum2`` sums.
gguf-py rides the scan extra (the pack extra includes it), so the
import is deferred to the first read — the base install keeps
``vramfit pack --help`` working, and a missing dependency names the
extra instead of tracebacking. A pack that names an imatrix
therefore needs the extra, where before it only shelled out.

The reader refuses a file it cannot vouch for: one that is no
imatrix, one that holds no counts, one that carries an unknown
suffix, and one whose counts are negative or not finite. Every one
of those would otherwise read as a healthy matrix, because a
healthy matrix reports no zero-count expert at all.

The scan reads the same counts through
[vramfit.adapters.outbound.scan.imatrix][], keyed by Hugging Face
parameter name and backed by torch. The pack path holds GGUF names
and no torch, so it reads here instead. Both round a stored count
half up, which is what ``std::lround`` does on the non-negative
counts an imatrix holds, so the two agree on which counts are zero.

Examples:
    Read a published matrix for starved experts:

    ```python
    counts = GgufImatrixCounts(Path("model.imatrix.gguf"))
    assert counts.zero_count_experts() == ()
    ```

See Also:
    - [vramfit.ports.outbound][]: `ImatrixCountSource`, which this
      satisfies.
    - [vramfit.domain.pack][]: `ZeroCountExpert`, the record it
      returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import ZeroCountExpert

# The two suffixes ``llama-imatrix`` gives an entry: the summed
# squares and the chunk tally (llama.cpp tools/imatrix/imatrix.cpp,
# checkout e9fa078). This reader touches only the second.
_COUNTS_SUFFIX: Final[str] = ".counts"
_SUMS_SUFFIX: Final[str] = ".in_sum2"
# An entry of one matrix is a dense tensor. A zero count there
# flattens the whole tensor, which is not the case this reports.
_STACK_MATRICES: Final[int] = 2


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
            "reading the imatrix counts needs gguf-py — install the pack "
            "extra: uv sync --extra pack (ADR-0026)"
        ) from exc
    return gguf


@dataclass(frozen=True, slots=True)
class GgufImatrixCounts:
    """`ImatrixCountSource` adapter reading a GGUF imatrix with gguf-py.

    The read validates before it reports: the file must declare
    itself an imatrix, hold at least one counts tensor, carry no
    unknown tensor suffix, and hold no count that is negative or not
    finite.

    Attributes:
        imatrix (Path): The importance matrix file the pack drives.

    Examples:
        The pack command wires one reader per matrix:

        ```python
        source: ImatrixCountSource = GgufImatrixCounts(Path("model.imatrix.gguf"))
        ```
    """

    imatrix: Path

    def zero_count_experts(self) -> tuple[ZeroCountExpert, ...]:
        """Name every routed expert the matrix covers at a count of zero.

        Returns:
            The zero-count experts, sorted by stack then expert
            index. Empty when every covered expert fired at least
            once (issue #162).

        Raises:
            PackError: If gguf-py is missing, the file cannot be read
                as a GGUF, it is not an imatrix, it holds no counts,
                it carries a tensor of neither known suffix, or a
                count is negative or not finite. Every one of those
                would otherwise return the empty tuple, which is what
                a healthy matrix returns — a silent read failure and
                a clean bill of health must never look alike.
        """
        gguf = _load_gguf()
        try:
            starved = self._starved(gguf.GGUFReader(self.imatrix))
        except (OSError, ValueError, OverflowError) as exc:
            raise PackError(f"cannot read the imatrix {self.imatrix}: {exc}") from exc
        return starved

    def _starved(self, reader: Any) -> tuple[ZeroCountExpert, ...]:
        """Scan an open reader's counts tensors for zero-count experts.

        Args:
            reader: An open ``GGUFReader`` over the imatrix.

        Returns:
            The zero-count experts, sorted by stack then expert index.

        Raises:
            ValueError: If the file is not an imatrix, holds a tensor
                of neither known suffix, holds no counts at all, or
                holds a count that is negative or not finite. The
                unknown-suffix refusal is deliberately stricter than
                the C loader, which skips what it does not recognize.
                A suffix rename in a future imatrix format must fail
                here. It must not report a pack's experts healthy
                without reading one.
        """
        general_type = reader.fields.get("general.type")
        if general_type is None or general_type.contents() != "imatrix":
            raise ValueError('general.type is not "imatrix"')
        starved: list[ZeroCountExpert] = []
        entries = 0
        for tensor in reader.tensors:
            if tensor.name.endswith(_SUMS_SUFFIX):
                continue
            if not tensor.name.endswith(_COUNTS_SUFFIX):
                raise ValueError(
                    f"unexpected tensor {tensor.name} — an imatrix holds "
                    f"only {_SUMS_SUFFIX}/{_COUNTS_SUFFIX} pairs"
                )
            entries += 1
            counts = [float(count) for count in tensor.data.reshape(-1).tolist()]
            if any(not math.isfinite(count) or count < 0 for count in counts):
                raise ValueError(
                    f"{tensor.name} holds a count that is negative or not finite"
                )
            if len(counts) < _STACK_MATRICES:
                continue
            stack = tensor.name.removesuffix(_COUNTS_SUFFIX)
            starved += [
                ZeroCountExpert(stack=stack, expert=expert)
                # Round before the test: the file stores a float, and
                # the C loader tests the rounded value
                # (llama.cpp common/imatrix-loader.cpp:158).
                for expert, count in enumerate(counts)
                if math.floor(count + 0.5) == 0
            ]
        if not entries:
            raise ValueError(f"the file holds no {_COUNTS_SUFFIX} tensors")
        return tuple(sorted(starved, key=lambda e: (e.stack, e.expert)))
