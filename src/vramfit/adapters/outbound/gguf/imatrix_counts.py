"""gguf-py imatrix reader: the experts a matrix covers at count zero.

Implements the `ImatrixCountSource` port for the pack path
(ADR-0026 decision 5). ``llama-imatrix`` writes one ``.counts``
tensor per matrix entry, holding the chunk tally per matrix. A
dense tensor holds one tally. A fused expert stack holds one per
routed expert, and a zero there fills that expert's row with ones —
the unassisted fit. The quantizer emits no warning for it, because
the stack itself is present. This reader is the only thing that
finds it.

The read costs about 24 KB on a 5,957-tensor matrix: only the
``.counts`` tensors are touched, never the ``in_sum2`` sums. gguf-py
rides the scan extra (the pack extra includes it), so the import is
deferred to the first read — the base install keeps
``vramfit pack --help`` working, and a missing dependency names the
extra instead of tracebacking.

The scan reads the same counts through
[vramfit.adapters.outbound.scan.imatrix][], keyed by Hugging Face
parameter name and backed by torch. The pack path holds GGUF names
and no torch, so it reads here instead. Both round a stored count
the way ``std::lround`` rounds — half away from zero — so the two
agree on which counts are zero.

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

# The suffix ``llama-imatrix`` gives the chunk-tally tensor of an
# entry (llama.cpp tools/imatrix/imatrix.cpp, checkout e9fa078).
_COUNTS_SUFFIX: Final[str] = ".counts"
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
            PackError: If gguf-py is missing, or the file cannot be
                read as a GGUF. The read never returns an empty
                result on failure — that is what a healthy matrix
                returns.
        """
        gguf = _load_gguf()
        try:
            reader = gguf.GGUFReader(self.imatrix)
        except (OSError, ValueError) as exc:
            raise PackError(f"cannot read the imatrix {self.imatrix}: {exc}") from exc
        starved: list[ZeroCountExpert] = []
        for tensor in reader.tensors:
            if not tensor.name.endswith(_COUNTS_SUFFIX):
                continue
            counts = tensor.data.reshape(-1).tolist()
            if len(counts) < _STACK_MATRICES:
                continue
            stack = tensor.name.removesuffix(_COUNTS_SUFFIX)
            starved += [
                ZeroCountExpert(stack=stack, expert=expert)
                # Round before the test: the file stores a float, and
                # the C loader tests the rounded value
                # (llama.cpp tools/imatrix/imatrix-loader.cpp:158).
                for expert, count in enumerate(counts)
                if math.floor(float(count) + 0.5) == 0
            ]
        return tuple(sorted(starved, key=lambda e: (e.stack, e.expert)))
