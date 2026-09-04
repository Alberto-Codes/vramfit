"""gguf-py reconstruction meter: per-tensor RMSE against the f16 base.

Implements the `ReconstructionChecker` port for GGUF packs
(ADR-0022). Each named tensor is dequantized from the packed file
with gguf-py's numpy dequantizers — no llama.cpp runtime, no GPU,
seconds of CPU — and compared against the same tensor in the f16
base. gguf-py rides the scan extra (the pack extra includes it), so
the import is deferred to the first measurement: the base install
keeps ``vramfit pack --help`` working, and a missing dependency
names the extra instead of tracebacking.

Examples:
    Measure two tensors of a packed model:

    ```python
    checker = GgufReconstructionChecker(
        packed=Path("packed.gguf"),
        base=Path("model-f16.gguf"),
    )
    errors = checker.rmse(("blk.4.attn_v.weight",))
    ```

See Also:
    - [vramfit.domain.pack][]: `collapsed_tensors`, the verdict on
      these measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vramfit.adapters.outbound.gguf.types import PackError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping


def _load_gguf() -> Any:
    """Import gguf-py on first use, naming the gguf extra when absent.

    Returns:
        The imported ``gguf`` module.

    Raises:
        PackError: If gguf-py is not installed.
    """
    try:
        import gguf  # noqa: PLC0415 - lazy: base install has no gguf-py (ADR-0005)
    except ImportError as exc:
        raise PackError(
            "the reconstruction check needs gguf-py — install the gguf "
            "extra: uv sync --extra gguf (ADR-0022)"
        ) from exc
    return gguf


def _tensor_values(reader: Any, gguf_module: Any, name: str, path: Path) -> Any:
    """Dequantize one tensor from an open GGUF reader.

    Args:
        reader: An open ``GGUFReader``.
        gguf_module: The imported ``gguf`` module.
        name: The tensor to read.
        path: The backing file, for error messages.

    Returns:
        The tensor's values as a flat float32 numpy array.

    Raises:
        PackError: If the tensor is missing from the file.
    """
    for tensor in reader.tensors:
        if tensor.name == name:
            return (
                gguf_module.quants.dequantize(tensor.data, tensor.tensor_type)
                .astype("float32")
                .ravel()
            )
    raise PackError(f'tensor "{name}" is not in {path}')


@dataclass(frozen=True, slots=True)
class GgufReconstructionChecker:
    """`ReconstructionChecker` adapter reading GGUF files with gguf-py.

    Attributes:
        packed (Path): The packed model to measure.
        base (Path): The f16 base GGUF the pack was quantized from.

    Examples:
        The pack command wires one checker per packed file:

        ```python
        checker: ReconstructionChecker = GgufReconstructionChecker(packed, base)
        ```
    """

    packed: Path
    base: Path

    def rmse(self, tensors: tuple[str, ...]) -> Mapping[str, float]:
        """Measure reconstruction error for the named tensors.

        Args:
            tensors: GGUF tensor names, e.g. ``blk.4.attn_v.weight``.

        Returns:
            Root-mean-square error against the f16 base, per tensor.

        Raises:
            PackError: If gguf-py is missing, a file cannot be read,
                a tensor is missing from either file, or the two
                files disagree about a tensor's element count.
        """
        gguf = _load_gguf()
        import numpy as np  # noqa: PLC0415 - lazy: rides in with gguf-py (ADR-0005)

        try:
            packed_reader = gguf.GGUFReader(self.packed)
            base_reader = gguf.GGUFReader(self.base)
        except (OSError, ValueError) as exc:
            raise PackError(f"cannot read GGUF for reconstruction: {exc}") from exc
        errors: dict[str, float] = {}
        for name in tensors:
            quantized = _tensor_values(packed_reader, gguf, name, self.packed)
            reference = _tensor_values(base_reader, gguf, name, self.base)
            if quantized.size != reference.size:
                raise PackError(
                    f'tensor "{name}" has {quantized.size} elements in '
                    f"{self.packed} but {reference.size} in {self.base}"
                )
            difference = quantized - reference
            errors[name] = float(np.sqrt(np.mean(difference * difference)))
        return errors
