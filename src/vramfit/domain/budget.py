"""VRAM budget math: size parsing, KV-cache cost, and the weight budget.

Implements the arithmetic from ``docs/explanation/vram-budget.md``:
``weight_budget = vram_total - kv_cache - runtime_overhead``. KV cost is a
sum over attention layers, not ``layers x constant``, because NAS-derived
models like the north-star target delete attention from some layers.

Attributes:
    KV_DTYPE_BYTES (dict[str, int]): Bytes per KV-cache element by dtype
        name (``fp16``, ``bf16``, ``fp8``).
    DEFAULT_RUNTIME_OVERHEAD_BYTES (int): Planning figure for CUDA
        context, workspace, and fragmentation (2 GiB).

Examples:
    Compute the weight budget for the north-star target:

    ```python
    from vramfit.domain.budget import Budget, ModelShape, kv_cache_bytes, parse_size

    shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
    budget = Budget(
        vram_total_bytes=parse_size("24GiB"),
        kv_cache_bytes=kv_cache_bytes(shape, context=16384, kv_dtype="fp8"),
        runtime_overhead_bytes=parse_size("2GiB"),
    )
    print(budget.weight_budget_bytes)
    ```

See Also:
    - [vramfit.domain.solver][]: Packs weights against the budget
      computed here.
    - [vramfit.adapters.outbound.hf_config][]: Builds a `ModelShape`
      from a Hugging Face ``config.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

KV_DTYPE_BYTES: Final[dict[str, int]] = {"fp16": 2, "bf16": 2, "fp8": 1}
DEFAULT_RUNTIME_OVERHEAD_BYTES: Final[int] = 2 * 2**30

_SIZE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]i?B|B)?\s*$", re.IGNORECASE
)
_UNIT_BYTES: Final[dict[str, int]] = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}


def parse_size(text: str) -> int:
    """Parse a human-readable size into bytes.

    Binary units (``GiB``) are powers of 1024; decimal units (``GB``) are
    powers of 1000. A bare number is bytes.

    Args:
        text: Size string, e.g. ``"24GiB"``, ``"4 GB"``, ``"1073741824"``.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the string is not a recognizable size, or the
            result does not fit the signed 64-bit range the artifacts
            carry (#260). A budget that large describes no machine,
            and refusing here names the option the operator typed
            rather than the artifact vramfit would write.

    Examples:
        Binary vs decimal units:

        ```python
        from vramfit.domain.budget import parse_size

        assert parse_size("1GiB") == 2**30
        assert parse_size("1GB") == 10**9
        ```
    """
    match = _SIZE_RE.match(text)
    if match is None:
        raise ValueError(f'not a recognizable size: "{text}"')
    number = float(match.group("number"))
    unit = (match.group("unit") or "B").lower()
    size = int(number * _UNIT_BYTES[unit])
    if not -(2**63) <= size <= 2**63 - 1:
        raise ValueError(f'size "{text}" does not fit a 64-bit byte count')
    return size


def format_size(n_bytes: int) -> str:
    """Render a byte count in binary units with two decimals.

    Args:
        n_bytes: Byte count to render.

    Returns:
        A string like ``"19.42 GiB"``; plain ``"512 B"`` below 1 KiB.

    Examples:
        Render two gibibytes:

        ```python
        from vramfit.domain.budget import format_size

        assert format_size(2 * 2**30) == "2.00 GiB"
        ```
    """
    magnitude = abs(n_bytes)
    for unit, size in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if magnitude >= size:
            return f"{n_bytes / size:.2f} {unit}"
    return f"{n_bytes} B"


@dataclass(frozen=True, slots=True)
class ModelShape:
    """The attention geometry that determines KV-cache cost.

    Attributes:
        kv_heads_per_layer (tuple[int, ...]): KV head count for each
            attention layer. Layers without attention (NAS ``no_op``
            blocks) have no entry.
        head_dim (int): Dimension of each attention head.

    Examples:
        The north-star target's shape, built by hand:

        ```python
        from vramfit.domain.budget import ModelShape

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        assert len(shape.kv_heads_per_layer) == 49
        ```
    """

    kv_heads_per_layer: tuple[int, ...]
    head_dim: int

    @classmethod
    def uniform(cls, attn_layers: int, kv_heads: int, head_dim: int) -> ModelShape:
        """Build a shape where every attention layer is identical.

        Args:
            attn_layers: Number of layers that have attention.
            kv_heads: KV heads per attention layer.
            head_dim: Dimension of each attention head.

        Returns:
            The uniform shape.
        """
        return cls(
            kv_heads_per_layer=(kv_heads,) * attn_layers,
            head_dim=head_dim,
        )


def kv_bytes_per_token(shape: ModelShape, kv_dtype: str = "fp16") -> int:
    """Compute KV-cache bytes stored per token across the whole stack.

    The formula is ``2 (keys + values) x head_dim x Σ kv_heads x
    bytes_per_element``, summing over attention layers only.

    Args:
        shape: The model's attention geometry.
        kv_dtype: KV-cache element dtype (``fp16``, ``bf16``, or ``fp8``).

    Returns:
        Bytes per token.

    Raises:
        KeyError: If ``kv_dtype`` is not a known dtype.

    Examples:
        The north-star target stores ~196 KiB per token at fp16:

        ```python
        from vramfit.domain.budget import ModelShape, kv_bytes_per_token

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        assert kv_bytes_per_token(shape) == 200_704
        ```
    """
    element_bytes = KV_DTYPE_BYTES[kv_dtype]
    return 2 * shape.head_dim * sum(shape.kv_heads_per_layer) * element_bytes


def kv_cache_bytes(
    shape: ModelShape,
    context: int,
    kv_dtype: str = "fp16",
    sequences: int = 1,
) -> int:
    """Compute total KV-cache bytes for a context length and batch.

    Args:
        shape: The model's attention geometry.
        context: Context length in tokens.
        kv_dtype: KV-cache element dtype.
        sequences: Concurrent sequences sharing the card.

    Returns:
        Total KV-cache bytes.

    Examples:
        Cache for 16k context on the north-star target:

        ```python
        from vramfit.domain.budget import ModelShape, kv_cache_bytes

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        total = kv_cache_bytes(shape, context=16384)
        ```
    """
    return kv_bytes_per_token(shape, kv_dtype) * context * sequences


@dataclass(frozen=True, slots=True)
class Budget:
    """The VRAM ledger that yields the weight budget.

    Attributes:
        vram_total_bytes (int): The card's total VRAM.
        kv_cache_bytes (int): Bytes reserved for the KV cache.
        runtime_overhead_bytes (int): Bytes reserved for CUDA context,
            workspace, and fragmentation.

    Examples:
        A ledger that leaves ~19 GiB for weights:

        ```python
        from vramfit.domain.budget import Budget, parse_size

        budget = Budget(
            vram_total_bytes=parse_size("24GiB"),
            kv_cache_bytes=parse_size("3GiB"),
            runtime_overhead_bytes=parse_size("2GiB"),
        )
        assert budget.weight_budget_bytes == parse_size("19GiB")
        ```
    """

    vram_total_bytes: int
    kv_cache_bytes: int
    runtime_overhead_bytes: int

    @property
    def weight_budget_bytes(self) -> int:
        """Bytes left for weights; negative when overcommitted.

        Returns:
            ``vram_total - kv_cache - runtime_overhead``.
        """
        return self.vram_total_bytes - self.kv_cache_bytes - self.runtime_overhead_bytes
