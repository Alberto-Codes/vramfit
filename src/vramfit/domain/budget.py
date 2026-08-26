"""VRAM budget math: size parsing, KV-cache cost, and the weight budget.

Implements the arithmetic from ``docs/explanation/vram-budget.md``:
``weight_budget = vram_total - kv_cache - runtime_overhead``. KV cost is a
sum over per-layer `KVLayer` entries, not ``layers x constant``. NAS-derived
models delete attention from some layers, and mixed-attention stacks
(Gemma 4, #421) vary window, head width, KV-head count, storage factor,
and KV sharing per layer.

A global layer's cache grows with context. A sliding layer's cache stops
at its window plus the runtime's `KV_WINDOW_PAD_TOKENS` padding (#431).
A shared layer allocates nothing. So one scalar "bytes per token" cannot
price these stacks: `kv_growth_bytes_per_token` carries the
context-scaled term and `kv_window_pool_bytes` the saturated window
term.

`parse_size` refuses a size the artifacts cannot carry, at the signed
64-bit range every reader bounds (ADR-0008 as amended 2026-08-16,
#260). Refusing here names the option the operator typed rather than
the artifact vramfit would write.

Attributes:
    KV_DTYPE_BYTES (dict[str, int]): Bytes per KV-cache element by dtype
        name (``fp16``, ``bf16``, ``fp8``).
    DEFAULT_RUNTIME_OVERHEAD_BYTES (int): Planning figure for CUDA
        context, workspace, and fragmentation (2 GiB).
    KV_WINDOW_PAD_TOKENS (int): Tokens the serving runtime adds to
        each sliding layer's cache past its window (512, the llama.cpp
        default ``n_ubatch``, measured on #431).

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

# The serving runtime sizes a sliding layer's cache at
# `window + n_ubatch` tokens, not `window` (#431). 512 is the
# llama.cpp default `n_ubatch`, measured at 1,200 MiB per sequence on
# the Gemma 4 31B fixture.
KV_WINDOW_PAD_TOKENS: Final[int] = 512

# How much of a rejected size string a message repeats back.
_SHOWN_SIZE_CHARS = 40

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
            carry (#260). A budget that large describes no machine.
            A digit string long enough to float to infinity refuses
            the same way, because `OverflowError` is not a
            `ValueError` the CLI catches. The message repeats at most
            `_SHOWN_SIZE_CHARS` of the input back.

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
    # The input reaches an operator-facing message, so cap it. A
    # 400-digit size would otherwise fill a terminal (#260).
    shown = text if len(text) <= _SHOWN_SIZE_CHARS else f"{text[:_SHOWN_SIZE_CHARS]}…"
    refusal = f'size "{shown}" does not fit a 64-bit byte count'
    try:
        size = int(number * _UNIT_BYTES[unit])
    except (OverflowError, ValueError) as exc:
        # A digit string long enough to float to infinity reaches
        # `int()` before the bound below can run, and `OverflowError`
        # is not a `ValueError` the CLI catches (ADR-0011).
        raise ValueError(refusal) from exc
    if not -(2**63) <= size <= 2**63 - 1:
        raise ValueError(refusal)
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
class KVLayer:
    """One attention layer's KV-cache geometry.

    Attributes:
        kv_heads (int): KV head count for this layer.
        head_dim (int): Dimension of each KV head in this layer.
        window (int | None): Sliding-window size in tokens. ``None``
            means global attention: the cache grows with context. An
            integer caps the cache at
            ``min(context, window + KV_WINDOW_PAD_TOKENS)`` tokens —
            the runtime pads each window with its batch size (#431).
            The config adapter admits only positive windows, and the
            domain does not re-check that bound.
        kv_tensors (int): KV tensors the runtime allocates per cached
            token: 2 for the K and V caches. The ruled runtime
            allocates both even under ``attention_k_eq_v`` and fills
            V with K (#431).
        shares_kv (bool): True when the layer reuses another layer's
            cache and allocates no KV of its own
            (``num_kv_shared_layers``).

    Examples:
        A Gemma 4 31B sliding layer:

        ```python
        from vramfit.domain.budget import KVLayer

        layer = KVLayer(kv_heads=16, head_dim=256, window=1024)
        ```
    """

    kv_heads: int
    head_dim: int
    window: int | None = None
    kv_tensors: int = 2
    shares_kv: bool = False


@dataclass(frozen=True, slots=True)
class ModelShape:
    """The attention geometry that determines KV-cache cost.

    Attributes:
        kv_layers (tuple[KVLayer, ...]): Per-layer KV geometry. Layers
            without attention (NAS ``no_op`` blocks) have no entry.
            Shared-KV layers keep an entry with ``shares_kv`` set, so
            the stack's layer count stays readable.

    Examples:
        The north-star target's shape, built by hand:

        ```python
        from vramfit.domain.budget import ModelShape

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        assert len(shape.kv_layers) == 49
        ```
    """

    kv_layers: tuple[KVLayer, ...]

    @classmethod
    def uniform(cls, attn_layers: int, kv_heads: int, head_dim: int) -> ModelShape:
        """Build a shape where every attention layer is identical.

        Every layer is global, stores a K and V pair, and shares
        nothing — the pre-#421 geometry.

        Args:
            attn_layers: Number of layers that have attention.
            kv_heads: KV heads per attention layer.
            head_dim: Dimension of each attention head.

        Returns:
            The uniform shape.
        """
        return cls(
            kv_layers=(KVLayer(kv_heads=kv_heads, head_dim=head_dim),) * attn_layers,
        )


def _layer_token_bytes(layer: KVLayer, kv_dtype: str) -> int:
    """Compute one layer's KV bytes per cached token.

    Args:
        layer: The layer's KV geometry.
        kv_dtype: KV-cache element dtype.

    Returns:
        Bytes per cached token, zero for a shared layer.
    """
    if layer.shares_kv:
        return 0
    return layer.kv_tensors * layer.kv_heads * layer.head_dim * KV_DTYPE_BYTES[kv_dtype]


def kv_growth_bytes_per_token(shape: ModelShape, kv_dtype: str = "fp16") -> int:
    """Compute the KV bytes each context token adds, windows excluded.

    Only global layers scale with context. Sliding layers stop at
    their padded window and belong to `kv_window_pool_bytes`. Shared
    layers add nothing.

    Args:
        shape: The model's attention geometry.
        kv_dtype: KV-cache element dtype (``fp16``, ``bf16``, or ``fp8``).

    Returns:
        Context-scaled bytes per token.

    Raises:
        KeyError: If ``kv_dtype`` is not a known dtype.

    Examples:
        The north-star target stores ~196 KiB per token at fp16:

        ```python
        from vramfit.domain.budget import ModelShape, kv_growth_bytes_per_token

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        assert kv_growth_bytes_per_token(shape) == 200_704
        ```
    """
    return sum(
        _layer_token_bytes(layer, kv_dtype)
        for layer in shape.kv_layers
        if layer.window is None
    )


def kv_window_pool_bytes(shape: ModelShape, kv_dtype: str = "fp16") -> int:
    """Compute the sliding layers' KV pool at window saturation.

    Each sliding layer caps its cache at its window plus the
    runtime's `KV_WINDOW_PAD_TOKENS` padding (#431). Past the largest
    padded window this pool is a constant per sequence.

    Args:
        shape: The model's attention geometry.
        kv_dtype: KV-cache element dtype (``fp16``, ``bf16``, or ``fp8``).

    Returns:
        Saturated window-pool bytes, zero for a uniform shape.

    Raises:
        KeyError: If ``kv_dtype`` is not a known dtype.

    Examples:
        Gemma 4 31B holds 1,200 MiB of saturated windows at fp16:

        ```python
        from vramfit.domain.budget import KVLayer, ModelShape, kv_window_pool_bytes

        shape = ModelShape(
            kv_layers=(KVLayer(kv_heads=16, head_dim=256, window=1024),) * 50
        )
        assert kv_window_pool_bytes(shape) == 1_258_291_200
        ```
    """
    return sum(
        _layer_token_bytes(layer, kv_dtype) * (layer.window + KV_WINDOW_PAD_TOKENS)
        for layer in shape.kv_layers
        if layer.window is not None
    )


def kv_cache_bytes(
    shape: ModelShape,
    context: int,
    kv_dtype: str = "fp16",
    sequences: int = 1,
) -> int:
    """Compute total KV-cache bytes for a context length and batch.

    Sums each layer's actual allocation: a global layer caches
    ``context`` tokens, a sliding layer
    ``min(context, window + KV_WINDOW_PAD_TOKENS)`` tokens (#431),
    and a shared layer none. Each sequence pays the full sum.

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
    total = 0
    for layer in shape.kv_layers:
        if layer.window is None:
            tokens = context
        else:
            tokens = min(context, layer.window + KV_WINDOW_PAD_TOKENS)
        total += _layer_token_bytes(layer, kv_dtype) * tokens
    return total * sequences


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
