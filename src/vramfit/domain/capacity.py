"""Capacity readout: the VRAM budget arithmetic run in reverse.

The forward ledger prices a serving shape and leaves a weight budget
(`vramfit.domain.budget`). This module inverts it (#422). Given the
KV headroom a packed recipe leaves, it reports the largest context,
the largest sequence count at a fixed context, and an image capacity
at a caller-supplied image token cost.

The inverse is piecewise on a mixed sliding/global stack. Sliding
layers saturate at their padded windows while global layers keep
growing,
so no single bytes-per-token scalar can invert the cost. The context
solver binary-searches `kv_cache_bytes` itself over integers, so the
result is exact at the fit boundary: the returned context fits and
one more token does not.

`max_context_tokens` and `max_sequences` return ``None`` when the
KV cache stops growing before the headroom runs out. Context or
concurrency is then not KV-limited, and a larger number always fits.

Examples:
    Read the context a 6 GiB KV headroom buys on the north-star
    target:

    ```python
    from vramfit.domain.budget import ModelShape
    from vramfit.domain.capacity import max_context_tokens

    shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
    tokens = max_context_tokens(shape, kv_headroom_bytes=6 * 2**30)
    ```

See Also:
    - [vramfit.domain.budget][]: The forward pricing this inverts.
"""

from __future__ import annotations

from vramfit.domain.budget import (
    KV_WINDOW_PAD_TOKENS,
    ModelShape,
    kv_cache_bytes,
    kv_growth_bytes_per_token,
    kv_window_pool_bytes,
)


def max_context_tokens(
    shape: ModelShape,
    kv_headroom_bytes: int,
    kv_dtype: str = "fp16",
    sequences: int = 1,
) -> int | None:
    """Solve for the largest context whose KV cache fits the headroom.

    Binary-searches `kv_cache_bytes` over integer contexts. The
    result is exact: the returned context fits the headroom and one
    more token does not.

    Args:
        shape: The model's attention geometry.
        kv_headroom_bytes: Bytes available for the KV cache.
        kv_dtype: KV-cache element dtype (``fp16``, ``bf16``, or
            ``fp8``).
        sequences: Concurrent sequences sharing the headroom. The CLI
            admits only positive counts, and the domain does not
            re-check that bound.

    Returns:
        The largest context in tokens — 0 when not even one token
        fits — or ``None`` when every layer saturates at its padded
        window inside the headroom and context is not KV-limited.

    Raises:
        KeyError: If ``kv_dtype`` is not a known dtype and any layer
            allocates KV.

    Examples:
        Ten tokens of uniform-stack headroom invert exactly:

        ```python
        from vramfit.domain.budget import ModelShape, kv_growth_bytes_per_token
        from vramfit.domain.capacity import max_context_tokens

        shape = ModelShape.uniform(attn_layers=2, kv_heads=2, head_dim=4)
        headroom = 10 * kv_growth_bytes_per_token(shape)
        assert max_context_tokens(shape, headroom) == 10
        ```
    """
    if kv_headroom_bytes < 0:
        return 0
    growth = kv_growth_bytes_per_token(shape, kv_dtype) * sequences
    if growth == 0:
        saturated = kv_window_pool_bytes(shape, kv_dtype) * sequences
        if saturated <= kv_headroom_bytes:
            return None
        # `saturated > headroom >= 0` proves a costing sliding layer
        # exists, and its padded window bounds the search from above.
        hi = (
            max(layer.window for layer in shape.kv_layers if layer.window is not None)
            + KV_WINDOW_PAD_TOKENS
        )
    else:
        # The cache costs at least `growth` per token, so this bound
        # always overshoots the headroom.
        hi = kv_headroom_bytes // growth + 1
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if kv_cache_bytes(shape, mid, kv_dtype, sequences) <= kv_headroom_bytes:
            lo = mid
        else:
            hi = mid
    return lo


def max_sequences(
    shape: ModelShape,
    kv_headroom_bytes: int,
    context: int,
    kv_dtype: str = "fp16",
) -> int | None:
    """Solve for the largest sequence count at a fixed context.

    Every sequence pays the same KV allocation, so the count is the
    headroom divided by one sequence's cache, floored. Exact by
    construction.

    Args:
        shape: The model's attention geometry.
        kv_headroom_bytes: Bytes available for the KV cache.
        context: Context length in tokens each sequence holds.
        kv_dtype: KV-cache element dtype (``fp16``, ``bf16``, or
            ``fp8``).

    Returns:
        The largest sequence count — 0 when not even one sequence
        fits — or ``None`` when one sequence allocates nothing and
        concurrency is not KV-limited.

    Raises:
        KeyError: If ``kv_dtype`` is not a known dtype and any layer
            allocates KV.

    Examples:
        Three sequences of 10-token cache fit a 2000-byte headroom:

        ```python
        from vramfit.domain.budget import ModelShape
        from vramfit.domain.capacity import max_sequences

        shape = ModelShape.uniform(attn_layers=2, kv_heads=2, head_dim=4)
        assert max_sequences(shape, 2000, context=10) == 3
        ```
    """
    if kv_headroom_bytes < 0:
        return 0
    per_sequence = kv_cache_bytes(shape, context, kv_dtype, sequences=1)
    if per_sequence == 0:
        return None
    return kv_headroom_bytes // per_sequence


def image_capacity(tokens: int, image_token_cost: int) -> int:
    """Convert a token capacity into whole images at a ruled cost.

    The caller supplies the image token cost — vramfit rules no
    vision policy. #236 owns the multimodal VRAM ledger, and #419
    owns vision-quality claims.

    Args:
        tokens: Token capacity to convert.
        image_token_cost: Decoder tokens one image consumes.

    Returns:
        The number of whole images the capacity carries.

    Raises:
        ValueError: If ``image_token_cost`` is not positive, or
            ``tokens`` is negative.

    Examples:
        A 1000-token capacity carries three 256-token images:

        ```python
        from vramfit.domain.capacity import image_capacity

        assert image_capacity(1000, image_token_cost=256) == 3
        ```
    """
    if image_token_cost <= 0:
        raise ValueError(f"image token cost must be positive, got {image_token_cost}")
    if tokens < 0:
        raise ValueError(f"token capacity must not be negative, got {tokens}")
    return tokens // image_token_cost
