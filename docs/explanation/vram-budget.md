---
status: draft
---

# VRAM budget math

> **Status: draft** — the formulas are standard; the overhead constants are
> estimates until we measure them on the reference box.

The plan step optimizes against a *weight budget*, which is what's left of
the card after everything else takes its cut.

```
weight_budget = vram_total − kv_cache − runtime_overhead
```

## Weights

```
weight_bytes ≈ Σ_groups (params_in_group × bits_group ÷ 8) × format_overhead
```

`format_overhead` covers quantization metadata (scales, zero-points, block
structure) — typically 3–10% depending on format and block size. Embedding
and output-projection tensors are usually kept at 8-bit or higher and must be
counted; on large-vocab models they are gigabytes, not a rounding error.

## KV cache

Per token, across the whole stack:

```
kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_elem
```

(2 = keys + values.) Multiply by context length × concurrent sequences.
Grouped-query attention (small `n_kv_heads`) is what makes long context
affordable; FP8 KV cache halves it again. This is why the budget must be
planned *jointly*: every GiB saved on weights is context length gained.

## Runtime overhead

CUDA context, allocator workspace, activation scratch, fragmentation.
Planning figure: **1.5–2 GiB** on a 24 GiB card until measured.

## Worked example: the north-star target

RTX 4090 (24 GiB = 24.56 GB), Nemotron Super 49B, 16k context, 1 sequence:

| Item | Estimate |
|------|----------|
| VRAM total | 24.5 GB |
| Runtime overhead | −2.0 GB |
| KV cache @16k, fp8 | −2 to −3 GB (needs the architecture's real head counts) |
| **Weight budget** | **~19.5–20.5 GB** |

Against 49B parameters, that budget forces an *average* of ~3.2–3.3
bits/parameter — below uniform-4-bit, above uniform-3. Selectivity is not an
optimization here; it's the only way the average can be spent unevenly enough
to preserve quality. This arithmetic is the entire reason the project exists.
