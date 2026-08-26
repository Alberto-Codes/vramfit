---
status: stable
---

# VRAM budget math

> **Status: stable** — implemented in `vramfit.domain.budget` and computable via
> `vramfit budget`; the worked example below uses the target model's real
> config. The runtime-overhead constant remains a planning figure until
> measured under the serving runtime (llama.cpp per
> [ADR-0010](../adr/0010-sub-4-bit-serving-path.md)) on the reference
> box.

The plan step optimizes against a *weight budget*, which is what's left of
the card after everything else takes its cut.

```
weight_budget = vram_total − kv_cache − runtime_overhead
```

## Weights

```
weight_bytes ≈ Σ_groups (params_in_group × effective_bits_group ÷ 8) × (1 + format_overhead)
```

`effective_bits` is what the runtime's quantization type really spends per
weight, block scales included — Q4_K stores a nominal 4-bit group at 4.5
bits/weight. When the target runtime has a measured table
([ADR-0014](../adr/0014-per-type-effective-bits.md)) the solver uses it and
`format_overhead` shrinks to a residual (~0.5%) for file metadata and
unquantized tensors. Without a table, nominal bits stand in and the scalar
(default 5%) has to cover the quantization metadata too. Embedding and
output-projection tensors are usually kept at 8-bit or higher and must be
counted; on large-vocab models they are gigabytes, not a rounding error.

## KV cache

Each attention layer prices its own cache (`KVLayer`, #421). Per layer
and sequence:

```
layer_kv_bytes = kv_tensors × n_kv_heads × head_dim × bytes_per_elem × cached_tokens
```

Four mechanisms decide `cached_tokens` and `kv_tensors`:

- A **global** layer caches `context` tokens — it grows with context.
- A **sliding** layer caches `min(context, window)` tokens. Past its
  window the allocation is a constant.
- A **shared-KV** layer reuses an earlier layer's cache and allocates
  nothing (`num_kv_shared_layers`).
- `kv_tensors` is 2 for an independent K and V pair, and 1 when the
  model stores one tensor for both (`attention_k_eq_v`).

The stack's total therefore splits into two terms: **KV growth**
(`kv_growth_bytes_per_token`, the global layers' bytes per context
token) and the **window pool** (`kv_window_pool_bytes`, the sliding
layers' saturated bytes per sequence). For a uniform full-attention
stack the pool is zero and the familiar formula holds:

```
kv_growth_bytes_per_token = 2 × n_attention_layers × n_kv_heads × head_dim × bytes_per_elem
```

(2 = keys + values.) Multiply by context length × concurrent sequences.
Grouped-query attention (small `n_kv_heads`) is what makes long context
affordable; FP8 KV cache halves it again. This is why the budget must be
planned *jointly*: every GiB saved on weights is context length gained.

### Worked example: Gemma 4 31B (mixed sliding/global)

From the official config (verified 2026-08-25, #423): 60 layers — 50
sliding (window 1024, 16 KV heads × width 256, K+V pair) and 10 global
(4 KV heads × width 512, one K=V tensor). At fp16, one sequence:

- KV growth: `10 × 4 × 512 × 1 × 2` = **40,960 B/token**;
- window pool: `50 × 16 × 256 × 2 × 2 × 1024` = **800 MiB**;
- total: **5.78 GiB at 128k context**, **10.78 GiB at 256k**.

Past ~1k tokens the card pays 40 KiB per extra token instead of the
~0.82 MiB the same 60 layers would charge priced fully global. That is the
arithmetic behind #423's capacity claim: every GiB of KV headroom buys
~26.2k tokens once the windows saturate.

## Capacity readout: the ledger in reverse

The forward ledger takes context and concurrency as inputs. But the
recipe's payoff — the VRAM its compression frees — deserves a readout of
its own (#422). Invert the ledger:

```
kv_headroom  = vram_total − weight_bytes(recipe) − runtime_overhead
max_context  = the largest context whose KV cache fits kv_headroom
max_sequences = kv_headroom ÷ kv_cache_bytes(shape, context, 1 sequence)
```

On a uniform stack the first inverse is one division by KV growth. On a
mixed stack it is piecewise: sliding terms saturate while global terms
grow, so `vramfit.domain.capacity` binary-searches `kv_cache_bytes`
itself over integer contexts. The result is exact at the fit boundary —
the returned context fits and one more token does not. An all-sliding
stack whose saturated pool fits the headroom has no finite maximum, and
the readout says so (`unbounded`) rather than inventing one.

On Gemma 4 31B the inverse reproduces the forward figures: a recipe
that leaves 5.78 GiB of KV headroom reads back exactly 128k tokens,
and each further GiB buys 26,214 tokens (2³⁰ ÷ 40,960). The same
headroom reads as concurrency at a fixed context — the serving
interpretation #68 parks — and as an image capacity when the caller
supplies a ruled image token cost per #236/#419. `vramfit capacity`
prints all three from a packed recipe.

## Runtime overhead

CUDA context, allocator workspace, activation scratch, fragmentation.
Planning figure: **1.5–2 GiB** on a 24 GiB card until measured.

## Worked example: the north-star target

Computed with `vramfit budget --model-config <nemotron config.json>` from
the real checkpoint config: 80 blocks of which **49 have attention** (31 are
NAS `no_op` blocks), GQA with 8 KV heads × head_dim 128 → **200,704 KV
bytes/token at fp16, 100,352 at fp8**.

RTX 4090 (24 GiB), 16k context, 1 sequence:

| Item | fp16 KV | fp8 KV |
|------|---------|--------|
| VRAM total | 24.00 GiB | 24.00 GiB |
| − KV cache @16k | 3.06 GiB | 1.53 GiB |
| − runtime overhead | 2.00 GiB | 2.00 GiB |
| **= weight budget** | **18.94 GiB** | **20.47 GiB** |

Against ~49B parameters (plus format overhead), the fp8-KV budget forces an
*average* of ~3.5 bits/parameter; fp16 KV forces ~3.3. Below uniform-4-bit,
above uniform-3. Selectivity is not an optimization here; it's the only way
the average can be spent unevenly enough to preserve quality. This
arithmetic is the entire reason the project exists — and it's also why
[ADR-0010](../adr/0010-sub-4-bit-serving-path.md) moves the benchmark's
serving path to GGUF, past vLLM's 4-bit kernel floor.
