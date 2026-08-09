---
status: stable
---

# VRAM budget math

> **Status: stable** — implemented in `quantfit.domain.budget` and computable via
> `quantfit budget`; the worked example below uses the target model's real
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

Per token, across the whole stack:

```
kv_bytes_per_token = 2 × n_attention_layers × n_kv_heads × head_dim × bytes_per_elem
```

(2 = keys + values.) Multiply by context length × concurrent sequences.
Grouped-query attention (small `n_kv_heads`) is what makes long context
affordable; FP8 KV cache halves it again. This is why the budget must be
planned *jointly*: every GiB saved on weights is context length gained.

## Runtime overhead

CUDA context, allocator workspace, activation scratch, fragmentation.
Planning figure: **1.5–2 GiB** on a 24 GiB card until measured.

## Worked example: the north-star target

Computed with `quantfit budget --model-config <nemotron config.json>` from
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
