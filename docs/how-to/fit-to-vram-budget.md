---
status: draft
---

# How to fit a model to a VRAM budget

> **Status: draft** — `quantfit plan` and `quantfit budget` are implemented
> and tested. No recipe has been packed and served yet, so the
> quality-review advice below is reasoned, not measured.

## Goal

Turn a sensitivity map into a mixed-precision recipe that lands a model under
a hard VRAM ceiling with acceptable damage.

## Work out the real budget

The card's sticker capacity is not the budget. Subtract:

- **KV cache** — grows with context length and batch size; see
  [VRAM budget math](../explanation/vram-budget.md)
- **Runtime overhead** — CUDA context, workspace, fragmentation (~1–2 GiB)

For a 24 GiB RTX 4090 serving a 49B model at moderate context, a realistic
weight budget is roughly 18–19 GiB.

## Basic invocation

```bash
uv run quantfit plan sensitivity.json \
  --vram 24GiB \
  --kv-headroom 4GiB \
  --out recipe.json
```

## Reading the output

The [recipe](../reference/recipe.md) lists each layer group, its assigned
precision, its size contribution, and its predicted damage score. Review the
extremes before packing:

- Groups pinned at high precision are where your budget went — sanity-check
  they match known-fragile structures (first/last blocks, attention).
- Groups crushed to 2-bit are where quality risk concentrates — if one looks
  load-bearing for your workload, pin it: `--pin "layers.0.*=8"`.

## When the solver can't fit

If no recipe satisfies the budget, the answer is honest failure with the gap
size — not a silently terrible recipe. Options then: shrink KV headroom
(shorter context), allow a lower precision floor, or accept CPU offload for
specific groups (out of scope for v1).
