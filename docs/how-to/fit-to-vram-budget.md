---
status: draft
---

# How to fit a model to a VRAM budget

> **Status: draft** — `vramfit plan` and `vramfit budget` are implemented
> and tested, and a planned 49B recipe packed to 20.30 GiB against a
> 20.47 GiB weight budget and served
> ([evidence](../explanation/evaluating-packed-models.md)). Run
> `vramfit validate` between plan and pack — it measures the whole
> recipe against the prediction.

## Goal

Turn a sensitivity map into a mixed-precision recipe that lands a model under
a hard VRAM ceiling with acceptable damage.

## Work out the real budget

The card's sticker capacity is not the budget. Subtract:

- **KV cache** — grows with context length and batch size; see
  [VRAM budget math](../explanation/vram-budget.md)
- **Runtime overhead** — CUDA context, workspace, fragmentation (~1–2 GiB)

For a 24 GiB RTX 4090 serving a 49B model at 16k context, the measured
loop budgeted 20.47 GiB for weights (fp8 KV) and packed 20.30 GiB. At
fp16 KV the budget drops to 18.94 GiB.

## Basic invocation

```bash
vramfit plan sensitivity.json \
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
- Since [ADR-0021](../adr/0021-runtime-frame-measurement.md) the
  solver buys a width only against a runtime-frame price. The 30B
  target's 2-bit price arrived 2026-08-14 and failed, so current
  practice there plans on a map copy without that column. If a group
  still looks load-bearing for your workload, pin it:
  `--pin "*.layers.0=8"` (patterns match the full group name,
  `model.layers.0`). A glob that sweeps a group the runtime holds at
  the F16 passthrough (a router's `mixer.gate`) skips that group and
  prints a warning naming it. A pin that resolves to that one held
  group refuses instead. The rule is in the
  [CLI reference](../reference/cli.md#vramfit-plan).
- When one tensor class inside otherwise-fine groups is the fragile
  part (the twelfth data point's `attn_v`), protect it instead of
  pinning whole groups:
  `--protect "*.self_attn.v_proj.weight=5"` holds the matched
  tensors at a 5-bit floor and prices the cost by size only
  ([ADR-0022](../adr/0022-within-layer-protections.md)). The map
  needs per-tensor sizes — new scans record them, and
  `scripts/backfill_tensor_sizes.py` annotates older maps from the
  checkpoint's safetensors headers.

## When the solver can't fit

If no recipe satisfies the budget, the answer is honest failure with the gap
size — not a silently terrible recipe. Options then: shrink KV headroom
(shorter context), allow a lower precision floor, or accept CPU offload for
specific groups (out of scope for v1).

## Reproduce the shipped 49B recipe

The published
[49B pack](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF)
ships the `recipe.json` it was packed from. The command below rebuilds
that file from the published map, field for field. The maintainer
ruled this section its permanent home on 2026-09-04
([#495](https://github.com/Alberto-Codes/vramfit/issues/495)). The
[first-run tutorial](../tutorials/first-run.md) solves a plainer
recipe on purpose and keeps these flags off its path.

Download the map first:

```bash
curl -LO https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps/resolve/main/sensitivity-64k-kquant-imx-no2-sized.json
```

Every flag below comes from the published recipe's `plan` block. A
script read that block and printed the command on 2026-09-04, with
`vramfit` 0.4.0. The output equalled the published file on every
field: assignments, protected tensors, the 162-step trace, predicted
bytes 21,957,337,301, and predicted damage 0.3905
([#483 checks](https://github.com/Alberto-Codes/vramfit/issues/483#issuecomment-5534333284)).
The glob form `--protect "*.self_attn.v_proj.weight=5"` does not
reproduce it. It matches the same 48 tensors but floors layer 3 at
5-bit, where the recipe holds 4-bit.

```bash
vramfit plan sensitivity-64k-kquant-imx-no2-sized.json \
  --vram 25769803776 \
  --kv-headroom 3791650816 \
  --format-overhead 0.005 \
  --runtime llama.cpp \
  --protect model.layers.1.self_attn.v_proj.weight=5 \
  --protect model.layers.2.self_attn.v_proj.weight=5 \
  --protect model.layers.4.self_attn.v_proj.weight=5 \
  --protect model.layers.5.self_attn.v_proj.weight=5 \
  --protect model.layers.8.self_attn.v_proj.weight=5 \
  --protect model.layers.9.self_attn.v_proj.weight=5 \
  --protect model.layers.10.self_attn.v_proj.weight=5 \
  --protect model.layers.12.self_attn.v_proj.weight=5 \
  --protect model.layers.13.self_attn.v_proj.weight=5 \
  --protect model.layers.14.self_attn.v_proj.weight=5 \
  --protect model.layers.15.self_attn.v_proj.weight=5 \
  --protect model.layers.16.self_attn.v_proj.weight=5 \
  --protect model.layers.17.self_attn.v_proj.weight=5 \
  --protect model.layers.18.self_attn.v_proj.weight=5 \
  --protect model.layers.19.self_attn.v_proj.weight=5 \
  --protect model.layers.20.self_attn.v_proj.weight=5 \
  --protect model.layers.21.self_attn.v_proj.weight=5 \
  --protect model.layers.22.self_attn.v_proj.weight=5 \
  --protect model.layers.23.self_attn.v_proj.weight=5 \
  --protect model.layers.24.self_attn.v_proj.weight=5 \
  --protect model.layers.25.self_attn.v_proj.weight=5 \
  --protect model.layers.26.self_attn.v_proj.weight=5 \
  --protect model.layers.27.self_attn.v_proj.weight=5 \
  --protect model.layers.28.self_attn.v_proj.weight=5 \
  --protect model.layers.29.self_attn.v_proj.weight=5 \
  --protect model.layers.30.self_attn.v_proj.weight=5 \
  --protect model.layers.31.self_attn.v_proj.weight=5 \
  --protect model.layers.32.self_attn.v_proj.weight=5 \
  --protect model.layers.33.self_attn.v_proj.weight=5 \
  --protect model.layers.34.self_attn.v_proj.weight=5 \
  --protect model.layers.35.self_attn.v_proj.weight=5 \
  --protect model.layers.36.self_attn.v_proj.weight=5 \
  --protect model.layers.37.self_attn.v_proj.weight=5 \
  --protect model.layers.38.self_attn.v_proj.weight=5 \
  --protect model.layers.39.self_attn.v_proj.weight=5 \
  --protect model.layers.40.self_attn.v_proj.weight=5 \
  --protect model.layers.41.self_attn.v_proj.weight=5 \
  --protect model.layers.52.self_attn.v_proj.weight=5 \
  --protect model.layers.71.self_attn.v_proj.weight=5 \
  --protect model.layers.72.self_attn.v_proj.weight=5 \
  --protect model.layers.73.self_attn.v_proj.weight=5 \
  --protect model.layers.74.self_attn.v_proj.weight=5 \
  --protect model.layers.75.self_attn.v_proj.weight=5 \
  --protect model.layers.76.self_attn.v_proj.weight=5 \
  --protect model.layers.77.self_attn.v_proj.weight=5 \
  --protect model.layers.78.self_attn.v_proj.weight=5 \
  --protect model.layers.79.self_attn.v_proj.weight=5 \
  --protect model.layers.3.self_attn.v_proj.weight=4 \
  --exclude-imatrix model.layers.1.self_attn.v_proj.weight \
  --exclude-imatrix model.layers.2.self_attn.v_proj.weight \
  --exclude-imatrix model.layers.3.self_attn.v_proj.weight \
  --exclude-imatrix model.layers.5.self_attn.v_proj.weight \
  --out recipe.json
```

The byte values are the recipe's own: `--vram 25769803776` is 24 GiB
and `--kv-headroom 3791650816` is 3616 MiB. The four `--exclude-imatrix`
flags are the exclusions the reconstruction check demanded
([ADR-0023](../adr/0023-imatrix-exclusions.md)).
