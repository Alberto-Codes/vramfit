---
status: draft
---

# First run

> **Status: draft** — every command below ran on 2026-09-03 with
> `vramfit` 0.4.0 from PyPI, in a clean virtual environment with no
> extras. The expected output comes from that run. A later release
> can move the figures, so check `vramfit version` against 0.4.0.
> The maintainer ruled this path on 2026-09-03 under chart #481
> ([#483](https://github.com/Alberto-Codes/vramfit/issues/483)).

This tutorial takes you from an empty directory to a mixed-precision
recipe for Nemotron Super 49B under 24 GiB. It then reads how much
context the recipe leaves. Both steps run on a CPU in under one
second.

You need no GPU, no CUDA, no torch, no llama.cpp, no model weights,
and no Hugging Face account. The inputs are two public files that
total 108 KB.

## Prerequisites

- Python 3.12 or newer
- `curl`, or a browser, for two downloads

## 1. Install vramfit with no extras

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install vramfit
vramfit version
```

Expected:

```
vramfit 0.4.0
```

The base install declares two dependencies, typer and structlog. It
resolves to vramfit plus 8 packages and no torch
([ADR-0005](../adr/0005-heavy-deps-as-extras.md)). `plan` and
`capacity` are pure Python. `scan` and `validate` need the scan extra
and a GPU. `pack` needs the gguf extra for its GGUF reads, the pack
extra for the convert interpreter, and a llama.cpp checkout. The
[getting-started tutorial](getting-started.md) covers all three.

## 2. Download the sensitivity map and the model config

```bash
curl -LO https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps/resolve/main/sensitivity-64k-kquant-imx-no2-sized.json
curl -LO https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/resolve/main/config.json
```

The first file is the published
[sensitivity map](../reference/sensitivity-map.md) the shipped 49B
pack solved from. Its name records the scan: 64k calibration tokens,
k-quant precisions, an importance matrix, no 2-bit column
([ADR-0021](../adr/0021-runtime-frame-measurement.md)), and per-tensor
sizes for protection rules
([ADR-0022](../adr/0022-within-layer-protections.md)). The map carries
each group's bytes, so `plan` prices the 82 groups it holds without a
checkpoint ([ADR-0029](../adr/0029-plan-independent-size-source.md)).
Its SHA-256 is
`3f0a914cc3b0889aa94fe2621f195fd398758c913d221eb4f5af19a7a08b6c36`.

The second file is the base model's `config.json`. Only `capacity`
reads it, for the attention shape.

## 3. Plan a recipe

```bash
vramfit plan sensitivity-64k-kquant-imx-no2-sized.json --vram 24GiB --kv-headroom 3616MiB --out recipe.json
```

Expected:

```
no --checkpoint: this plan prices the 82 groups the map carries and reads no other size source (ADR-0029)
planned 82 groups for llama.cpp: 20.46 GiB of 20.47 GiB weight budget, predicted damage 0.2776, 161 downgrades -> recipe.json
```

The first line states the size source. The second line is the
result. The solver started every group at 8-bit and took 161
downgrade steps until the recipe fit the 20.47 GiB weight budget.
The [recipe](../reference/recipe.md) holds 80 groups at 3-bit, one
at 4-bit (`model.layers.3`), and one at 8-bit (`model.layers.0`).
Its `plan.trace` block records every step with its damage delta, the
bytes it freed, and the damage-per-byte ratio that chose it.

[Damage](../reference/glossary.md) is the output divergence a group
takes at a precision, measured by the scan. Higher is worse. The
predicted damage sums the chosen groups.

The headroom figure is the shipped recipe's. Its
[budget table](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF#what-fit24gib-means)
reserves 3,791,650,816 B for KV cache and runtime, which is 3616 MiB.
The default headroom of 4 GiB leaves 20.00 GiB for weights. No recipe
fits that budget on this map, so `plan` exits 1 and reports a
56.39 MiB gap. Drop `--kv-headroom 3616MiB` from the command above
and run it to see the refusal.

## 4. Read the capacity

```bash
vramfit capacity recipe.json --model-config config.json --kv-dtype fp8 --context 16384
```

Expected:

```
attention layers      49  (KV grows 100352 bytes/token, fp8)
VRAM total            24.00 GiB
- weights (recipe)    20.46 GiB
- runtime overhead    2.00 GiB
vision                none claimed — nothing subtracted
= KV headroom         1.54 GiB
max context           16482 tokens  (1 sequence)
max sequences         1  (at 16384 tokens)
```

The [capacity readout](../reference/glossary.md) runs the budget
ledger in reverse. The card holds 24 GiB. The recipe's predicted
weights take 20.46 GiB and the default runtime reservation takes
2 GiB. The remaining 1.54 GiB holds a KV cache of 16,482 tokens at
fp8, so the recipe serves one 16,384-token sequence. The 49 attention
layers come from the config. Nemotron Super 49B carries 80 decoder
layers, and its neural architecture search removed attention from 31
of them.

## What the damage figure means

The plain solve predicts damage 0.2776. The shipped `recipe.json`
predicts 0.3905. Both figures sum per-group measurements from one
map, so they compare under
[ADR-0027](../adr/0027-instrument-frame-matching.md). Do not read the
lower number as the better pack. The shipped solve buys quality the
prediction never prices.

The shipped solve ends one step further. It holds 47 `attn_v`
tensors at a 5-bit floor and one at a 4-bit floor, all inside 3-bit
groups ([ADR-0022](../adr/0022-within-layer-protections.md)). The 47
floors cost about 97 MiB. To pay for them, the solver downgraded
`model.layers.3` from 4-bit to 3-bit as a 162nd step, at a predicted
damage of 0.113.
The two recipes agree on every other group. Protections price by size
only, and predicted damage stays the group-level sum. Their benefit
shows in measurement. The protected recipe won the runtime-frame
head-to-head against the size-matched community quant, 7.8σ paired
([evidence](../explanation/evaluating-packed-models.md#the-fifteenth-data-point-the-pipeline-packs-its-own-winner)).

## Optional: reproduce the shipped recipe

The shipped
[`recipe.json`](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF/blob/main/recipe.json)
records its full solve in the `plan` block. That block holds the
budget bytes, the 0.005 format overhead, 48 protection rules, 4
imatrix exclusions, and the 162-step trace. The
[#483 checks](https://github.com/Alberto-Codes/vramfit/issues/483#issuecomment-5534333284)
passed those values back to `plan` as flags. The output matched the
published file on every field. The rebuilt command runs to 113
arguments, so this page does not carry it. The
[fit-to-VRAM how-to](../how-to/fit-to-vram-budget.md#reproduce-the-shipped-49b-recipe)
carries it in full, as its permanent home since 2026-09-04.

One trap: the glob `--protect "*.self_attn.v_proj.weight=5"` matches
the same 48 tensors but holds `model.layers.3.self_attn.v_proj.weight`
at 5-bit, where the recipe holds it at 4-bit. It predicts
21,958,391,121 bytes instead of 21,957,337,301. Name the 48 tensors
one by one, as the recipe does.

## Where next

- [Getting started](getting-started.md) runs the full loop on a
  small model: scan, plan, validate, pack.
- [How to fit a model to a VRAM budget](../how-to/fit-to-vram-budget.md)
  covers headroom, pins, and protections.
- The [CLI reference](../reference/cli.md) lists every flag of
  `plan` and `capacity`.
