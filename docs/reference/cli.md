---
status: draft
---

# CLI reference

> **Status: draft** — `version`, `budget`, and `plan` are implemented.
> `scan` is a stub and `pack` is design-stage.

## `quantfit version`

Implemented. Prints the installed package version.

```console
$ quantfit version
quantfit 0.1.0
```

## `quantfit budget`

Implemented. Prints the VRAM budget breakdown. The `--kv-headroom` value
for `plan` is the sum of the KV-cache and runtime-overhead lines. The attention shape comes from
exactly one source: `--model-config` (a Hugging Face `config.json` —
DeciLM NAS configs with skipped-attention blocks are handled) or the
manual triple.

```
quantfit budget
  --vram SIZE            Total VRAM  [default: 24GiB]
  --context INT          Context length in tokens  [default: 16384]
  --kv-dtype TEXT        fp16 | bf16 | fp8  [default: fp16]
  --sequences INT        Concurrent sequences  [default: 1]
  --overhead SIZE        Runtime overhead reservation  [default: 2GiB]
  --model-config PATH    Model config.json to derive the shape from
  --attn-layers INT      Attention layer count (manual shape)
  --kv-heads INT         KV heads per layer (manual shape)
  --head-dim INT         Head dimension (manual shape)
```

Exits 1 when nothing is left for weights, and 2 on conflicting or missing
shape sources.

```console
$ quantfit budget --model-config config.json --vram 24GiB --kv-dtype fp8
attention layers      49  (KV 100352 bytes/token, fp8)
VRAM total            24.00 GiB
- KV cache            1.53 GiB  (16384 tokens x 1 seq)
- runtime overhead    2.00 GiB
= weight budget       20.47 GiB
```

## `quantfit plan`

Implemented. Solves a sensitivity map into a recipe under a VRAM budget.

```
quantfit plan SENSITIVITY_MAP
  --vram SIZE            Hard VRAM ceiling (e.g. 24GiB)  [required]
  --kv-headroom SIZE     Reserved for KV cache + runtime  [default: 4GiB]
  --pin TEXT             Pin groups to a precision, repeatable (glob=bits)
  --out PATH             Output recipe  [default: recipe.json]
  --format-overhead F    Quantization-format overhead fraction  [default: 0.05]
```

Pin semantics: patterns are case-sensitive `fnmatch` globs matched against
the full group name (`--pin "model.layers.0.*=8"`). A pattern that matches
no group is an error (typo detection). Later pins override earlier ones for
overlapping groups — repeating a pattern moves it to the last position.
Pins are recorded in the recipe in their effective order.

Exit codes: 1 when the map is invalid, the output is unwritable, or no
recipe fits the budget (the gap is reported). Exit 2 on malformed options
(`--pin` not of the form `pattern=bits` with positive bits, unparseable
sizes, negative `--format-overhead`).

## `quantfit scan` *(stub)*

Exists as a stub that exits with code 1. Planned signature:

```
quantfit scan MODEL_ID
  --calibration TEXT     Calibration dataset name or path  [default: wikitext]
  --precisions TEXT      Comma-separated candidate bit-widths; the useful set
                         depends on the target runtime's kernels (vLLM today:
                         8, 4-int, 4-fp — see ADR-0004)  [default: 8,4]
  --group-by TEXT        Layer grouping granularity (layer | tensor)  [default: layer]
  --resume               Continue an interrupted scan from its checkpoint
  --out PATH             Output sensitivity map  [default: sensitivity.json]
```

## `quantfit pack` *(planned)*

```
quantfit pack MODEL_ID
  --recipe PATH          Recipe produced by `quantfit plan`
  --runtime TEXT         Target runtime (vllm)  [default: vllm]
  --out PATH             Output checkpoint directory
```
