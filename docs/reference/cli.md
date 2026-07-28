---
status: draft
---

# CLI reference

> **Status: draft** — `version` and the `scan` stub exist in code; `plan` and
> `pack` are design-stage.

## `quantfit version`

Implemented. Prints the installed package version.

```console
$ quantfit version
quantfit 0.1.0
```

## `quantfit scan` *(stub)*

Exists as a stub that exits with code 1. Planned signature:

```
quantfit scan MODEL_ID
  --calibration TEXT     Calibration dataset name or path  [default: wikitext]
  --precisions TEXT      Comma-separated candidate bit-widths  [default: 8,4,3,2]
  --group-by TEXT        Layer grouping granularity (layer | tensor)  [default: layer]
  --resume               Continue an interrupted scan from its checkpoint
  --out PATH             Output sensitivity map  [default: sensitivity.json]
```

## `quantfit plan` *(planned)*

```
quantfit plan SENSITIVITY_MAP
  --vram SIZE            Hard VRAM ceiling (e.g. 24GiB)
  --kv-headroom SIZE     Reserved for KV cache + runtime  [default: 4GiB]
  --pin TEXT             Pin groups to a precision, repeatable (glob=bits)
  --out PATH             Output recipe  [default: recipe.json]
```

Exits non-zero if no recipe fits the budget, reporting the gap.

## `quantfit pack` *(planned)*

```
quantfit pack MODEL_ID
  --recipe PATH          Recipe produced by `quantfit plan`
  --runtime TEXT         Target runtime (vllm)  [default: vllm]
  --out PATH             Output checkpoint directory
```
