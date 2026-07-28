---
status: draft
---

# Sensitivity map format

> **Status: draft** — implemented in
> `quantfit.adapters.outbound.sensitivity_map_json`, whose loader enforces
> everything described here. The scan pipeline that *produces* these
> files does not exist yet.

The sensitivity map is the output of `quantfit scan` and the input to
`quantfit plan`: JSON, one entry per (layer group × candidate precision).

```json
{
  "quantfit_schema": 1,
  "model_id": "nvidia/Nemotron-Super-49B",
  "scan": {
    "metric": "kl_divergence",
    "calibration": "wikitext",
    "calibration_tokens": 131072,
    "precisions": [8, 4, 3, 2],
    "group_by": "layer",
    "started_at": "2026-07-27T00:00:00Z"
  },
  "groups": [
    {
      "name": "model.layers.0.self_attn",
      "tensors": ["q_proj", "k_proj", "v_proj", "o_proj"],
      "bytes_fp16": 100000000,
      "sensitivity": {
        "8": 0.0001,
        "4": 0.0042,
        "3": 0.0311,
        "2": 0.4170
      }
    }
  ]
}
```

## Field notes

- **`sensitivity`** — divergence of the perturbed model's output from the
  full-precision reference, measured per
  [ADR-0006](../adr/0006-sensitivity-metric.md) (metric choice still open).
  Higher = more damage. Values are comparable *within* a scan, not across
  scans with different calibration sets.
- **`bytes_fp16`** — group size at reference precision; the solver derives
  per-precision sizes from this plus quantization-format overhead.
- **`groups`** — granularity is set by `--group-by`. Marginal (one group at a
  time) measurement is assumed — interaction effects between groups are a
  known blind spot recorded in ADR-0006. Group names must be unique, and
  every group's `sensitivity` keys must equal `scan.precisions` exactly
  (the v1 loader rejects partially-scanned groups).
