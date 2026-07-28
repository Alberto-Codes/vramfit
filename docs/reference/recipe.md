---
status: sketch
---

# Recipe format

> **Status: sketch** — proposed schema; will change as the plan/pack pipeline
> lands.

The recipe is the output of `quantfit plan` and the input to `quantfit pack`:
JSON, one precision assignment per layer group, plus the budget accounting
that produced it.

```json
{
  "quantfit_schema": 1,
  "model_id": "nvidia/Nemotron-Super-49B",
  "plan": {
    "vram_budget_bytes": 25769803776,
    "kv_headroom_bytes": 4294967296,
    "weight_budget_bytes": 20401094656,
    "predicted_total_bytes": 20208459776,
    "predicted_damage": 0.0871,
    "solver": "greedy-damage-per-byte",
    "pins": {}
  },
  "assignments": [
    {
      "group": "model.layers.0.self_attn",
      "bits": 8,
      "bytes": 50000000,
      "damage": 0.0001
    }
  ]
}
```

## Field notes

- **`assignments`** — every group from the sensitivity map appears exactly
  once. `bytes` includes quantization-format overhead (scales, zero-points).
- **`predicted_damage`** — sum of per-group damage at the chosen precisions.
  A *prediction* from marginal measurements, not a guarantee; the pack step's
  post-quantization eval is the ground truth.
- **`solver`** — which strategy produced the recipe (see
  [ADR-0007](../adr/0007-recipe-solver-strategy.md)); recorded so recipes are
  reproducible and comparable.
- **`pins`** — user-forced precision overrides, kept verbatim for provenance.
