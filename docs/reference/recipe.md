---
status: draft
---

# Recipe format

> **Status: draft** — implemented in `quantfit.artifacts.Recipe`; the
> loader enforces everything described here.

The recipe is the output of `quantfit plan` and the input to `quantfit pack`:
JSON, one precision assignment per layer group, plus the budget accounting
that produced it.

```json
{
  "quantfit_schema": 1,
  "model_id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
  "plan": {
    "vram_budget_bytes": 25769803776,
    "kv_headroom_bytes": 4294967296,
    "weight_budget_bytes": 21474836480,
    "predicted_total_bytes": 20208459776,
    "predicted_damage": 0.0871,
    "solver": "greedy-damage-per-byte",
    "pins": {"model.layers.0.*": 8},
    "format_overhead": 0.05,
    "trace": [
      {
        "step": 1,
        "group": "model.layers.17.mlp",
        "from_bits": 8,
        "to_bits": 4,
        "damage_delta": 0.0004,
        "bytes_freed": 210000000,
        "ratio": 1.9e-12
      }
    ]
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
  once, in map order. `bytes` includes quantization-format overhead
  (scales, zero-points). `damage` is the *measured* value at the assigned
  precision — an all-8-bit recipe still carries the measured 8-bit damage.
- **`predicted_damage`** — sum of per-group damage at the chosen precisions.
  A *prediction* from marginal measurements, not a guarantee; the pack step's
  post-quantization eval is the ground truth.
- **`solver`** — which strategy produced the recipe (see
  [ADR-0007](../adr/0007-recipe-solver-strategy.md)); recorded so recipes are
  reproducible and comparable.
- **`pins`** — user-forced precision overrides, kept verbatim for
  provenance. Patterns are case-sensitive `fnmatch` globs against the full
  group name; later pins override earlier ones.
- **`format_overhead`** — the overhead fraction used for every size
  prediction. Recorded so a recipe is reproducible from map + pins +
  overhead alone.
- **`trace`** — the solver's ordered downgrade log: replaying it from
  all-highest-precision reproduces the assignments exactly. This is the
  human-readable answer to "why did this group end up at 4-bit?".
