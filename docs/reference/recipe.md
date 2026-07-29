---
status: draft
---

# Recipe format

> **Status: draft** — implemented in `quantfit.adapters.outbound.recipe_json`.
> The loader enforces the structural rules below (required fields, types,
> positive sizes, unique groups). Cross-artifact claims — map order, trace
> consistency — are properties of `quantfit plan`, not of loading.

The recipe is the output of `quantfit plan` and the input to `quantfit pack`:
JSON, one precision assignment per layer group, plus the budget accounting
that produced it.

```json
{
  "quantfit_schema": 2,
  "model_id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
  "runtime": "llama.cpp",
  "plan": {
    "vram_budget_bytes": 25769803776,
    "kv_headroom_bytes": 4294967296,
    "weight_budget_bytes": 21474836480,
    "predicted_total_bytes": 20208459776,
    "predicted_damage": 0.0871,
    "solver": "greedy-damage-per-byte",
    "pins": {"model.layers.0.*": 8},
    "format_overhead": 0.005,
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

- **`quantfit_schema`** — 2 since recipes gained the `runtime` field
  ([ADR-0013](../adr/0013-runtime-capability-in-recipes.md)). Schema
  versions advance per artifact — the sensitivity map stays at 1.
- **`runtime`** — the target runtime the plan was made for, or null for
  an unconstrained plan. `quantfit plan` always sets it. The solver
  filtered its candidates to this runtime's capability, and pack
  backends refuse a recipe recorded for a runtime they do not serve.
- **`assignments`** — every group from the sensitivity map appears exactly
  once, in map order. `bytes` is the predicted size at the runtime's
  effective bits when the runtime has a table
  ([ADR-0014](../adr/0014-per-type-effective-bits.md)), at nominal bits
  otherwise — format overhead included either way. `damage` is the
  *measured* value at the assigned precision — an all-8-bit recipe still
  carries the measured 8-bit damage.
- **`predicted_damage`** — sum of per-group damage at the chosen precisions.
  A *prediction* from marginal measurements, not a guarantee — the pack step's
  post-quantization eval is the ground truth.
- **`solver`** — which strategy produced the recipe (see
  [ADR-0007](../adr/0007-recipe-solver-strategy.md)); recorded so recipes are
  reproducible and comparable.
- **`pins`** — user-forced precision overrides, kept verbatim for
  provenance. Patterns are case-sensitive `fnmatch` globs against the full
  group name, and later pins override earlier ones.
- **`format_overhead`** — the overhead fraction used for every size
  prediction, resolved from the size model's default when `--format-overhead`
  is not given (0.005 with an effective-bits table, 0.05 without). Together
  with the map, the pins, the runtime, and the recorded weight budget, it
  makes the recipe reproducible.
- **`trace`** — the solver's ordered downgrade log. Replaying it from the
  starting state (all groups at highest precision, pinned groups at their
  pin) reproduces the assignments exactly. This is the human-readable
  answer to "why did this group end up at 4-bit?".
