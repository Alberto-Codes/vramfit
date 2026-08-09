---
status: stable
---

# Recipe format

> **Status: stable** — implemented in `quantfit.adapters.outbound.recipe_json`.
> The loader enforces the structural rules below (required fields, types,
> positive sizes, unique groups), and real recipes carried the full 49B
> loop. Cross-artifact claims — map order, trace
> consistency — are properties of `quantfit plan`, not of loading.

The recipe is the output of `quantfit plan` and the input to `quantfit pack`:
JSON, one precision assignment per layer group, plus the budget accounting
that produced it.

```json
{
  "quantfit_schema": 4,
  "model_id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
  "runtime": "llama.cpp",
  "within_group": "kquant-imx",
  "imatrix": "/runs/nemotron-49b-f16.imatrix.gguf",
  "plan": {
    "vram_budget_bytes": 25769803776,
    "kv_headroom_bytes": 4294967296,
    "weight_budget_bytes": 21474836480,
    "predicted_total_bytes": 20208459776,
    "predicted_damage": 0.0871,
    "solver": "greedy-damage-per-byte",
    "pins": {"model.layers.0.*": 8},
    "protections": {"*.self_attn.v_proj.weight": 5},
    "imatrix_exclusions": ["model.layers.1.self_attn.v_proj.weight"],
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
  ],
  "protected_tensors": [
    {
      "tensor": "model.layers.1.self_attn.v_proj.weight",
      "bits": 5,
      "exclude_imatrix": true
    },
    {
      "tensor": "model.layers.4.self_attn.v_proj.weight",
      "bits": 5,
      "exclude_imatrix": false
    }
  ]
}
```

## Field notes

- **`quantfit_schema`** — 4 since recipes gained imatrix exclusions
  ([ADR-0023](../adr/0023-imatrix-exclusions.md)); 3 added protections
  ([ADR-0022](../adr/0022-within-layer-protections.md)). A reader that
  dropped either record would silently pack a
  different artifact than the recipe intends — ADR-0013 ruled that
  case breaking. Schema versions advance per artifact — the
  sensitivity map stays at 1.
- **`runtime`** — the target runtime the plan was made for, or null for
  an unconstrained plan. `quantfit plan` always sets it. The solver
  filtered its candidates to this runtime's capability, and pack
  backends refuse a recipe recorded for a runtime they do not serve.
- **`within_group`** — the within-group method token of the map that
  priced the recipe ([ADR-0019](../adr/0019-kquant-priced-maps.md)),
  or null when the provenance is unknown. `quantfit plan` copies it
  from the map. `quantfit validate` resolves its frame from this
  field and refuses flags that contradict it. The loader accepts an
  absent field as null — recipes written before the field existed do
  not record their map's method.
- **`imatrix`** — the imatrix path of the map that priced the recipe
  ([ADR-0020](../adr/0020-imatrix-assisted-pricing.md)), or null.
  Pairs with the `kquant-imx` token, like the map's `scan.imatrix`.
  `quantfit validate` and `quantfit pack` warn when their
  `--imatrix` names a different file — a different file breaks the
  frame the map priced. The loader accepts an absent field as null.
- **`assignments`** — every group from the sensitivity map appears exactly
  once, in map order. `bytes` is the predicted size at the runtime's
  effective bits when the runtime has a table
  ([ADR-0014](../adr/0014-per-type-effective-bits.md)), at nominal bits
  otherwise — format overhead included either way. `damage` is the
  *measured* value at the assigned precision — an all-8-bit recipe still
  carries the measured 8-bit damage.
- **`predicted_damage`** — sum of per-group damage at the chosen precisions.
  A *prediction* from marginal measurements, not a guarantee —
  `quantfit validate` measures the whole recipe against it
  ([ADR-0006](../adr/0006-sensitivity-metric.md)).
- **`solver`** — which strategy produced the recipe (see
  [ADR-0007](../adr/0007-recipe-solver-strategy.md)). Recorded so recipes are
  reproducible and comparable.
- **`pins`** — user-forced precision overrides, kept verbatim for
  provenance. Patterns are case-sensitive `fnmatch` globs against the full
  group name, and later pins override earlier ones.
- **`protections`** — the `--protect` rules, kept verbatim
  ([ADR-0022](../adr/0022-within-layer-protections.md)). Patterns are
  case-sensitive `fnmatch` globs against full tensor names, and later
  rules override earlier ones for overlapping tensors. Empty for an
  unprotected recipe.
- **`imatrix_exclusions`** — the `--exclude-imatrix` globs, kept
  verbatim ([ADR-0023](../adr/0023-imatrix-exclusions.md)). Each marks
  matched *protected* tensors to quantize without their imatrix rows —
  the fit-collapse remedy that keeps the promotion. Empty when the
  recipe excludes nothing.
- **`protected_tensors`** — the resolved (tensor, precision) pairs, in
  map order. Each precision is the higher of the tensor's group
  assignment and its protection floor. `quantfit pack` drives these
  pairs, never the raw patterns — a resolved pair cannot demote a
  tensor whose group assignment exceeds the floor. Present exactly
  when `protections` is. `exclude_imatrix` marks the pairs the
  exclusion globs resolved to — pack emits `--exclude-weights` for
  each marked pair when it runs with an imatrix.
- **`format_overhead`** — the overhead fraction used for every size
  prediction, resolved from the size model's default when `--format-overhead`
  is not given (0.005 with an effective-bits table, 0.05 without). Together
  with the map, the pins, the runtime, and the recorded weight budget, it
  makes the recipe reproducible.
- **`trace`** — the solver's ordered downgrade log. Replaying it from the
  starting state (all groups at highest precision, pinned groups at their
  pin) reproduces the assignments exactly. This is the human-readable
  answer to "why did this group end up at 4-bit?".
