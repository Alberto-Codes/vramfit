---
status: draft
---

# How sensitivity scanning works

> **Status: draft** — the method below is implemented in the scan
> adapter and verified on a tiny model. Nothing full-size has been
> measured yet; the open questions at the bottom are still the real
> content of this page.

## The core loop

For each layer group `g` and candidate precision `b`:

1. Quantize *only* `g` to `b` bits; leave the rest of the model at reference
   precision.
2. Run the calibration set through the perturbed model.
3. Measure divergence of its outputs from the full-precision reference.
4. Restore `g`; record `(g, b) → divergence`.

The result is a per-group *damage curve* — how quality falls as bits drop —
which is exactly the cost function the plan step's solver needs.

The v1 implementation (per [ADR-0006](../adr/0006-sensitivity-metric.md),
accepted): divergence is mean KL of next-token distributions at the final
logits, and the perturbation is round-to-nearest quantization with
per-block scales (32 elements per scale). The reference distribution is
computed once and cached, so each cell costs one calibration pass.

## Why marginal measurement (and its blind spot)

Measuring one group at a time is `O(groups × precisions)` — tractable.
Measuring *combinations* is exponential — not. So we assume damage is
approximately additive across groups. That assumption is known to leak:
quantization errors compound through depth, and two individually-tolerant
groups can be jointly fragile.

Planned mitigation: after the solver picks a recipe, run one full-model
evaluation of that exact recipe (cheap — it's a single configuration) and
compare against the sum of marginal predictions. Large gaps mean the
additivity assumption is failing for this model, and the recipe needs a
safety margin.

## What "divergence" means here

[ADR-0006](../adr/0006-sensitivity-metric.md) fixed the ladder:

- **KL divergence** at the final logits — dense signal per token, cheap.
  This is the scan metric.
- **Whole-recipe validation pass** — one extra configuration after
  `plan`, guarding the additivity assumption.
- **Task eval** — once, on the final packed model, as ground truth.

## Open questions

1. **Granularity** — per-layer groups (fast, coarse) vs per-tensor (slow,
   precise)? Both are implemented (`--group-by`); the plan remains a
   layer-level first pass, then tensor-level refinement for the groups the
   solver puts on the budget boundary.
2. **Calibration data** — generic text vs workload-matched, and how many
   tokens before the measurement stabilizes? The first real scan reports
   per-group KL at 1/4, 1/2, and full calibration to answer this.
3. **Quantization method within a group** — round-to-nearest is v1;
   AWQ-style scaling changes both the damage and the cost of scanning. A
   method change is a new scan, not a new schema.
4. **Streaming** — the v1 meter relies on `device_map=auto` sharding.
   Streaming one group at a time to the GPU (with cached reference
   activations) is the planned speedup for models larger than VRAM.
