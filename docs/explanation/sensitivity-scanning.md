---
status: sketch
---

# How sensitivity scanning works

> **Status: sketch** — method design; nothing measured yet. The open
> questions at the bottom are the real content of this page.

## The core loop

For each layer group `g` and candidate precision `b`:

1. Quantize *only* `g` to `b` bits; leave the rest of the model at reference
   precision.
2. Run the calibration set through the perturbed model.
3. Measure divergence of its outputs from the full-precision reference.
4. Restore `g`; record `(g, b) → divergence`.

The result is a per-group *damage curve* — how quality falls as bits drop —
which is exactly the cost function the plan step's solver needs.

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

## What "divergence" should mean

Candidates, in increasing order of cost and faithfulness:

- **KL divergence** of next-token distributions vs the reference — dense
  signal per token, cheap, current front-runner.
- **Perplexity delta** on held-out text — the literature's standard, coarser.
- **Task evals** (MMLU-style) — closest to what users feel, far too slow to
  run per (group × precision).

Likely answer: KL for the scan, a task eval once for the final packed model.
Decision tracked in [ADR-0006](../adr/0006-sensitivity-metric.md).

## Open questions

1. **Granularity** — per-layer groups (fast, coarse) vs per-tensor (slow,
   precise)? Probably layer-level first pass, tensor-level refinement for the
   groups the solver puts on the budget boundary.
2. **Calibration data** — generic text vs workload-matched, and how many
   tokens before the measurement stabilizes?
3. **Reference activations** — cache them once and replay per perturbation
   (fast, memory-hungry) vs re-run the reference each time (slow, simple)?
4. **Quantization method within a group** — RTN is cheapest; AWQ-style
   scaling changes both the damage and the cost of scanning.
