---
status: draft
---

# How sensitivity scanning works

> **Status: draft** — the method below is implemented in the scan
> adapter and has produced two full-size maps: Qwen2.5-3B (148 cells)
> and the 49B north-star target (328 cells, offload-aware per
> [ADR-0015](../adr/0015-offload-aware-scanning.md)). The open
> questions at the bottom now carry measured partial answers.

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
damage compounds through depth, and two individually-tolerant groups
can be jointly fragile.

The mitigation is implemented as `quantfit validate`: replay the exact
recipe in one pass and compare against the sum of marginal predictions.
Both real measurements came in **sub-additive** — the safe direction:
2.05× over-prediction on Qwen2.5-3B (measured 0.0322 vs predicted
0.0661) and 2.94× on the 49B (0.1682 vs 0.4949). The dangerous
direction — measured above predicted — has not appeared
([ADR-0006](../adr/0006-sensitivity-metric.md)).

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
   tokens before the measurement stabilizes? A dedicated convergence
   scan answers this: the 49B is being re-scanned at 32,768 tokens for
   comparison against its 8,192-token map.
3. **Quantization method within a group** — round-to-nearest is v1;
   AWQ-style scaling changes both the damage and the cost of scanning. A
   method change is a new scan, not a new schema. The 49B loop measured
   this question's teeth: at 2/3-bit the RTN stand-in over-promises the
   runtime frame (packed-model KL 0.375 landed above the scan-frame
   0.168, inverting the Qwen ordering), and one 3-bit-heavy recipe the
   scan priced at damage 1.44 packed into a destroyed artifact. The
   [fourth data point](evaluating-packed-models.md) records both — and
   the post-pack smoke test they force.
4. **Streaming** — the v1 meter relies on `device_map=auto` sharding,
   and groups offloaded to host RAM measure through accelerate's
   weights map (ADR-0015). Streaming one group at a time to the GPU
   (with cached reference activations) is the planned speedup for
   models larger than VRAM.
