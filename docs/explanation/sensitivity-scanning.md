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
Four real measurements have come in, and both directions have
appeared. Three were **sub-additive** — the safe direction: 2.05×
over-prediction on Qwen2.5-3B (0.0322 measured vs 0.0661 predicted),
2.94× and 1.6× on the 49B. One was **super-additive by 11.9×** on a
2-bit-heavy 49B recipe — the dangerous direction, driven by which
groups sat at 2-bit, and caught by this pass before packing
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
   tokens before the measurement stabilizes? First measurement
   ([ADR-0006](../adr/0006-sensitivity-metric.md)): 8,192 tokens is
   not enough on the 49B — the 32,768-token re-scan moves median
   cell damage up to 4.5× and flips 41 of 82 planned assignments.
   Whether 32,768 suffices needs a point beyond it. The
   131,072-token default stands.
3. **Quantization method within a group** — round-to-nearest is v1,
   and a method change is a new scan, not a new schema. The 49B loop
   measured this question's teeth twice. First the frame gap:
   packed-model KL landed above the scan-frame prediction at
   2/3-bit, and one 3-bit-heavy recipe the scan priced at damage
   1.44 packed into a destroyed artifact — the
   [fourth data point](evaluating-packed-models.md) records both,
   and the post-pack smoke test they force. Then the direct
   measurement ([ADR-0018](../adr/0018-kquant-within-group-method.md),
   the [seventh data point](evaluating-packed-models.md)): against
   the real K-quant round trip, RTN *over*-prices 2-bit cells
   2.0–3.9x and 3-bit cells up to 1.7x, per-cell. Its symmetric
   absmax grid spends three usable levels at 2-bit where `Q2_K`
   fits four plus a minimum. Sub-4-bit recipes now price with the
   kquant method
   ([ADR-0019](../adr/0019-kquant-priced-maps.md), Proposed).
4. **Streaming** — the v1 meter relies on `device_map=auto` sharding,
   and groups offloaded to host RAM measure through accelerate's
   weights map (ADR-0015). Streaming one group at a time to the GPU
   (with cached reference activations) is the planned speedup for
   models larger than VRAM.
