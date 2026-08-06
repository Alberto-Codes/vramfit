# ADR-0021: Sub-4-bit damage is measured in the runtime frame

- **Status:** Accepted
- **Date:** 2026-08-06
- **Supersedes:** [ADR-0019](0019-kquant-priced-maps.md),
  [ADR-0020](0020-imatrix-assisted-pricing.md)

## Context

Four data points closed the scan-side elimination ledger (the
seventh through the tenth in
[evaluating packed models](../explanation/evaluating-packed-models.md)).
Granularity recovers at most ~14 % of the baseline gap. Super-block
pricing (ADR-0019) packed worse than RTN pricing. The evaluation
set does not change the ranking. Imatrix-assisted pricing
(ADR-0020) packed worst of all.

The trend across the ledger is monotone. Each refinement cheapened
in-frame low-bit prices. Each re-plan converted the cheaper prices
into more 2-bit breadth: 35, then 52, then 56 of 82 groups. Each
packed artifact lost worse: 9.156, then 9.251, then 9.607 PPL
against the 8.532 baseline.

The instruments stayed self-consistent throughout. The validation
pass measured sub-additive on all three losing recipes (2.9x, 2.0x,
1.87x). A gate that clears three consecutive packed losers does not
measure what the runtime punishes.

The leak is not an ingredient inside the scan frame. The leak is
the frame's transfer: an in-frame 2-bit price does not predict the
packed artifact, however faithfully the frame imitates the pack's
arithmetic.

## Decision

1. **ADR-0019 and ADR-0020 are superseded.** The scan-frame
   refinement lane is closed. The `kquant-ref` and `kquant-imx`
   methods remain valid scan options — ADR-0018 stands. Withdrawn
   is the claim that in-frame refinement licenses sub-4-bit
   assignments.
2. **Sub-4-bit damage is measured in the runtime frame.** Quantize
   the candidate group to its real packed type inside a real GGUF.
   Measure damage under the runtime's own numerics (issue #40).
3. **An instrument check precedes any runtime-frame campaign.**
   Cross-process re-measurement moved identical cells 2.7–4.1x
   (the ninth data point). The lane must measure its own noise
   floor first. Only same-frame comparisons carry weight.
4. **The solver does not buy 2-bit until a runtime-frame price
   exists.** Recipes solve on maps without a 2-bit column. The
   mechanism today is a derived map variant — the eleventh data
   point measures what the constraint costs.

## Open questions

- Does the 3-bit-floored recipe transfer? The eleventh data point
  (in flight) answers this. A packed result under 9.156 PPL marks
  the transfer failure as 2-bit-specific. A loss to flat `Q3_K`
  extends the failure above 2 bits and strengthens decision 2.
  Caution: the 8k-era no-2-bit diagnostic packed destroyed, and
  the cause was never isolated — the smoke gate (ADR-0017) guards
  the rerun.
- Runtime-frame tooling shape. A per-group override pack through
  `llama-quantize` costs ~12 minutes — 328 cells price at ~2.7
  days of packing alone. Candidates: a targeted subset (the 2- and
  3-bit frontier cells only), or a repack tool that holds the f16
  weights resident.
- Where the lane runs: the reference box, or a rented H100 NVL /
  H200 (~$10–30 per loop, issue #40). The instrument check decides
  whether rented numbers compare with box numbers at all.
- Whether `plan` grows a precision-exclusion flag, or derived map
  variants stay the mechanism for decision 4.

## Consequences

- The sensitivity map keeps its pricing role at 3 bits and above,
  pending the eleventh data point. 2-bit assignment waits for
  runtime-frame prices.
- The measurement bottleneck moves from GPU forward passes to CPU
  quantize passes — a different resource, and one that rents
  cheaply.
- The ADR-0018 and ADR-0020 quantizer ports stay in the codebase.
  The validation pass still runs frame-matched to its map.
- The sub-additive validation result loses its standing as a
  pack-quality signal. It checks additivity inside the scan frame,
  nothing more.
