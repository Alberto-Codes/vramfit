# ADR-0006: Sensitivity metric for the scan step

- **Status:** Accepted
- **Date:** 2026-07-27 (accepted 2026-07-28)

## Context

The scan's damage numbers drive every downstream decision, so the metric
choice is load-bearing. Candidates:

1. **KL divergence** of next-token distributions (perturbed vs reference),
   averaged over calibration tokens. Dense signal, one forward pass per
   perturbation, no generation needed. Front-runner.
2. **Perplexity delta** on held-out text. The literature standard, coarser
   per-token signal, similar cost.
3. **Task evals** (MMLU-style accuracy). Closest to user-felt quality.
   Orders of magnitude too slow per (group × precision).
4. **Layer-local error** (MSE on the group's own output). Cheapest — no full
   forward pass — but ignores how errors propagate through depth, which is
   the thing we care about.

There's also a structural question: marginal scanning assumes per-group
damage is additive, which is known to leak (errors compound through depth).

## Decision

Use **mean KL divergence over the calibration set** as the scan metric, with
a **single whole-recipe validation pass** after `plan` to check the additive
prediction, and **one task eval on the final packed model** as ground truth.
Record the metric name in the sensitivity map (`scan.metric`) so it can
change without breaking consumers.

Points fixed at acceptance (2026-07-28):

- Measure divergence at the **final logits only**. Intermediate-state
  probes are an optimization to explore after a first full scan exists,
  not a v1 requirement.
- Default calibration size is **131,072 tokens** until measured. The
  map records `scan.calibration_tokens`, so maps stay comparable and
  the default can move without a schema change.
- The scan quantizes groups by **round-to-nearest with per-block
  scales** in v1. It approximates the pack formats without depending on
  any of them. A within-group method change (GPTQ, AWQ) is a new scan,
  not a new schema.

## Open questions

- How many calibration tokens before per-group KL stabilizes? Plan:
  the first real scan reports per-group KL at 1/4, 1/2, and full
  calibration, and the observed convergence sets the new default.
- How large a marginal-vs-whole-recipe gap invalidates a scan? The
  validation pass exists (`quantfit validate`). First measurement
  (2026-07-28, Qwen2.5-3B, the 6/5/4 mix, 32,768 tokens): measured
  0.0322 against predicted 0.0661 — the marginal damages are
  sub-additive by 2.05x. Second measurement (2026-07-29, Nemotron
  Super 49B, the 8/4/3/2 recipe at the 20.47 GiB budget, 8,192
  tokens): measured 0.1682 against predicted 0.4949 — sub-additive
  by 2.94x, at 80-layer depth and a 3-and-2-bit-dominant mix. Both
  measurements over-predict, which is the safe direction. The open
  threshold narrows to under-prediction: how much measured damage
  above predicted invalidates a scan.

## Consequences

- Scan cost stays `O(groups × precisions)` forward passes — tractable
  overnight on the reference box for the 49B target.
- Damage values are calibration-set-relative. Maps must record their
  calibration provenance and are not comparable across different sets.
