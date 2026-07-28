# ADR-0006: Sensitivity metric for the scan step

- **Status:** Proposed
- **Date:** 2026-07-27

## Context

The scan's damage numbers drive every downstream decision, so the metric
choice is load-bearing. Candidates:

1. **KL divergence** of next-token distributions (perturbed vs reference),
   averaged over calibration tokens. Dense signal, one forward pass per
   perturbation, no generation needed. Front-runner.
2. **Perplexity delta** on held-out text. The literature standard, coarser
   per-token signal, similar cost.
3. **Task evals** (MMLU-style accuracy). Closest to user-felt quality;
   orders of magnitude too slow per (group × precision).
4. **Layer-local error** (MSE on the group's own output). Cheapest — no full
   forward pass — but ignores how errors propagate through depth, which is
   the thing we care about.

There's also a structural question: marginal scanning assumes per-group
damage is additive, which is known to leak (errors compound through depth).

## Decision (proposed)

Use **mean KL divergence over the calibration set** as the scan metric, with
a **single whole-recipe validation pass** after `plan` to check the additive
prediction, and **one task eval on the final packed model** as ground truth.
Record the metric name in the sensitivity map (`scan.metric`) so it can
change without breaking consumers.

## Open questions

- How many calibration tokens before per-group KL stabilizes? (Determines
  scan cost directly.)
- Should divergence be measured at the final logits only, or also at
  intermediate hidden states (cheaper early-exit signal)?
- How large a marginal-vs-whole-recipe gap invalidates a scan?

## Consequences (if accepted)

- Scan cost stays `O(groups × precisions)` forward passes — tractable
  overnight on the reference box for the 49B target.
- Damage values are calibration-set-relative; maps must record their
  calibration provenance and are not comparable across different sets.
