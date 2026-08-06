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

- How many calibration tokens before per-group KL stabilizes? First
  measurement (2026-07-29, the 49B at 8,192 vs 32,768 tokens, same
  calibration file): **8,192 tokens is not enough.** Median cell
  damage falls to 0.77× at 8-bit and 0.22× at 2-bit. Five
  front-stack layers drop 50×–180×. Re-planning the same budget
  flips 41 of 82 assignments. Rank order holds better than
  magnitudes (Spearman 0.99 at 8-bit down to 0.79 at 2-bit): the
  U-curve and the worst-group set survive while the middle
  reorders. Second measurement (2026-07-31, 32,768 vs 65,536
  tokens): **32,768 tokens suffices at 3-bit and above** — the
  median 65,536/32,768 damage ratio is 1.00/1.06/1.10 at 8/4/3-bit,
  Spearman ≥ 0.97, and the re-plan flips 15 of 82 assignments,
  mostly 2↔3-bit swaps. **2-bit cells are not converged at
  32,768**: their median ratio is 1.29 and still rises with
  tokens. Whether 65,536 converges 2-bit needs a
  further point. The 131,072-token default stands for scans whose
  recipes may assign 2-bit. Scans confined to ≥ 3-bit may stop at
  32,768.
- The "one task eval as ground truth" leg has not run as decided.
  What ran on the first packed 49B: a perplexity and whole-model-KL
  head-to-head, plus a now-mandatory post-pack smoke test (ADR-0012,
  second 2026-07-29 amendment). The task-eval leg (tier 3) stays
  open.
- How large a marginal-vs-whole-recipe gap invalidates a scan? The
  validation pass exists (`quantfit validate`). First measurement
  (2026-07-28, Qwen2.5-3B, the 6/5/4 mix, 32,768 tokens): measured
  0.0322 against predicted 0.0661 — the marginal damages are
  sub-additive by 2.05x. Second measurement (2026-07-29, Nemotron
  Super 49B, the 8/4/3/2 recipe at the 20.47 GiB budget, 8,192
  tokens): measured 0.1682 against predicted 0.4949 — sub-additive
  by 2.94x, at 80-layer depth and a 3-and-2-bit-dominant mix. Both
  measurements over-predict, which is the safe direction.
  **Third measurement (2026-07-29, the same budget re-planned on the
  32,768-token map): measured 1.1234 against predicted 0.0940 —
  super-additive by 11.9x.** The dangerous direction exists. The
  recipe that produced it moves 18 more groups to 2-bit (42 of 82)
  on the strength of collapsed marginal damages, and the joint
  measurement says those marginals do not add. Sub-additivity is a property of
  particular recipes, not of the meter. The packed artifact
  confirmed the warning: 10.48 PPL against the pilot-map recipe's
  9.92 at the same budget (the fifth data point).
  **Fourth measurement (2026-07-31, the controlled A/B):** the
  65,536-token map's recipe, measured in the same 32,768-token
  frame, predicts 0.0946 and measures 0.0589 — sub-additive by
  1.6×. Same budget, same predicted sum as the super-additive
  recipe (0.0940 → 1.1234), 19× apart in measured joint damage.
  The only difference is *which* groups sit at 2-bit (35 vs 42,
  different membership). Additivity failure is driven by 2-bit
  group selection, and converged marginals steered the solver back
  to a sub-additive recipe without any interaction modeling. The
  invalidation threshold question stands, but the operational rule
  is now clear: a super-additive validation is a solve-again
  signal, not a pack input.
  **Fifth measurement (2026-08-02, the kquant-priced recipe,
  ADR-0019):** the honest map's re-plan moves 52 of 82 groups to
  2-bit, more than the super-additive recipe's 42. It still
  validates **sub-additive by 2.0x** (measured 0.0610 against
  predicted 0.1221, kquant perturbation, 32,768-token frame
  against a 65,536-token map). The absolute damages are
  kquant-frame and do not compare with the RTN-frame measurements
  above — the ratio does. Membership quality, not membership
  count, drives additivity: honest per-cell prices select
  compatible 2-bit members even at wider breadth.
  **Sixth measurement (2026-08-06, the assisted-priced recipe,
  ADR-0020):** 56 of 82 groups at 2-bit — the widest breadth yet —
  validates **sub-additive by 1.87x** (measured 0.0651 against
  predicted 0.1215, assisted kquant perturbation, 32,768-token
  frame against a 65,536-token map). The frame was fully matched:
  method and imatrix, enforced by the recipe's provenance record.
  The packed artifact lost anyway (the tenth data point) — the
  third consecutive recipe the sub-additive gate cleared before a
  packed loss. The gate answers "do the marginals add in this
  frame", not "does this frame transfer to the runtime". The two
  questions are now known to be different.

## Consequences

- Scan cost stays `O(groups × precisions)` forward passes — tractable
  overnight on the reference box for the 49B target.
- Damage values are calibration-set-relative. Maps must record their
  calibration provenance and are not comparable across different sets.
