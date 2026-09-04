# ADR-0021: Sub-4-bit damage is measured in the runtime frame

- **Status:** Accepted
- **Date:** 2026-08-06 (accepted 2026-08-06)
- **Supersedes:** [ADR-0019](0019-kquant-priced-maps.md),
  [ADR-0020](0020-imatrix-assisted-pricing.md)
- **Amendment (2026-08-14, issue #248):** decision 4 changes shape.
  The original clause barred the solver from 2-bit **until a
  runtime-frame price exists**. The #229 gate produced that price, so
  the condition discharged and the clause needed a successor rather
  than a deletion. Maintainer ruling (2026-08-14): the bar becomes a
  measured one. A price must show the width beats the alternatives
  **available at or below the budget**. The bar reads per target and
  per width, and it bans no number of bits anywhere. A first draft
  compared at the same total packed size. That reading barred every
  width on this target, because no two widths here pack to one size,
  so the maintainer replaced it the same day. The same amendment
  states the evaluation set as the full WikiText-2 test set rather
  than a chunk count.
  Record:
  [#229 closing comment](https://github.com/Alberto-Codes/vramfit/issues/229#issuecomment-5300460635).
- **Amendment (2026-08-22, issue #301):** decision 4's bar lifts for
  expert-stack groups on a target where a runtime-frame campaign
  priced the width's mixed use. The maintainer ruled it 2026-08-22
  in the #301 exchange. On the 30B target the campaign ran across
  #300, #372, and #387. It measured nine mixes, each holding 11 expert
  stacks at nominal 2, and the falsifier arm beats the smallest
  published build on both ruled metrics
  ([the nineteenth data point](../explanation/evaluating-packed-models.md#the-nineteenth-data-point-the-recipe-beats-the-published-build-and-serves-under-the-cap)).
  The solver may buy nominal 2 on expert-stack groups under the
  [ADR-0007](0007-recipe-solver-strategy.md) placement rule. Dense
  groups keep the bar, and pins at nominal 8 hold them in the ruled
  form. One acceptance clause guards the transfer. The first
  solver-emitted recipe must pack byte-identical to the measured
  falsifier arm, both packs on one machine. A divergence stops the
  lane. The divergent recipe earns nothing until measured —
  [ADR-0027](0027-instrument-frame-matching.md) keeps damage
  magnitudes on one instrument. Record:
  [#301 ruling comment](https://github.com/Alberto-Codes/vramfit/issues/301#issuecomment-5377920461).
- **Amendment (2026-09-04, issue #277):** decision 4's worked example
  argued from the 10.5 GiB weight budget. #257 superseded that
  constant on 2026-08-15 with 14.5 GiB. #284 superseded 14.5 GiB on
  2026-08-16 with **15.776 GiB**, from #266's measured 228.99 MiB
  runtime overhead. The example is restated in place against
  15.776 GiB. Its reading holds: uniform `Q4_0` at 17.600 GiB is
  still unreachable. Its breadth figure changes. A mix at 15.776 GiB
  relieves 35 of 46 expert stacks to `Q4_0` at dense nominal 8, so
  24 % of stacks stay cheap, not 82 %. The example also deferred the
  `Q2_0` price to #249, which is closed. The price arrived on #328
  (2026-08-19). `sensitivity-32k-q0-ref-stacks.json` prices all 46
  expert stacks at nominal 2 and 4 on #163's instrument, in the frame
  `tensor_type_fallback` applies. Median damage reads 0.156083 at
  nominal 2 and 0.007031 at nominal 4, a median 18.4 times apart.
  That is the per-stack price the example said no measurement
  supplied. Every arm #300 packed holds 11 of 46 expert stacks at
  `Q2_0`, which is the 2026-08-22 (#301) expert-stack lift above.
  Record:
  [#284 closing comment](https://github.com/Alberto-Codes/vramfit/issues/284#issuecomment-5308544054),
  [#328 closing comment](https://github.com/Alberto-Codes/vramfit/issues/328#issuecomment-5336224466).
- **Correction (2026-08-31, issue #415):** the 2026-08-22 amendment
  above calls the comparator the smallest published build. It was
  not. A 2026-08-22 Hub-wide query found eight other publishers'
  full-model builds below the falsifier arm's 15.76 GiB. The
  measured win over `IQ2_XXS` and the amendment's decision stand.

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
pass measured sub-additive on all three losing recipes (1.6x, 2.0x,
1.87x). A gate that clears three consecutive packed losers does not
measure what the runtime punishes.

The leak is not an ingredient inside the scan frame. The leak is
the frame's transfer: an in-frame 2-bit price does not predict the
packed artifact, however faithfully the frame imitates the pack's
arithmetic.

## Decision

1. **ADR-0019 and ADR-0020 are superseded.** The scan-frame
   refinement lane is closed. The `kquant-ref` and `kquant-imx`
   methods remain valid scan options — ADR-0018 stands. This
   record withdraws the claim that in-frame refinement licenses
   sub-4-bit assignments.
2. **Sub-4-bit damage is measured in the runtime frame.** Quantize
   the candidate group to its real packed type inside a real GGUF.
   Measure damage under the runtime's own numerics (issue #40).
3. **An instrument check precedes any runtime-frame campaign.**
   Cross-process re-measurement moved identical cells 2.7–4.1x
   (the ninth data point). The lane must measure its own noise
   floor first. Only frame-matched comparisons carry weight.
4. **The solver buys a width only against a runtime-frame price.
   That price must show the width beats the alternatives available
   at or below the budget.** Amended 2026-08-14 (#248). Recipes
   solve on maps with each barred width's column removed. The
   mechanism today is a copy of the sensitivity map without those
   columns — on the 30B target that is the 2-bit column (#229). The
   eleventh data point measures what the constraint costs.

   The bar reads per target and per width. It bans no number of bits
   anywhere. It compares a width against what the budget can
   actually buy, and not against a width the budget cannot reach.

   Measured 2026-08-14 (#229) on Nemotron 3.5 Lightning 30B-A3B. A
   whole-frontier `Q2_0` pack costs 4.097 times the reference
   perplexity, at 9.906 GiB. `Q4_0` costs 1.009 times at 17.600 GiB,
   which the chart's ~~10.5 GiB~~ **15.776 GiB** weight budget cannot
   reach (restated 2026-09-04, #277). So `Q4_0` is not an alternative
   the bar reads here.

   Amended 2026-08-22 (#301): the bar lifts for expert-stack
   groups where a runtime-frame campaign priced the width's mixed
   use. On this target that is nominal 2, under the ADR-0007
   placement rule. Dense groups keep the bar, held by pins at
   nominal 8. The header amendment carries the acceptance clause.

   `Q2_0` still fails. Mixed recipes also fit at or below
   ~~10.5 GiB, at roughly 82 % of stacks cheap~~ **15.776 GiB, at
   11 of 46 expert stacks cheap** (restated 2026-09-04, #277) and the
   rest higher, and the gate measured none of them. ~~No price yet
   shows `Q2_0` beats those alternatives, so the solver buys no 2-bit
   on this target. #249 carries the measurements that would settle
   it.~~ **Superseded 2026-08-22 (#301) for expert-stack groups.**
   The #249 campaign priced nine mixed recipes and the falsifier arm
   won (the nineteenth data point). Dense groups keep the bar. The
   per-stack `Q2_0` price against `Q4_0` now sits on #328's
   stack-keyed map (2026-09-04 amendment, #277).

- **ADR-0018's 2026-08-17 amendment (#319) adds a fourth method,
  whose token is `q0-ref` since #332. Decision 1 above still
  stands.** That method replaces a frame the
  pack cannot apply, on rows where `tensor_type_fallback` rewrites every
  K-quant. It refines no frame toward a type the pack applies, so it
  does not reopen the closed lane.

## Open questions

- ~~Does the 3-bit-floored recipe transfer? The eleventh data
  point (in flight) answers this. A packed result under 9.156 PPL
  marks the transfer failure as 2-bit-specific. A loss to flat
  `Q3_K` extends the failure above 2 bits and strengthens
  decision 2. Caution: the 8k-era no-2-bit diagnostic packed
  cleanly and then scored PPL ~10⁶, and the cause was never
  isolated — the smoke gate (ADR-0017) guards the rerun.~~
  **Measured (2026-08-06, the eleventh data point): it
  transfers.** The recipe packed to 8.597 ± 0.064 PPL /
  0.1703 mean KLD — a statistical tie with the baseline's 8.532
  on PPL, 7.5 % behind on KLD, and at least 0.55 PPL ahead of
  every 2-bit recipe. The frame's own prediction (flat-3 at 2.3x
  the 2-bit mix's damage) was falsified packed. The transfer
  failure is 2-bit-specific. The smoke gate read 15.35 — the
  8k-era destruction did not recur.
- Runtime-frame tooling shape. Per-group override packs through
  `llama-quantize` measured 4.6–17 minutes on the reference box
  (probe and control quantize logs) — 328 cells price at roughly
  2–4 days of packing alone. Candidates: a targeted subset (the 2-
  and 3-bit frontier cells only), or a repack tool that holds the
  f16 weights resident.
- Where the lane runs: the reference box, or a rented H100 NVL /
  H200 (~$10–30 per loop, issue #40). The instrument check decides
  whether rented numbers compare with box numbers at all.
  **Resolved (2026-08-14, #163 and #220): they do not.** The
  instruments disagree 0.3–10.6 % per cell with zero
  same-instrument noise, and ordering survives.
  [ADR-0027](0027-instrument-frame-matching.md) states the
  frame-match rule, the ordering bar for rented maps, and the
  per-instrument noise floor.
- Whether `plan` grows a precision-exclusion flag, or map copies
  with the excluded column removed stay the mechanism for
  decision 4.
- Whether the bar needs a quality floor beside its comparison
  (added 2026-08-14, #248). Decision 4 compares a width against the
  alternatives a budget can buy. A width that is the only thing
  fitting a budget therefore meets the bar by default, however bad
  its measured damage. The #164 serve test catches that case today,
  and no record says whether the solver should.
- Evaluation breadth in the runtime-frame lane (added 2026-08-07).
  Two WikiText chunks carry the twelfth data point's full-set PPL
  loss. They sit at positions 347 and 502 of 564. The 100-chunk
  tier-2 window does not reach them. A runtime-frame damage
  measure needs evaluation text that reaches such instabilities.
  **Resolved (2026-08-14, #234): the runtime-frame lane evaluates
  the full WikiText-2 test set per cell.** The damage numbers are
  full-set PPL and KLD against the f16 reference. The runtime-frame
  lane reports KLD as mean, 99.9th percentile, and maximum. The
  tail-metric rationale and source links sit on #234.

  The chunk count belongs to the target's tokenizer and not to the
  clause (amended 2026-08-14, #248). At `n_ctx` 512 the same file
  tokenizes to 564 chunks on Nemotron Super 49B. It tokenizes to 594
  chunks on Nemotron 3.5 Lightning 30B-A3B, which is 304,128 tokens.
  Read the count the tool reports, and state it beside the numbers.

## Consequences

- The sensitivity map keeps its pricing role at 3 bits and above —
  the eleventh data point confirmed it. The 30B target's 2-bit price
  arrived 2026-08-14 and failed (#229).
- The measurement bottleneck moves from GPU forward passes to CPU
  quantize passes — a different resource, and one that rents
  cheaply.
- The ADR-0018 and ADR-0020 quantizer ports stay in the codebase.
  The validation pass still runs frame-matched to its map.
- The sub-additive validation result loses its standing as a
  pack-quality signal. It checks additivity inside the scan frame,
  nothing more.
- The seventh data point's ~14 % granularity ceiling is specific
  to the 2-bit lineage (noted 2026-08-07). On the 3-bit-floored
  recipe, `attn_v` protection crossed the baseline's mean KLD
  (the twelfth data point).
- Full-set KLD needs a saved f16 base-logits file per model
  (noted 2026-08-14). The 30B target's file measures
  **39,709,379,972 bytes** over 594 chunks (#229), against the 76 GB
  this record first estimated. llama.cpp stores about one byte per
  vocabulary entry, so an estimate at two bytes doubles the real
  size. The 49B's file measures 36,893,861,492 bytes over 564
  chunks. Both stay pod-side.
- The measured bar in decision 4 costs one whole-frontier pack per
  width per target, before any per-cell grid (noted 2026-08-14).
  The #229 gate's own work ran in under 10 minutes. Three parallel
  quantize passes took 1 minute 56 seconds. The f16 base-logits pass
  took 2 minutes 24 seconds. Four evaluations took 5 minutes 22
  seconds, one of them the noise-floor repeat. One f16 convert at 15
  minutes 3 seconds dominates the bill, and it amortizes across
  every later cell on that target. RunPod billed the pod $2.16, at
  38.6 minutes of an H100 at $3.29 per hour.
