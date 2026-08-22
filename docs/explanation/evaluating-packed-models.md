---
status: stable
---

# Evaluating packed models: the endgame scoreboard

> **Status: stable** — tiers 1 and 2 ran for real on 2026-07-28,
> against the first packed model and again against the
> runtime-capability mix (Qwen2.5-3B, see
> [the first data point](#the-first-data-point) and
> [the second data point](#the-second-data-point-the-runtime-capability-mix)
> below). The validation pass — the middle leg — ran the same night
> ([the third data point](#the-third-data-point-the-first-validation-pass)).
> On 2026-07-29 the full loop ran on the 49B acceptance target and
> **lost to the size-matched baseline**
> ([the fourth data point](#the-fourth-data-point-the-north-star-attempt-lost-honestly)).
> The imatrix rematch ran the same night and **lost again, by less,
> for a different reason**
> ([the fifth data point](#the-fifth-data-point-the-imatrix-rematch-and-the-map-that-made-things-worse)).
> On 2026-07-31 the converged map fixed the additivity failure and
> tied the pilot's packed quality — moving the open frontier to the
> scan-to-runtime frame transfer
> ([the sixth data point](#the-sixth-data-point-the-converged-map-and-where-the-leak-moved)).
> The same night, a two-probe duel settled which named leak gets
> the next build: within-layer granularity saturates at ~14 % of
> the gap, and the scan frame turned out to *over*-price low bits,
> not under-price them
> ([the seventh data point](#the-seventh-data-point-the-frontier-duel)).
> On 2026-08-02 the full loop ran on the honest map and **lost
> again** — eliminating super-block structure as the frame leak
> and promoting the imatrix weighting to prime suspect
> ([the eighth data point](#the-eighth-data-point-honest-prices-worse-artifact)).
> The elimination ledger then closed on the scan side — eval set,
> imatrix assistance — and banning 2-bit outright **tied the
> baseline** (ninth through eleventh data points), before the
> granularity probes opened a split decision: first in-budget KLD
> win, PPL still behind
> ([the twelfth data point](#the-twelfth-data-point-the-granularity-probes-and-a-split-decision)).
> The thirteenth data point mapped the split fully and fixed the
> ruling 564-chunk window
> ([the thirteenth data point](#the-thirteenth-data-point-the-imatrix-swap-and-the-instabilitys-address)).
> On 2026-08-08 the fit-collapse root cause fell — imatrix rows
> with extreme column dynamic range, not flags or vintage — and
> probe G1c **ended the split decision**: an outright 100-window KLD win, statistical
> ties on full-window KLD and PPL, and chunks 347/502 silenced
> at baseline level — though the knife edge, it later turned
> out, had moved rather than vanished
> ([the fourteenth data point](#the-fourteenth-data-point-the-pack-path-gap-closed-and-the-split-decision-ended)).
> On 2026-08-09 the CLI ran the whole loop itself — protections,
> imatrix exclusions, all-green reconstruction gate — and its
> artifact **set the best full-window KLD on record**, beating
> the baseline at 7.8σ paired with the scoreboard's first
> spike-free chunk profile; the hunt for its mean gap to G1c
> uncovered a knife-edge spike (chunk 137) the 347/502 watch
> had missed in G1c itself
> ([the fifteenth data point](#the-fifteenth-data-point-the-pipeline-packs-its-own-winner)).
> Tier 3's slice is fixed
> ([ADR-0024](../adr/0024-tier3-task-slice.md)), its harness lane
> is decided and spot-checked against the stock backend (that
> record's open questions), and both slice runs completed
> 2026-08-10: **five tasks, five statistical ties** against the
> standing baseline, none past 0.8σ — the slice certifies the
> candidate at equal size
> ([the sixteenth data point](#the-sixteenth-data-point-five-tasks-five-ties)).
> Results ship as an evals sidecar
> ([ADR-0025](../adr/0025-evals-sidecar.md)).
> On 2026-08-11 a 25-point conversational probe **could not tell the
> two packs apart** — that tie is an argument for measuring
> ([the seventeenth data point](#the-seventeenth-data-point-the-probe-that-could-not-tell-them-apart)).
> The f16 original answered the same fifteen prompts later that day and
> **failed the same way**, which places the shared failures in the
> base model rather than in either pack (issue #143). The control ran
> hours after the public writeup published, which named the gap —
> issue #174 carries that post's correction.
> The publication gates that consume
> these evaluations live in [the artifact ecosystem](artifact-ecosystem.md)
> and issue #11.
> Note 2026-08-12: publication #1 shipped on this evidence, and the
> public writeup cites this page as its record. The page is now
> `stable` — the f16 control closed the last open question, and the
> record carries none.
> Note 2026-08-14: the ledger leaves the 49B. On a rented H100 the
> runtime-frame lane priced the 2-bit frontier of the 30B MoE target
> and **`Q2_0` lost by 4.097x perplexity**, which is the measurement
> ADR-0021 decision 4 had been waiting for since 2026-08-06
> ([the eighteenth data point](#the-eighteenth-data-point-2-bit-fails-its-gate-on-a-new-target)).
> MXFP4 lost the 4-bit row to `Q4_0` on the same runs. The page stays
> `stable` and gains one open question, on whether a whole-frontier
> gate price predicts a mixed recipe. The 2026-08-12 note above holds
> for the 49B lane it describes, and this entry supersedes its
> "carries none" clause.
> Note 2026-08-22: the 30B campaign measured both of the
> Destination's outstanding clauses. The falsifier arm **beats the
> smallest published GGUF on both ruled damage metrics** — 1.161096
> against 1.320914 on the PPL ratio, 0.204318 against 0.370257 on
> mean KLD — at 1.78 GiB less packed weight, and **serves fully
> offloaded under a 16 GiB ballast cap on the 24 GiB 4090**, a card
> budget the published build cannot fit
> ([the nineteenth data point](#the-nineteenth-data-point-the-recipe-beats-the-published-build-and-serves-under-the-cap)).
> One bound travels with the headline: the `q0-ref` map derives the
> identical arm, so the win credits the stack-keyed ranking and not
> the imatrix assistance. The campaign's nine KLD-measured mixed
> arms also mapped the interior of the gate's range, which closes
> the eighteenth entry's open question. The page stays `stable`,
> and the record again carries none.

Scanning and planning happen inside vramfit's own frame: damage,
measured per cell, on our calibration set. The moment a packed model
exists, that frame is no longer enough. A skeptical downloader does
not care what our meter says — they care whether the model still
works, judged by yardsticks the community already trusts. This page
records the three-tier scoreboard for that judgment, cheapest first.

## Tier 1: perplexity — the lingua franca

The community standard is `llama-perplexity` over the WikiText-2 test
set (`wiki.test.raw`): run the packed model across held-out text and
score how surprised it is by each real next token. Lower is better.
Every k-quant comparison table on Hugging Face model cards quotes this
number, which is exactly why we must too — it makes a vramfit packed
model comparable against every heuristic GGUF ever published, on their
home turf. Cost: minutes for a 3B on the reference box, CPU or GPU.

Perplexity's known blind spot: a packed model can hold perplexity flat
while shifting *which* tokens it predicts. It measures surprise, not
faithfulness. That is why tier 1 is necessary but never sufficient.

## Tier 2: whole-model KL — our own metric, closing the loop

The stronger comparison is KL divergence of the packed model's output
distribution against the reference model's, over the same text.
llama.cpp ships this natively: `llama-perplexity --kl-divergence`
reads a base-model logit file (written first with
`--kl-divergence-base <file>` on the reference) and reports the
divergence statistics.

Here is the part that matters strategically: **this is the same
divergence family the damage metric already uses**
([ADR-0006](../adr/0006-sensitivity-metric.md)). The scan measures
KL per (group × precision) cell under marginal perturbation. The
whole-recipe validation pass (`vramfit validate`, ADR-0006) then
replays the exact recipe through the scan's own quantization and
compares against the summed marginal damages — that pre-pack check
is what isolates the additivity assumption leaking. Tier 2 complements it from the
other side: whole-model KL of the *packed* result, through the
runtime's real quantization types. One metric family will run
end-to-end from scan to verdict:

- Cell damage *predicts* the recipe's cost (under the additivity
  assumption).
- The validation pass *checks the prediction* in the scan's frame.
- Packed-model KL *confirms the shipped artifact* in the runtime's
  frame.

A model card that shows the prediction next to the confirmation is a
card no heuristic quant can write. Cost: one reference-logit pass plus
one packed-model pass — tens of minutes at 3B scale.

## Tier 3: task benchmarks — the credibility tier

lm-evaluation-harness (EleutherAI) is the de facto standard for
capability evaluation: MMLU, HellaSwag, GSM8K, and friends, run
against the served model. This answers the question distributional
metrics cannot: does the model still *do things*. It is also what
skeptical readers trust most, precisely because it is furthest from
our own machinery.

It is the expensive tier — measured at 5.70 h per 49B artifact on
the reference box (the sixteenth data point), and scores carry
enough noise that small deltas mean nothing. Reserve it for the head-to-head
that matters: vramfit's packed model versus the size-matched
heuristic GGUF ([ADR-0010](../adr/0010-sub-4-bit-serving-path.md)
names IQ3-class baselines for the 49B). A small, fixed slice of tasks,
reported honestly with the noise acknowledged, beats a sprawling suite
nobody re-runs. The slice is fixed in
[ADR-0024](../adr/0024-tier3-task-slice.md), and the results ship as
an evals sidecar ([ADR-0025](../adr/0025-evals-sidecar.md)).

## The first data point

Both tiers ran on 2026-07-28 against the first packed model:
Qwen2.5-3B under a 4 GiB VRAM budget (2 GiB weight budget), packed
from the measured sensitivity map by `vramfit pack`
([ADR-0012](../adr/0012-gguf-type-mapping.md)). Tier 1 is the full
WikiText-2 test set. Tier 2 is whole-model KL against the f16
reference over the first 100 chunks (51,200 tokens). All runs on the
reference box (llama.cpp b10172, Vulkan).

| Model | File size | Fits 2.00 GiB budget | PPL ↓ | Mean KLD ↓ | Same top token ↑ |
|-------|-----------|----------------------|-------|------------|------------------|
| f16 reference | 5.75 GiB | no | 8.422 ± 0.057 | — | — |
| **vramfit recipe** (7×8-bit incl. embed, 30×4-bit)* | 1.98 GiB | **yes** (17 MiB under) | **8.661 ± 0.058** | **0.0382** | **90.5 %** |
| Q4_K_M heuristic | 1.80 GiB | yes | 8.790 ± 0.060 | 0.0494 | 88.9 % |
| Q5_K_S heuristic | 2.02 GiB | no (21 MiB over) | 8.520 ± 0.057 | 0.0161 | 93.3 % |

\* A 10 %-overhead re-plan of the same map's 2 GiB solve — the
default-overhead variant (9×8-bit, 28×4-bit) in
[why selective quantization](why-selective-quantization.md) is the
same recipe family before the overhead re-plan (pre-ADR-0014).

Reading it honestly, in both directions:

- Among the artifacts that fit the weight budget, the measured
  recipe beats the heuristic on every metric: 35 % less of the
  f16→quant perplexity climb, 23 % lower mean KL, more of the
  reference's top tokens preserved.
- Q5_K_S beats the recipe on quality and loses the budget test by
  21 MiB. It is this benchmark's over-budget quality reference, the
  same role NVFP4 plays for the 49B target (ADR-0010). The recipe's
  candidate set was {8, 4} — the runtime-capability milestone
  (ADR-0013) added 6- and 5-bit candidates, which is exactly the
  ground Q5_K_S occupies. The second data point below is that
  milestone's rematch.

The size lesson from the same run: the first pack of this recipe,
planned with the default 5 % format overhead, came out 56 MiB over
budget and `vramfit pack` refused it — GGUF's effective bits exceed
nominal bits (ADR-0012). Re-planning at 10 % produced the table's
artifact on the first try.

## The second data point: the runtime-capability mix

The runtime-capability milestone (ADR-0013) opened 6- and 5-bit
candidates, so the natural question was whether a measured mix could
take Q5_K_S's ground while staying inside the budget Q5_K_S misses.
On 2026-07-28 the scan re-ran at `--precisions 8,6,5,4,3,2` — 37
groups × 6 precisions = 222 cells, 32,768 calibration tokens,
1 h 24 m on the reference box. The re-plan under the same 4 GiB VRAM
budget (2 GiB weight budget) chose 3 groups at 6-bit (including the
embedding), 29 at 5-bit, and 5 at 4-bit. Same harness as the first
data point: full WikiText-2 for tier 1, whole-model KL against the
same f16 logit file over 100 chunks for tier 2, llama.cpp b10172 on
Vulkan.

| Model | File size | Fits 2.00 GiB budget | PPL ↓ | Mean KLD ↓ | Same top token ↑ |
|-------|-----------|----------------------|-------|------------|------------------|
| f16 reference | 5.75 GiB | no | 8.422 ± 0.057 | — | — |
| **vramfit 6/5/4 mix** (3×6-bit incl. embed, 29×5-bit, 5×4-bit) | **1.995 GiB** | **yes** (5.3 MiB under) | **8.534 ± 0.057** | **0.0180** | **93.3 %** |
| vramfit {8, 4} recipe (first data point) | 1.983 GiB | yes | 8.661 ± 0.058 | 0.0382 | 90.5 % |
| Q4_K_M heuristic | 1.80 GiB | yes | 8.790 ± 0.060 | 0.0494 | 88.9 % |
| Q5_K_S heuristic | 2.02 GiB | no (21 MiB over) | 8.520 ± 0.057 | 0.0161 | 93.3 % |

Reading it honestly, in both directions:

- Against its own predecessor, the mix is a rout: 53 % less of the
  f16→quant perplexity climb (0.112 vs 0.239) and 53 % lower mean
  KL, at nearly the same size. The 6- and 5-bit candidates are worth
  measuring.
- Against Q5_K_S, the mix is 26.4 MiB smaller, fits the budget
  Q5_K_S misses, and ties on perplexity within noise (8.534 vs
  8.520, overlapping ± 0.057 intervals) and on top-token agreement
  (93.31 % vs 93.35 %). It **loses narrowly on mean KL** (0.0180 vs
  0.0161, 12 % higher). The claim the numbers support: near-parity
  with the over-budget quality reference, at smaller size, inside
  the budget — not an outright quality win.
- The distribution tails split: the mix's worst chunk is better
  (maximum KLD 1.02 vs 1.41) but its 99.9th percentile is worse
  (0.445 vs 0.305).

The size lesson repeated, with sharper teeth: planned at the 10 %
overhead that fit the {8, 4} recipe with 17 MiB to spare, the
5-bit-dominant mix packed 4.5 MiB **over** budget and `vramfit
pack` refused it. Re-planning at 10.5 % fit with 5.3 MiB to spare.
The mechanism: one scalar `format_overhead` has to match the
mix-weighted drift of whatever types the solver happens to pick.
The {8, 4} recipe's largest tensor was a `Q8_0` embedding at
+6.25 % drift, which pulled its aggregate under the 10 % scalar.
The new mix packs only types drifting +9.4 % (`Q6_K`) to +12.5 %
(`Q4_K`), so its aggregate lands just above 10 % — and the scalar
that fit one recipe overflowed the next. That was direct evidence
for the open question ADR-0012 and ADR-0013 both carried: the
solver should consume per-type effective-bit tables instead of one
fraction.

[ADR-0014](../adr/0014-per-type-effective-bits.md) closed that
question the same day. The solver now prices llama.cpp recipes at
per-type effective bits, and the overhead fraction shrinks to a
0.5 % residual for file metadata and unquantized tensors. The
rematch of the rematch: re-planning this same map and budget with
pure defaults — no `--format-overhead` at all — reproduced the
6/5/4 mix assignment for assignment and packed first try. Predicted
2,145,721,400 bytes, real 2,141,968,896: a 0.18 % over-reserve, in
the safe direction. The packed file is byte-identical in size to
the hand-tuned artifact this section evaluates, so every number in
the table above stands for it.

## The third data point: the first validation pass

The three-leg story above — cell damage *predicts*, the validation
pass *checks the prediction*, packed-model KL *confirms the
artifact* — ran without its middle leg until `vramfit validate`
existed. On 2026-07-28 the pass ran for the first time, against the
same 6/5/4 mix the second data point evaluates: all 37 assignments
replayed through the scan's own quantization in one pass, over the
scan's own 32,768 calibration tokens, 34 s on the reference box.

| Quantity | Frame | Mean KL |
|----------|-------|---------|
| Summed marginal damages (predicted) | scan | 0.0661 |
| Whole-recipe damage (measured) | scan | 0.0322 |
| Packed-model KL (tier 2) | runtime | 0.0180 |

The additivity assumption over-predicts by 2.05×. The marginal
damages are **sub-additive**: quantize 37 groups at once and the
joint damage is half the sum of the one-at-a-time damages. The
damage did not compound through depth for this recipe — it partially
cancelled.

Reading it honestly, in both directions:

- The gap is large. A prediction off by half is not a calibrated
  estimate of recipe damage, and any claim built on the predicted
  number should say so.
- The gap points the safe way. The solver ranks groups by marginal
  damage per byte and promises the sum; a sub-additive reality means
  the recipe lands *better* than promised. The dangerous failure
  mode — super-additive damage, measured above predicted — did not
  appear, and ADR-0006's invalidation threshold now narrows to that
  direction.
- One recipe, one model, one candidate mix is one data point.
  Whether sub-additivity holds at 3-bit floors, at 49B depth, or
  under tighter budgets is what running `validate` after every
  `plan` will accumulate.

The ordering across the three legs is coherent: the packed model
(0.0180, real K-quants with per-block and super-block scales) lands
below the scan-frame measurement (0.0322, plain round-to-nearest),
and both land below the additive prediction (0.0661). The runtime's
quantizer should beat the scan's approximation — and does — while
the prediction stays a conservative upper bound. That is the shape
you want the three numbers to have on a model card.

## The fourth data point: the north-star attempt, lost honestly

The first full loop on the acceptance target ran 2026-07-29:
Nemotron Super 49B, the 8,192-token sensitivity map, planned at the
real deployment budget (24 GiB card, 16k context at fp8 KV → a
20.47 GiB weight budget), packed to GGUF, and scored against
size-matched community baselines. The packed file came out at
20.30 GiB — the effective-bits prediction over-reserved by 0.44 %
(ADR-0014) and the pack fit first try, 169.7 MiB under budget. The
recipe: 3 groups at 8-bit, 6 at 4-bit, 35 at 3-bit, 38 at 2-bit —
~3.50 effective bits/parameter.

The baselines are bartowski's community GGUFs, both **imatrix**
quants: Q3_K_S (20.45 GiB, fits the budget 21.7 MiB under — the
size match) and IQ3_M (21.10 GiB, 648 MiB over budget — the
over-budget quality reference). Tier 1 is the full WikiText-2 test
set (564 chunks — corrected from "584", a transcription error the
twelfth data point's log audit caught). Tier 2 is whole-model KL
against the f16
reference over the first 100 chunks (51,200 tokens), the f16 pass
GPU-assisted on the reference box (llama.cpp b10172, Vulkan). The
f16 reference measures PPL 8.228 ± 0.141 on those 100 chunks.

| Model | File size | Fits 20.47 GiB budget | imatrix | PPL ↓ | Mean KLD ↓ | Same top token ↑ |
|-------|-----------|----------------------|---------|-------|------------|------------------|
| f16 reference | 93 GiB | no | — | 8.228 ± 0.141* | — | — |
| **vramfit recipe** (8/4/3/2 mix) | **20.30 GiB** | **yes** (169.7 MiB under) | no | 9.917 ± 0.075 | 0.3748 | 75.4 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | yes (21.7 MiB under) | yes | **8.532 ± 0.064** | **0.1584** | **83.8 %** |
| IQ3_M heuristic (bartowski) | 21.10 GiB | no (648 MiB over) | yes | 8.300 ± 0.060 | 0.1633 | 84.1 % |
| control Q3_K_S (ours, same f16 base) | 20.45 GiB | yes | no | 9.655 ± 0.073 | 0.3451 | 76.9 % |

\* The f16 PPL covers the tier-2 100 chunks only — the 93 GiB
reference is too slow for the full 564-chunk set on this box. All
other PPLs are full-set.

**The recipe lost, and not narrowly**: 1.39 PPL and 2.4× mean KLD
behind the size-matched baseline. Under the artifact ecosystem's
hard gate this is a negative result, recorded as such.

Reading it honestly, in both directions:

- **The control experiment locates the gap.** Our own uniform
  Q3_K_S, quantized from the same f16 base with default mixing and
  no imatrix, lands at 9.655 — within 352 bytes of the baseline's
  size, same tensor types, and still 1.12 PPL behind it. The
  importance matrix alone accounts for ~81 % of the recipe's gap.
  Against same-conditions competition the measured mix loses by
  0.26 PPL, not 1.39. The v1 pack path packs plain K-quants
  (ADR-0012 deferred the i-quant/imatrix table until the scan emits
  an importance matrix) — at the 3-bit class, that deferral costs
  more than every other decision in the loop combined.
- **The three-number chain inverted.** On Qwen, the packed artifact
  (real K-quants) beat the scan-frame measurement: 0.0180 < 0.0322
  < 0.0661. Here the packed model's KL (0.3748, WikiText) sits
  *above* the scan-frame whole-recipe measurement (0.1682,
  calibration text) — different text, so indicative rather than
  exact, but the direction flipped. The scan's round-to-nearest
  2-bit is an optimistic stand-in for un-assisted `--pure` Q2_K.
  At 6/5/4-bit the scan frame under-promised the runtime; at
  3/2-bit it over-promises.
- **The additive prediction still over-predicted its own frame.**
  Validation measured 0.1682 against the predicted 0.4949
  (sub-additive by 2.94×, the safe direction — ADR-0006's second
  measurement). The prediction machinery behaved; the frame
  transfer is what leaked.
- **Size math held.** Every predicted byte count over-reserved by
  under half a percent, the budget re-check passed first try, and
  the packed file serves the card the loop planned for. The defeat
  is a quality gap at equal size, not a budget failure.

### The diagnostic that broke: a warning the meter cannot give

To isolate the 2-bit contribution, the same map was re-planned with
its 2-bit cells removed. The solver produced a near-uniform 3-bit
recipe (79 groups at 3-bit, layers 0–2 at 4-bit, 20.24 GiB,
predicted damage 1.44) and `vramfit pack` packed it cleanly. The
artifact is **destroyed**: PPL ~10⁶, same-top-token 0.3 %, on both
the Vulkan and CPU backends. A second variant with the output head
at Q6_K instead of Q3_K is equally destroyed, which rules the
output tensor out. Every Q3_K tensor in the broken file
dequantizes to finite, sane-magnitude values — the payloads are
fine, so the failure is an inference-time interaction with the
type layout, not a corrupt file.

What separates the broken layout from every working one is narrow:
the working control keeps `attn_v` at Q5_K everywhere (the
quantizer's GQA heuristic) and the working recipe keeps layers 3,
4, 5, and 79 at 4-bit or above. The broken layout is the only one
that takes those layers' attention tensors to Q3_K. The exact
mechanism is not isolated, and the honest statement is the scarier
one: **the scan predicted damage 1.44 for a recipe whose real
damage is total.** The damage meter measures round-to-nearest
simulation, and no gate between `plan` and the eval tier would
have caught this. A packed artifact needs a cheap post-pack smoke
test — a few perplexity chunks — before anything downstream trusts
it.

One more contributor surfaced the same evening: the 32,768-token
convergence re-scan showed the 8,192-token map this recipe was
planned from had not converged — re-planning the same budget on the
32,768-token map flips 41 of 82 assignments
([ADR-0006](../adr/0006-sensitivity-metric.md)). The imatrix
remains the measured dominant cause. Map quality joins the
candidate list for the residual 0.26 PPL against same-conditions
competition.

What this changes: the i-quant/imatrix open question in ADR-0012
stops being deferred housekeeping and becomes the gating item for
the north-star claim. The scan already runs the full calibration
set through the model — emitting the importance matrix as a
byproduct was the design intent. Second, the scan's quantization
stand-in needs a 2-bit-honest variant (or the solver a
runtime-frame correction) before 2-bit cells can be trusted at
pack time. Third, the broken diagnostic above makes a post-pack
smoke test a hard requirement, not hygiene.

## The fifth data point: the imatrix rematch, and the map that made things worse

The fourth data point ended with two removable handicaps: the
baseline quantized with an importance matrix and ours didn't, and
the 8,192-token map turned out to be a pilot. On 2026-07-29 both
were removed — ADR-0016 wired `--imatrix` through pack, and the
32,768-token map's recipe existed — and the rematch ran as a full
2×2 over (map × imatrix), so the two factors could not hide behind
each other. The importance matrix came from `llama-imatrix` over
the f16 base and our own calibration text (345 chunks,
`--process-output` so the untied head is covered). Same tiers as
before: full WikiText-2 perplexity, 100-chunk KL against the same
f16 logit file.

Before any of it packed, the validation pass earned its place in
the loop. The 32k recipe's whole-recipe damage measured **1.1234
against a predicted 0.0940 — super-additive by 11.9×**, the first
dangerous-direction measurement after two sub-additive ones
(ADR-0006's third data point). The 32k map's collapsed front-stack
marginals let the solver move 18 more groups to 2-bit (42 of 82),
and the joint measurement said those marginals do not add. The
packed numbers below confirm the warning was real.

| Model | Size | imatrix | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|-------|------|---------|-------|------------|------------|
| vramfit 8k map | 20.30 GiB | no | 9.917 ± 0.075 | 0.3748 | 75.4 % |
| **vramfit 8k map** | **20.30 GiB** | **yes** | **9.061 ± 0.067** | **0.2701** | **79.2 %** |
| vramfit 32k map | 19.99 GiB | no | 10.483 ± 0.081 | 0.4288 | 73.8 % |
| vramfit 32k map | 19.99 GiB | yes | 10.412 ± 0.082 | 0.4708 | 77.0 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | yes | **8.532 ± 0.064** | **0.1584** | **83.8 %** |
| control Q3_K_S (ours) | 20.45 GiB | no | 9.655 ± 0.073 | 0.3451 | 76.9 % |

**The rematch is lost, and the loss is now informative.** The best
cell — 8k map with imatrix — lands 0.53 PPL and 1.7× mean KLD
behind the size-matched baseline. That is down from 1.39 PPL and
2.4× in the fourth data point, but both artifacts now quantize
imatrix-assisted from the same f16 base, so no toolchain handicap
remains to blame. Reading the grid honestly, in both directions:

- **The imatrix factor is real and recipe-dependent.** On the 8k
  recipe it bought 0.86 PPL (9.917 → 9.061). On the uniform-Q3
  control it bought 1.12 (9.655 → 8.532). On the 32k recipe it
  bought 0.07 (10.483 → 10.412). Importance-weighted rounding
  improves every tensor a little — it cannot rescue a bit
  allocation whose damage comes from interactions between 2-bit
  groups, and it helps a 2-bit-heavy mix *less* than it helps the
  baseline's 3-bit mix. The gap against the baseline actually
  widened under fair conditions (0.53 vs 0.26 same-conditions in
  the fourth data point) for exactly that reason.
- **The better map made a worse recipe.** The 32k map is closer to
  converged per cell (PR #34), its recipe's *predicted* damage is
  5× lower (0.094 vs 0.495) — and its packed artifact is worse in
  every cell of the grid. The mechanism is the super-additivity
  above: more accurate marginals read the front stack as nearly
  free, the solver spent those bits elsewhere and paid in 2-bit
  interactions the additive model cannot see. Map convergence and
  recipe quality are not the same axis.
- **The three-number chain now points at one culprit.** Validation
  caught the 32k recipe before packing (measured 1.12 in the scan
  frame; the packed artifact confirms at 0.43–0.47 KL on held-out
  text). The scan's marginal cells, the additive sum over them, and
  the runtime's real 2-bit types each tell a different story about
  the same recipe. The binding constraint on the north-star claim
  is no longer the pack toolchain — it is the additive damage model
  at 2-bit, plus the scan's round-to-nearest stand-in for Q2_K.
- **The operational gates behaved.** All three packs fit first try
  (ADR-0014's margins: 488.6 MiB and 169.7 MiB under). The smoke
  test (ADR-0017) gated every artifact — 20.96, 16.50, 15.97 over
  its two calibration chunks, all under the ceiling, none destroyed.
  The imatrix coverage scan reported exactly one uncovered tensor
  (`token_embd`, expected — embeddings have no activation
  statistics). The imatrix pair differ from their blind twins by
  ~350 bytes of embedded provenance, confirming ADR-0016's
  size-invariance claim.

What this changes: ADR-0016 and ADR-0017 carried their weight and
are Accepted. The i-quant table stays open but is no longer the
gating item. The gating item is now solver-shaped: a recipe whose
validation pass measures super-additive must not reach pack
unchallenged, and 2-bit assignments need either an
interaction-aware solve, a validation-in-the-loop correction, or a
runtime-frame damage measurement before the north-star claim can
close. The residual 0.53 PPL is the price of those missing pieces,
measured.

## The sixth data point: the converged map, and where the leak moved

The 65,536-token convergence scan finished on 2026-07-31 (~24.6 h,
issue #37) and answered ADR-0006's calibration question: 32,768
tokens suffices at 3-bit and above, and 2-bit cells are still
rising (median ×1.29). The re-planned recipe moved 7 groups off
2-bit (35 of 82, different membership than either predecessor) and
set up the cleanest experiment of the week.

**The controlled A/B on additivity.** Measured in the identical
32,768-token frame, the two recipes carry the same predicted
marginal sum and land 19× apart:

| Recipe | Predicted (32k frame) | Measured | Direction |
|--------|----------------------|----------|-----------|
| 32k map (42 groups at 2-bit) | 0.0940 | 1.1234 | super-additive by 11.9× |
| 64k map (35 groups at 2-bit) | 0.0946 | **0.0589** | **sub-additive by 1.6×** |

Factors read at or above 1 with the direction named: super-additive
divides measured by predicted, sub-additive the reverse.

Which groups sit at 2-bit decides whether damages add — and
converged marginals alone steered the solver back to a
sub-additive recipe, no interaction modeling required. The
operational rule this fixes: a super-additive validation is a
solve-again signal, not a pack input.

**The packed result.** The 64k recipe packed imatrix-assisted
(20.32 GiB, 152.4 MiB under, smoke 16.22) and scored:

| Model | Size | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|-------|------|-------|------------|------------|
| vramfit 64k map + imatrix | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
| vramfit 8k map + imatrix | 20.30 GiB | 9.061 ± 0.067 | 0.2701 | 79.2 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | **8.532 ± 0.064** | **0.1584** | **83.8 %** |

Reading it honestly, in both directions:

- **The catastrophic recipe is fixed, the ceiling is not moved.**
  The converged map's artifact ties the pilot map's within noise
  (overlapping PPL intervals, marginally better KL and top-token).
  They sit 0.62 and 0.53 PPL behind the baseline.
- **The scan frame over-promises at low bits.** Each map's recipe
  promises less damage than its predecessor in the scan frame: the
  64k recipe measures 0.0589 at 32,768 tokens against the 8k
  recipe's 0.1682 at 8,192 tokens. The frames differ, so that gap
  is indicative, not exact. Packed, the two artifacts tie —
  round-to-nearest is an optimistic stand-in for real Q2_K/Q3_K
  types, and that transfer leak now bounds packed quality more
  than map quality does.
- **The suspect list reorders.** Additivity: handled by converged
  maps plus the validation gate. Map convergence: answered. What
  remains between 9.06 and 8.53 is the runtime-frame gap
  (a 2-bit-honest scan variant, or measuring damage through the
  packed types directly) and allocation granularity (the baseline
  mixes precision *within* layers — tensor-level groups are
  ADR-0012's declared v1 boundary).
- **The operational record stays clean**: another first-try size
  fit, the smoke gate on everything, one expected imatrix coverage
  miss (`token_embd`).

One operational limit surfaced: a 65,536-token validation pass
does not fit the 24 GiB card at any weight cap tried — the
embed-sized fp32 buffer plus 64k-token bookkeeping fragments the
allocator. The frame-matched measurement above ran at 32,768
tokens instead. Bigger-VRAM measurement is tracked in issue #40.

## The seventh data point: the frontier duel

The sixth data point left two named suspects for the remaining
0.53–0.62 PPL: allocation granularity (the baseline protects
tensors inside layers we crush whole) and frame honesty (RTN as an
optimistic stand-in for real K-quant types). Both had cheap
falsification probes. Both ran on 2026-07-31, before building
anything. Both verdicts surprised.

**The granularity probe: the lever is real and too short.** First,
the baseline's within-layer cleverness was read directly off its
tensors: bartowski's Q3_K_S is flat `Q3_K` everywhere except
`attn_v` at `Q5_K` and the output head at `Q6_K`. No 2-bit
anywhere — and it fits the same weight budget with 21.8 MiB to
spare. Second, the NAS architecture shrinks the lever's reach:
only 10 of our 35 2-bit layers have attention tensors at all
(layers 6, 7, 11, 42–51, and 53–70 are FFN-only blocks — corrected
from "42–70" by the twelfth data point's tensor census: layer 52
keeps its attention, and 6, 7, and 11 lack it. The 10-layer count
stands: 6, 7, and 11 sat in the 2-bit set without attention
tensors, and layer 52 sat at 3-bit). Hand-driven packs of
recipe-64k with baseline-mirroring holds inside those 10 layers:

| Model | Size | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|-------|------|-------|------------|------------|
| recipe-64k + imatrix (sixth data point) | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
| + `attn_v`→`q4_k` in 2-bit layers (18.75 MiB) | 20.34 GiB | 9.103 ± 0.068 | 0.2449 | 81.2 % |
| + `attn_output`→`q3_k` too (87 MiB total) | 20.40 GiB | 9.068 ± 0.068 | 0.2367 | 81.5 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | **8.532 ± 0.064** | **0.1584** | **83.8 %** |

Per byte, the holds are excellent — 0.088 PPL and 27 % of the KLD
gap for 87 MiB confirms attention tensors inside 2-bit layers are
disproportionately fragile. As the milestone candidate, they
saturate: the full within-layer ceiling recovers ~14 % of the PPL
gap. Tensor-level groups (ADR-0012's v1 boundary) stay worth
lifting eventually — they are not what separates 9.07 from 8.53.

**The frame-honesty probe: the leak is real and points the other
way.** ADR-0018 gave the meter a K-quant-faithful within-group
method (torch ports of llama.cpp's reference quantizers, verified
against `ggml_quantize_chunk` — `Q3_K`/`Q8_0` bit-exact,
`Q2_K`/`Q4_K` equal within representation-tie noise). Sixteen
cells re-measured on the 65,536-token frame, against the RTN map's
values, frame-noise-corrected by two RTN re-measurements:

| Cells | kquant / RTN damage | Reading |
|-------|--------------------|---------|
| 2-bit, attention-bearing layers (9–36) | 0.26–0.41 | RTN over-prices 2.4–3.9x |
| 2-bit, FFN-only layers (46–68) | 0.50–0.72 | RTN over-prices 1.4–2.0x |
| 3-bit (9, 46, 61, 75) | 0.59–1.19 | 0.8–1.7x, non-uniform |

Every prior framing assumed RTN flatters low bits. It does the
opposite. RTN's symmetric absmax grid cannot reach its lowest
level at 2-bit — three usable levels stand in for `Q2_K`'s four
fitted levels plus a minimum — so the scan charged up to 3.9x the
real price for 2-bit cells, non-uniformly, worst exactly where the
attention tensors live. The solver bought its whole allocation —
two `q8_0` layers and eight `q4_k` layers paid for by 35 crushed
layers — at that distorted exchange rate. The membership problem
(ADR-0006, fourth measurement) compounds on the same wrong prices.

**The verdict.** Granularity recovers 14 %. Re-pricing invalidates
the arithmetic behind every sub-4-bit decision the solver has
made. Re-pricing gets the build ([ADR-0019](../adr/0019-kquant-priced-maps.md)):
a full kquant-priced re-scan of the 49B at 65,536 tokens is in
flight, and the re-planned recipe walks the full loop against the
same baselines. The falsifiable prediction on record: honest
prices pull the recipe toward the baseline's flat-3-bit region,
and the packed result closes most of what granularity could not.

## The eighth data point: honest prices, worse artifact

The seventh data point's build ran end to end on 2026-08-02: a full
328-cell kquant-priced re-scan at 65,536 tokens (ADR-0019), re-plan,
method-matched validation (kquant perturbation, 32,768 tokens),
imatrix pack, smoke, and both tiers. Two
predictions on record went down, and the second one reorders the
suspect list again.

**The re-plan went the other way.** Against the RTN map, the kquant
map's median cell ratio is 0.74 at 2-bit and 1.28–1.43x at 8/4/3-bit
— raw cross-process ratios, uncorrected for the ~20 % frame offset
between scan runs (the seventh data point's bound). The offset
multiplies every cell in a map roughly equally, so it cancels in the
*relative* prices the solver actually consumes: 2-bit is ~40 %
cheaper relative to 3-bit than RTN claimed, whatever the absolute
offset. (That relative shift also reconciles the full scan with the
16-cell probe — the probe's corrected ratios and the full map agree
on the 2-versus-3 discount, and its per-precision magnitudes were
sampled on 16 selected cells, not 328.) So the solver moved 52 of 82
groups to 2-bit — not toward the baseline's flat-3-bit shape. The
growth came mostly from attention-bearing mid-stack layers (15 of
the 20 entrants); only the three front-most attention layers
(9, 14, 15) left the 2-bit set.

**The validation gate held at record breadth.** 52 groups at 2-bit
— ten more than the recipe that went super-additive 11.9x on RTN
prices — measured sub-additive by 2.0x (0.0610 against predicted
0.1221, kquant perturbation, a 32,768-token pass against the
65,536-token map, ADR-0006 fifth measurement). Membership quality,
not membership count, drives additivity.

**The packed artifact lost anyway.** Another first-try fit
(20.21 GiB, 269 MiB under) and a passed smoke, then:

| Model | Size | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|-------|------|-------|------------|------------|
| vramfit kquant map + imatrix | 20.21 GiB | 9.251 ± 0.069 | 0.3056 | 77.8 % |
| vramfit RTN 64k map + imatrix | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | **8.532 ± 0.064** | **0.1584** | **83.8 %** |

Reading it honestly: matching the pack's super-block structure made
the packed result *worse*, not better — and the gap to the baseline
widens from 0.62 to 0.72 PPL. The scan frame and the packed
artifact now quantize with the same block layout, the recipe is
internally consistent in its own frame, and the transfer still
inverts the ranking. Super-block structure is therefore eliminated
as the frame leak. The within-group method is *not* yet matched:
the pack fits with the importance matrix and the meter prices
unassisted, so the method's remaining unmatched half is exactly
where the suspicion moves. Granularity stays bounded at its
measured ~14 % ceiling from the seventh data point — struck from
the frontier, not erased from the ledger.

**What remains between the frames.** Three differences survive:
the imatrix weighting inside the within-group method (ADR-0018's
first open question), the measurement set (calibration text
against held-out WikiText-2), and llama.cpp's runtime numerics —
all read through a ~20 % cross-process noise floor that any future
elimination argument must clear. The imatrix is the prime
suspect, with prior evidence: assistance was worth 0.86 PPL on
one allocation and 0.07 on another (the fifth data point), so it
is violently allocation-dependent, and a solver pricing without
it optimizes an objective the packed artifact does not ship. The
mechanism also fits this loss's direction: assistance recovers
more of the damage in cells with more levels to re-weight, so an
unassisted map overvalues exactly the 2-bit breadth this recipe
bought. An imatrix-aware meter is the named next step, priced for
the rented measurement lane (issue #40) if the reference box
cannot carry it.

ADR-0019 stays Proposed: its first full measurement contradicts
its decision as stated, and the record says so.

## The ninth data point: the imatrix duel

The eighth data point left one prime suspect and one cheap control.
Both probes ran on 2026-08-03, before any full re-scan
(ADR-0020) — and both moved the story.

**Probe A: the eval-set control — mismatch distorts PPL, not the
ranking.** Tier-2 KLD re-ran on the scan's own calibration text
(first 100 chunks, fresh f16 base) for both live artifacts and the
baseline. If the kquant recipe won in-set and lost on wiki, the
meter's objective would need held-out text before any pricing fix
could show up in the tiers.

| Model (calibration text) | PPL ↓ | Mean KLD ↓ |
|--------------------------|-------|------------|
| f16 base | 8.170 | — |
| vramfit RTN 64k map + imatrix | 8.829 | 0.5611 |
| vramfit kquant map + imatrix | 8.673 | 0.6114 |
| Q3_K_S heuristic (bartowski) | 7.758 | **0.4538** |

By PPL the ordering *does* flip: the kquant recipe wins in-set and
loses on wiki, and the baseline lands below the f16 base itself —
a quantized model "beating" its reference is the tell that PPL on
near-calibration text measures affinity, not fidelity. By mean KLD
— the meter's own damage metric, which cannot go below zero against
its own base — nothing flips: the kquant recipe loses on both texts
and the baseline wins on both, with an imatrix built from a
*different* dataset. The frame leak survives text matching. The
eval-set mismatch is struck from the suspect list, with a standing
caution: cross-text PPL comparisons do not transfer.

**Probe B: assistance re-prices per cell, by structure, at every
precision.** Sixteen cells spanning the kquant recipe's 52-group
2-bit set, re-measured on the 65,536-token frame with the ported
`_impl` quantizers (ADR-0020) — assisted and unassisted in one
process, paired per cell. The pair ratio is the honest number; the
sanity cells below say why.

| Cells (in-process, assisted / unassisted) | median | range |
|--------------------------------------------|--------|-------|
| 2-bit, attention-bearing (10, 17, 22, 31, 41) | **0.55** | 0.47–0.67 |
| 2-bit, FFN-only (46, 50, 55, 61, 68) | 0.90 | 0.83–0.99 |
| 3-bit (9, 14, 70, 75) | 0.82 | 0.40–0.95 |
| 4-bit (76, 79) | 0.60 | 0.41–0.79 |

Three readings:

1. **The re-pricing clears the gate.** Attention-bearing cells
   re-price 1.8–2.1x at 2-bit, up to 2.5x at 3-bit (layer 9) and
   2.5x at 4-bit (layer 79). Per-cell, not scalar — no rescale of
   the unassisted map reproduces these prices.
2. **The distortion is structural.** Assistance concentrates where
   attention tensors are — the FFN-only mid-stack (the NAS
   architecture's layers 42–70) gets almost none (0.83–0.99). The
   unassisted map therefore overprices attention-layer low-bit
   cells ~1.65x *relative* to FFN-only cells. The solver bought 52
   groups of 2-bit on those tilted relative prices.
3. **Assistance is not a 2-bit story.** Layer 9 at 3-bit (0.40)
   and layer 79 at 4-bit (0.41) re-price hardest of all. An
   assisted map changes prices everywhere the imatrix covers.

**The sanity pair widened the noise floor.** Two RTN re-measurements
of stored RTN-map cells — the seventh data point's trick — read
2.7x and 4.1x their map values, against 0.79–0.81 last time. The
environment is unchanged (the lockfile diff since the maps is one
reading-only dependency), so cross-process frame offsets on this
box are larger and less stable than the ~20 % previously bounded,
and they compound between maps scanned in different processes.
Consequences: absolute damages do not transfer across processes,
every cross-map ratio in earlier data points carries wider error
bars than stated, and the in-process pairing this probe used is
the only comparison that survives. The rented-lane instrument
check (issue #40) stops being a formality and becomes the first
task of any off-box scan.

**Verdict.** The re-pricing is material, per-cell, and
structure-dependent: the full imatrix-assisted re-scan is
justified (ADR-0020's gate). Probe timing prices it at ~40–50 h
on the reference box (assisted cells averaged ~9 min against ~6.5
unassisted), which strengthens the #40 case. The follow-up loop
carries the usual burden: re-plan, frame-matched validation —
method-matched *and* imatrix-matched — pack, smoke, both tiers.
Until an assisted-priced recipe beats an unassisted one packed,
ADR-0020 stays Proposed, exactly as ADR-0019 waits beside it.

## The tenth data point: assisted prices, deepest loss

The build the ninth data point justified ran end to end on
2026-08-06: the full
328-cell imatrix-assisted re-scan (ADR-0020 — 37 h at ~7 min per
cell, 437 of 438 tensors covered, `token_embd` the expected
miss), re-plan, the first fully frame-matched validation pass —
method-matched *and* imatrix-matched, with the pairing enforced by
the recipe's own provenance record rather than the operator's
memory — then an imatrix pack with the same file, smoke, and both
tiers. The result is the program's clearest negative to date: the
most honest map yet produced the worst artifact yet.

**The re-plan bought more 2-bit, not less.** The hoped-for rotation
toward the baseline's flat-3-bit region did not happen. Assistance
cheapened most low-bit cells — 65 % of 2- and 3-bit cells, median
ratio 0.91, with per-cell ratios against the unassisted map
spanning 0.23–8.6x in both directions (cross-process, so
indicative only). The *relative* prices the solver consumes
tilted further toward breadth: the median per-group 2-bit/3-bit
price ratio dropped from 2.83 to 2.19, and 56 of 82 groups landed
at 2-bit (52 on unassisted kquant prices), mix
5×8 / 8×4 / 13×3 / 56×2, predicted damage 0.1215 against the
kquant recipe's 0.1221. Twenty-one assignments flipped — six
groups left the 2-bit set, ten joined, and the interior
reshuffled — so membership quality changed even as breadth grew.

**The validation gate held again, for the third time before a
packed loss.** Sub-additive by 1.87x — 0.0651 measured against
0.1215 predicted (32,768-token pass against the 65,536-token map,
ADR-0006 sixth measurement). Record 2-bit breadth, no alarm. The
in-frame gate has now cleared three consecutive recipes that went
on to lose packed, which is itself a finding: whatever the gate
measures, it is not the thing the runtime punishes.

**The packed artifact lost by the most yet.** Another first-try fit
(20.37 GiB, 97 MiB under) and a passed smoke, then:

| Model | Size | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|-------|------|-------|------------|------------|
| vramfit assisted map + imatrix | 20.37 GiB | 9.607 ± 0.072 | 0.3437 | 76.3 % |
| vramfit kquant map + imatrix | 20.21 GiB | 9.251 ± 0.069 | 0.3056 | 77.8 % |
| vramfit RTN 64k map + imatrix | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | **8.532 ± 0.064** | **0.1584** | **83.8 %** |

The baseline gap widened from 0.62 to 0.72 to 1.08 PPL across the
three vramfit artifacts. Imatrix-blind pricing is eliminated as
the frame leak: pricing *with* the pack's imatrix made the packed
result worse. The elimination ledger now reads granularity (~14 %
ceiling, the seventh data point), super-block structure (the
eighth), the evaluation set (the ninth), and imatrix assistance
(the tenth) — and the trend across them is monotone. Every
refinement that made in-frame prices more faithful moved low-bit
prices *down*, every re-plan converted that into more 2-bit
breadth, and every packed artifact got worse. The mechanism is
hard to miss at this point: the scan frame — perturbing weights
inside the bf16 model and measuring calibration KL — under-prices
what 2-bit costs in the packed runtime, and no refinement of the
frame's arithmetic fixes a leak in the frame's *transfer*.

**A smoke caution for the record.** The smoke gate read 7.47 over
its two chunks — less than half the previous artifacts' smoke
readings (16.22, 17.14) — on the worst full-set artifact of the
three. The smoke is a liveness gate (ADR-0017). Two chunks carry
no ranking information, and this data point is the proof.

**Where this leaves the program.** The instruments are
self-consistent and the bookkeeping held: provenance enforced end
to end, first-try fits, sub-additive validation. What the in-frame
instruments are not is predictive of packed reality at 2-bit. Two
directions survive, and they are the same direction at different
depths. Measure damage *through the packed types* — issue #40's
runtime-frame lane, where a cell is quantized into a real GGUF and
measured under the runtime's own numerics. And constrain the
solver away from the region the frame cannot price — the
baseline's flat-3-with-protections shape sits inside the recipe
space and remains unbeaten by anything the maps have bought.
ADR-0019's and ADR-0020's acceptance bars were both measured and
both failed. The records say so. What their status becomes is the
next decision, not a footnote here. *(That decision is now made:
[ADR-0021](../adr/0021-runtime-frame-measurement.md) supersedes
both records toward runtime-frame measurement, 2026-08-06.)*

## The eleventh data point: ban 2-bit, tie the baseline

The constrained diagnostic
[ADR-0021](../adr/0021-runtime-frame-measurement.md) committed to
ran the same day, and it is the cheapest data point on the ledger:
no scan, no new instrument. The mechanism is a copy of the assisted
map with the 2-bit column removed
(`sensitivity-64k-kquant-imx-no2.json`, marked derived), re-planned
at the identical budget. Everything downstream is the standard
loop: frame-matched validation, an imatrix pack, smoke, both tiers.

**The solver rediscovered the baseline's shape.** Given {8, 4, 3},
the plan came out 1×8 / 1×4 / 80×3 — `layers.0` at 8-bit,
`layers.3` at 4-bit, everything else including the embedding flat
at 3-bit. That is, to within two protected groups, the shape the
Q3_K_S heuristic ships. Predicted damage: 0.2776, which is 2.3x
the 2-bit mix's 0.1215 — by the frame's marginal sums, this recipe
should be far worse than the one that packed 9.607.

**Validation: the deepest sub-additivity yet, and a diagnostic
reversal.** Sub-additive by 4.87x — 0.0570 measured against 0.2776
predicted (32,768-token frame-matched pass, ADR-0006 seventh
measurement). The more telling number: the measured whole-recipe
damage, 0.0570, sits *below* the 2-bit recipe's 0.0651. In its own
frame, whole-recipe measurement already ranks the flat-3 recipe
ahead — it is the summed marginals that invert the order. The
additivity failure and the transfer failure point at the same
place: 2-bit membership.

**Packed: a photo finish with the baseline.** First-try fit
(20.37 GiB, 102 MiB under budget), smoke 15.35 — squarely in
the healthy band, and the 8k-era no-2-bit destruction did not
recur. Then:

| Model | Size | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|-------|------|-------|------------|------------|
| vramfit no-2 (assisted map) | 20.37 GiB | 8.597 ± 0.064 | 0.1703 | 82.7 % |
| vramfit assisted map + imatrix | 20.37 GiB | 9.607 ± 0.072 | 0.3437 | 76.3 % |
| vramfit kquant map + imatrix | 20.21 GiB | 9.251 ± 0.069 | 0.3056 | 77.8 % |
| vramfit RTN 64k map + imatrix | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | **8.532 ± 0.064** | **0.1584** | **83.8 %** |

The PPL gap to the baseline is 0.065 — inside one standard error
of the difference (±0.090), a statistical tie. Mean KLD loses
narrowly and honestly: 0.1703 against 0.1584, outside the error
bars but 7.5 % relative, against the 67–117 % relative losses of
the three 2-bit recipes. Every 2-bit-carrying artifact is beaten
by at least 0.55 PPL and 0.095 KLD.

**What it settles.** The frame-transfer failure is 2-bit-specific.
The same scan frame that turned three progressively refined 2-bit
recipes into progressively worse artifacts priced an {8, 4, 3}
recipe well enough to land within noise of the unbeaten baseline.
ADR-0021's interim constraint — no 2-bit assignments until a
runtime-frame price exists — now carries its own supporting
measurement, and the in-frame prediction it overrode (flat-3 is
2.3x worse) is falsified in the packed runtime. The remaining
edge — 0.065 PPL, 0.012 KLD — is the same order as the
within-layer granularity lever the seventh data point measured
(`attn_v`/`attn_output` protections, ~0.09 PPL for 87 MiB), which
suggests the baseline's residual advantage is granularity, not
allocation. One operational note: the full post-plan loop —
validation, pack, smoke, both tiers — completed in 104 minutes on
the reference box. *(The granularity suspicion is measured in
[the twelfth data point](#the-twelfth-data-point-the-granularity-probes-and-a-split-decision):
partly right, and the residual splits in a way nobody predicted.)*

## The twelfth data point: the granularity probes, and a split decision

The eleventh data point left a 0.065 PPL / 0.012 KLD residual and
one named suspect: within-layer granularity, the baseline's
`attn_v`@`Q5_K` + `output`@`Q6_K` toolkit. On 2026-08-07 the suspicion
met the seventh data point's probe pattern, applied to the no-2
recipe: hand-driven `llama-quantize` tensor overrides on the
recipe's exact pack layout, each candidate sized to the byte before
packing.

**The budget arithmetic gutted the ladder first.** The no-2 pack
sits 102.1 MiB under the 21,978,152,960-byte weight budget, and 49
of the 80 layers carry attention tensors (the NAS gap is layers 6,
7, 11, 42–51, and 53–70 — layer 52 keeps its attention, a small
correction to the seventh data point's 42–70). Against that
headroom:

| Candidate | Cost | Verdict |
|-----------|------|---------|
| `attn_v`→`q5_k`, 3-bit attention layers | +96.9 MiB | fits |
| output→`q6_k` | +391.4 MiB | 3.8× the headroom |
| output→`q4_k` (the cheapest head promotion) | +133.1 MiB | does not fit |
| `attn_output`→`q4_k`, same layers | +399.5 MiB | does not fit |

The baseline can afford its head because it protects nothing else:
no 8-bit layer 0, no 4-bit layer 3. Inside the no-2 allocation, the
head promotion is unreachable without a demotion trade — an
allocation change, out of probe scope. So one probe ran in budget
(G1, the `attn_v` hold) and the head promotion ran as an explicitly
**out-of-budget diagnostic** (G2 = G1 + output@`q6_k`, 20.84 GiB) to
measure the lever it cannot ship.

**The detour: a quantizer fit collapse, caught by a reconstruction
check.** The first G1 build promoted all 47 in-budget attention
layers and scored 9.594 PPL — a full point *worse* than the artifact
it modified. Per-tensor reconstruction against the f16 base located
it: with our importance matrix, `Q5_K` quantization *collapses* on
the outlier-heavy front-stack `attn_v` tensors. Layer 1 reconstructs
5.1× worse at `q5_k` than at `q3_k` (RMSE 0.0241 against 0.0048,
max element error 8.9 on a tensor whose values reach 12.1); layers
2 and 5 degrade 1.9× and 1.3×. The other 44 promotions all improve
4×, as 5-bit should. Three cross-checks pinned the cause. A newer
llama.cpp build (b10172) reproduces the collapse bit-for-bit, so it
is not a toolchain bug. The baseline's own `Q5_K` layer-1 `attn_v` —
same tensor, same type, bartowski's importance matrix — reconstructs
10× better than ours, so the collapse is imatrix-dependent: the
weighted fit sacrifices outlier channels our calibration set marks
unimportant (the thirteenth data point corrects this reading — his
matrix collapses the same way in our pack path, and the 10× gap
moves to the open questions). And the damage signature — median KLD equal to the
final build's 0.061, mean KLD 2.2× worse — is exactly what a
localized tensor failure looks like.
Two lessons for the ledger: **under a fixed importance matrix, a
type promotion is not guaranteed to improve a tensor**, and a
per-tensor reconstruction check — seconds of CPU — catches what the
smoke test cannot (the collapsed build would have smoked clean).
The final G1 leaves layers 1, 2, and 5 at `q3_k` and promotes the
other 44.

**The results, and the first metric the baseline loses.** Tier 1 is
the full WikiText-2 set, and tier 2 is the standard 100-chunk KL against
the f16 base:

| Model | Size | Fits 20.47 GiB budget | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|-------|------|----------------------|-------|------------|------------|
| **G1** = no-2 + `attn_v`@`q5_k` ×44 | 20.46 GiB | **yes** (11.3 MiB under) | 8.650 ± 0.064 | **0.1512** | 83.8 % |
| vramfit no-2 (eleventh data point) | 20.37 GiB | yes | 8.597 ± 0.064 | 0.1703 | 82.7 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | yes | **8.532 ± 0.064** | 0.1584 | **83.8 %** |
| G2 diagnostic = G1 + output@`q6_k` | 20.84 GiB | no (380 MiB over) | 8.620 ± 0.065 | 0.1368 | 85.5 % |

Bold marks the best in-budget value — G2's better fidelity numbers
do not compete, because G2 does not fit.

G1 is the first in-budget vramfit artifact to beat the baseline on
mean KLD — 0.1512 against 0.1584, 4.5 % lower, outside the combined
error bars — and it matches the baseline's top-token agreement
(83.81 % against 83.85 %). It loses full-set perplexity, 8.650
against 8.532. By the duel's stated bar (beat both numbers), this is
not the outright win. It is a split decision, and the split has a
structure worth recording. (The thirteenth data point bounds the
KLD win to this 100-chunk window — the full 564-chunk window loses
the mean on the same two unstable chunks as PPL, and G1 leads both
metrics across the other 562.)

**Where the PPL loss lives: two chunks.** Per-chunk decomposition of
the 564-chunk run puts the entire full-set loss in chunks 347 and
502, where G1's per-token NLL jumps to 7.3 and 8.4 against the
baseline's 2.4 and 2.3. Exclude those two chunks and G1 *leads* the
baseline across the other 562. The chunks are unremarkable
encyclopedic prose. The no-2 artifact already ran hot there (3.3
and 3.5) — those two chunks alone carried roughly half of the
eleventh data point's 0.065 PPL gap — and the promotion amplified
the weak spot rather than causing it. The amplification is not a
weight defect: every promoted channel of every promoted tensor
reconstructs strictly closer to f16, and the collapsed 47-layer
build — a strictly *worse* set of weights — shows no spike at all on
either chunk. Two passages sit near a dynamical instability of this
recipe lineage, where small weight perturbations in either
direction swing the trajectory. The scan frame has no concept of
this, the validation pass has no concept of this, and 100-chunk
tier 2 does not reach chunk 347 — the scoreboard caveat is
recorded: the KLD window and the PPL window disagree about G1
partly because the instability sits outside the KLD window.

**The head is not the lever.** G2 spent 391 MiB promoting the
output head to the baseline's `Q6_K` and bought 0.031 PPL and
0.014 KLD over G1 — and the two unstable chunks barely moved
(7.3 and 8.5, against G1's 7.3 and 8.4). The probe ladder was drawn
up expecting the head to be the biggest single lever — it is the
baseline's most visible extravagance. Measured, it is a modest
fidelity lever and no stabilizer. The baseline's PPL edge does not
live in its head.

**What this changes.** The granularity suspicion was half right.
Within-layer `attn_v` protection closes the KLD residual and
crosses the baseline — on the metric the ninth data point
established as the ranking-stable one, a measured in-budget recipe
now beats the heuristic. That is the evidence tensor-level groups
(ADR-0012's declared v1 boundary) have waited for, and lifting the
boundary is a solver feature, scoped separately. What granularity
does not close is the PPL residual, because that residual is not
smooth quality — it is two unstable passages that no tensor
promotion in either probe touched. Whole-model stability under
weight perturbation is a phenomenon none of our instruments
measure: not the scan frame, not the validation pass, not a
100-chunk tier 2, not the smoke test. The runtime-frame lane
(issue #40, ADR-0021) inherits a second job description:
runtime-frame prices for 2-bit re-entry, and enough evaluation
breadth to see instabilities the calibration-sized windows miss.

## The thirteenth data point: the imatrix swap, and the instability's address

The twelfth data point left one untested lever that plausibly
touched chunks 347 and 502: the importance matrix. The case was
circumstantial but consistent — the fit collapse looked
imatrix-dependent (the baseline's own layer-1 `attn_v` reconstructs
10× better than ours at the same type), imatrix effects are
allocation-dependent and large (the eighth and ninth data points),
and the no-2 artifact already ran hot on both chunks under our
matrix. On 2026-08-08 the probe ran: bartowski's published
importance matrix (483 chunks of his calibration set, a 17.5 MiB
fetch — no rescan), swapped into the G1 pack pattern. Two builds:
**G1-bimx**, the exact 44-layer promotion set, isolating the
imatrix variable; and **G1b-bimx**, the full 47-layer set including
layers 1, 2, and 5, testing whether his matrix removes the
collapse. The reconstruction check ran before any eval, and this
time its raw output is saved (`probes/recon-*.jsonl` — the twelfth
data point's collapse ratios were computed live and never written
to disk, a gap this run closes).

**The collapse is not the imatrix's, and not the promotion set's.**
Under bartowski's matrix, in our pack path, layer 1's `attn_v` at
`Q5_K` reconstructs at RMSE 0.0245 — the same signature as ours
(0.0241), 5.1× worse than the `Q3_K` fit (0.00476 under our matrix,
0.00474 under his). Layers 2 and 5 repeat the pattern at 1.87× and
1.32×. The promotion set is exonerated outright: between the
44-layer and 47-layer builds under the same matrix, every shared
(tensor, type) pair reconstructs to identical RMSE — which layers
are promoted has zero effect on any individual fit. Within our pack
path, then, the collapse is a **(tensor, type)** property. What the
run does *not* explain is the twelfth data point's 10× cross-check,
which it reproduces and sharpens
(`probes/recon-baseline-q3ks.jsonl`): bartowski's published pack
fits the same layer-1 tensor at the same `Q5_K` 10.1× cleaner
(RMSE 0.00241) against the same f16 base, while its deep-layer fits
match ours digit for digit (layer 4 at 0.000493, layer 9 at
0.000440 in both). Same tensor, same type, same base — and his own
matrix collapses when *our* quantize invocation drives it. The
difference lives somewhere in the pack path (toolchain vintage or
quantizer flags), and it moves to the open questions. The twelfth
data point read the cross-check as "the collapse is
imatrix-dependent" — this run corrects that. His published pack
also fits our worst constant cleanly: our layer-3 `attn_v` at
`Q4_K` sits at relative RMSE 0.52–0.53 under both matrices, his at
`Q5_K` sits at 0.061. The front stack is hostile to our fits, not
to K-quants as such.

**A collapsed signature is not a destroyed model.** The build that
carried this signature under our matrix cost +0.94 PPL (9.594
against G1's 8.650). The same signature under bartowski's cost
nothing — G1b-bimx scored 0.11 PPL *better* than its 44-layer
sibling. RMSE magnitude does not decide model-level damage; the
identity of the sacrificed channels does. The reconstruction check
stays a conservative gate: it refuses some packs that would serve
fine, and it catches the one catastrophic case on record.

**The results.** Tier 1 is the full WikiText-2 set, tier 2 the
standard 100-chunk KL window. Every build fits the 20.47 GiB
weight budget (G1 and G1-bimx 11.3 MiB under, G1b-bimx 5.2 MiB
under — the size checks are in the pipeline console logs):

| Model | PPL ↓ | Mean KLD (100) ↓ | Same top ↑ |
|-------|-------|------------------|------------|
| Q3_K_S heuristic (bartowski) | **8.532 ± 0.064** | 0.1584 | **83.8 %** |
| G1 (ours, 44 layers) | 8.650 ± 0.064 | **0.1512** | **83.8 %** |
| G1b-bimx (bartowski imx, 47 layers) | 8.646 ± 0.065 | 0.1551 | 83.6 % |
| G1-bimx (bartowski imx, 44 layers) | 8.752 ± 0.066 | 0.1550 | 83.6 % |

The probe misses. Neither bartowski-imatrix build beats the
baseline's PPL, and neither touches G1's KLD. The apparent
inversion — under his matrix the 47-layer build outscores the
44-layer one — is not a quality inversion: it is chunks 347 and
502 again. Excluding those two chunks, the 44-layer build *leads*
its sibling 8.551 to 8.640, and the 100-chunk KLD column — a
window that reaches neither chunk — scores them a dead tie.

**The instability has an address, and it is not the imatrix.**
Per-chunk NLL at the two hot chunks:

| Build | chunk 347 | chunk 502 |
|-------|-----------|-----------|
| baseline Q3_K_S | 2.4 | 2.3 |
| no-2 | 3.3 | 3.5 |
| G1 (ours, 44 layers) | 7.3 | 8.4 |
| G1-bimx (bartowski, 44 layers) | 8.3 | 9.1 |
| G1b-bimx (bartowski, 47 layers) | 2.4 | 2.3 |

Swapping the matrix in the 44-layer set made both chunks *worse*.
Adding layers 1, 2, and 5 — the "collapsed" tensors — returned
both chunks to baseline level: the table's 2.4 and 2.3 are 2.40
and 2.27 at two decimals, against the baseline's 2.36 and 2.28. The instability is
a property of the joint weight state: the front-stack
`attn_v` fits and the mid-stack promotions have to agree, and
which side a build lands on is decided by interactions no
per-tensor instrument sees. This is the twelfth data point's
dynamical-instability reading, and G1b-bimx is now the *second*
independent build that sits quiet on both chunks — the collapsed
47-layer our-matrix build was the first ("no spike at all on
either chunk"). Both off switches run through the front stack, and
both cost quality elsewhere to flip.

**The full-window KLD localizes the two-window disagreement.** A
564-chunk KL base (34.4 GiB of f16 logits, 81 minutes) put tier 2
over the same window as tier 1 for the first time — and on the
full window the baseline wins mean KLD too:

| Model | Mean KLD (564) ↓ | chunk 347 | chunk 502 | Mean KLD excl. those two | Same top ↑ |
|-------|------------------|-----------|-----------|--------------------------|------------|
| Q3_K_S heuristic | **0.2959** | 0.120 | 0.107 | 0.2966 | **83.4 %** |
| G1 | 0.3066 | 5.963 | 6.864 | **0.2849** | 82.8 % |
| no-2 | 0.3088 | 1.558 | 1.674 | 0.3042 | 81.8 % |

So the two windows do rank G1 opposite on the same metric —
0.1512 against 0.1584 inside 100 chunks, 0.3066 against 0.2959
over 564 — and the decomposition says exactly why. Two chunks out
of 564 carry G1's entire full-window loss: exclude them and G1
leads on mean KLD (0.2849 against 0.2966) *and* on PPL (8.476
against 8.526). The disagreement is not between windows or
between metrics — it is between the two instability sites and
everything else. G1 is the better quantization of 562 of 564
chunks on both metrics, and the baseline wins both full-set means
because two passages sit on the wrong side of a knife edge G1's
weight state lands on. no-2 loses the exclusion comparison too
(0.3042) — the `attn_v` protection is what closes the
smooth-quality gap, confirmed on a second metric.

**What this changes.** The imatrix lane is closed: swapping the
matrix neither removes the fit collapse in our pack path nor
stabilizes the hot chunks in the 44-layer set, and the
G1b-bimx pack pattern that does stabilize them pays for it in
mean KLD. The twelfth data point's headline needs its bound
stated: "first in-budget artifact to beat the baseline on mean
KLD" holds inside the 100-chunk window and loses on the full
window — window-dependent, like everything the instability
touches. The evidence for within-layer protections does not rest
on that headline; it rests on the exclusion analysis, where the
protection closes the smooth-quality gap on both metrics across
562 of 564 chunks. The honest scoreboard claim is the split
decision with its structure fully mapped: an in-budget measured
recipe that beats the heuristic everywhere except two unstable
passages. Whole-model
stability under weight perturbation remains the one phenomenon no
instrument in the pipeline prices, and the runtime-frame lane
(issue #40, ADR-0021) keeps its second job description unchanged.

Raw logs: `eval/ppl-probeG1{,b}-bimx.log`,
`eval/kl-probeG1{,b}-bimx.log`, `eval/kl564-{baseline-q3ks,probeG1-attnv5,no2}.log`,
`eval/kl-base-564.log`,
`probes/recon-{g1-ourimx,g1-bimx,g1b-bimx,baseline-q3ks}.{jsonl,txt}`,
and `eval/ppl-recipe-no2-assisted.log` for the no-2 rows — not
`eval/ppl-no2.log`, which is the destroyed 2026-07-29 build.

## The fourteenth data point: the pack-path gap closed, and the split decision ended

The thirteenth data point left one fact standing between the
project and an outright win: bartowski's published pack fits the
front-stack `attn_v` tensors 10× cleaner than every pack we
quantize — same tensor, same type, same f16 base, same imatrix.
The suspects were quantizer flags and toolchain vintage. On
2026-08-08 both were run down, and both are innocent. The real
mechanism is uglier and more useful.

**Flags exonerated in one run.** Test A: a stock `Q3_K_S`
quantize on our build — no `--pure`, no `--tensor-type`
overrides, bartowski's own imatrix — the closest replica of his
invocation our toolchain can produce. Layer 1's `attn_v` at
`Q5_K` reconstructs at RMSE 0.02448, digit-identical to the
G1b-bimx probe and 10× worse than his 0.00241
(`probes/recon-testA-stock-bimx.jsonl`). Every flag we ever
passed is irrelevant: the stock path collapses the same way.

**Vintage exonerated at the source level.** His model card names
llama.cpp release b5962 (2025-07). Diffing b5962 against our
build (e9fa078, 2026-07-28): `make_qkx3_quants` and
`quantize_row_q5_K_impl` — the entire weighted `Q5_K` fit — are
byte-identical, `make_qp_quants` differs only in two epsilon
guards no healthy scale triggers, and the imatrix loader computes
the same per-column means in both (a global scale the fit is
invariant to). There is no vintage to blame.

**The mechanism, isolated.** A 53-line harness
(`probes/qfit.c`) calls `ggml_quantize_chunk` on the single
extracted tensor with a single extracted imatrix row — no
llama-quantize, no driver, no flags. It reproduces the pack
fits to eight significant digits (blk.1 under bartowski's row:
RMSE 0.0244839988, max error 11.641464 — the pack recon says
0.0244839974, 11.641464). The fit is fully determined by
(tensor, type, imatrix row). And the rows are the problem: the
calibration activations for the front-stack attention inputs
span up to 4×10¹³ from smallest to largest column (blk.0
4.1×10¹³, blk.1 1.1×10¹², blk.2 2.4×10¹¹, blk.3 1.5×10¹⁰,
blk.5 5.6×10⁸), in both our matrix and his. Under a row like that, the weighted
super-block scale fit (`make_qkx3_quants` feeding
`make_qp_quants`) collapses at `Q4_K`/`Q5_K` — 5.8× to 14.7×
worse than the unweighted fit — while the `Q3_K` path only
inflates ~1.4× (0.00476 against 0.00346 on blk.1). Raw range
only loosely predicts the failure: blk.5 collapses 5.8× at
5.6×10⁸ while blk.23 fits clean at 3.8×10⁹, and blk.4's ffn
row spans 2.0×10¹⁰ untested at the collapsing types — how the
extreme columns land inside the 32-wide sub-blocks decides,
and the reconstruction check, not a range threshold, is the
instrument. The type-dependence is why the twelfth data
point saw promotions *hurt*: promote a front-stack tensor from
`Q3_K` to `Q5_K` under a pathological row and the fit gets
worse, not better.

| fit of blk.N `attn_v` | weighted (our imatrix) | unweighted | ratio |
|-----------------------|------------------------|------------|-------|
| blk.1 @ `Q5_K` | 0.02413 | 0.00164 | 14.7× |
| blk.2 @ `Q5_K` | 0.00426 | 0.00057 | 7.4× |
| blk.3 @ `Q4_K` | 0.01182 | 0.00114 | 10.4× |
| blk.5 @ `Q5_K` | 0.00289 | 0.00050 | 5.8× |

**What the harness says about his pack.** Bartowski's published
row reproduces *our* collapse (0.0245), and the unweighted fit
gives 0.00164 — his published 0.00241 matches neither branch of
code that has not changed since his release. His `attn_output`
fits match ours within ~4 %, his `attn_q`/`attn_k` within a few
percent everywhere except blk.0 (~25 % off — the front stack
again). The simplest explanation consistent with all of it: the
imatrix he fed `llama-quantize` is not byte-for-byte the file he
published — the front-stack rows differed. A patched non-release
build would fit the observations too. Either way his pack is
unreproducible from his published artifacts, and the gap is off
the project's critical path. The open question below closes.

**The fix, and probe G1c.** `llama-quantize --exclude-weights
<tensor>` drops the imatrix row for named tensors and takes the
clean unweighted fit. Probe G1c is G1b's exact 47-layer pattern
— the one that silences chunks 347 and 502 — with four
exclusions: `blk.{1,2,3,5}.attn_v.weight`
(`eval/run-probeG1c-cleanfit-quantize.sh`). Exclusions change
no tensor layout, so the size holds: 21,972,739,584 B, 5.2 MiB
under budget. The
reconstruction check comes back all-green for the first time on
any pack we quantize: every promoted `attn_v` sits in the
0.036–0.050 relative band (blk.0's `Q8_0` far cleaner at
0.005), blk.1 at RMSE 0.00164 — cleaner than bartowski's own
0.00241 (`probes/recon-g1c-cleanfit.jsonl`).

**The results.** Tier 1 full set, tier 2 both windows:

| Model | PPL ↓ | KLD (100) ↓ | KLD (564) ↓ | chunk 347 | chunk 502 | KLD (564) excl. | Same top (564) ↑ |
|-------|-------|-------------|-------------|-----------|-----------|------------------|-------------------|
| Q3_K_S heuristic | **8.532 ± 0.064** | 0.1584 | 0.2959 | 0.120 | 0.107 | 0.2966 | **83.4 %** |
| G1 (44 layers) | 8.650 ± 0.064 | 0.1512 | 0.3066 | 5.963 | 6.864 | **0.2849** | 82.8 % |
| G1c (47 layers, clean fits) | 8.549 ± 0.063 | **0.1509** | **0.2949** | 0.120 | 0.122 | 0.2956 | 82.9 % |

G1c wins the 100-chunk KLD window outright — 0.1509 against
0.1584, better on 67 of 100 chunks in a paired per-chunk
comparison — and sits at exactly baseline level on both
instability chunks: 0.120 and 0.122 against the baseline's
0.120 and 0.107, where G1 scored 5.963 and 6.864. The knife
edge is off. **[Correction, the fifteenth data point: off at
347 and 502 — not gone. G1c's own full-window log carries a
fresh knife-edge spike at chunk 137, own-chunk KLD 6.09 against
the baseline's 0.108, which the 347/502 watch never flagged.
The fifteenth data point's paired comparison found it while
hunting down a mean gap. The instability moved, and a
two-chunk watchlist could not see it.]** The full-window mean KLD is a statistical tie at
a nominal lead (0.2949 against 0.2959, 0.3σ — the same margin
this page calls a tie when PPL wears it), but the tie
decomposes G1c's way: chunk by chunk, G1c is better on 416 of
564 (74 %), and the mean only ties because chunk-level KLD
differences are heavy-tailed. PPL is the same kind of tie in
the other direction — Δ0.017 against a ±0.064 interval, with
the nominal lead on the baseline's side. The baseline's one
clear remaining lead is 564-window top-token agreement
(83.4 % against 82.9 %).

Two structural notes. First, the result decomposes cleanly:
G1b-bimx proved the 47-layer pattern silences the hot chunks
even with collapsed fits, and the twelfth data point proved the
same pattern with collapsed fits *under our matrix* costs +0.94
PPL — clean fits are what let the pattern pay. Second, G1's
excluded-window KLD (0.2849) is still the best smooth-text
number on record: the collapsed 44-layer state sacrificed
channels the smooth window never exercises. G1c gives up that
edge (0.2956) to be uniformly good. A recipe cannot currently
express "collapse on purpose", and nothing measured suggests it
should.

**What this changes.** The scoreboard headline: the split
decision is over. An in-budget measured recipe wins one KLD
window outright, ties the size-matched heuristic on full-window
mean KLD and on PPL while leading 74 % of chunks on the former,
silences the instability, and passes every reconstruction gate.
What remains on the baseline's side is a real half-point of
564-window top-token agreement. ADR-0022's refuse-and-name remedy gains a
second option: instead of dropping the protection, exclude the
named tensor's imatrix row and keep the promotion — the pack
path should learn to emit `--exclude-weights` (a new lane).
*(Built: [ADR-0023](../adr/0023-imatrix-exclusions.md), exercised
end-to-end by the fifteenth data point.)*
The reconstruction check graduates from conservative gate to
the instrument that found, diagnosed, and verified the fix for
a 10× fit defect — without a single GPU eval until the final
scoreboard run.

Raw receipts: `probes/qfit.c` (the harness),
`probes/qfit-runs-2026-08-08.txt` (every fit variant),
`probes/recon-{testA-stock-bimx,g1c-cleanfit}.jsonl`,
`eval/run-{testA-stock-bimx,probeG1c-cleanfit}-quantize.sh`,
`eval/{ppl,kl,kl564}-probeG1c-cleanfit.log`,
`eval/testA-stock-bimx-quantize.log`, `probeG1c-quantize.log`.

## The fifteenth data point: the pipeline packs its own winner

Every scoreboard entry above G1's came from hand-written
`llama-quantize` invocations — the probes proved what the right
pack looks like, and ADR-0022/0023 built the CLI to express it.
The outstanding proof was the end-to-end run: `plan --protect
--exclude-imatrix`, `pack --imatrix`, and an all-green
reconstruction check, with no hand-edited flags anywhere. On
2026-08-09 that run happened. It did not replicate G1c — and the
divergence is the interesting part.

**The plan, and the one-step divergence.** The solve took the
sized no-2 map at the no-2 budget (24 GiB, KV headroom
3,791,650,816 B), G1c's 47 `attn_v` layers as explicit
`--protect …=5` rules, one `--protect
model.layers.3.self_attn.v_proj.weight=4` rule, and the four
`--exclude-imatrix` tensor names. The solver reproduced blk.0 at
8-bit and every other group of G1c's layout — except blk.3. The
47 floors price at ~97 MiB, the unprotected no-2 solve had
finished with 9.4 MiB of predicted headroom, so the greedy took
exactly one more step: blk.3's group from 4-bit to 3-bit (step
162, freeing 113,087,938 B at a predicted damage of 0.113 —
29 % of the recipe's whole predicted damage, the largest share
any archived solve trace has spent on one step). Pinning blk.3
would not restore
G1c: every other group already sits at floor, so the same bytes
would have to come out of blk.0. G1c itself really does fit —
5.2 MiB under budget — but only the 0.005 format-overhead margin
keeps predicted sizes honest against real GGUF bytes, and that
margin is ~104 MiB on this file. The hand layout lives inside
the safety margin the solver refuses to spend. So the artifact
is a sibling of G1c, not a clone: same protections, same
exclusions, same 47-layer pattern, blk.3's unprotected tensors
one step lower. The recipe records all of it
(`recipe-g1c-replication.json`, schema 4, 48 protected pairs, 4
marked `exclude_imatrix`).

One trap is worth naming for the next protection author. The
first draft used a single glob, which also matched blk.0's
`attn_v` — a per-tensor no-op, floor 5 under a group assigned 8.
The reconstruction check demands a protected tensor reconstruct
*strictly* better than the unprotected reference, and a no-op
pair quantizes identically in both packs: equal RMSE, verdict
collapsed, on a perfectly healthy tensor. The plan-time no-op
warning is per-pattern, and a pattern that lifts 47 real floors
does not warn about its 48th silent match. Enumerating the
layers avoided it. Issue #59 closed the trap for the next
author: `resolve_protected` now drops a pair the floor never
moved, and the CLI warns per tensor.

**The pack, gated.** `pack --imatrix` emitted `--pure`, the
per-group `--tensor-type` overrides, and the four
`--exclude-weights` flags on its own: 21,860,214,272 B
(20.36 GiB), 112.48 MiB under the weight budget — inside the
band the predicted 19.9 MiB margin plus the ~104 MiB overhead
cushion allows. The reconstruction check packed its unprotected
reference and measured all 48 protected tensors: green across
the board, the first all-green gate on a pack this pipeline
produced end-to-end. The four excluded tensors reproduce the
qfit.c harness fits to the digit — blk.1 at RMSE 0.001641
against the receipt's 0.00164, blk.2 0.000574, blk.3 0.001136,
blk.5 0.000501 — against unprotected references of 0.004755,
0.002229, 0.002303, and 0.002178. The remedy the gate proposes
is now the remedy the pipeline executes.

**The results.** Tier 1 full set, tier 2 both windows, against
the two artifacts that matter:

| Model | PPL ↓ | KLD (100) ↓ | KLD (564) ↓ | chunk 347 | chunk 502 | chunk 137 | KLD (564) excl. | Same top (564) ↑ |
|-------|-------|-------------|-------------|-----------|-----------|-----------|------------------|-------------------|
| Q3_K_S heuristic | 8.532 ± 0.064 | 0.1584 | 0.2959 | 0.120 | 0.107 | 0.108 | 0.2966 | **83.4 %** |
| G1c (hand-quantized) | 8.549 ± 0.063 | **0.1509** | 0.2949 | 0.120 | 0.122 | 6.086 | 0.2956 | 82.9 % |
| CLI pack (this data point) | **8.517 ± 0.063** | 0.1538 | **0.2873** | 0.126 | 0.124 | 0.106 | **0.2878** | 82.9 % |

The chunk 137 column is new to this table, and the reason it is
new is the finding below.

Against the baseline the verdict is unambiguous, and stronger
than any before it: the CLI pack is better on 369 of 564 chunks
(65 %), the mean gap is 0.0086, and a paired per-chunk test
puts the difference at 7.8σ — the first time a vramfit
artifact beats the size-matched heuristic beyond argument on
the window this page says rules.

Against G1c the mean also leads (0.2873 to 0.2949), but the
decomposition dissolves that lead into a single chunk — and the
chunk is a discovery. G1c is better on 429 of 564 chunks by a
hair. The whole mean gap of 0.0077 lives at **chunk 137, where
G1c spikes to own-chunk KLD 6.09 against the baseline's 0.108
and the CLI pack's 0.106**. Exclude that chunk and G1c leads
the mean too (by 0.0029, at 9.6σ paired). Chunk 137 is exactly
the knife-edge class this page built the 347/502 watch for, in
the artifact that watch declared clean: the fourteenth data
point's claim that G1c silenced the instability was too broad
— G1c moved the knife edge, from 347/502 to 137, and nothing
flagged it until this comparison went hunting for its mean gap.
The honest statement of the full window is: G1c wins the
per-chunk grind, the CLI pack is the scoreboard's first
spike-free profile — its worst excess over the baseline
anywhere in 564 chunks is +0.05 — and the mean prefers
spike-free. All three known knife-edge chunks stay quiet on the
CLI pack: 0.126 at 347, 0.124 at 502, 0.106 at 137.

The 100-window KLD lands between its parents — 0.1538 beats the
baseline's 0.1584 (59 of 100 chunks) and gives back nearly two
fifths of G1c's 0.1509 lead. PPL reads 8.517 ± 0.063: a tie by
the interval, but the nominal lead over the baseline (8.532)
sits on a measured recipe's side for the first time in the 49B
lane. Top-token agreement holds at G1c's 82.9 % against the
baseline's 83.4 % — still the baseline's one clear lead.

Step 162 deserves its own paragraph, carefully. The solver
predicted 0.113 damage for demoting blk.3 — 29 % of the
recipe's total, a step it took expecting to hurt — and the
measured ledger says the demotion taxed most chunks a hair
(G1c's 429-chunk edge, the two-fifths giveback on the
100-window) while the knife-edge spike the hand layout carried
is gone. The demotion is the only layout difference, so the
ledger is cleanly attributable. What is not clean is causality
on one flip: the 347/502 history says knife-edge chunks move
under small perturbations, and one artifact is one sample. The
defensible claim: the solver's forced trade cost nothing the
intervals can see, and on the deciding window it came out
ahead. The validation gap (the eleventh data point's −79.5 %
overprediction) has always said marginal damage overprices
demotions — step 162 is that gap steering a live allocation,
and this time the measurement did not punish it.

**What this changes.** The outstanding proof closes. What is
proven: the full loop — sensitivity map to recipe to gated pack
— runs from the CLI with protections and imatrix exclusions,
passes the reconstruction gate without a single hand-edited
flag, and produces the best full-window KLD on record with the
scoreboard's first spike-free chunk profile. What is *not*
proven: byte-replication of G1c at the same budget — the
solver's overhead margin will not buy the hand layout, and
pinning cannot force it without moving the bytes somewhere
worse. Nothing measured argues it should be forced: against
the baseline the CLI artifact is the strongest result this
page has recorded, and against G1c the hand layout's remaining
edge is a per-chunk grind that comes packaged with a knife
edge. The publication candidate is now a pipeline artifact,
not a probe.

Raw receipts: `recipe-g1c-replication.json` (schema 4, the
trace, 48 protected pairs),
`pack-g1c-replication.console.log`,
`nemotron-49b-g1c-replication.runlog.jsonl` (the
`reconstruction_checked` event, all 48 tensors),
`eval/run-g1c-replication-evals.sh`,
`eval/{ppl,kl,kl564}-g1c-replication.log`. The `g1c-replication`
in the filenames is the lane's name from before the divergence
was measured — the artifact is G1c's sibling, not a
byte-replication, and this page is the record of that
distinction.

## The sixteenth data point: five tasks, five ties

The first tier-3 runs completed 2026-08-10: the fixed slice
([ADR-0024](../adr/0024-tier3-task-slice.md)) on the publication
candidate, then the identical slice on the standing baseline,
chained back to back in one detached run on the reference box. Same
instruments end to end — lm-evaluation-harness 0.4.12 running
through the recorded in-process llama-cpp-python lane on the
b10172 Vulkan build, the build behind every tier-1 and tier-2
number on this page. Full evaluation splits, no `--limit`, zero context
truncations, zero task failures, and both artifacts' SHA-256s,
per-task versions, and per-item outputs in the raw JSON.

| Task (metric) | Candidate | Baseline Q3_K_S | Δ | Combined σ | Verdict |
|---|---|---|---|---|---|
| MMLU 5-shot (acc) | 0.7829 ± 0.0033 | 0.7827 ± 0.0033 | +0.0002 | 0.0047 | tie |
| GSM8K 5-shot (strict) | 0.9318 ± 0.0069 | 0.9242 ± 0.0073 | +0.0076 | 0.0101 | tie |
| HellaSwag 10-shot (acc_norm) | 0.8412 ± 0.0036 | 0.8379 ± 0.0037 | +0.0033 | 0.0052 | tie |
| Winogrande 5-shot (acc) | 0.7845 ± 0.0116 | 0.7861 ± 0.0115 | −0.0016 | 0.0163 | tie |
| ARC-Challenge 25-shot (acc_norm) | 0.6493 ± 0.0139 | 0.6604 ± 0.0138 | −0.0111 | 0.0196 | tie |

Every delta sits inside the combined standard error — the largest
is 0.8σ (GSM8K, candidate nominally ahead) — so the card says
"tie" five times, per ADR-0024 decision 4. The candidate leads
nominally on three tasks and trails on two, a split consistent
with noise. Nobody cherry-picked: the slice was fixed before any
run, and both nominal deficits print here with their error bars.

**The certification reads clean.** Tier 3 asked the one question
the distributional tiers cannot: does the packed model still
perform tasks. At equal size the answer is indistinguishable from
the standing baseline on all five axes — knowledge breadth,
generative math, commonsense continuation, coreference, science
reasoning. That is the outcome the field's largest quantization
evaluation predicted (quantized checkpoints recovering over 99 %
of baseline scores), and it is the outcome that lets tier 2 carry
the ranking claim: the 7.8σ full-window KLD win now stands on a
certified-capable artifact. The one failure mode this slice was
built to catch — decode-compounding damage that multiple-choice
scoring masks — did not appear in this run: GSM8K ties the
baseline, and strict-versus-flexible answer extraction differ by
at most 0.4 percentage points on either artifact, so both models
follow the answer format cleanly.

**The cost surprised in the right direction.** 5.70 h per
artifact — both slices, within three minutes of each other —
against the 8–14 h estimate and the lane probe's 9–12 h
projection. Both projections priced MMLU (3–5 h) and GSM8K
(4–5 h) high: MMLU's fourteen thousand questions spread across
57 subjects, each subject sharing one few-shot prefix the lane's
KV-prefix reuse makes nearly free (1.06 h measured), and GSM8K
decoded at ~3.6 s per item (1.35 h). HellaSwag's ten thousand
long continuations (2.88 h) are the slice's real cost center.
Two artifacts ran back to back in 11.4 h of wall-clock: one
night.

**What this changes.** The publication procedure's last
measurement exists. The tiers now read as one story: tier 2
ranks (the 7.8σ full-window KLD win), tier 3 certifies (five
ties at equal size), and no task lost outside noise, so the
procedure's publish-the-negative-result branch has no trigger.
Issue #80 consumes this table as its go/no-go evidence, and the
evals sidecars (ADR-0025, issue #65) will carry it beside the
weights.

Raw receipts: `eval/tier3/{candidate,baseline}/<task>.json`
(scores, stderr, wall-clock, artifact SHA-256, lm-eval and
llama.cpp versions, per-item samples),
`eval/tier3/{candidate,baseline}.console.log`,
`eval/tier3/run-chain.sh`, and the lane's cross-checks in
`eval/tier3/probe-receipts-2026-08-09.md`. The measured hours are
annotated in ADR-0024's open questions.

## The seventeenth data point: the probe that could not tell them apart

This one measures what conversation can reveal, not the artifact. It
ran 2026-08-11, after the rename and the v0.1.0 release, to answer a
question the writeup has to answer: can a person tell the publication
candidate from the standing baseline by talking to both?

The answer is no, and the run says so with a tie.

Both packs served from the b10172 Vulkan build, the build behind
every tier-1 and tier-2 number on this page, loaded one at a time
through `llama-server` at `-ngl 99 -c 8192` — every layer on the GPU,
on the reference box. Fifteen prompts went to each, decoded greedily —
temperature 0, top-k 1, seed 1234 — so nothing here is decoding luck.
The f16 original answered the same fifteen prompts on the same night,
through the same build and the same sampler, on the split lane
described below. The prompts split into three categories, scored 25
points total.

| Category | Prompts | Points | f16 original | Candidate | Baseline Q3_K_S |
|---|---|---|---|---|---|
| Factual recall (JSON, 3 fields each) | 5 | 15 | 11 | 11 | 11 |
| Acronym expansion | 5 | 5 | 5 | 5 | 4 |
| Code, executed against fixed tests | 5 | 5 | 4 | 3 | 4 |
| **Total** | **15** | **25** | **20** | **19** | **19** |

**The two packs do not merely tie — they agree.** On factual
recall they scored the same 11 of 15 points by dropping the *same four
fields*. Three are release years, each one year early: Mistral 7B to
2022, Qwen2.5 to 2023, Gemma 2 to 2023. The fourth is Qwen2.5's
developer, where both named the model family instead of Alibaba — the
candidate said "QwenAI", the baseline said "Qwen". That is the only
field in the category where their answers differ at all, and both
answers are wrong. Both named the parameter count correctly on all
five models. On code, both wrote `merge_intervals` with the same
shallow copy and then mutated the caller's inner lists through it —
the same defect, reached by the same reasoning. Two quantizations that
agree on four wrong fields and one wrong mechanism are not showing
their own damage. The likeliest reading is the model underneath, and
the control below confirms it.

The two *scored* points that separated them went one each way, and
greedy decoding makes both reproducible. The baseline expanded RoPE as
"Relative Position Encoding" instead of Rotary Position Embedding.
The candidate wrote `dict_keys + dict_keys` in `parse_size`, a Python
2 habit that raises `TypeError` on contact. Neither failure has a
family in this set — but five prompts per category cannot separate a
real difference from which five prompts got chosen.

**What this data point may not claim.** It is not a tier and does not
join the publication procedure. Twenty-five points cannot resolve the
gap tier 2 measured — a 7.8σ full-window KLD win is a statement about
how much more closely the packed artifact tracks the f16 original's
output distribution than the baseline does, and a distribution is not
a thing fifteen questions sample. The categories were also chosen
adversarially: each one is a place where an earlier informal pass at
temperature 0.6, whose replies nobody recorded, had suggested the
baseline led. Under greedy decoding and tighter instructions, that
lead did not survive. Nobody kept the earlier replies, and the
decoding and the wording changed together, so the vanished lead has no
clean cause — one sample per prompt at temperature 0.6 is the likeliest
one. The selection cuts both ways: because the categories are the ones
where the baseline had appeared to lead, the tie says nothing about
categories nobody probed.

**The control ran, and it agrees too.** The f16 original answered the
same fifteen prompts later the same day (issue #143), decoded greedily
at the same settings through the same b10172 build. It could not serve
at `-ngl 99` — 92.9 GiB does not fit a 24 GiB card — so it ran a split
lane, 14 layers on the GPU and the rest on the CPU, pinned to the 4090
with `--device Vulkan0`. It generated at 0.54 to 0.77 tokens per
second, and the batch took 35 minutes for 1,161 tokens. It dropped all
four factual fields: the same three release years, and the same
"QwenAI" for Qwen2.5's developer, character for character with the
candidate. It wrote the same `merge_intervals`, the same shallow copy
under the same comment claiming to avoid the mutation it causes. On
these prompts the shared failures belong to the base model. Neither
pack caused them.

The split lane makes that agreement harder, not softer. The f16 served
through a different device mix than either pack, and **8 of its 15
replies still came back byte-identical to the candidate's** (6 to the
baseline's) — same tokens, across a different backend split. Identical
text through an unmatched lane is a stronger statement about the model
underneath than the score table is.

**The original scores 20, one point above both.** It does not score
perfectly either — it wrote the same `merge_intervals`. Each pack then
gave up exactly one further point, and they are the two points that
already separated the packs from each other: `parse_size` for the
candidate, RoPE for the baseline. The original got both right.

A hostile reading is available here: quantization cost each pack a
point, and that is real. This probe cannot refute it and does not try.
One sample per prompt, one draw of fifteen prompts, no re-runs, and a
comparison that crosses serving lanes — one point out of 25 sits well
inside what prompt selection alone can move, which is the same
weakness that makes the tie between the packs a weak result. Tier 2
measures the shift this probe cannot size.

**What this changes.** The writeup (issue #84) has to explain why this
project built an eval harness at all, and this is the cleanest
available answer. At equal size, the candidate's advantage is
invisible to conversation. A reader who downloads both and chats with
them will find nothing, conclude the recipe bought nothing, and be
wrong — the difference is real, it just does not surface in
conversation. Tier 2 sees it because it measures the whole output
distribution against the f16 reference. Tier 3 certifies that the
difference costs no capability. Neither result is reachable by reading
answers and forming an impression, which is precisely the ecosystem
habit the scoreboard exists to replace.

Raw receipts: `eval/probe-2026-08-11/` holds the prompt set and ground
truth (`prompts.py`), the runner (`run_batch.py`), the scorer with its
executed test suite (`score.py`), all three models' full replies with
per-prompt token counts and wall-clock (`out-candidate.json`,
`out-baseline.json`, `out-f16.json`), the scored output
(`score.console.log` for the two packs,
`score-f16-vs-candidate.console.log` and
`score-f16-vs-baseline.console.log` for the control), and all three
`llama-server` logs. The control reused `prompts.py`, `run_batch.py`,
and `score.py` unchanged from the pack runs, so the *scoring*
instrument is one instrument. The *serving* lane is not —
`server-f16.cmd` records both deviating flags, `-ngl 14` and
`--device Vulkan0`, against the packs' `-ngl 99`. That file is a launch
record rather than a log, and the offload it states is not separately
attested by the server's own output.
`artifacts.json` records the served paths and SHA-256s — the
candidate's `48271199…0122` is the published file, and the f16's is
`a16d46d3…3638d`. `SHA256SUMS` covers the set. The unrecorded earlier
pass left only its runner (`informal-pass-runner.sh`).

## The eighteenth data point: 2-bit fails its gate on a new target

The first seventeen entries all measure Nemotron Super 49B. This one
does not. It ran 2026-08-14 on NVIDIA Nemotron 3.5 Lightning 30B-A3B,
the MoE target of chart #158, and it is the first runtime-frame price
this project has ever put on a 2-bit width.

That price is the whole point.
[ADR-0021](../adr/0021-runtime-frame-measurement.md) decision 4 had
barred the solver from 2-bit until a runtime-frame measurement
existed, because the scan frame's 2-bit prices kept transferring
badly — three consecutive packed losers, the eighth through the tenth
data points. The bar was a promise to measure rather than a verdict.
This is the measurement.

**The method is a gate, not a grid.** The maintainer ruled
(issue #233) that three whole-frontier packs run before any per-cell
spend: every one of the 46 routed-expert stacks at `Q2_0`, then every
stack at `Q4_0`, then every stack at MXFP4, with all 52 dense layer
groups pinned at nominal 8. Three packs answer the question three
hundred would, and they answer it in an afternoon.

Everything ran on one instrument, which for the runtime frame means
one llama.cpp release on one card. That is an H100 80 GB HBM3 on
rented hardware, llama.cpp release **b10326** built for CUDA sm_90,
every layer on the GPU at `-ngl 99`. Damage is full-set PPL and
KLD against the f16 reference over the whole WikiText-2 test set,
which tokenizes to **594 chunks** on this target — 304,128 tokens,
against the 49B's 564. The f16 base-logits file is 39,709,379,972
bytes.

Two notes on reading the table. The f16 base run's own headline is
`Final estimate: PPL = 6.8314`, a token-weighted figure over all 594
chunks. The table instead uses **6.8192**, the per-chunk mean that the
KLD runs print beside every cell, because the ratios come from the
same paired statistic. Both describe the same reference. Bits per
parameter divides packed bytes by the **31,577,554,944**-parameter
backbone, which excludes the MTP block the converter drops.

| cell | bytes | GiB | bits/param | PPL | PPL / f16 | mean KLD | 99.9 % KLD | max KLD | top-1 agree |
|---|---|---|---|---|---|---|---|---|---|
| f16 reference | 63,181,504,640 | 58.842 | 16.007 | 6.8192 | — | — | — | — | — |
| `Q2_0` | 10,636,427,168 | 9.906 | 2.695 | **27.9380** | **4.097** | 1.604130 | 12.3312 | 22.206 | 51.65 % |
| MXFP4 | 17,980,129,184 | 16.745 | 4.555 | 6.9087 | 1.013 | 0.030277 | 0.9003 | 3.183 | 92.79 % |
| `Q4_0` | 18,898,091,936 | 17.600 | 4.788 | 6.8784 | 1.009 | 0.017825 | 0.5416 | 4.703 | 94.42 % |

**`Q2_0` does not survive contact with a bf16 checkpoint.** It costs
4.097 times the reference perplexity and ninety times `Q4_0`'s mean
KLD. It agrees with the reference's top token barely more often than
it disagrees. This is not a marginal loss to be recovered by a better
allocation — it is a different model. The type landed upstream in July
2026 as a carrier for ternary QAT weights, where the weights were
already ternary before anyone quantized them, and no upstream number
had ever covered post-hoc use. Now one does.

**MXFP4 loses the 4-bit row it was auditioning for.** It packs 4.86 %
fewer bytes and carries 1.699 times `Q4_0`'s mean KLD.
[ADR-0028](../adr/0028-expert-stack-type-table.md) had guessed the
reason in advance and the guess held: `quantize_q4_0` consumes the
per-expert importance matrix and `quantize_mxfp4` ignores it, so the
cheaper format throws away the one signal the expensive one uses. One
number runs the other way, and it is worth keeping — MXFP4's *maximum*
KLD is 3.183 against `Q4_0`'s 4.703. Its worst token is better while
its bulk is worse, which is the signature of a format that clips
outliers well and models the body poorly.

**The instrument's noise floor is zero.** The `Q2_0` evaluation ran
twice, back to back, as the first measured cell. All 594 per-chunk
rows matched, and so did every summary statistic. The runtime frame
on one instrument repeats itself exactly, the way the scan frame did
on the 4090 (issue #163) and on the H100 (issue #220). That matters
because the ninth data point's 2.7–4.1× cross-process drift is what
put the instrument in the frame in the first place.

**The budget arithmetic is where this hurts.** When the gate ran,
chart #158 wanted roughly 2.9 bits per parameter to fit a 12 GiB card,
which is a 10.5 GiB weight budget. The whole frontier at `Q2_0`
*fits*, at 9.906 GiB with 0.594 GiB to spare. The cheapest cell that
holds quality needs 17.600 GiB. There is nothing between them, because
the expert-stack palette holds no type at all between 2.25 and 4.25
bits per weight on rows of 2688 and 1856. So the empty band is no
longer a gap between two usable widths. It is the entire distance
between a pack that fits and a pack that works, and no allocation
policy can cross it.

That arithmetic moved the target rather than the recipe. On 2026-08-15
the maintainer ruled the card up to 16 GiB (issue #257). Why the *card*
moved and not the recipe is the part worth recording, because no
runtime-overhead figure could rescue 12 GiB. Take the counterfactual
as far as it goes: grant the weights the entire card, all 12 GiB at
zero runtime overhead. Drop every quantizable dense group to nominal 4
on top of that. 26 of the 46 expert stacks still land on `Q2_0`. The
gate ruled out a target parameter rather than a design, and the card
turns out to have been the binding constraint the whole time.

The weight budget under that card took two rulings, and the second one
is a lesson about assumptions. Issue #257 set 14.5 GiB by subtracting
an *assumed* 1.5 GiB of runtime overhead, a figure no run had ever
measured on this model. Issue #266 measured it on 2026-08-16 and it is
**228.99 MiB**: 96.00 MiB of KV across the six layers that hold any,
47.62 MiB of recurrent state that does not grow with context, and
85.37 MiB of compute buffers. A Mamba-2 hybrid does not spend memory
the way a dense transformer teaches. The assumption over-reserved by
1.2764 GiB, which buys 7.63 expert-stack upgrades. Issue #284 ruled
the budget to **15.776 GiB** on 2026-08-16, which the realized 35-stack
mix spends at 4.287 bits per parameter, and the Destination's own text
moved with it. The measurement is a single-sequence figure: the
recurrent state allocates one cell per sequence, so `n_seq_max` 4 costs
about 371.85 MiB rather than 228.99 MiB.

The same run found a second thing worth keeping. llama.cpp reports
`offloaded 53/53 layers to GPU` at `-ngl 99` and still leaves the
357.00 MiB token embedding in host memory, because it assigns the
input layer to the CPU buffer list unconditionally. So device VRAM
runs *below* a pack's file size rather than above it. The ruling
declined to bank that credit, since its size tracks whatever type the
recipe packs the embedding at, and a budget that moves with the recipe
cannot constrain it.

RunPod billed **$2.16** for the pod, which is the cheapest decisive
result on this page. The gate work itself ran in under ten minutes —
two minutes for three parallel quantize passes, two and a half for
the base-logits pass, five and a half for four evaluations. The
fifteen-minute f16 conversion dominated the bill, and it amortizes
across every later cell on this target.

One caveat bounds all of it. A whole-frontier pack sets every stack to
one width, and the recipe chart #158 wants is a *mix*. At the measured
budget, that means 35 of 46 stacks at `Q4_0` and the rest at `Q2_0`, or
41 of 46 if issue #183 relieves the dense classes to nominal 4, which
inverts the mostly-cheap mix the old budget forced. Either way, the
gate prices the corners of that space and not its interior. Nothing
here measures whether a mixed
recipe's damage interpolates between the corners or sits somewhere
worse, and the gap between 1.604130 and 0.017825 is wide enough that
the shape of the curve between them matters. No published evaluation
fills it either — the most recent unified survey of llama.cpp
quantization ([arXiv 2601.14277](https://arxiv.org/abs/2601.14277), on
Llama-3.1-8B-Instruct) stops at 3 bits and runs no mixed recipe at
all. Issue #249 carries what the campaign buys next.

## The nineteenth data point: the recipe beats the published build and serves under the cap

This is the entry the eighteenth data point was building toward. It
closed 2026-08-22 in two halves, and together they measure the two
clauses of chart #158's Destination no earlier entry had measured:
that the recipe beats the smallest published GGUF of the target on
measured damage, and that it serves on a 16 GiB card no published
build fits. The damage half ran on the ruled campaign instrument —
llama.cpp b10362 on a rented H100, same-pod f16 base logits, 594
WikiText-2 chunks
([ADR-0027](../adr/0027-instrument-frame-matching.md)). Issue #387's
pod measured the arm and the probe, issue #372's pod measured the
published build, and the probe eval reproduces bit-identically across
the two pods. The fit half ran on the reference box under a 16 GiB
ballast cap, at $0 (issue #389).

The arm is the falsifier arm of
[ADR-0018](../adr/0018-kquant-within-group-method.md)'s `q0-imx`
clause: 11 `down_proj` expert stacks at nominal 2 on layers 22, 24,
27, 29, 31, 34, 43, 45, 47, 49, 51, the other 35 stacks at nominal 4,
every dense group at nominal 8. It is hand-authored in the campaign
form by applying
[ADR-0007](../adr/0007-recipe-solver-strategy.md)'s placement rule to
the `q0-imx` stack-keyed sensitivity map. No solver code buys 2-bit
([ADR-0021](../adr/0021-runtime-frame-measurement.md) decision 4).
One bound travels with the result wherever it is quoted: ADR-0018's
observed consequence records that the `q0-ref` map derives the
identical arm, so the win credits the stack-keyed ranking under the
ruled placement policy and not the imatrix-assisted repricing.

| build | bytes | GiB | bits/param | PPL | PPL / f16 | mean KLD | 99.9 % KLD | max KLD | top-1 agree |
|---|---|---|---|---|---|---|---|---|---|
| f16 reference | 63,181,504,640 | 58.842 | 16.007 | 6.8192 | — | — | — | — | — |
| **falsifier arm** | 16,922,476,448 | 15.760 | 4.287 | 7.9177 | **1.161096** | **0.204318** | 5.1596 | 10.507 | 83.13 % |
| spread-map probe (#321) | 16,922,476,448 | 15.760 | 4.287 | 8.0370 | 1.178594 | 0.219037 | 5.3804 | 10.767 | 82.73 % |
| bartowski `IQ2_XXS` (#372) | 18,838,022,112 | 17.544 | — | 9.0075 | 1.320914 | 0.370257 | 6.5350 | 14.823 | — |

The published build's bits-per-parameter cell stays empty because its
bytes include the `blk.52` MTP block that our `--no-mtp` converts
drop — the division would run over different weights.

**The comparison binds on both metrics, and both agree.** The
maintainer ruled (issue #380) that the Destination's published-build
clause reads the PPL ratio and mean KLD together, from one
instrument, and that a split refuses the scoreboard row rather than
letting anyone choose a metric after the measurement. No split
happened. The falsifier arm beats bartowski's `IQ2_XXS` by 15.98
points of PPL ratio — the unit that ADR-0016's standing comparison
uses, where the probe's margin reads 14.23 — and by 44.8 % on mean
KLD, at
1,915,545,664 fewer packed bytes. It is the best arm measured on this
target, on both metrics.

**The imatrix confound is stated, not bought off** (issue #278's
ruling, recorded in
[ADR-0016](../adr/0016-imatrix-in-the-pack-path.md)). Both sides
consume the same bartowski importance matrix, 185 entries over 822
chunks. The published build quantizes 91.53 % of its bytes assisted.
The arm quantizes 74.44 % assisted, because no type takes an assisted
fit at 2.25 bits on rows of 2688 and 1856, and `quantize_q8_0`
discards the matrix outright. The asymmetry is a cost of the widths
the recipe chose. It runs against the recipe, and the recipe wins
anyway. The label on the other side deserves its own sentence:
`IQ2_XXS` names 12 of the build's 417 tensors, because
`tensor_type_fallback` rewrites every row 256 does not divide, which
sends all 46 backbone expert stacks to `IQ4_NL` at 4.5 bits per
weight. The shelf's smallest build spends 4.5-bit experts and loses
to a recipe holding 11 stacks at 2.25.

**The serve half passed its bar with 225 MiB to spare.** The bar
(issue #164) is fit, not speed: the pack loads fully offloaded inside
the cap at 16k context and generates. The method is the ballast cap
issue #164 proved binds at the boundary — a real card's
`ErrorOutOfDeviceMemory`, not a soft warning — run at issue #266's
16 GiB precedent. `scripts/vram_ballast.py` holds a CUDA allocation
sized from free VRAM until llama.cpp's Vulkan device query reads
16,383 MiB free on the 24 GiB 4090. The pack was rebuilt box-side
from the recipe alone. It landed at this machine's exact byte count
for the shared campaign composition, 32 B of metadata away from the
pod's, which is issue #300's measured cross-machine variance. Under
the cap, llama.cpp b10326 reported `offloaded 53/53 layers to GPU`
and put 15,774.00 MiB of weights on the device, with the 357.00 MiB
token embedding host-mapped as always. Device buffers totaled
16,157.88 MiB against 16,383 MiB visible: KV 96.00 MiB at 16,384
cells across the six layers that hold any, recurrent state 190.47 MiB
at four server slots, compute 97.41 MiB. `llama-server` answered a
32-token completion request from inside that envelope. The published
build cannot take this test, because its 17.544 GiB of weights exceed
the card before the first buffer allocates. Per issue #164's standing
rule no tokens-per-second figure from a capped 4090 publishes, ever —
the figure reads 2 to 3 times optimistic against real smaller
silicon.

**The campaign also mapped the interior of the range the eighteenth
entry could only bracket.** Nine mixed arms carry measured KLD on the
b10362 lane, every one at the identical composition — 11 of 46 stacks
at nominal 2, 35 at nominal 4, dense at nominal 8 — differing only in
where the cheap width lands: falsifier 0.204318, spread-map probe
0.219037, the three blind draws at 0.234003, 0.251240, and 0.306273,
spread-matched 0.299049, the measured-map arm 1 at 0.360932,
class-wise 0.402953, and the deliberately inverted arm 5 at 0.559473.
A straight line between the gate's corners predicts 0.397 at this
stack count, and the measured arms straddle it — from half the
chord's height to 1.41 times it, a 2.74x span on allocation alone.
The magnitude license for reading b10362 arms against b10326 corners
is issue #372's finding that every re-run arm reproduced its b10326
figure to every printed decimal. So a gate price bounds the range and
predicts nothing inside it: allocation decides which side of the
chord a mix lands on, and the ruled-policy arm halves it — which is
the answer the eighteenth entry's open question was waiting on.

The damage half cost about $1.25 — a fresh H100 pod for 22 minutes,
17.7 of them working. The serve half cost nothing. Receipts sit in
the box's run archive under `nemotron-30b-a3b/falsifier-q0-imx/`:
eval logs, pack logs, the serve logs with device-memory samples, and
the served completion. Three things travel with this entry:

- The serve result is a ballast-cap measurement on a 4090, and the
  published claim states that method.
- The best blind draw also beats the published build, and a blind
  recipe is a scoreboard row and never a published claim
  (issue #265).
- The `q0-ref` attribution bound above travels with the headline.

## Provenance is not evidence

Hashes answer a different question and must not be confused with
quality. Hugging Face stores SHA-256 per file, and GGUF embeds
metadata in-file — those prove *this is the exact file*. vramfit's
fingerprint proves less: it ties a scan checkpoint to that scan's
recorded provenance, not to content — swapping weights under an
unchanged path defeats it, and no content evidence covers that gap
today. None of these proves the artifact is any good. The project's claim is
that a publication should carry both: provenance (hashes,
fingerprint, run log) and evidence (the three tiers above). Shipping
either alone is the current ecosystem's failure mode — evidence
without provenance is unreproducible, provenance without evidence is
a checksum on folklore.

## The publication procedure

For publication number one (the 49B pipeline pack):

> **Amended 2026-08-10 (#78).** This line named a Qwen-class packed
> model as publication number one, per
> [the artifact ecosystem](artifact-ecosystem.md). Gate 3 ended that
> plan. The Qwen2.5-3B artifact tied the Q5_K_S baseline on
> perplexity and lost KL divergence by 12% — it never earned a
> publication. The 49B pack beats its size-matched baseline at 7.8σ
> and carries the only win. A Qwen-class model publishes only with a
> winning recipe of its own.

1. Tier 1 and tier 2 on every candidate, against the same-size
   heuristic GGUF.
2. The fixed tier-3 slice ([ADR-0024](../adr/0024-tier3-task-slice.md))
   on the winner only.
3. Every number on the card next to its baseline counterpart, with
   the losing numbers included if any lose.

All three tiers run on the reference box. None require training
compute.

### The Hugging Face conventions

> **Decided 2026-08-10 (#79),** on the research in #71 and #72.

> **Corrected 2026-08-10 (#82).** The #79 record wrote `v1` in the
> repo id and the `base_model` line. Every measured artifact derives
> from `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5`: the checkpoint
> (revision `420ba7d`), the f16 conversion, the candidate pack, and
> the four baselines. The maintainer ruled that the publication
> carries `v1_5`. The repo id and metadata below carry the
> correction. The weight file uploads as
> `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf` (the repo id
> minus `-GGUF`, plus `.gguf`).

The packed model publishes as
`Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF`, from the
maintainer's personal account — no org, matching the
individual-quantizer precedent (bartowski, mradermacher). The id
starts with "Llama", which satisfies Meta's §1.b naming clause under
its strictest reading, so the repo drops bartowski's `nvidia_`
prefix.

The `fit24gib` marker names the VRAM budget the recipe was solved
for. It sits in the slot single-quant repos give the scheme
(`<Model>-Q4_K_M-GGUF`) — a vramfit pack has no single scheme, and
the budget is the claim. No surveyed repo encodes a budget in its
name (#72). The novelty is deliberate: fit-to-budget is the
differentiator, and the name should carry it.

Card metadata follows the #71 license findings plus community
convention:

- `license: other`, `license_name: nvidia-open-model-license`,
  `license_link` to the NVIDIA page.
- `base_model: nvidia/Llama-3_3-Nemotron-Super-49B-v1_5`,
  `base_model_relation: quantized`.
- `quantized_by: Alberto-Codes`, `pipeline_tag: text-generation`.
- Tags: `vramfit` (tool tag, precedent: `unsloth`, `gguf-my-repo`),
  `gguf`, `imatrix`.
- No `library_name` — post-2024 GGUF-only repos get no
  auto-detection either way. Revisit only if the "Use this model"
  widget matters.

The repo carries both license texts, both notice files, and "Built
with Llama" on the card.

The artifact set splits by role. The model repo holds the weights
and everything specific to this pack: the imatrix (published, per
3-of-4 precedent), the recipe, the evals sidecar
([ADR-0025](../adr/0025-evals-sidecar.md)), and the run log.
Baseline sidecars publish under `baselines/` with their upstream
file names (the #65 ruling in that ADR). The
sensitivity map lives only in the linked dataset repo — it describes
the base model, not this pack, and the dataset copy is canonical.
The card links it.

### The identity grammar from publication #2

> **Decided 2026-08-22 (#401),** on the
> [checks comment](https://github.com/Alberto-Codes/vramfit/issues/401#issuecomment-5382545492)
> there. Publication #1's record above is historical and stays as
> shipped. This section carries the reusable convention. #401
> rules identity only — publication #2's artifact set is not
> settled by this record.

A vramfit publication names its repository
`<family-stem>-fit<N>gib-GGUF` and its canonical weight file the
repo id minus `-GGUF`, plus `.gguf` — the #82 rule, kept. The
[family stem](../reference/glossary.md#publication) is the
upstream repository name after the org namespace, with its variant
suffix removed. The card's H1 is the repo id, as on publication
#1. The repo publishes from the maintainer's personal account, per
#79.

**`fit<N>gib` is a ruled deployment claim, not a file size.** It
states that the ruled serve validation ran the artifact inside an
N GiB VRAM boundary, under the runtime and context contract the
model card states. This sharpens #79's wording: the marker named
the solved-for budget there, and from publication #2 it names a
validated deployment boundary, which publication #1 also met. The
card states what substantiates the name: the weight budget, the
serve configuration, the context, the toolchain, and the
concurrency bound. The card publishes no throughput figure
(#164). The fit claim's counter, CPU offload, stays open on #279 —
the card cites it and settles nothing there.

**Variant rule.** While one artifact occupies a model-and-budget
family, the name carries no further marker: no context suffix, no
quality tier, no GPU model, no recipe version. If a second
artifact enters the same family, the maintainer settles the
minimum discriminator then.

For publication #2 the grammar yields:

- repo: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib-GGUF`
- weight: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf`

The stem begins with `NVIDIA-` because upstream's own repository
name does. The upstream release publishes no bare-stem repo. Its
five repositories carry variant suffixes, and the measured
checkpoint is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
(#401 checks). The derived identity drops `BF16`, as the family's
existing GGUF repos do. Card metadata states `base_model:
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` with
`base_model_relation: quantized` — the full checkpoint id keeps
its suffix there. The card's provenance section states revision
`ce38b6a`, because upstream `main` moved to `d468880` after the
download (#401 checks).

Card metadata copies the upstream license triple, as every GGUF
repo in this family does: `license: other`, `license_name:
openmdw-1.1`, `license_link` to the OpenMDW 1.1 page. The license
requires a distributor to retain the license text and the origin
notices (#401 checks). It has no naming clause, so nothing forces
or forbids the stem, unlike Meta's §1.b on publication #1. It has
no guardrail clause, so the #86 stance below has no counterpart
here. The rest of the metadata block carries over from publication
#1: `quantized_by: Alberto-Codes`, `pipeline_tag:
text-generation`, tags `vramfit`, `gguf`, `imatrix`, no
`library_name`.

Two publication-#2 facts the card must state plainly:

- **The pack carries no MTP block.** The f16 conversion ran
  `--no-mtp`. The artifact serves no speculative decoding, while
  the comparator's builds carry the MTP block, stored at `Q4_0` in
  its imatrix quants (#401 checks). No sibling artifact exists, so
  the name takes no MTP marker.
- **The stated serve contract.** llama.cpp b10326, Vulkan, at
  `-ngl 99` with the 357.00 MiB token embedding still host-side,
  16k context, under the 16 GiB ballast cap, at four server slots
  (#389, #284). The budget derives from a single-sequence overhead
  measurement and over-reserves by 364.53 MiB, so a pack at the
  full budget tolerates 593.52 MiB of overhead. That tolerance
  holds to `n_seq_max` 8, and above 8 the budget needs restating
  (#284 caveat 3, chart #158 Notes). The figure is a pass-bar
  number, not a serving-deployment number, and the card says so.

### The guardrail-efficacy stance

> **Decided 2026-08-10 (#86),** on the #71 license findings.

The NVIDIA Open Model License terminates rights when a user reduces
a Guardrail's efficacy. The #71 record quoted the clause without its
carve-out. The full §2.1 text conditions termination on acting
"without a substantially similar Guardrail appropriate for your use
case", and it lists guardrails beside encryption, DRM, and
authentication — anti-circumvention language, not a compression ban.

Publication #1 takes the comply-and-disclose stance:

- The pack modifies no Guardrail. Quantization compresses every
  weight tensor with one uniform lossy procedure. The base model's
  safety training ships intact at lower precision.
- The published damage numbers are the compliance evidence, not a
  liability. Tier 2 measures the output-distribution shift against
  the f16 base, and tier 3 holds five statistical ties at equal
  size. Both support the "substantially similar" carve-out.
- The card states the limit plainly: damage measures general
  distribution shift on WikiText-2, not guardrail behavior. The
  card claims no separate guardrail measurement.
- The card directs deployers to keep the application-layer
  protections they would use with the base model.

The card's "Guardrails and damage disclosure" section carries this
stance. The rejected alternative was silence — community quant repos
carry no guardrail language, but they publish no damage numbers
either. Numbers without framing invite the hostile reading.

## Open questions

- ~~Which lm-evaluation-harness tasks form the fixed slice, and at what
  few-shot settings.~~ **Decided (2026-08-09,
  [ADR-0024](../adr/0024-tier3-task-slice.md)): five tasks at
  leaderboard settings.** MMLU 5-shot, GSM8K 5-shot, HellaSwag
  10-shot, Winogrande 5-shot, ARC-Challenge 25-shot — full
  evaluation splits, and deltas inside the combined standard error
  report as ties. The harness lane followed on the same date: an
  in-process llama-cpp-python class over the b10172 Vulkan build,
  after both lanes named in the record failed the probe (no prompt
  logprobs from llama-server, 64 s per request through the stock
  `gguf` backend). The lane's cross-checks and the launch note live
  in that ADR's open questions.
- ~~Whether tier 2 uses the scan's calibration set, WikiText-2, or
  both — same-set confirms the additivity story, held-out text guards
  against calibration overfit.~~ **Measured (the ninth data point):
  held-out wiki.test stays the scoreboard.** In-set PPL rewards
  calibration affinity (the baseline scored below f16), and in-set
  KLD ranked the artifacts the same as wiki — same-set adds noise,
  not signal.
- ~~Whether evaluation results become a versioned artifact of their own
  (an "evals" sidecar) or stay embedded in the model card.~~
  **Decided (2026-08-09,
  [ADR-0025](../adr/0025-evals-sidecar.md)): a versioned evals
  sidecar.** One JSON document per evaluated artifact carries all
  three tiers with their settings and toolchain versions, and
  model-card numbers trace to it. The schema settled and the
  writer landed on 2026-08-10 (that record's amendment, issue
  #65). Every card artifact has a sidecar, the i-quant baselines
  included.
- ~~Which window rules when the two disagree. The twelfth data point's
  G1 wins the 100-chunk KLD window and loses the 564-chunk PPL
  window against the same baseline — and the unstable chunks sit
  only in the second.~~ **Measured (the thirteenth data point): the
  full window rules, and the disagreement is two chunks.** The
  564-chunk KLD ranks G1 behind the baseline, like full-set PPL —
  and excluding chunks 347 and 502, G1 leads on both metrics
  across the other 562. Report the full window with the
  decomposition beside it. The 564-chunk KL base (34.4 GiB, 81
  minutes of GPU-assisted f16) stays on disk for future
  full-window tier-2 runs.
- ~~Why bartowski's published pack fits the front-stack `attn_v`
  tensors 10× cleaner than every pack we quantize — same tensor,
  same type, same f16 base, and the gap survives using his own
  imatrix in our invocation (the thirteenth data point). The
  difference lives in the pack path: toolchain vintage or
  quantizer flags.~~ **Measured (the fourteenth data point):
  flags and vintage are both innocent.** The collapse is the
  weighted `Q4_K`/`Q5_K` fit itself under imatrix rows with
  extreme column dynamic range — a stock quantize on our
  build reproduces it, a single-tensor harness reproduces it
  to eight significant digits, and the fit code is
  byte-identical to his b5962 release. His published imatrix reproduces *our* collapse, so
  his clean pack traces to an imatrix that differs from the one
  he published — unreproducible from his artifacts, and closed.
  The remedy is `--exclude-weights` per collapsed tensor (probe
  G1c). Front-stack promotions stay gated by the reconstruction
  check.
- ~~Whether the pack path should emit `--exclude-weights` from a
  recipe, so a protection whose reconstruction check fails can
  keep its promotion and drop only the imatrix row — the
  fourteenth data point applied the fix by hand.~~
  **Resolved (2026-08-09,
  [ADR-0023](../adr/0023-imatrix-exclusions.md)):** it should, and
  does — `plan --exclude-imatrix` marks protected pairs, the recipe
  schema bumped to 4 (now 5, after the no-op pair drop of issue
  #59), pack emits the flags under an imatrix, and the
  gate's refusal now prints the exact flags for the re-plan. ~~The
  CLI replication of G1c is the outstanding proof.~~ **Answered
  (the fifteenth data point), with a twist:** the CLI ran the loop
  end-to-end and passed the gate all-green, but the solver refused
  G1c's exact layout — its overhead margin demotes blk.3 one step
  — and the sibling artifact set the best full-window KLD on
  record with the scoreboard's first spike-free chunk profile.
  The mechanism replicates. The exact layout does not, and the
  hand layout turned out to carry a knife edge (chunk 137) that
  the sibling does not.
- ~~Whether the f16 original answers the seventeenth data point's
  fifteen prompts the way both quantizations did (issue #143). The
  candidate and the baseline dropped the same four factual fields and
  wrote the same shallow-copy defect, which reads as inherited from
  the base model rather than caused by either pack. Nothing has tested
  that reading.~~ **Measured (2026-08-11, issue #143): the base model
  does it too.** The f16 dropped the same four factual fields and
  wrote the same shallow-copy `merge_intervals`, so the shared
  failures are inherited. It scored 20 of 25 against 19 for both
  packs, and each pack's one further loss is its own — `parse_size`
  for the candidate, RoPE for the baseline. The control ran on a
  split lane, 14 layers on the GPU and the rest on the CPU, because
  the f16 is 92.9 GiB and does not fit the 24 GiB card.
- ~~Whether a whole-frontier gate price predicts a mixed recipe's
  damage (added 2026-08-14, the eighteenth data point). The gate sets
  every expert stack to one width and measures the corners. A recipe
  that fits chart #158's budget mixes widths across stacks, and the
  corners sit 90 times apart on mean KLD. Nothing measures the shape
  of the curve between them. Issue #249 rules what the campaign
  spends next, and the answer decides whether that shape gets
  measured directly or inferred.~~ **Measured (the nineteenth data
  point): the shape got measured directly, and the chord predicts
  nothing.** Nine mixed arms at 11 of 46 stacks straddle the 0.397
  that a straight line between the corners predicts, from 0.204318 to
  0.559473 — a 2.74x span on allocation alone. The ruled-policy arm
  lands at roughly half the chord and the inverted arm lands 1.41
  times above it, so a gate price bounds a mixed recipe without
  predicting it.
