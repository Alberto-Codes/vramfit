---
status: draft
---

# Evaluating packed models: the endgame scoreboard

> **Status: draft** — tiers 1 and 2 ran for real on 2026-07-28,
> against the first packed model and again against the
> runtime-capability mix (Qwen2.5-3B, see
> [the first data point](#the-first-data-point) and
> [the second data point](#the-second-data-point-the-runtime-capability-mix)
> below). The validation pass — the middle leg — ran the same night
> ([the third data point](#the-third-data-point-the-first-validation-pass)).
> On 2026-07-29 the full loop ran on the 49B acceptance target and
> **lost to the size-matched baseline**
> ([the fourth data point](#the-fourth-data-point-the-north-star-attempt-lost-honestly)).
> Tier 3 has not run. The publication gates that consume
> these evaluations live in [the artifact ecosystem](artifact-ecosystem.md)
> and issue #11.

Scanning and planning happen inside quantfit's own frame: damage,
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
number, which is exactly why we must too — it makes a quantfit packed
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
whole-recipe validation pass (`quantfit validate`, ADR-0006) then
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

It is the expensive tier — hours per model, and scores carry enough
noise that small deltas mean nothing. Reserve it for the head-to-head
that matters: quantfit's packed model versus the size-matched
heuristic GGUF ([ADR-0010](../adr/0010-sub-4-bit-serving-path.md)
names IQ3-class baselines for the 49B). A small, fixed slice of tasks,
reported honestly with the noise acknowledged, beats a sprawling suite
nobody re-runs.

## The first data point

Both tiers ran on 2026-07-28 against the first packed model:
Qwen2.5-3B under a 4 GiB VRAM budget (2 GiB weight budget), packed
from the measured sensitivity map by `quantfit pack`
([ADR-0012](../adr/0012-gguf-type-mapping.md)). Tier 1 is the full
WikiText-2 test set. Tier 2 is whole-model KL against the f16
reference over the first 100 chunks (51,200 tokens). All runs on the
reference box (llama.cpp b10172, Vulkan).

| Model | File size | Fits 2.00 GiB budget | PPL ↓ | Mean KLD ↓ | Same top token ↑ |
|-------|-----------|----------------------|-------|------------|------------------|
| f16 reference | 5.75 GiB | no | 8.422 ± 0.057 | — | — |
| **quantfit recipe** (7×8-bit incl. embed, 30×4-bit)* | 1.98 GiB | **yes** (17 MiB under) | **8.661 ± 0.058** | **0.0382** | **90.5 %** |
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
budget and `quantfit pack` refused it — GGUF's effective bits exceed
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
| **quantfit 6/5/4 mix** (3×6-bit incl. embed, 29×5-bit, 5×4-bit) | **1.995 GiB** | **yes** (5.3 MiB under) | **8.534 ± 0.057** | **0.0180** | **93.3 %** |
| quantfit {8, 4} recipe (first data point) | 1.983 GiB | yes | 8.661 ± 0.058 | 0.0382 | 90.5 % |
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
5-bit-dominant mix packed 4.5 MiB **over** budget and `quantfit
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
artifact* — ran without its middle leg until `quantfit validate`
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
set (584 chunks). Tier 2 is whole-model KL against the f16
reference over the first 100 chunks (51,200 tokens), the f16 pass
GPU-assisted on the reference box (llama.cpp b10172, Vulkan). The
f16 reference measures PPL 8.228 ± 0.141 on those 100 chunks.

| Model | File size | Fits 20.47 GiB budget | imatrix | PPL ↓ | Mean KLD ↓ | Same top token ↑ |
|-------|-----------|----------------------|---------|-------|------------|------------------|
| f16 reference | 93 GiB | no | — | 8.228 ± 0.141* | — | — |
| **quantfit recipe** (8/4/3/2 mix) | **20.30 GiB** | **yes** (169.7 MiB under) | no | 9.917 ± 0.075 | 0.3748 | 75.4 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | yes (21.7 MiB under) | yes | **8.532 ± 0.064** | **0.1584** | **83.8 %** |
| IQ3_M heuristic (bartowski) | 21.10 GiB | no (648 MiB over) | yes | 8.300 ± 0.060 | 0.1633 | 84.1 % |
| control Q3_K_S (ours, same f16 base) | 20.45 GiB | yes | no | 9.655 ± 0.073 | 0.3451 | 76.9 % |

\* The f16 PPL covers the tier-2 100 chunks only — the 93 GiB
reference is too slow for the full 584-chunk set on this box. All
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
predicted damage 1.44) and `quantfit pack` packed it cleanly. The
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

## Provenance is not evidence

Hashes answer a different question and must not be confused with
quality. Hugging Face stores SHA-256 per file, and GGUF embeds
metadata in-file — those prove *this is the exact file*. quantfit's
fingerprint proves less: it ties a scan checkpoint to that scan's
recorded provenance, not to content (swapping weights under an
unchanged path defeats it — content evidence is an open item in
issue #8). None of these proves the artifact is any good. The project's claim is
that a publication should carry both: provenance (hashes,
fingerprint, run log) and evidence (the three tiers above). Shipping
either alone is the current ecosystem's failure mode — evidence
without provenance is unreproducible, provenance without evidence is
a checksum on folklore.

## The publication procedure

For publication number one (a Qwen-class packed model, per
[the artifact ecosystem](artifact-ecosystem.md)):

1. Tier 1 and tier 2 on every candidate, against the same-size
   heuristic GGUF.
2. A fixed tier-3 slice on the winner only.
3. Every number on the card next to its baseline counterpart, with
   the losing numbers included if any lose.

All three tiers will run on the reference box. None require training
compute.

## Open questions

- Which lm-evaluation-harness tasks form the fixed slice, and at what
  few-shot settings.
- Whether tier 2 uses the scan's calibration set, WikiText-2, or
  both — same-set confirms the additivity story, held-out text guards
  against calibration overfit.
- Whether evaluation results become a versioned artifact of their own
  (an "evals" sidecar) or stay embedded in the model card.
