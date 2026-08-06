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
| quantfit 8k map | 20.30 GiB | no | 9.917 ± 0.075 | 0.3748 | 75.4 % |
| **quantfit 8k map** | **20.30 GiB** | **yes** | **9.061 ± 0.067** | **0.2701** | **79.2 %** |
| quantfit 32k map | 19.99 GiB | no | 10.483 ± 0.081 | 0.4288 | 73.8 % |
| quantfit 32k map | 19.99 GiB | yes | 10.412 ± 0.082 | 0.4708 | 77.0 % |
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
| quantfit 64k map + imatrix | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
| quantfit 8k map + imatrix | 20.30 GiB | 9.061 ± 0.067 | 0.2701 | 79.2 % |
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
(layers 42–70 are FFN-only blocks). Hand-driven packs of
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
| quantfit kquant map + imatrix | 20.21 GiB | 9.251 ± 0.069 | 0.3056 | 77.8 % |
| quantfit RTN 64k map + imatrix | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
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
| quantfit RTN 64k map + imatrix | 8.829 | 0.5611 |
| quantfit kquant map + imatrix | 8.673 | 0.6114 |
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
| quantfit assisted map + imatrix | 20.37 GiB | 9.607 ± 0.072 | 0.3437 | 76.3 % |
| quantfit kquant map + imatrix | 20.21 GiB | 9.251 ± 0.069 | 0.3056 | 77.8 % |
| quantfit RTN 64k map + imatrix | 20.32 GiB | 9.156 ± 0.068 | 0.2653 | 80.1 % |
| Q3_K_S heuristic (bartowski) | 20.45 GiB | **8.532 ± 0.064** | **0.1584** | **83.8 %** |

The baseline gap widened from 0.62 to 0.72 to 1.08 PPL across the
three quantfit artifacts. Imatrix-blind pricing is eliminated as
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
- ~~Whether tier 2 uses the scan's calibration set, WikiText-2, or
  both — same-set confirms the additivity story, held-out text guards
  against calibration overfit.~~ **Measured (the ninth data point):
  held-out wiki.test stays the scoreboard.** In-set PPL rewards
  calibration affinity (the baseline scored below f16), and in-set
  KLD ranked the artifacts the same as wiki — same-set adds noise,
  not signal.
- Whether evaluation results become a versioned artifact of their own
  (an "evals" sidecar) or stay embedded in the model card.
