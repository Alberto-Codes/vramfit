---
status: sketch
---

# Evaluating packed models: the endgame scoreboard

> **Status: sketch** — evaluation strategy recorded 2026-07-28, before
> `quantfit pack` exists. Nothing here is implemented. The publication
> gates that consume these evaluations live in
> [the artifact ecosystem](artifact-ecosystem.md) and issue #11.

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
whole-recipe validation pass (ADR-0006, issue #8) then replays the
exact recipe through the scan's own quantization and compares against
the summed marginal damages — that pre-pack check is what isolates
the additivity assumption leaking. Tier 2 complements it from the
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
