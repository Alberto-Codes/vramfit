# ADR-0024: Tier 3 runs a fixed five-task slice at leaderboard settings

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

Two of the three publication gates are satisfied, and the third
closed on 2026-08-09 (the artifact ecosystem's Power user #0). The
fifteenth data point's pipeline artifact beats the size-matched
baseline on full-window KL divergence at 7.8σ
([evaluating packed models](../explanation/evaluating-packed-models.md)).
The remaining step before publication number one is tier 3: task
benchmarks through lm-evaluation-harness.

Tier 3 certifies, tier 2 ranks. Whole-model KL divergence separates
candidates at fractions of a percent. Task scores carry sampling
noise of ±0.4 % to ±1.4 %, so small task deltas mean nothing. The
slice answers one question a distributional metric cannot: does the
packed model still do things.

Two constraints size the slice. It runs on the reference box per
publication candidate, so it must fit one overnight run. And it
sits on a model card next to baseline numbers, so readers must
recognize every task without a footnote.

The field's practice, surveyed 2026-08-09. Unsloth headlines
5-shot MMLU beside KL divergence for its Dynamic v2.0 GGUFs and
argues KL divergence is the ranking metric — the same division of
labor as our tiers 2 and 3. bartowski model cards report
perplexity and KL divergence only, so any fixed task slice
out-reports the standing baseline's card. The largest published
quantization evaluation (Neural Magic, ~500,000 evaluations) used
the Open LLM Leaderboard v1 suite and found that well-made quants
recover over 99 % of baseline scores. That finding calibrates
expectations: at equal size, task scores confirm capability rather
than separate candidates.

## Decision

1. **The tier-3 slice is five lm-evaluation-harness tasks at fixed
   few-shot settings.**

   | Task | Few-shot | Items | Axis |
   |------|----------|-------|------|
   | MMLU | 5 | 14,042 | knowledge breadth |
   | GSM8K (strict-match) | 5 | 1,319 | generative math reasoning |
   | HellaSwag | 10 | 10,042 | commonsense continuation |
   | Winogrande | 5 | 1,267 | coreference |
   | ARC-Challenge | 25 | 1,172 | science reasoning |

2. **Each pick earns its hours.** MMLU is the number a card reader
   looks for first, and the community's de facto quant-quality
   task. GSM8K is the slice's only generative task: damage
   compounds over a ~250-token decode, which multiple-choice
   scoring masks. HellaSwag buys the tightest standard error
   (±0.5 %) of the cheap tasks. Winogrande and ARC-Challenge cost
   under 1 h combined at 49B scale and widen coverage to
   coreference and science reasoning. The few-shot settings match
   the Open LLM Leaderboard v1 conventions, so every number is
   comparable to the largest published quant-evaluation corpus.
   TruthfulQA, the sixth leaderboard task, is dropped: it measures
   alignment behavior, not quantization damage, and carries the
   widest noise of the six.

3. **Full test sets, always.** No `--limit`, no subsampling. The
   sample count is the noise floor, and every reported score
   carries its standard error.

4. **Deltas inside the combined standard error are ties, and the
   card says "tie".** Tier 2 ranks candidates. Tier 3 certifies
   the winner.

5. **The baseline runs the identical slice, once per baseline
   artifact.** Baseline scores are measured on the reference box,
   not copied from third-party reports, and reused across
   candidates.

6. **Every run pins its instruments.** The lm-evaluation-harness
   version, per-task versions, and llama.cpp build are recorded
   with the results
   ([ADR-0025](0025-evals-sidecar.md)).

## Consequences

- Estimated cost per 49B candidate on the reference box: 10–14 h.
  MMLU 3–5 h, GSM8K 4–5 h, the other three under 4 h combined.
  That is one overnight run. A 3B-class candidate costs 1–2 h.
- The multiple-choice tasks are prefill-bound, and prompt caching
  makes the shared few-shot prefixes nearly free. GSM8K is
  decode-bound and sets the floor no caching removes.
- A slice fixed before any run cannot be cherry-picked after one.
  The card reports all five scores, including any losses, per the
  publication procedure.
- The model card gains five task rows next to baseline
  counterparts, each with a standard error — a comparison the
  standing baseline's card does not offer.

## Open questions

- Which harness lane drives the packed GGUF on the reference box.
  The loglikelihood tasks need prompt logprobs. Candidates:
  `local-completions` against llama-server (`echo` plus
  `logprobs`), or the harness's `gguf` backend through
  llama-cpp-python. The first tier-3 run decides and records the
  lane.
- Measured hours per task on the reference box. The estimates
  above are projections from tier-1 throughput, not measurements.
- Whether instruct-tuned targets also get a chat-template variant
  of the slice. The fixed slice runs harness-default prompts. The
  packed-versus-baseline delta is the claim, and absolute scores
  are secondary.
- Whether a flip count — answers changed against the f16
  reference, Unsloth's fidelity metric — joins the card. It needs
  per-item outputs from both runs (ADR-0025's per-item open
  question).
