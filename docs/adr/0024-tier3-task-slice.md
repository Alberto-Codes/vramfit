# ADR-0024: Tier 3 runs a fixed five-task slice at leaderboard settings

- **Status:** Accepted
- **Date:** 2026-08-09
- **Note (2026-08-09):** the lane probe ran, and both candidate
  lanes failed. The recorded lane is an in-process third path —
  see the first open question. The first slice runs launched the
  same day. The runner chains the baseline after the candidate.

## Context

All three publication gates
([artifact ecosystem](../explanation/artifact-ecosystem.md),
Power user #0) are satisfied. The third closed on 2026-08-09: the
fifteenth data point's pipeline artifact beats the size-matched
baseline on full-window KL divergence at 7.8σ
([evaluating packed models](../explanation/evaluating-packed-models.md)).
The remaining step before publication number one is tier 3: task
benchmarks through lm-evaluation-harness.

Tier 3 certifies, tier 2 ranks. Whole-model KL divergence resolved
a 3 % relative gap at 7.8σ (the fifteenth data point). The slice's
task scores carry sampling noise of ±0.4 % (MMLU) to ±1.4 %
(Winogrande), so small task deltas mean nothing. The slice answers
one question a distributional metric cannot: does the packed model
still perform tasks.

Two constraints size the slice. It runs on the reference box for
the winner and its baseline, so each artifact must fit one
overnight run. And it sits on a model card next to baseline
numbers, so readers must recognize every task without a footnote.

We surveyed the field's practice on 2026-08-09. Unsloth headlines
5-shot MMLU beside KL divergence for its Dynamic v2.0 GGUFs. It
argues KL divergence is the ranking metric — the same division of
labor as our tiers 2 and 3. The standing baseline's model card
publishes no quality numbers of its own, only links to external
perplexity and KL-divergence charts. Any fixed task slice
out-reports it. The largest published quantization evaluation
(Neural Magic, ~500,000 evaluations) included the Open LLM
Leaderboard v1 suite. On that suite, quantized checkpoints
recovered over 99 % of baseline scores. That finding calibrates
expectations: at equal size, task scores confirm capability rather
than separate candidates.

Each pick below earns its hours. MMLU is the number a card reader
looks for first, and the community's de facto quant-quality task.
GSM8K is the slice's only generative task: damage compounds over a
decode of up to 256 tokens, which multiple-choice scoring masks.
HellaSwag buys the tightest standard error (±0.5 %) of the cheap
tasks. Winogrande and ARC-Challenge cost under 1 h combined at 49B
scale and widen coverage to coreference and science reasoning. The
few-shot settings match the Open LLM Leaderboard v1 conventions,
so every number is comparable to the largest published
quant-evaluation corpus.

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

2. **The slice drops TruthfulQA.** The sixth leaderboard-v1 task
   measures alignment behavior, not quantization damage, and
   carries the widest noise of the six.

3. **Full evaluation splits, always.** No `--limit`, no
   subsampling. The split size is the noise floor, and every
   reported score carries its standard error.

4. **Deltas inside the combined standard error are ties, and the
   card says "tie".** Tier 2 ranks candidates. Tier 3 certifies
   the winner.

5. **The baseline runs the identical slice, once per baseline
   artifact.** The reference box measures baseline scores rather
   than copying third-party reports, and later publications reuse
   them.

6. **Every run pins its instruments.** The lm-evaluation-harness
   version, per-task versions, and llama.cpp build are recorded
   with the results
   ([ADR-0025](0025-evals-sidecar.md)).

## Consequences

- Estimated cost per evaluated 49B artifact on the reference box:
  8–14 h. MMLU 3–5 h, GSM8K 4–5 h, the other three 1–4 h combined.
  That is one overnight run. A 3B-class artifact costs 1–2 h.
- The multiple-choice tasks are prefill-bound, and the server
  processes each shared few-shot prefix once through prompt
  caching. GSM8K is decode-bound and sets the floor no caching
  removes.
- Nobody can cherry-pick a slice fixed before any run. The card
  reports all five scores, including any losses, per the
  publication procedure.
- The model card gains five task rows next to baseline
  counterparts, each with a standard error — a comparison the
  standing baseline's card does not offer.

## Open questions

- ~~Which harness lane drives the packed GGUF on the reference box.
  The loglikelihood tasks need prompt logprobs. Candidates:
  `local-completions` against llama-server (`echo` plus
  `logprobs`), or the harness's `gguf` backend through
  llama-cpp-python. The first tier-3 run decides and records the
  lane.~~ **Decided (2026-08-09): in-process llama-cpp-python on
  the b10172 Vulkan build.** Both named candidates failed the
  probe. llama-server returns no prompt logprobs, and upstream
  rejects `echo` ("Only no echo is supported"). The stock `gguf`
  backend scores correctly but spends 64 s per request in a
  per-token Python sort over the 131k vocab — MMLU alone would
  cost ~six weeks. The recorded lane: lm-evaluation-harness
  0.4.12 drives an in-process model class over llama-cpp-python
  0.3.34, which loads the b10172 Vulkan libraries through
  `LLAMA_CPP_LIB_PATH`. That is the build behind every tier-1 and
  tier-2 number. The class reads continuation logprobs from the
  logits buffer with vectorized numpy and reuses the KV prefix
  across requests. Cross-checks against the stock backend ran at
  one seed. Aggregate scores matched on eight ARC items.
  Per-item logprobs differ by small numeric deltas, and the
  aggregates and rankings are unchanged. Greedy flags matched on
  four handmade scoring pairs. The ARC batch ran 150× faster
  (13.6 s against 34 min). The driver lives beside the run
  artifacts (`eval/tier3/llamacpp_lm.py`,
  `eval/tier3/run_tier3.py`) in the `lm-eval-venv` side venv.
  ADR-0005 keeps the harness out of the project env.
- Measured hours per task on the reference box. The estimates
  above are projections from tier-1 throughput, not measurements.
  **Note (2026-08-09):** the first runs record wall-clock per task
  (`eval/tier3/*/<task>.json`). Probe measurements: ARC-Challenge
  25-shot scores at 3.7 requests/s, and GSM8K decodes at ~5 s per
  item. The candidate's first task measured 0.32 h for the full
  ARC-Challenge split. Projection for the full slice: 9–12 h per
  artifact, narrowing the 8–14 h estimate in Consequences. The
  remaining measured hours land with the results.
- Whether instruct-tuned targets also get a chat-template variant
  of the slice. The fixed slice runs harness-default prompts. The
  packed-versus-baseline delta is the claim, and absolute scores
  are secondary.
- Whether a flip count (answers changed against the f16 reference,
  Unsloth's fidelity metric) joins the card. It needs per-item
  outputs from both runs (ADR-0025's per-item open question).
