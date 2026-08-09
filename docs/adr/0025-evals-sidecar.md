# ADR-0025: Evaluation results ship as a versioned evals sidecar

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

Publication number one ships weights with receipts: sensitivity
map, recipe, run log
(the [artifact ecosystem](../explanation/artifact-ecosystem.md)). Evaluation results
have no artifact of their own. Tier numbers live in raw eval logs
and the scoreboard page, and a model card would hand-copy them.

Hand-copied numbers drift. The scoreboard already corrected one
transcription error in a chunk count (564, not 584 — the twelfth
data point's log audit). The model card is the highest-visibility
surface the project writes, and it must not depend on
transcription.

The scoreboard's own rule is that a publication carries provenance
*and* evidence
([evaluating packed models](../explanation/evaluating-packed-models.md)).
The map, recipe, and run log make the provenance machine-readable
and versioned. The evidence deserves the same treatment.

## Decision

1. **Evaluation results become a versioned artifact: the evals
   sidecar.** One JSON document per evaluated packed model,
   carrying the `quantfit_schema` envelope, published beside the
   weights.
2. **The sidecar records all three tiers.** Per tier: metric
   values with standard errors, the dataset or task list, few-shot
   settings (ADR-0024), sample or chunk counts, and the tier-2
   window sizes.
3. **The sidecar names what produced the numbers.** The evaluated
   file's SHA-256, the lm-evaluation-harness version, per-task
   versions, the llama.cpp build, and the run date.
4. **Model-card numbers trace to a sidecar.** A card number
   without a sidecar entry is a defect. Card tooling can come
   later — the rule binds now.
5. **No schema code lands with this record.** The shape below is
   an open item, and the envelope stays with the JSON adapters
   when a writer is built (ADR-0008).

## Consequences

- A re-run produces a new sidecar, never an edit to prose. Two
  sidecars for one artifact diff cleanly.
- The baseline's scores get a sidecar too (ADR-0024 decision 5),
  so every card comparison is a pair of machine-readable
  documents.
- When the writer lands, it is an outbound adapter behind a port
  with a verified-fake contract suite (ADR-0009). The schema then
  joins the `quantfit_schema` versioning rule: breaking changes
  bump it.
- One more artifact rides every publication. The upload checklist
  grows by one file.

## Open questions

- The schema shape. A candidate sketch, not committed:

  ```json
  {
    "quantfit_schema": 1,
    "artifact": {"file": "…", "sha256": "…", "size_bytes": 0},
    "toolchain": {"lm_eval": "…", "llama_cpp": "…", "date": "…"},
    "tier1": {"dataset": "wikitext-2-test", "chunks": 564,
              "ppl": 0.0, "ppl_stderr": 0.0},
    "tier2": {"reference": "f16",
              "windows": [{"chunks": 564, "mean_kld": 0.0,
                           "same_top": 0.0}]},
    "tier3": {"tasks": [{"name": "mmlu", "few_shot": 5,
                         "n": 14042, "score": 0.0,
                         "stderr": 0.0}]}
  }
  ```

- Whether per-item and per-chunk detail rides beside the
  aggregates. Flip counts need per-item outputs (ADR-0024's open
  question), and the knife-edge chunk story lives in per-chunk
  KLD.
- Whether the sidecar embeds the baseline comparison or the card
  joins two sidecars at render time.
- Where the sidecar sits in the published repo, and whether the
  planned `quantfit` HF tag points at it (the
  [artifact ecosystem's](../explanation/artifact-ecosystem.md)
  conventions-to-settle list).
