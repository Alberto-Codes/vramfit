# ADR-0025: Evaluation results ship as a versioned evals sidecar

- **Status:** Accepted
- **Date:** 2026-08-09
- **Note (2026-08-10):** the open items below are settled, and the
  writer landed (issue #65). Schema version 1 carries aggregates
  only, each tier block is optional, and the card joins sidecar
  pairs at render time. The baseline set grew: the #90 i-quant
  comparison put three more baselines on the card, so decision 4
  now requires sidecars for them too. Publication #1 carries five
  sidecars.

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

- ~~The schema shape.~~ **Decided (2026-08-10, issue #65): schema
  version 1, committed with the writer.**

  ```json
  {
    "quantfit_schema": 1,
    "artifact": {"file": "…", "sha256": "…", "size_bytes": 0},
    "toolchain": {"llama_cpp_build": "…", "lm_eval": "…",
                  "llama_cpp_python": "…", "lane": "…"},
    "tier1": {"date": "…", "dataset": "wikitext-2-test",
              "chunks": 564, "ppl": 0.0, "ppl_stderr": 0.0},
    "tier2": {"reference": "f16", "dataset": "wikitext-2-test",
              "windows": [{"date": "…", "chunks": 564,
                           "mean_kld": 0.0, "kld_stderr": 0.0,
                           "same_top_pct": 0.0,
                           "same_top_stderr_pct": 0.0}]},
    "tier3": {"tasks": [{"date": "…", "name": "mmlu",
                         "version": "2", "few_shot": 5, "n": 14042,
                         "metric": "acc", "score": 0.0,
                         "stderr": 0.0,
                         "wall_clock_seconds": 0.0}]}
  }
  ```

  Four changes against the sketch. Each tier block is nullable,
  and at least one tier must be present — the #90 i-quant
  baselines carry tiers 1–2 only, and decision 4 still binds
  their card rows to a sidecar. Each task row names its metric
  and its harness version (decision 3) — GSM8K reports
  `exact_match,strict-match`, HellaSwag and ARC-Challenge report
  `acc_norm`. The run date sits on each
  measured leaf — the tier-1 block, the tier-2 window, the tier-3
  task — because one artifact's windows and tasks run on different
  days. The Q3_K_S windows ran ten days apart, and the baseline
  slice crossed midnight. The toolchain names the
  llama.cpp build for every tier, and the three harness fields
  (`lm_eval`, `llama_cpp_python`, `lane`) are null without
  tier 3. Every numeric field records what the instrument
  printed, including the percent-unit same-top rates — the
  sidecar exists to end transcription.
- ~~Whether per-item and per-chunk detail rides beside the
  aggregates.~~ **Decided (2026-08-10): aggregates only.** The raw
  lm-eval documents run 3 MB to 197 MB per task because of
  per-item samples, and the sidecar publishes beside the weights.
  Per-item and per-chunk outputs stay in the run archive's raw
  eval logs. A future flip count lands as a per-task aggregate
  under a schema bump. ADR-0024's flip-count question stays open.
- ~~Whether the sidecar embeds the baseline comparison or the card
  joins two sidecars at render time.~~ **Decided (2026-08-10):
  render-time join.** One sidecar describes one artifact. A
  baseline re-run replaces one file and never edits the
  candidate's — the diff-cleanly consequence above requires this.
- ~~Where the sidecar sits in the published repo.~~ **Decided
  (2026-08-10):** the pack's sidecar publishes beside its weight
  file as `<artifact-file>.evals.json`, in the model repo the
  Hugging Face conventions name (#79). Baseline sidecars publish
  under `baselines/` with their upstream file names — they
  describe files this repo does not carry, and each names its
  subject by `sha256`. The card links every sidecar it renders
  from.
