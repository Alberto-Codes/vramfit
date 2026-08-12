# ADR-0025: Evaluation results ship as a versioned evals sidecar

- **Status:** Accepted
- **Date:** 2026-08-09
- **Note (2026-08-10):** the open items below are settled, and the
  writer landed (issue #65). Decision 5 is superseded: the writer
  lives at `quantfit.adapters.outbound.evals_sidecar_json`. Schema
  version 1 carries aggregates only, each tier block is optional,
  and the card joins sidecar pairs at render time. The baseline set grew: the #90 i-quant
  comparison put three more baselines on the card, so decision 4
  now requires sidecars for them too. Publication #1 carries five
  sidecars.
- **Note (2026-08-10, issue #99):** the aggregates-only rule left
  the card's derived tier-2 statistics without a record. Ruling: a
  cross-artifact derivation lands as its own analysis artifact,
  never in a sidecar. Decision 4 extends: a derived card number
  traces to its analysis artifact instead. The analysis artifact
  carries the `quantfit_schema` envelope, the method, the input log
  hashes, the results, and the derived per-chunk KLD pairs. The
  pairs let a reader recompute every derived card number without
  the unpublished logs. This narrows the aggregates-only decision's
  archive rule to raw outputs — derived pairs may publish. The
  artifact publishes in the model repo under `analysis/`. Its
  generator sits beside it in the run archive.
  Publication #1 carries one: `analysis/kld564-paired-q3ks.json`,
  the paired candidate-vs-Q3_K_S comparison.
- **Amendment (2026-08-11):** the `quantfit_schema` envelope key
  became `vramfit_schema` with the rename to vramfit (#118, chart
  #114). The sidecar schema version bumped to 2 with it. The
  rename executed in #120. This record keeps every dated word
  (#119). Throughout it, read `quantfit_schema` as
  `vramfit_schema`. Read every sidecar schema version 1 as 2. The
  example block therefore emits `"vramfit_schema": 2`. #121
  re-uploaded the five published sidecars at that key and
  version.

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

  The committed shape changes the sketch as follows. Each tier
  block is nullable, and at least one tier must be present. The
  #90 i-quant baselines carry tiers 1-2 only, and decision 4
  still binds their card rows to a sidecar. Each task row names
  its metric and its harness version (decision 3). GSM8K reports
  `exact_match,strict-match`. HellaSwag and ARC-Challenge report
  `acc_norm`. The run date sits on each measured leaf: the tier-1
  block, the tier-2 window, the tier-3 task. One artifact's
  windows and tasks run on different days — the Q3_K_S windows
  ran ten days apart, and the baseline slice crossed midnight.
  The tier-2 block names its `dataset`, because the windows need
  not share tier 1's text. The toolchain names the llama.cpp
  build for every tier as `llama_cpp_build` (the sketch's
  `llama_cpp`, renamed for precision). Its three harness fields
  (`lm_eval`, `llama_cpp_python`, `lane`) are null without
  tier 3. Every numeric field records what the instrument
  printed, including the percent-unit same-top rates. The sidecar
  exists to end transcription.
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
- Whether the planned `quantfit` HF tag points at the sidecar
  (the [artifact ecosystem's](../explanation/artifact-ecosystem.md)
  conventions-to-settle list). The #79 conventions record adopts
  the tag but names no target.
- ~~Where the sidecar sits in the published repo.~~ **Decided
  (2026-08-10):** the pack's sidecar publishes beside its weight
  file as `<artifact-file>.evals.json`, in the model repo the
  Hugging Face conventions name (#79). Baseline sidecars publish
  under `baselines/` with their upstream file names — they
  describe files this repo does not carry, and each names its
  subject by `sha256`. The card links every sidecar it renders
  from.
