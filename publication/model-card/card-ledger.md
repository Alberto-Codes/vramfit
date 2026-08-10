# Card number ledger

Status: draft — companion to `README.md` (issue #81).

ADR-0025 binds the rule: a card number without a sidecar entry is a
defect. This ledger maps every number on the card to its source record
on the reference box and to its sidecar destination. The run root is
`~/quantfit-runs/nemotron-49b-v1_5/`, and every listed path is relative
to that root. Issue #65 settles the sidecar schema. Its ruling decides every row marked **pending #65**.

ADR-0025's consequences already commit the Q3_K_S baseline to a sidecar
of its own (ADR-0024 decision 5: measured once, reused). For its rows,
only the mechanics are open: the candidate sidecar embeds the comparison,
or the card joins two sidecars at render time. The three i-quant rows are
the fully open case — those artifacts were measured in #90 and deleted,
and no ADR commits them to a sidecar.

## Candidate numbers (sidecar-bound, source records exist)

| Card numbers | Source record | Sidecar destination |
|---|---|---|
| PPL 8.517 ± 0.063 | `eval/ppl-g1c-replication.log` | candidate sidecar, tier 1 |
| KLD 100 0.1538 | `eval/kl-g1c-replication.log` | candidate sidecar, tier 2 |
| KLD 564 0.2873, same top 82.9 % | `eval/kl564-g1c-replication.log` | candidate sidecar, tier 2 |
| MMLU 0.7829 ± 0.0033 | `eval/tier3/candidate/mmlu.json` | candidate sidecar, tier 3 |
| GSM8K 0.9318 ± 0.0069 | `eval/tier3/candidate/gsm8k.json` | candidate sidecar, tier 3 |
| HellaSwag 0.8412 ± 0.0036 | `eval/tier3/candidate/hellaswag.json` | candidate sidecar, tier 3 |
| Winogrande 0.7845 ± 0.0116 | `eval/tier3/candidate/winogrande.json` | candidate sidecar, tier 3 |
| ARC-Challenge 0.6493 ± 0.0139 | `eval/tier3/candidate/arc_challenge.json` | candidate sidecar, tier 3 |
| File size 20.36 GiB (21,860,214,272 B) | runlog `model_packed` event, `stat`, tier-3 artifact block | candidate sidecar, artifact block |
| SHA-256 `48271199…0122` | `eval/tier3/candidate/*.json` artifact block, `eval/tier3/candidate/nemotron-49b-g1c-replication.gguf.sha256` | candidate sidecar, artifact block |
| Toolchain versions (lm-eval 0.4.12, llama-cpp-python 0.3.34, b10172) | tier-3 JSON toolchain blocks | candidate sidecar, toolchain block |

## Baseline numbers (pending #65)

| Card numbers | Source record | Sidecar destination |
|---|---|---|
| Q3_K_S row: 20.45 GiB, 8.532 ± 0.064, 0.1584, 0.2959, 83.4 % | `eval/{ppl,kl,kl564}-baseline-q3ks.log`, bytes 21,955,339,776 (local `stat`, tier-3 artifact block) | baseline sidecar (ADR-0025) — embed vs join **pending #65** |
| Tier-3 baseline column (0.7827, 0.9242, 0.8379, 0.7861, 0.6604 with stderr) | `eval/tier3/baseline/<task>.json` | baseline sidecar (ADR-0025) — embed vs join **pending #65** |
| IQ3_XS row: 19.47 GiB, 8.554 ± 0.063, 0.1982, 0.3309, 81.7 % | `eval/{ppl,kl,kl564}-baseline-iq3xs.log`, bytes 20,908,008,960 (HF API, #90) | **pending #65** — no ADR commits i-quants to a sidecar |
| IQ3_XXS row: 18.18 GiB, 8.723 ± 0.065, 0.2302, 0.3665, 80.1 % | `eval/{ppl,kl,kl564}-baseline-iq3xxs.log`, bytes 19,519,022,592 (HF API, #90) | **pending #65** — no ADR commits i-quants to a sidecar |
| UD-IQ3_XXS row: 18.34 GiB, 8.697 ± 0.065, 0.1805, 0.3439, 82.0 % | `eval/{ppl,kl,kl564}-baseline-udiq3xxs.log`, bytes 19,692,431,264 (HF API, #90) | **pending #65** — no ADR commits i-quants to a sidecar |

Baseline provenance: Q3_K_S is bartowski's published community GGUF
(the fourth data point's size match). The i-quants are
`bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-GGUF` and
`unsloth/Llama-3_3-Nemotron-Super-49B-v1_5-GGUF`, evaluated on the
recorded instrument in #90. The i-quant artifacts were deleted after
measurement (disk) — logs and HF file hashes are the records.

## Recipe and run-log numbers (not sidecar-bound — trace to shipped artifacts)

| Card numbers | Source record | Ships as |
|---|---|---|
| Budget table: 25,769,803,776 / 3,791,650,816 / 21,978,152,960 / 21,957,337,301 B, overhead 0.005 | `recipe-g1c-replication.json` `plan` block | recipe, in this repo |
| Predicted damage 0.3905, 82-group damage table, layout counts (1× Q8_0, 81× Q3_K, 47× Q5_K floors, 1× Q4_K floor) | recipe `assignments` + `protected_tensors` | recipe |
| Solver trace: 162 steps, step 162 = `model.layers.3` 4→3, 113,087,938 B freed, damage 0.1129 | recipe `plan.trace` | recipe |
| 29 % share of step 162 | derived: 0.1129 / 0.3905 | derived from recipe |
| Margin 112.48 MiB under budget | `pack-g1c-replication.console.log`, runlog | run log, in this repo |
| Reconstruction RMSE pairs (0.001641/0.004755, 0.000574/0.002229, 0.001136/0.002303, 0.000501/0.002178), 48/48 green | runlog `reconstruction_checked`, pack console log | run log |
| 128 type overrides, base type Q3_K_S, token_embd uncovered | runlog `model_packed` event | run log |

## Flagged: numbers with no single source record

These are derived statistics. No artifact on disk records them — they
were computed from per-chunk KLD logs during the fifteenth data point
and from tier-3 JSON during the sixteenth. Options: the #65 schema
carries per-chunk / per-item detail, the sidecar stores derived
comparison stats, or a recorded analysis artifact lands before upload.

| Card claim | Computed from | Risk |
|---|---|---|
| 369 of 564 chunks (65 %), mean gap 0.0086, 7.8σ paired | per-chunk KLD in `eval/kl564-g1c-replication.log` vs `eval/kl564-baseline-q3ks.log` | **no recorded artifact** |
| Spike-free: worst excess +0.05; chunks 347/502/137 = 0.126/0.124/0.106 | same per-chunk logs | **no recorded artifact** |
| Tier-3 Δ and combined σ columns, "largest delta 0.8σ" | tier-3 JSON pairs, computed in #88 | derived — decide store vs render |
| Reconstruction improvement range 2.0–4.3× | derived: console-log RMSE pairs (ratios 2.90 / 3.88 / 2.03 / 4.34) | trivial derivation |
| i-quant sizes in GiB (19.47 / 18.18 / 18.34) | HF API bytes in #90, converted | trivial derivation |

Independent recompute (2026-08-10, PR #96 review cycle): every derived
tier-2 value reproduces from the per-chunk logs — 369 of 564 (65.4 %),
mean gap 0.00864, paired t 7.78σ, worst excess +0.0509 (chunk 263). The
recompute is not a recorded artifact. The flag stands.

## Other open flags

- Repo identity: #79 record says `…-49B-v1-…` and `base_model …-v1`.
  All artifacts derive from v1_5. The card uses v1_5. Maintainer call.
- Publication file names (`recipe.json`, `imatrix.gguf`, sidecar and
  run-log names) are placeholders until the #82 dry run.
- `license_link` URL copied from community convention — verify at #82.
- Sensitivity-map dataset link is a placeholder until #85.
- ~~The guardrails section is empty until #86.~~ Resolved 2026-08-10:
  comply-and-disclose stance (#86). The section cites the tables and
  adds no numbers, so it needs no ledger rows.
