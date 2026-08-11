# Card number ledger

Status: draft — companion to `README.md` (issue #81).

ADR-0025 binds the rule: a card number without a sidecar entry is a
defect. This ledger maps every number on the card to its source record
on the reference box and to its sidecar destination. The run root is
`~/quantfit-runs/nemotron-49b-v1_5/`, and every listed path is relative
to that root.

The #65 ruling (2026-08-10, ADR-0025 amendment) settled the schema:
version 1, aggregates only, render-time baseline join, sidecars for the
i-quant baselines, baseline sidecars under `baselines/`. All five
sidecars exist in `eval/sidecars/`. The #82 dry run verified every
sidecar number against the card tables on 2026-08-10.

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

## Baseline numbers (resolved by the #65 ruling — render-time join)

| Card numbers | Source record | Sidecar destination |
|---|---|---|
| Q3_K_S row: 20.45 GiB, 8.532 ± 0.064, 0.1584, 0.2959, 83.4 % | `eval/{ppl,kl,kl564}-baseline-q3ks.log`, bytes 21,955,339,776 (local `stat`, tier-3 artifact block) | `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q3_K_S.gguf.evals.json`, tiers 1–2 |
| Tier-3 baseline column (0.7827, 0.9242, 0.8379, 0.7861, 0.6604 with stderr) | `eval/tier3/baseline/<task>.json` | same Q3_K_S sidecar, tier 3 |
| IQ3_XS row: 19.47 GiB, 8.554 ± 0.063, 0.1982, 0.3309, 81.7 % | `eval/{ppl,kl,kl564}-baseline-iq3xs.log`, bytes 20,908,008,960 (HF API, #90) | `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ3_XS.gguf.evals.json` |
| IQ3_XXS row: 18.18 GiB, 8.723 ± 0.065, 0.2302, 0.3665, 80.1 % | `eval/{ppl,kl,kl564}-baseline-iq3xxs.log`, bytes 19,519,022,592 (HF API, #90) | `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ3_XXS.gguf.evals.json` |
| UD-IQ3_XXS row: 18.34 GiB, 8.697 ± 0.065, 0.1805, 0.3439, 82.0 % | `eval/{ppl,kl,kl564}-baseline-udiq3xxs.log`, bytes 19,692,431,264 (HF API, #90) | `baselines/Llama-3_3-Nemotron-Super-49B-v1_5-UD-IQ3_XXS.gguf.evals.json` |

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

## Derived tier-2 statistics (analysis artifact, #99)

The #99 ruling (2026-08-10, ADR-0025 dated note): a cross-artifact
derivation lands as its own analysis artifact with the derived
per-chunk KLD pairs. The artifact is
`eval/analysis/kld564-paired-q3ks.json` (SHA-256
`38032b73512326d2708bee1b4cd7463ce09a7deb6f6586b909028350f4dc5f4d`),
generated by `eval/analysis/make-paired-analysis.py` from the two
kl564 logs. It publishes in the model repo as
`analysis/kld564-paired-q3ks.json`. Every statistic below recomputes
from the artifact's per-chunk block alone. The artifact's two
`final_mean_kld` fields (0.28727, 0.29591) link the per-chunk pairs
to the published sidecar aggregates.

| Card claim | Artifact field |
|---|---|
| Better on 369 of 564 chunks (65 %) | `results.candidate_wins` 369, `results.candidate_win_pct` 65.4 |
| Mean gap 0.0086 | `results.mean_gap` 0.00864 |
| 7.8σ paired | `results.paired_t` 7.78 |
| Worst excess +0.05 | `results.worst_excess` +0.0509, `results.worst_excess_chunk` 263 |
| Spike chunks 347/502/137 read 0.126/0.124/0.106 | `results.spike_chunks[].candidate_kld` |

## Derived numbers: trivial or render-time derivations

| Card claim | Computed from | Risk |
|---|---|---|
| Tier-3 Δ and combined σ columns, "largest delta 0.8σ" | tier-3 JSON pairs, computed in #88 | render-time derivation from the two sidecars' score and stderr fields (#65 join ruling) |
| Reconstruction improvement range 2.0–4.3× | derived: console-log RMSE pairs (ratios 2.90 / 3.88 / 2.03 / 4.34) | trivial derivation |
| i-quant sizes in GiB (19.47 / 18.18 / 18.34) | HF API bytes in #90, converted | trivial derivation |

## Upload file list (#82 dry run, 2026-08-10)

Repo: `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF`. Local paths
are relative to the run root.

| Upload path | Local source | SHA-256 |
|---|---|---|
| `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf` | `nemotron-49b-g1c-replication.gguf` (rename) | `48271199ee97d5559caa6bb963162265a9fc35cb5c7ec2b181513f7c4c810122` |
| `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf.evals.json` | `eval/sidecars/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf.evals.json` (regenerated with the final name, #82) | `fdc48ad3210ea23333c51aeea42fb62d3c95baf9c9d395929e46d4cd2a2f8a7c` |
| `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.runlog.jsonl` | `nemotron-49b-g1c-replication.runlog.jsonl` (rename) | `969537bb5319bb97f90cf241eca0385ff960c1b5f38834b622c9288d830f9d5d` |
| `recipe.json` | `recipe-g1c-replication.json` (rename) | `c418c24cf7830815f121b3c64a470392b841dc4dbc5a5dc2ed53cf957a06e5f9` |
| `imatrix.gguf` | `nemotron-49b-f16.imatrix.gguf` (rename) | `eb9b5ffd362b9b1c7a1ae3557804b62c5953756c2db53f7dc5ca6a364ff6d08c` |
| `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q3_K_S.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `25c5cb6de32bc17be3190354dc3f5a5811c7847a314d399aec0d922b44f750d3` |
| `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ3_XS.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `6850ad842785c3277c4d9331cb438747799f64b9df3faa04e840f17169ac0dae` |
| `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ3_XXS.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `395cd40f27b0ed2b40369d32543a4b12e1751d3dfb524c06deabafcf2febdadd` |
| `baselines/Llama-3_3-Nemotron-Super-49B-v1_5-UD-IQ3_XXS.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `23f230076a438a03d611c5c6dbefe6ddddb52e1a48e98e5f1ba8495d396fe755` |
| `analysis/kld564-paired-q3ks.json` | `eval/analysis/`, same file name (#99) | `38032b73512326d2708bee1b4cd7463ce09a7deb6f6586b909028350f4dc5f4d` |
| `README.md` | this card, finalized at upload | — |
| `LICENSE` | NVIDIA Open Model License text — **not on disk, #83 gathers** | — |
| `LICENSE-llama3.3` | Llama 3.3 Community License text — **not on disk, #83 gathers** | — |
| `NOTICE` | NVIDIA notice line (#71) — **not on disk, #83 writes** | — |
| `NOTICE-llama3.3` | Llama 3.3 notice line (#71) — **not on disk, #83 writes** | — |

The i-quant baseline weights are not on disk (deleted, #90). Their
sidecar `artifact.sha256` values match the download receipts in
`eval/iquant-baselines.console.log` and the HF tree API check of
2026-08-10. No re-hash is possible without a re-download — the receipts
are the record.

## Other open flags

- ~~Repo identity: v1 vs v1_5.~~ **Resolved 2026-08-10 (#82): the
  maintainer ruled v1_5. Dated correction on the #79 record.**
- ~~Publication file names.~~ **Resolved 2026-08-10 (#82): the table
  above.**
- ~~`license_link` URL.~~ **Verified 2026-08-10 (#82): HTTP 200, no
  redirect.**
- ~~Sensitivity-map dataset link is a placeholder until #85.~~
  **Resolved 2026-08-10 (#85): the dataset repo
  `Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps`
  exists, private until ship. The card links it. File list and
  SHA-256s live in the dataset card
  (`publication/maps-dataset/README.md`) and on the #85 record.**
- ~~The guardrails section is empty until #86.~~ Resolved 2026-08-10:
  comply-and-disclose stance (#86). The section cites the tables and
  adds no numbers, so it needs no ledger rows.
- ~~Derived tier-2 statistics need the #99 analysis artifact.~~
  **Resolved 2026-08-10 (#99): `eval/analysis/kld564-paired-q3ks.json`,
  recorded by an ADR-0025 dated note. The table above carries its
  upload row.**
