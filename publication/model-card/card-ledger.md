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

Dated note 2026-08-11 (#121): the sidecar schema now reads version 2.
The #118 ruling renamed the sidecar's envelope key to `vramfit_schema`
and bumped every schema version. The key rename changed no
measurement, so every number on this page stands.

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
kl564 logs. The generator stamps the current date, so only a
2026-08-10 run reproduces that hash (#134). It publishes in the
model repo as
`analysis/kld564-paired-q3ks.json`. The published copy carries the
renamed envelope key and hashes
`90d0b813c6a44c91378adc31e7a501d486ce3bfc1194501a1ae7ce837a864a6f`
(#121). Every statistic below recomputes
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

Task #121 re-hashed eight of these files on 2026-08-11 after the key
rename: the candidate sidecar, the run log, the recipe, the analysis
artifact, and the four baseline sidecars. The weights and the imatrix
did not change.

The SHA-256 column records the published file, not the local source.
Task #121 renamed the envelope key at upload. Every local source in
the run root still carries the pre-rename key and its pre-rename
hash. Re-running a generator on the reference box reproduces the old
key, and no longer the archived hash (#134).

| Upload path | Local source | SHA-256 |
|---|---|---|
| `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf` | `nemotron-49b-g1c-replication.gguf` (rename) | `48271199ee97d5559caa6bb963162265a9fc35cb5c7ec2b181513f7c4c810122` |
| `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf.evals.json` | `eval/sidecars/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf.evals.json` (regenerated with the final name, #82) | `f4c06fe2e24d217df72851069b66713293822ad8aef4ddfa6ff1e206350a670a` |
| `Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.runlog.jsonl` | `nemotron-49b-g1c-replication.runlog.jsonl` (rename) | `00a739ced874a0513824d0559c12073db236476b47a84e2ad8547376064fa511` |
| `recipe.json` | `recipe-g1c-replication.json` (rename) | `75f36c1835c9f768ef03da3160db9a44d95205b8da8f18d966c6f60dcf92b9b6` |
| `imatrix.gguf` | `nemotron-49b-f16.imatrix.gguf` (rename) | `eb9b5ffd362b9b1c7a1ae3557804b62c5953756c2db53f7dc5ca6a364ff6d08c` |
| `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q3_K_S.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `69c9e04380ff628ec28ab9bd79cdf4d24320e245f50b395886d42c50b27ad531` |
| `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ3_XS.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `6c2bf9b65b7c30cf5a1d499e989e28489875ee8772b7e3b6464030a0c4d0a182` |
| `baselines/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ3_XXS.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `6264c64c429608db858fcb0244a2f2987d9fd06fbb027d01351c930f93130f1b` |
| `baselines/Llama-3_3-Nemotron-Super-49B-v1_5-UD-IQ3_XXS.gguf.evals.json` | `eval/sidecars/baselines/`, same file name | `74e47690541518a16d84494fc5e7455b90804d2123d241ac7de05c6f0446b85f` |
| `analysis/kld564-paired-q3ks.json` | `eval/analysis/`, same file name (#99) | `90d0b813c6a44c91378adc31e7a501d486ce3bfc1194501a1ae7ce837a864a6f` |
| `README.md` | `publication/model-card/README.md`, uploaded verbatim | — |
| `LICENSE` | NVIDIA Open Model License text, gathered by #83 | — |
| `LICENSE-llama3.3` | Llama 3.3 Community License text, gathered by #83 | — |
| `NOTICE` | NVIDIA notice line (#71), written by #83 | — |
| `NOTICE-llama3.3` | Llama 3.3 notice line (#71), written by #83 | — |

The i-quant baseline weights are not on disk (deleted, #90). Their
sidecar `artifact.sha256` values match the download receipts in
`eval/iquant-baselines.console.log` and the HF tree API check of
2026-08-10. No re-hash is possible without a re-download — the receipts
are the record.

## The run root is a frozen archive (#134, 2026-08-11)

The maintainer does not re-key the run root. It stays a pre-rename
archive. A reader migrates a copy on load. The files keep their
pre-rename keys and their pre-rename hashes, so the "Local source"
column above stays true.

The schema version blocks the reader, not the key. Every run-root
artifact predates the rename by one to four schema versions.

| Artifact | Run-root version | Current version |
|---|---|---|
| Sensitivity map (9 files) | 1 | 2 |
| Scan checkpoint (5 files) | 1 | 2 |
| Recipe (7 files) | 2 | 6 |
| Recipe `recipe-g1c-replication.json` | 4 | 6 |
| Evals sidecar (5 files) | 1 | 2, writer only |
| Analysis artifact (1 file) | 1 | 2 on the Hub copy |

The last two rows name no reader. `evals_sidecar_json` writes version
2 and parses nothing. The analysis artifact has no vramfit constant.

A key rename alone repairs nothing. The rejection moves from the key
to the version. Two loads on 2026-08-11 measured it:

- A re-keyed `recipe-g1c-replication.json` fails with
  `unsupported schema version 4 — this vramfit reads version 6`.
- A re-keyed schema-1 map fails with
  `unsupported schema version 1 — this vramfit reads version 2`.

A re-key therefore buys no re-plan. It also destroys the pre-rename
hashes that tie each published file to its source.

### How a reader migrates a copy

Migrate a copy. Never edit the archive. The steps live on the format
pages. This section records what #134 measured on 2026-08-11.

- **Sensitivity maps and scan checkpoints.** Migrate per
  [the map format page](../../docs/reference/sensitivity-map.md). A
  #134 load check read all nine maps and all five checkpoints: 82
  groups and 328 measurements each.
- **`recipe-g1c-replication.json`.** Take the published copy. The Hub
  serves a migrated copy as `recipe.json` at
  `75f36c1835c9f768ef03da3160db9a44d95205b8da8f18d966c6f60dcf92b9b6`.
  It matches the local source at every top-level field except the
  envelope. To migrate the local file instead, follow
  [the recipe format page](../../docs/reference/recipe.md). A #134
  load check read that copy. The recipe protects 48 tensors at 5
  bits inside 3-bit groups, so it carries no no-op pair (issue #59).
- **The seven schema-2 recipes.** A reader cannot migrate these. They
  predate `plan.protections`
  ([ADR-0022](../../docs/adr/0022-within-layer-protections.md)). A
  version stamp would claim a field the run never recorded. Re-plan
  instead.
- **Sidecars, run logs, and the analysis artifact.** These need no
  migration. `evals_sidecar_json` is a write-only adapter.
  `read_run_log` parses each line with `json.loads` and checks no
  envelope. The 23 run logs carry `quantfit_runlog`, a different key.
  No path reaches the envelope check.

A migrated checkpoint faces a second gate. The reader compares
`fingerprint` against the running scan. That fingerprint stores the
model, calibration, and imatrix paths as the invocation spelled them,
and ADR-0020 resolves the imatrix path to absolute. A resume must
therefore reproduce the original command line. The five archived
checkpoints already disagree: two spell the calibration path
`/home/…` and three spell it `/var/home/…`. No single command line
resumes all five.

This ruling freezes `eval/analysis/make-paired-analysis.py` too. It
still writes `"quantfit_schema": 1`. It also stamps
`datetime.date.today()`, so it reproduced `38032b73…` only on
2026-08-10. A #134 re-run on 2026-08-11 produced `4f781e4a…`, which
differs from the archived file in the `date` field alone. The
published `90d0b813…` stays the migrated copy.

This ruling leaves `_reject_renamed_envelope_key` in
`vramfit.adapters.outbound.json_common` a live subject. The run root
keeps 28 artifacts that carry `quantfit_schema`, plus the generator
script that writes it. A reader that opens one of the 28 reports the
rename and #118, not a bare missing-field error. #154 tier 2 rules
on the guard itself against that fact.

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
