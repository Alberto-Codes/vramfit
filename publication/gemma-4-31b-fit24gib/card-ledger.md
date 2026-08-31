# Card number ledger

Status: draft — companion to `README.md` (issue #446).

ADR-0025 binds the rule: a card number without a sidecar entry is a
defect. This ledger maps every number on the card to its source
record on the reference box and to its shipped destination. The run
root is `~/quantfit-runs/gemma-4-31b/`, and every listed path is
relative to that root. The upload staging area is
`publication-upload/model/` under the run root.

## Candidate numbers (sidecar-bound)

| Card numbers | Source record | Shipped destination |
|---|---|---|
| PPL 37.4552, ratio 1.0681 ± 0.0027 | `validate-inframe.console.log`, kv9 STEP 2 block | candidate sidecar, tier 1 |
| Mean KLD 0.0446 ± 0.0004, same top 92.04 % (92.036 ± 0.090) | same log block | candidate sidecar, tier 2 |
| Tier-3 five tasks with stderr | `eval/tier3/kv9-decoder/<task>.json` | candidate sidecar, tier 3 |
| File size 14.92 GiB (16,015,862,144 B) | `stat`, run log `size_checked` | candidate sidecar, artifact block |
| SHA-256 `2a7bd7a7…b1e` | `sha256sum` 2026-08-29 on the staged pack, `eval/tier3/kv9-decoder/*.sha256` | candidate sidecar, artifact block |
| Toolchain (lm-eval 0.4.12, llama-cpp-python 0.3.34, b10362 lane) | tier-3 JSON `quantfit_tier3` blocks | candidate sidecar, toolchain block |

The BF16 reference row (KLD-pass PPL 35.0668, 61,413,171,264 B =
57.20 GiB) traces to the same log's `Mean PPL(base)` line and the
run log's `gguf_converted` byte count. The reference is not a
shipped artifact and carries no sidecar.

## Comparator numbers (QAT Q4_0 baseline sidecar)

| Card numbers | Source record | Shipped destination |
|---|---|---|
| PPL 38.7227, ratio 1.1043 ± 0.0029 | `validate-inframe.console.log`, QAT STEP 2 block | baselines sidecar, tier 1 |
| Mean KLD 0.0420 ± 0.0003, same top 92.32 % | same log block | baselines sidecar, tier 2 |
| Tier-3 comparator column | `eval/tier3/qat-q4_0/<task>.json` | baselines sidecar, tier 3 |
| 16.44 GiB (17,651,001,568 B), SHA-256 `179cfb99…74b` | local `stat` and `sha256sum`, `eval/tier3/qat-q4_0/*.sha256` | baselines sidecar, artifact block |

## Vision-bound numbers (analysis artifact)

| Card numbers | Source record | Shipped destination |
|---|---|---|
| Content class 0.0045 mean / 0.0239 p95 / 99.2 % (both arms) | `campaign-breakdown.json` | `analysis/vision-campaign-kv9.json`, results |
| All-position means 0.0489 (candidate) and 1.1092 (QAT) over 178 positions | `campaign-metrics.json` | same artifact, results |
| Ratio 10.9x content class to all-position mean | ADR-0030 amendment, recomputed from the two figures above | card prose |
| The 178 per-position KLD pairs | recomputed 2026-08-29 from the campaign force logs with `campaign-breakdown.py`'s `kld()`, aggregates verified against the shipped breakdown | same artifact, `results.pairs` (ADR-0025 derived-number rule) |
| Noise floor 1.07e-4, ratios 8.3x and 42x | [#419 checks comment](https://github.com/Alberto-Codes/vramfit/issues/419) 2026-08-29, `campaign-mm-kv9-self.force.jsonl` | same artifact, method block |
| Q4_K_M-sidecar arm: content 0.0050 mean / 0.0193 p95 / 99.2 %, all-position 0.0483 | `campaign-breakdown.json` `mm-kv9q4km` block, [#451 findings comment](https://github.com/Alberto-Codes/vramfit/issues/451) 2026-08-30 | card prose and table (raw logs stay in the run archive) |
| 10 of 10 held-out answers (reference and candidate) | `campaign-breakdown.json` `own_path` block | same artifact, results |
| Campaign input hashes | `sha256sum` 2026-08-29 on the six campaign logs and `heldout/manifest.json` | same artifact, inputs block |

The metric framing (truncated top-20, position classes, the
ADR-0027 second-instrument caveat, the no-transfer scope) follows
the [ADR-0030 amendment](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0030-vision-budget-sidecar.md)
word for word.

## Recipe and serve numbers (trace to shipped artifacts and records)

| Card numbers | Source record | Ships as |
|---|---|---|
| Budget table: 25,769,803,776 / 9,663,676,416 / 16,106,127,360 / 16,074,691,830 / 16,015,862,144 B | `recipe.json` `plan` block, run log `size_checked` | recipe, run log, in this repo |
| Margin 86.08 MiB (90,265,216 B) | run log `size_checked` `margin_bytes` | run log |
| Allocation 6/9/46 groups at nominal 2/3/4 (46 = 45 layers + the token embedding), trace 81 steps, damage 0.184 | `recipe.json` `assignments` and `plan` | recipe |
| Pack flags: `--pure`, Q2_K base, 60 overrides, Q4_K embedding and output | run log `model_packed` | run log |
| Text ladder 86,016 vs 65,536 (+31.25 %), fail rungs 90,112 / 69,632 | ADR-0030 2026-08-31 amendment, `vline-clean-*.out`, `vline-ext-*.out`, `qat-line-*.out`, `server-vline-*.log` (2026-08-31 frame, desktop 481 MiB) | card prose |
| Image ladder 73,728 vs 49,152 (+50 %), encode fails 77,824 / 53,248 | same amendment and out files, `server-vline-q4km-*.log`, `server-vline-qat-mm-*.log` | card prose |
| Prior-frame pair 81,920 / 61,440 (2026-08-28, BF16 sidecar) and its 81,920 reproduction | [#423 serve-proof comment](https://github.com/Alberto-Codes/vramfit/issues/423#issuecomment-5459430288), `server-kv9-ctx*.log`, `vline-clean-*.out` 81,920 rung (2026-08-31) | card prose |
| Boundary generation check (five decoded tokens, no throughput claim) | `server-gencheck-86016.log`, `server-gencheck-86016.response.json` (2026-08-31) | card prose |
| Device size 24,564 MiB | `server-kv9-ctx80k.log`, #423 serve-proof comment | card prose |
| Vision line 960 MiB (Q4_K_M sidecar, 12,288 displaced tokens), 772 MiB same-context load delta | ADR-0030 2026-08-31 amendment, `vline-clean-*.out` free_at_load deltas | card prose |
| BF16-sidecar line 1,280 MiB in the same frame (1,022.8 MiB weights + 150.63 MiB CLIP reserve + transient), 256 tokens per 768×768 image | ADR-0030 decision 3 and 4 plus the 2026-08-31 amendment, `bf16mm-line-*.out`, #236 checks comments | card prose |
| `-np 1` trap (~2,400 MiB SWA at `-np 4`), ~200 MiB encode headroom | #423 serve-proof comment, chart #441 Notes | card prose |
| Imatrix coverage: no `token_embd`, `attn_v` on 50 of 60 layers | the matrix's own entry names ([#439](https://github.com/Alberto-Codes/vramfit/issues/439) body), run log `imatrix_uncovered` | card prose (matrix not carried) |
| Frame: 357 blocks, 182,404 tokens, 356 chunks, n_ctx 512 | [#423 results comment](https://github.com/Alberto-Codes/vramfit/issues/423#issuecomment-5451599611) | card prose |

#439 (open) records that the pack run log named only `token_embd`
as uncovered while the ten full_attention `attn_v` tensors also
quantized unassisted. The card derives coverage from the matrix's
own entries and reports beside that ticket. This publication does
not absorb it.

## The upload set (model repo)

Every file stages in `publication-upload/model/` and uploads under
the name below. `README.md` uploads byte-verbatim from
`publication/gemma-4-31b-fit24gib/README.md` — the published card
and this source must match.

| File | SHA-256 | Bytes |
|---|---|---|
| `gemma-4-31B-it-fit24gib.gguf` | `2a7bd7a7be6979c858258618ab576db573a7b671b45ee5e9785247341b8c3b1e` | 16,015,862,144 |
| `gemma-4-31B-it-mmproj-q4km.gguf` | `4a03ccaeaaa49cde65a97addac0b2ccd07df4617858aac1472048589ab672033` | 659,537,504 |
| `recipe.json` | `2730692845959b457211c5bd23a4d67acb8744aaa15e5eda8e7f825ed1e3b320` | 29,951 |
| `gemma-4-31B-it-fit24gib.runlog.jsonl` | `8da670782e6ae96ef3cce4a2bc00c0962f91b5ab083a19f11a8c836c0ade5b6a` | 2,036 |
| `gemma-4-31B-it-fit24gib.gguf.evals.json` | `eaefcf7c6b6d40afde6ea275cd7f6b6474525d389036bdbf6a5012c61a9a62d9` | 2,714 |
| `baselines/gemma-4-31B_q4_0-it.gguf.evals.json` | `2d8561c1d9d30b5b99b586dd3b2485c51e8d49a03d58884c1bfad6efc4928f9f` | 2,710 |
| `analysis/vision-campaign-kv9.json` | `2b705017870668ba248eea36ecb837c91d88ba0e78299ba7af9a7ce2ee709b4d` | 55,286 |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | 11,358 |
| `README.md` | recorded at ship | — |

The staged decoder is the ruled kv9 pack (#423 stop-rule verdict,
2026-08-28) byte-for-byte — the staged copy's `sha256sum` matches
the tier-3 lane's stored hash. The pack run (`run_id 6a07c8cda968`,
2026-08-27) predates PR #443, so no `sidecar_shipped` event exists
in the run log. Both weight hashes were computed directly on
2026-08-29 (#446 checks comment). The maintainer ruled on #446 to
publish the ruled bytes rather than re-pack for the event.

**The importance matrix is not carried.** The in-frame matrix
(`gemma-4-31b-bf16-framed.imatrix.gguf`, 13,753,984 B) stays in the
run archive, following the publication-#2 precedent of not carrying
matrices in the model repo. The card describes its coverage from
its own entry names.

The `recipe.json` is `gemma-4-31b-24gib-kv9-decoder.recipe.json`
byte-for-byte. Its paths record the reference box run root. The run
log is `gemma-4-31b-24gib-kv9-decoder.runlog.jsonl` byte-for-byte,
renamed at staging.

## Provenance pins

The first source below lives outside the run root — the path reads
from `~`.

| Claim | Source |
|---|---|
| Checkpoint revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f` | `~/models/gemma-4-31B-it-qat-q4_0-unquantized/.cache/huggingface/download/config.json.metadata` |
| Projector source downloaded from the vendor GGUF repo at revision `59dde24573e7e61570dba08b18a2e1fe246955ed`, then converted to Q4_K_M by llama-quantize b10362 (swap ruling 2026-08-31) | `.cache/huggingface/download/gemma-4-31B-it-mmproj.gguf.metadata` in the run root — the etag equals the BF16 source SHA-256 (`6bd60bdb…07a4`). Conversion record: [#451 findings](https://github.com/Alberto-Codes/vramfit/issues/451), `sha256sum` 2026-08-31 on `gemma-4-31B-it-mmproj-q4km.gguf` |
| QAT comparator from the same repo at the same revision | `.cache/huggingface/download/gemma-4-31B_q4_0-it.gguf.metadata` in the run root — the etag equals `179cfb99…74b` |
| Projector: 190 BF16 + 166 F32 tensors, roots `v.` and `mm.` | ADR-0030 Context (header verification, 2026-08-28) |
| Calibration corpus: Project Gutenberg text under the Gemma channel frame | `calibration.txt`, `calibration-framed.txt` in the run root |
