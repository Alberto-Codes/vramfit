# Card number ledger

Status: draft — companion to `README.md` (issue #404).

ADR-0025 binds the rule: a card number without a sidecar entry is a
defect. This ledger maps every number on the card to its source
record on the reference box and to its sidecar destination. The run
root is `~/quantfit-runs/nemotron-30b-a3b/`, and every listed path
is relative to that root. The upload staging area is
`publication-upload/` under the run root.

## Candidate numbers (sidecar-bound)

| Card numbers | Source record | Sidecar destination |
|---|---|---|
| PPL 7.9177 (7.917699 ± 0.054005) | `falsifier-q0-imx/eval-logs/eval-falsifier-q0-imx.log` | candidate sidecar, tier 1 |
| Mean KLD 0.204318, same top 83.13 % (83.127 ± 0.096) | same log | candidate sidecar, tier 2 |
| Tier-3 five tasks with stderr | `eval/tier3/candidate/<task>.json` | candidate sidecar, tier 3 |
| File size 15.76 GiB (16,922,476,480 B) | `publication-upload/pack-upload.log`, `stat`, tier-3 artifact block | candidate sidecar, artifact block |
| SHA-256 `85ed06fa…c062` | `sha256sum` on the staged pack, `falsifier-q0-imx/falsifier-q0-imx.gguf.sha256`, #400 | candidate sidecar, artifact block |
| Toolchain (lm-eval 0.4.12, llama-cpp-python 0.3.34, b10362 lane) | tier-3 JSON toolchain blocks | candidate sidecar, toolchain block |

The f16 reference row (PPL 6.8192, 63,181,504,640 B, 16.007
bits/param) traces to the same eval log's `Mean PPL(base)` line and
the nineteenth data point. The reference is not a shipped artifact
and carries no sidecar.

## Comparator numbers (render-time join, the #65 ruling)

| Card numbers | Source record | Sidecar destination |
|---|---|---|
| PPL 9.0075 (9.007521 ± 0.063939) | `campaign-b10362/eval-logs/eval-arm3-published-IQ2_XXS.log` | baselines sidecar, tier 1 |
| Mean KLD 0.370257, same top 76.09 % (76.086 ± 0.110) | same log | baselines sidecar, tier 2 |
| Tier-3 comparator column | `eval/tier3/iq2xxs/<task>.json` | baselines sidecar, tier 3 |
| 17.54 GiB (18,838,022,112 B), SHA-256 `3d16c415…f1bb` | local `stat` and `sha256sum` on #400, upstream LFS hash | baselines sidecar, artifact block |

## Recipe and serve numbers (not sidecar-bound — trace to shipped artifacts and records)

| Card numbers | Source record | Ships as |
|---|---|---|
| Budget table: 17,179,869,184 / 240,518,169 / 16,939,351,015 / 16,929,873,667 B | `recipe.json` `plan` block, the #284 ruling (228.99 MiB) | recipe, in this repo |
| Margin 16.09 MiB | `publication-upload/pack-upload.log` | pack log line, run log |
| 210-group allocation table, 11/35/118/46 split, nine pins, 0.002 overhead, 11-step trace | recipe `assignments` and `plan` | recipe |
| Serve numbers (16,383 MiB visible, 53/53, 15,774.00, 357.00, 96.00, 190.47, 97.41, 16,157.88 MiB) | `falsifier-q0-imx/` serve logs (#389) | card prose, nineteenth data point |
| `n_seq_max` 8 bound | #284 caveat 3 and ruling, chart #158 Notes | card prose |
| Imatrix SHA-256 `fbd36e4f…aac5`, 55,314,688 B | `sha256sum` 2026-08-22, upstream LFS etag at the pinned revision | linked, never carried (see below) |
| Five publishers' repositories checked, comparator smallest of them | chart #158 Notes: four on #265 (2026-08-15), a fifth on #276 (2026-08-16), ruled the bar on #393 | card prose, comparator section |
| Eight other full-model GGUFs below 15.76 GiB, Hub query 2026-08-22 | issue #415 evidence table (correction, 2026-08-31) | card prose, comparator section and fit16gib section |
| `general.file_type` Q4_0 at 74.3 % of bytes | issue #414 composition table, read from the published file at `0c72c8a` on 2026-08-22, ruled 2026-09-04 (ADR-0012 decision 3 amendment) | card prose, recipe section; the packed file's metadata after re-stamp |

## The upload set (model repo)

Every file stages in `publication-upload/` and uploads under the
name below. `README.md` uploads byte-verbatim from
`publication/nemotron-30b-a3b-fit16gib/README.md` — the published
card and the source must match.

| File | SHA-256 | Bytes |
|---|---|---|
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf` | `85ed06fac2f879ee83f83264f3b7cad9bde4947983976205ea8c2c5d6291c062` | 16,922,476,480 |
| `recipe.json` | `634b788ceb0def60bd837e9be28bd82452b569696a47044317c927d1cdb174b3` | 31,433 |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.runlog.jsonl` | `f3115a4df06cfeb3e005d445b1c7df17397a37d6419ac4c122dd402dbed4b7e4` | 1,971 |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf.evals.json` | `bbaf8e76888f0e521896b91de1f39c359e8bc6f376d8186c4ca1f02179efdaf5` | 2,365 |
| `baselines/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-IQ2_XXS.gguf.evals.json` | `f928e8d0ef05bfe6ea8298e06b2a48f812acfcd6c03efb432f2c18f137245912` | 2,366 |
| `LICENSE` | `d7c8a9e5d1896d0a9588319cc7b1433e64645ad6d9e55632c30b78d8c038c23b` | 2,693 |
| `README.md` | recorded at ship | — |

The staged pack came from `vramfit pack` on `recipe.json` with the
#300 box toolchain on 2026-08-22, and its bytes equal the evaluated
file. `recipe.json` is `falsifier-q0-imx/plan-falsifier-oh002.recipe.json`
byte-for-byte. Its `plan` block records the measuring pod's imatrix
path — the pack passed the box path and logged the recorded
ADR-0020 warning, and the bytes reproduced.

**The importance matrix is linked, never carried** — maintainer
ruling 2026-08-22
([#404 comment](https://github.com/Alberto-Codes/vramfit/issues/404#issuecomment-5382725059)),
superseding the imatrix item of the step-1 artifact-set
confirmation. No license grants rehosting a matrix built on
another's calibration text. The card links
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf` at pinned
revision `f0eec2267ae843d9eb21ea3926ab0046da0a8628` of
`bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`. SHA-256
`fbd36e4fa9be8324062a041ba5cb6247e9f68594168596257a85deb86438aac5`,
55,314,688 B. The pinned URL's LFS etag matches, checked
2026-08-22.

## The dataset repo

`Alberto-Codes/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps`
carries five files plus its card. The card's own hashes table is
the ledger:
[`publication/nemotron-30b-a3b-sensitivity-maps/README.md`](../nemotron-30b-a3b-sensitivity-maps/README.md).
Sources under the run root: `q0-imx-rescan/`, `q0-rescan/`, and
`calibration.txt`.
