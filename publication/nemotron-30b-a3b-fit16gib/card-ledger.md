# Card number ledger

Status: draft — companion to `README.md` (issue #404).

ADR-0025 binds the rule: a card number without a sidecar entry is a
defect. This ledger maps every number on the card to its source
record and to its sidecar destination.

**This revision's run root is not the reference box.** Every tier-1
and tier-2 number on the card was measured on one rented H100 pod on
2026-09-17, and the artifacts were uploaded from that same pod
before it was deleted. The records live under
`data/vramfit-30b-v2-publish/artifacts/` in the operator's run
archive, and the paths below are relative to that directory.

The serve rows were measured on **two** RTX 4090s against the file
downloaded back from the Hub and hashed first: the CUDA row on a
separately rented card, and the Vulkan row on the maintainer's own
card under a one-time authorisation, because no rented pod available
to this run could initialise a Vulkan driver. Nothing was resident on
that card when the measurement began and nothing was evicted.

That ordering is the point of the revision. The previous revision's
card cited figures measured on a file that was not the uploaded bytes
(issue #598). Measuring and uploading in one sequence removes that
class of defect rather than patching the instance.

## Candidate numbers (sidecar-bound)

| Card numbers | Source record | Sidecar destination |
|---|---|---|
| PPL 7.0689 (7.068906 ± 0.047531) | `results.txt`, block `v2-puballoc-q0imx2`; `logs/eval-v2-puballoc-q0imx2.log` | candidate sidecar, tier 1 |
| Mean KLD 0.071403, same top 89.418 % | same block and log | candidate sidecar, tier 2 |
| PPL ratio 1.036624 | same block, `Mean PPL(Q)/PPL(base)` | card prose (derived, printed by the instrument) |
| File size 16,922,476,480 B, 15.76 GiB | `logs/pack-v2.log`, `stat` on the packed file, `SHA256SUMS` | candidate sidecar, artifact block |
| SHA-256 `187858b0…cd75` | `SHA256SUMS`, re-read from the Hub in `logs/upload-verify.txt` | candidate sidecar, artifact block |
| Corpus digest `173c87a5…dd08`, 1,290,590 B | `sha256sum` on the staged corpus, pod and box agreeing | candidate sidecar, `corpora` |
| Toolchain: llama.cpp b10362 `4801e3c56`, CUDA | `logs/versions.txt`, `logs/toolchain-sha256.txt` | candidate sidecar, toolchain block |

Tier 3 has no row because **tier 3 did not run for this revision**.
The card's Evaluation section states that, with the cost that kept it
out. No tier-3 number is carried over from the previous revision.

The f16 reference row (PPL 6.8192, 63,181,504,640 B, 16.007
bits/param) traces to the same eval logs' `Mean PPL(base)` line. The
reference is not a shipped artifact and carries no sidecar.

## Comparator numbers (render-time join, the #65 ruling)

Re-measured in this run's own sequence rather than joined from an
older lane, so the card's three rows share one instrument and one set
of reference logits.

| Card numbers | Source record | Sidecar destination |
|---|---|---|
| PPL 9.0075, ratio 1.320914 | `results.txt`, block `ctl-iq2xxs` | baselines sidecar, tier 1 |
| Mean KLD 0.370257, same top 76.086 % | same block | baselines sidecar, tier 2 |
| 17.54 GiB (18,838,022,112 B), SHA-256 `3d16c415…f1bb` | `results.txt` block, `sha256sum` on the downloaded file | baselines sidecar, artifact block |

The comparator's previously published tier-3 column is not reproduced
on this card, because this card publishes no tier-3 table.

## The superseded revision's row

| Card numbers | Source record | Ships as |
|---|---|---|
| PPL 7.9174, ratio 1.161055, mean KLD 0.204322, same top 83.146 % | `results.txt`, block `ctl-v1-published`; the file was downloaded from this repository and hashed before evaluation | card prose, the "revision this replaces" row |

This row is also the run's **control**. It re-measures the exact bytes
this repository published previously, on the new instrument, and it
reproduced the recorded 0.204322 to every printed digit. A
measurement is not readable until its control reproduces the
published frame, so this row is what licenses the other two.

## Recipe and serve numbers (not sidecar-bound)

| Card numbers | Source record | Ships as |
|---|---|---|
| Budget table: 17,179,869,184 / 240,518,169 / 16,939,351,015 B | `recipe.json` `plan` block, the #284 ruling (228.99 MiB) | recipe, in this repo |
| Margin 16.09 MiB under the weight budget | `logs/pack-v2.log`, run log `size_checked` event | pack log, run log |
| No predicted size printed | the recipe's `plan` block is hand-derived and its `solver` field says so; the card states the bound rather than printing a stale prediction | card prose, fit16gib section |
| 210-group allocation table, 11/35/118/46 split, nine pins, 0.002 overhead | recipe `assignments` | recipe |
| Pre-encoder cost: whole pack stage 8 m 35 s, about 50 GB scratch | `logs/pack-v2.log`, the pack run log's `model_packed` event (`seconds` 515.183) | card prose, recipe section |
| Encoder peak resident set 16.22 GB, NOT re-measured here | the separately instrumented pack under issue #607 | card prose, labelled on the card as not re-measured |
| Serve rows: Vulkan and CUDA rows: 15,774.00 / 15,774.05 MiB model buffer, 16,002.99 / 16,002 MiB device buffers, Vulkan fits under a 16,380 MiB cap and CUDA does not | `serve-vulkan-local/` (Vulkan, prebuilt b10362 ubuntu-vulkan-x64, the captain's own RTX 4090 under his explicit one-time authorisation) and `serve-4090/` (CUDA, rented RTX 4090); both served the file downloaded back from the Hub after a digest check | card prose, fit16gib section |
| `n_seq_max` 8 bound | #284 caveat 3 and ruling, chart #158 Notes | card prose |
| Imatrix SHA-256 `fbd36e4f…aac5`, 55,314,688 B | `sha256sum` on the staged matrix, matching the pinned revision's recorded value | linked, never carried (see below) |
| Five publishers' repositories checked, comparator smallest of them | chart #158 Notes, ruled the bar on #393 | card prose, comparator section |
| Eight other full-model GGUFs below 15.76 GiB, Hub query 2026-08-22 | issue #415 evidence table | card prose |
| `general.file_type` Q4_0 at 74.3 % of bytes | issue #414 composition table, ruled 2026-09-04 (ADR-0012 decision 3 amendment); `logs/pack-v2.log` `file_type` field | card prose, recipe section |
| Usage section reasoning and slot figures | issue #410 body and comments, measured 2026-08-22 on b10573 against the PREVIOUS revision's bytes | card prose, Usage section, labelled as such on the card |

## The upload set (model repo)

Every file staged in `upload/` on the pod and uploaded under the name
below, in one `hf upload` of that directory. `README.md` uploads
byte-verbatim from
`publication/nemotron-30b-a3b-fit16gib/README.md` through
`scripts/publish_card.py`, which reads the published copy back and
compares — the published card and this source must match.

| File | SHA-256 | Bytes |
|---|---|---|
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf` | `187858b04dccae82a8c6fbf8bc5f0a62cfedb21d2f5aef3b4589456b09b6cd75` | 16,922,476,480 |
| `recipe.json` | `7fa0d6b028a1ef3642fd4e478a922af9fb4460d47f7bb21bdc8a00ec3c464a1c` | 28,039 |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.runlog.jsonl` | `17d6d6c61a4d6b32a9fbd982675a98fc6114c87b2845a14d0ae14d6038623faa` | 2,444 |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf.evals.json` | `537799b59a7b1c6f4b7d02bbc37cca9eed8f13374354a5f60c0789fa7b778478` | 1,146 |
| `baselines/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-IQ2_XXS.gguf.evals.json` | `8bbb31ba73b0149c20b0f5154e5107b5ded5ebd9c2cdd43d23170ee5a5195072` | 1,144 |
| `README.md` | `e5a170efd873a0a0aba1094f4707bb3dd549db7bfa13689bea90f920ee673dcf` | 41,329 |

`LICENSE` is unchanged from the previous revision and was not
re-uploaded.

**The published digests were read back from the Hub**, not assumed
from the upload call: `logs/upload-verify.txt` records the repository
revision and every file's LFS SHA-256 as the Hub reports it, and the
packed file's digest there equals the digest hashed on the pod before
the upload. The card's own row was checked the same way, by fetching
the published `README.md` back and hashing it, so every digest in this
table is one the Hub returned rather than one this file asserts.

**The importance matrix is linked, never carried** — maintainer
ruling 2026-08-22
([#404 comment](https://github.com/Alberto-Codes/vramfit/issues/404#issuecomment-5382725059)).
No license grants rehosting a matrix built on another's calibration
text. The card links
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf` at pinned
revision `f0eec2267ae843d9eb21ea3926ab0046da0a8628` of
`bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`. SHA-256
`fbd36e4fa9be8324062a041ba5cb6247e9f68594168596257a85deb86438aac5`,
55,314,688 B, re-verified on the pod in this run.

## The dataset repo

`Alberto-Codes/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps`
carries the published maps plus its card. The card's own hashes table
is the ledger:
[`publication/nemotron-30b-a3b-sensitivity-maps/README.md`](../nemotron-30b-a3b-sensitivity-maps/README.md).
The `q0-imx2` map this revision's allocation was checked against is
not yet published there; issue #558's schema ruling gates that upload.
