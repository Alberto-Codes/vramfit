---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-31B-it-qat-q4_0-unquantized
base_model_relation: quantized
quantized_by: Alberto-Codes
pipeline_tag: image-text-to-text
tags:
  - vramfit
  - gguf
  - imatrix
---

<!--
Authored for issue #446 under the #401 identity grammar. Upload this
file verbatim — the published card and this source must match. The
number ledger lives beside this file in the vramfit repository:
publication/gemma-4-31b-fit24gib/card-ledger.md.
-->

# gemma-4-31B-it-fit24gib-GGUF

This repository carries a mixed-precision GGUF of
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it),
packed from the
[QAT unquantized checkpoint](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
and solved to serve images inside a 24 GiB card.
[vramfit](https://github.com/Alberto-Codes/vramfit) measured each
decoder layer's quantization damage — the shift in the model's
output distribution when that layer quantizes — then allocated bits
under the budget and packed through llama.cpp's quantizer. The
`fit24gib` marker is a deployment claim, not a file size: a serve
test ran these exact files inside a 24 GiB VRAM boundary under the
configuration stated below. This pack has no single quantization
scheme. The budget is the claim.

**Two files, one artifact.** The decoder
(`gemma-4-31B-it-fit24gib.gguf`, 14.92 GiB) carries the quantized
language model. The projector sidecar
(`gemma-4-31B-it-mmproj-q4km.gguf`, 629 MiB) carries the vision
tower, converted from the vendor projector with llama-quantize's
Q4_K_M recipe. The recipe name labels the command, not the
contents. Every quantizable tensor falls back on this geometry.
The file holds 150 Q5_0, 13 Q8_0, 27 F16, and 166 F32 tensors at
9.16 effective bits per parameter, read from its own header.
Text-only serving needs the decoder alone. Image serving needs
both files. The sidecar shipped in BF16 until 2026-08-31, when a
measurement priced the trade. The converted sidecar matches the
BF16 sidecar on content-class KLD (0.0050 against 0.0045, 99.2 %
top-token agreement both). It buys 4,096 tokens of context at the
2026-08-31 serve boundary. The record is ADR-0030 open question 2
and its 2026-08-31 amendment.

## The headline

The official [QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf)
of this model is 16.44 GiB. This pack is 14.92 GiB at comparable
measured text quality — four benchmark ties and one win, tables
below — and lower measured vision-conditioned divergence. The freed
weights buy context on the same card:

| Serving shape | This pack | QAT Q4_0 | Gain |
|---|---|---|---|
| Text-only, measured max load | 86,016 tokens | 65,536 tokens | +20,480 (+31.25 %) |
| One image aboard, measured max load | 73,728 tokens | 49,152 tokens | +24,576 (+50 %) |

Both rows are served results on an RTX 4090 (24 GiB), llama.cpp
b10362, `-ngl 99 -np 1`, measured 2026-08-31 at 4,096-token ladder
rungs in one frame — one day, one idle VRAM share, both packs. The
QAT image row serves the vendor's own BF16 projector, so that row
moves two variables: the decoder and the sidecar. The same-sidecar
comparison holds one variable. Behind the same BF16 projector this
pack serves one image at 69,632 against QAT's 49,152, a gain of
20,480 tokens (+41.7 %). The sidecar conversion adds the last
4,096. Every measured number below sits beside its baseline
counterpart. The card prints the losing numbers too.

## Quality beside size — text

Measured on the framed calibration corpus: 357 blocks of public-
domain text under the Gemma channel frame, 182,404 tokens, 356
chunks at n_ctx 512. This is the pack's measurement frame — the
same corpus the importance matrix consumed — not held-out text. The
held-out check is the benchmark table below. The instrument is a
CPU-only llama-perplexity built from the b10362 source with
`parse_special` enabled, and it measured both sides of every ratio.
All three rows share the KLD-pass perplexity convention. Lower is
better for PPL and KLD. "Same top" is the share of tokens where the
pack and the BF16 reference agree on the top token.

| Model | File size | PPL ↓ | PPL / bf16 ↓ | Mean KLD ↓ | Same top ↑ |
|---|---|---|---|---|---|
| BF16 reference | 57.20 GiB | 35.0668 | — | — | — |
| **This pack** | **14.92 GiB** | 37.4552 | **1.0681 ± 0.0027** | 0.0446 ± 0.0004 | 92.04 % |
| QAT Q4_0 | 16.44 GiB | 38.7227 | 1.1043 ± 0.0029 | **0.0420 ± 0.0003** | 92.32 % |

Read the split honestly: this pack holds the better PPL ratio, and
the QAT baseline holds a slightly better mean KLD (0.0420 against
0.0446) and top-token agreement (92.32 % against 92.04 %). One
asymmetry runs in this pack's favor: its importance matrix consumed
this same corpus, and the QAT baseline consumed no matrix from this
frame, so tier 1 and tier 2 lean toward this pack. The two text
metrics disagree at the margin, and the held-out benchmarks
arbitrate.

## Held-out benchmarks

lm-evaluation-harness 0.4.12 over llama-cpp-python 0.3.34 with the
b10362 Vulkan libraries, full splits, no sampling limit. A delta
inside the combined standard error is a tie. The Δ column computes
from the sidecars' unrounded scores.

| Task (shots, metric) | This pack | QAT Q4_0 | Δ | Combined σ | Verdict |
|---|---|---|---|---|---|
| MMLU (5, acc) | **71.36 ± 0.37** | 70.20 ± 0.38 | +1.15 | 0.53 | **win** |
| GSM8K (5, strict exact match) | 92.34 ± 0.73 | 92.42 ± 0.73 | −0.08 | 1.03 | tie |
| HellaSwag (10, acc_norm) | 58.71 ± 0.49 | 59.34 ± 0.49 | −0.63 | 0.69 | tie |
| Winogrande (5, acc) | 68.27 ± 1.31 | 68.03 ± 1.31 | +0.24 | 1.85 | tie |
| ARC-Challenge (25, acc_norm) | 61.77 ± 1.42 | 61.09 ± 1.42 | +0.68 | 2.01 | tie |

Four ties and one win, at 1.5 GiB fewer packed bytes. The two
nominal deficits — HellaSwag −0.63 and GSM8K −0.08 — sit inside
their combined standard errors. Per-task records ship in the evals
sidecars beside the weights.

## The measured vision bound

This card states a vision-quality bound because a campaign measured
one for this exact artifact. The sensitivity map that solved this
recipe measured text damage only — a text-measured map licenses no
vision-quality claim — so the bound below comes from its own
image-conditioned measurement, not from the map.

**The metric is a truncated top-20 KLD in nats at the serve
boundary.** The llama.cpp server caps `n_probs` at 20, so this is
not a full-vocabulary KLD. A BF16 reference decoder generated
greedily over 10 held-out 768×768 images, and the harness
teacher-forced each quantized arm on the reference sequence. The
campaign's three decoder arms served through the same BF16
projector. A fourth arm measured the shipped pair — this decoder
behind the Q4_K_M sidecar — with the projector as the only
variable. Position classes
separate image-grounded `content` tokens from channel-frame policy
tokens (`markup`, `pos0`). Frame-policy positions dominate
all-position means on this channel-locked model, so the content
class carries the vision claim.

Content-class results (n = 120 positions):

| Arm | Mean KLD ↓ | p95 KLD ↓ | Top-token agreement ↑ |
|---|---|---|---|
| **This pack as shipped (Q4_K_M sidecar)** | **0.0050** | **0.0193** | **99.2 %** |
| This pack behind the BF16 sidecar | 0.0045 | 0.0239 | 99.2 % |
| QAT Q4_0 behind the BF16 sidecar | 0.0373 | 0.1928 | 97.5 % |

The shipped arm measures 0.0050 against the BF16-sidecar arm's
0.0045 at the content mean, and 0.0193 against 0.0239 at p95. The
all-position means sit at 0.0483 (shipped) and 0.0489 (BF16
sidecar) — 9.7 and 10.9 times their content-class figures, with
frame-policy positions dominating both. The QAT baseline's
all-position mean over the same 178 positions is 1.1092. The
instrument noise floor is 1.07e-4 mean KLD, measured by
teacher-forcing the BF16-sidecar arm on its own greedy sequence.
The shipped arm measures 7.5 times below the QAT baseline and 47
times above the floor. The BF16-sidecar arm measures 8.3 and 42.
The BF16 reference and the BF16-sidecar arm each answered 10 of 10
held-out image questions correctly on their own generation paths.

Two caveats travel with these numbers:

- The BF16 reference ran on CPU — a second instrument. The KLD
  figures read as divergence from the reference distribution, never
  as same-instrument damage numbers. Only the pack-against-QAT
  comparison is same-instrument.
- The bound belongs to this artifact and this metric. A different
  recipe on the same checkpoint measures its own bound or states
  none. No measurement covers transfer between artifacts.

The full campaign record — per-image tables, position-class
breakdown, the 178 per-position KLD pairs, input log hashes —
ships in `analysis/vision-campaign-kv9.json`. The pairs recompute
every derived number for the BF16-sidecar arms. The shipped arm's
record lives in the vramfit tracker
([#451](https://github.com/Alberto-Codes/vramfit/issues/451)) and
in ADR-0030 open question 2, with raw logs in the run archive.

## The real-GUI campaign

The vision bound above rests on 10 held-out 768×768 images. A
second campaign measured three arms on real GUI screenshots at
1280×720. The dataset is PSAI Computer Use Data
([anaisleila/computer-use-data-psai](https://huggingface.co/datasets/anaisleila/computer-use-data-psai),
MIT): the 1,349 tasks that carry both screenshots and interaction
events. The set is 1,318 `EASY` tasks and 1,346 browser tasks out
of 1,349. The
`MEDIUM` and `HARD` rows exist in the artifact at 15 and 16 tasks,
too few to read, and do not print here.

One H100 SXM 80 GB ran llama.cpp b10362 CUDA at `-ngl 99`, context
8,192, `-np 1`, greedy, with a 48-token generation cap. All three
arms (the BF16 reference, this pack, and the QAT Q4_0 baseline)
served through the same BF16 projector on that one instrument. The
vision bound above could not claim that: its reference ran on CPU.

**Divergence.** The metric is the vision bound's: a teacher-forced
truncated top-20 KLD in nats at the serve boundary, 27,962
positions, with the same position classes. The content class
carries the claim.

| Arm | Class | Positions | Mean KLD ↓ | Median ↓ | p95 ↓ | Top-token agreement ↑ |
|---|---|---|---|---|---|---|
| **This pack** | content | 21,216 | **0.0973** | **0.00096** | **0.485** | **93.4 %** |
| QAT Q4_0 | content | 21,216 | 0.1143 | 0.00342 | 0.556 | 92.8 % |
| **This pack** | all positions | 27,962 | **0.0852** | **0.0042** | **0.384** | **95.0 %** |
| QAT Q4_0 | all positions | 27,962 | 0.3824 | 0.0125 | 1.953 | 89.7 % |

Read the two scales apart. On content positions the QAT baseline's
mean is 1.18 times this pack's, and its median 3.6 times, from the
artifact's unrounded values. This campaign measured no instrument
noise floor. The prompt-prefix difference disclosed below rides
inside the mean margin. Read the content-class gap as a bound, not
as a separation. The all-position gap of 4.5 times is a
frame-policy number. The QAT baseline diverges on the channel-frame
positions (markup mean 1.13, `pos0` mean 1.61), where this pack
measures 0.03 and 0.11. The vision bound's 0.0045 content mean
comes from another frame and another instrument. The two figures
do not compare.

**Task identification.** A judge scored each arm's one-sentence
answer against the task's ground-truth name. A `MATCH` needs the
same application or site and the same user goal. Over 1,349 tasks
the BF16 reference scored 49.9 %, this pack 51.2 %, and the QAT
baseline 51.4 %. The QAT scores come from a 512-token
regeneration. All 1,349 of its 48-token generations ended with no
answer. No arm separates. The judge's noise floor is 6.9 %. On 598
tasks the reference and this pack wrote byte-identical answers.
The judge returned different verdicts on 41 of them. Every gap
between the three scores sits inside that floor. The ceiling is the
frame — one middle screenshot against a full-task name left the
BF16 reference itself at 49.9 %. The content-class divergence gap
does not surface as an accuracy gap in this frame.

**Throughput.** Per-request server timings, means over generations,
one slot, both quantized arms behind the BF16 projector:

| Instrument | This pack | QAT Q4_0 | BF16 reference |
|---|---|---|---|
| H100 SXM 80 GB, CUDA, 1,349 generations per arm | 66.1 tok/s | 75.9 tok/s | 39.4 tok/s |
| RTX 4090, Vulkan, 20 generations per arm | **47.8 tok/s** | 43.3 tok/s | — |

The H100 timings are incidental, not a controlled timing arm, and
the H100 is not the target card. The 4090 arms ran the same
harness, prompt, generation cap, and 20-task subset. That card is
this pack's target: b10362 Vulkan, `-ngl 99 -np 1`, context 8,192,
under desktop sharing. No BF16 reference arm ran on the 4090. The
gap inverts. This pack decodes 10 % faster than the baseline on the
4090, where the H100 CUDA build had the baseline 15 % ahead. The
spread over 20 generations is 0.5 tok/s standard deviation (pack)
and 0.2 tok/s (QAT). The shipped sidecar measures 47.7 tok/s,
inside that spread. One asymmetry runs against this pack. Its
generations stop at a mean 20 tokens, and the QAT generations hit
the 48-token cap. The QAT arm therefore amortizes per-request
overhead over more tokens.

Five disclosures ride every number in this section:

- Screenshot policy. One middle screenshot per task, 1280×720. One
  such screenshot is 271 decoder tokens on this pack, measured at
  the server. The vision bound's images are 768×768.
- Reference origin. `convert_hf_to_gguf.py` produced the BF16
  reference decoder from the QAT unquantized checkpoint at b10362.
  The H100 reference and the vision bound's CPU reference are two
  instruments.
- Template confound. The three GGUFs render one identical request
  to 319 (reference), 324 (this pack), and 312 (QAT) prompt tokens,
  uniform across all 1,349 tasks. Each arm's KLD folds that prefix
  difference into the measured divergence.
- Generation budget. Every QAT 48-token generation ended inside
  the thought channel, the model's reasoning block before its
  answer. The 512-token regeneration ran on a second H100 SXM pod
  of the same SKU and build. 49 of its 1,349 generations still hit
  the cap, and the judge scored those as fragments. The reference
  and this pack answered under 48 tokens on every task.
- Judge transport. The judge is `claude-haiku-4-5` through the
  Claude Code CLI (`claude -p`), which exposes no temperature
  control. All 4,047 verdicts persist in the run archive.

The full record ships in `analysis/psai-gui-kv9.json`: method, 13
input log hashes, aggregates, and 1,349 per-task rows with every
forced position. The per-task rows recompute the divergence tables
and the accuracy scores. The noise floor traces to the answer files
in the run archive, with the raw generations, force logs, and judge
outputs. The tracker record is
[#462](https://github.com/Alberto-Codes/vramfit/issues/462).

## What fit24gib means

The claim: the decoder loads fully offloaded on a 24 GiB card at
86,016 tokens of context and generates, and the decoder plus the
projector load at 73,728 tokens and answer an image prompt. Both
are measured serve results under the stated configuration, not a
promise about every runtime setup.

The budget arithmetic behind the recipe:

| Quantity | Bytes | GiB |
|---|---|---|
| VRAM budget | 25,769,803,776 | 24.000 |
| KV headroom | 9,663,676,416 | 9.000 |
| Weight budget | 16,106,127,360 | 15.000 |
| Predicted pack size | 16,074,691,830 | 14.971 |
| Real packed file | 16,015,862,144 | 14.916 |

The packed file lands 86.08 MiB under the weight budget.

The serve ladders, llama.cpp b10362 Vulkan on an RTX 4090
(24,564 MiB) under desktop sharing, `-ngl 99 -np 1`, KV cache f16,
4,096-token rungs, measured 2026-08-31 with 23,629–23,631 MiB free
before each load:

| Pack | Text-only max load | One image aboard |
|---|---|---|
| **This pack** | **86,016** (fails at 90,112) | **73,728** (encode fails at 77,824) |
| QAT Q4_0 | 65,536 (fails at 69,632) | 49,152 (encode fails at 53,248) |

The ladder boundary moves with the box's idle VRAM share. The
2026-08-28 card published 81,920 text-only with 86,016 as its
failing rung, and 61,440 with one image behind the BF16 sidecar.
Both boundaries are real in their frames. The published 81,920
rung reproduced on 2026-08-31 (465 MiB free at load) before these
ladders ran, and 86,016 then passed with 143 MiB free. The text
boundary moved one rung with the frame. This pack's image boundary
moved three rungs: one from the frame, one from the BF16 line
re-measuring lower in this frame, and one from the sidecar
conversion. At the 86,016-token boundary the decoder answered a
completion request from inside that envelope. The serve ladder is
a fit bar, not a speed bar — the boundary check decoded five tokens
under desktop sharing. The throughput figures sit in the real-GUI
campaign section above, measured at context 8,192. Throughput at
the 86,016-token boundary is unmeasured.

Serving images costs a measured 960 MiB beyond text-only serving
with the shipped sidecar: a 772 MiB load-time delta at matching
context, plus the image-encode transient. The line quantizes to
the 4,096-token rung, 320 MiB on this geometry. The BF16 sidecar's
line measured 1,280 MiB in the same frame. Its components measured
2026-08-28: 1,022.8 MiB of projector weights and a 150.63 MiB CLIP
compute reserve, plus the transient. One 768×768 image consumes
256 decoder tokens, measured at the server — the checkpoint config
claims 280, and the measured cost wins.

To serve:

```
# Text only, at the measured boundary
llama-server -m gemma-4-31B-it-fit24gib.gguf -c 86016 -ngl 99 -np 1

# Images, at the measured boundary
llama-server -m gemma-4-31B-it-fit24gib.gguf \
  --mmproj gemma-4-31B-it-mmproj-q4km.gguf -c 73728 -ngl 99 -np 1 \
  --mtmd-batch-max-tokens 264
```

Three reproduction traps, stated because each cost a failed load
or a crashed server:

- Pass `-np 1`. The b10362 server defaults to `-np 4` with unified
  KV, which adds ~2,400 MiB of SWA cache for this geometry and
  fails loads that fit at one slot.
- Keep ~200 MiB free beyond the load when serving images. The
  image-encode transient allocates at request time, and the b10362
  server crashes on the failure path instead of refusing.
- Cap the encode batch at one image with
  `--mtmd-batch-max-tokens 264`. A 1280×720 image is 264 image
  tokens, 271 with its wrapper. The b10362 server packs up to
  1,024 image tokens into one encode graph by default. Two such
  images then share one graph, and its compute buffer asks 328 MiB
  against the 150.63 MiB one-image reserve. At this boundary the
  server holds ~100 MiB free after one image on this pack and
  ~65 MiB on the QAT baseline. A 2026-09-02 multi-image ladder
  (23,549–23,556 MiB free before each load) crashed the server on
  the second image at both configurations in the table above. With
  one image per batch the same ladder filled the window to the
  context refusal on both, and the server survived every request.
  The ~200 MiB rule above is a one-image rule.

## The recipe

The solver — greedy damage-per-byte over the sensitivity map —
allocated 61 decoder groups: the token embedding and 60 layers. A
nominal width names the solver's assignment, and the packer maps it
to a k-quant type under the recipe's `kquant-imx` within-group
policy. The allocation:

- 6 layers at nominal 2 (Q2_K): layers 1–5 and 11.
- 9 layers at nominal 3 (Q3_K): layers 0, 6–10, and 12–14.
- 45 layers at nominal 4 (Q4_K): layers 15–59.
- The token embedding at nominal 4 (Q4_K), with the output head at
  Q4_K by the same override.

The map priced the early layers cheapest, and the solve spent its
budget protecting depth. The plan's trace records 81 steps and
replays from `recipe.json`, which ships in this repository. The
predicted whole-recipe damage is 0.184 on the map's frame. Damage
values are one scan's measurements on one measurement frame — do
not compare them across scans or across models.

The pack ran llama-quantize b10362 with `--pure`, a Q2_K base type,
and 60 per-tensor overrides from the recipe.

## The importance matrix

The pack consumed an in-frame importance matrix: 356 chunks of the
framed calibration corpus through the pinned prebuilt llama-imatrix
b10362 with `--parse-special`. The matrix stays in the run archive
and is not carried in this repository.

Coverage derives from the matrix's own entry names, never from a
label. Two absences matter:

- The matrix carries no `token_embd.weight` entry, so the token
  embedding quantized unassisted. This is expected at b10362.
- The matrix carries `attn_v` entries for 50 of 60 layers. The 10
  full_attention layers (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
  receive no `attn_v` activation from the b10362 graph, so those
  ten tensors quantized unassisted. This is a property of the
  instrument, not a defect in the matrix.

## Reproduce it

The repository ships the recipe, the run log, the evals sidecars,
and the two analysis artifacts beside the weights. Two commands
rebuild the decoder from the base checkpoint:

```
python convert_hf_to_gguf.py <checkpoint dir> \
  --outfile gemma-4-31b-it-qat-unquantized-bf16.gguf --outtype bf16
uv run vramfit pack recipe.json --llama-cpp <llama.cpp checkout> \
  --model <checkpoint dir> \
  --base-gguf gemma-4-31b-it-qat-unquantized-bf16.gguf \
  --imatrix gemma-4-31b-bf16-framed.imatrix.gguf \
  --out gemma-4-31B-it-fit24gib.gguf
```

The recipe records the full solve: the 15 GiB weight budget, the
0.005 format overhead, and the 81-step trace, so the type placement
replays from `recipe.json` alone. Two inputs stay in the project's
run archive and are not published: the importance matrix and the
sensitivity map. Without the matrix the pack step reproduces the
type placement but not this file's exact bytes, and without the map
`vramfit plan` cannot re-derive the recipe or solve a different
budget for this model.

## Damage disclosure

Quantization compresses every weight tensor with one uniform lossy
procedure. It does not bypass or disable the base model's safety
training, which ships in these weights at lower precision — and it
can shift any model behavior. The tables above are the measured
bound on that shift. Tier 2 measures whole-model KL divergence
against the BF16 reference on the measurement frame. Tier 3 holds
four ties and one win on held-out benchmarks. The vision campaign
bounds the distributional shift on 120 content-class positions over
10 held-out images. The real-GUI campaign bounds it on 21,216
content-class positions over 1,349 screenshots.

One limit, stated plainly: damage measures output distributions,
not safety behavior separately, and the vision bounds are
distributional bounds on images, not a safety evaluation of
image inputs. Read this card as a damage disclosure, not a safety
certificate. Deploy this pack with the same system-prompt and
application-layer protections you would give the base model.

## Files

| File | SHA-256 | Bytes |
|---|---|---|
| `gemma-4-31B-it-fit24gib.gguf` | `2a7bd7a7be6979c858258618ab576db573a7b671b45ee5e9785247341b8c3b1e` | 16,015,862,144 |
| `gemma-4-31B-it-mmproj-q4km.gguf` | `4a03ccaeaaa49cde65a97addac0b2ccd07df4617858aac1472048589ab672033` | 659,537,504 |
| `recipe.json` | `2730692845959b457211c5bd23a4d67acb8744aaa15e5eda8e7f825ed1e3b320` | 29,951 |
| `gemma-4-31B-it-fit24gib.runlog.jsonl` | `8da670782e6ae96ef3cce4a2bc00c0962f91b5ab083a19f11a8c836c0ade5b6a` | 2,036 |
| `gemma-4-31B-it-fit24gib.gguf.evals.json` | `eaefcf7c6b6d40afde6ea275cd7f6b6474525d389036bdbf6a5012c61a9a62d9` | 2,714 |
| `baselines/gemma-4-31B_q4_0-it.gguf.evals.json` | `2d8561c1d9d30b5b99b586dd3b2485c51e8d49a03d58884c1bfad6efc4928f9f` | 2,710 |
| `analysis/vision-campaign-kv9.json` | `2b705017870668ba248eea36ecb837c91d88ba0e78299ba7af9a7ce2ee709b4d` | 55,286 |
| `analysis/psai-gui-kv9.json` | `8f25d7e3add46dab0cd95db161323d07c0c0cc5e216018a7778857b72cf96363` | 2,850,490 |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | 11,358 |
| `README.md` | recorded at ship | — |

The run log ends at `pack_finished`: this pack predates the vramfit
release that records the sidecar hash in the run log, so the
projector hash above was computed directly on the shipped bytes.
Until 2026-08-31 this repository shipped the vendor's BF16
projector (`gemma-4-31B-it-mmproj.gguf`, SHA-256 `6bd60bdb…07a4`,
1,200,726,368 bytes). ADR-0030's 2026-08-31 amendment swapped the
sidecar to Q4_K_M. The vendor repository carries the identical
BF16 bytes.

## Provenance and license

The source checkpoint is
[google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
at revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`. Conversion
to the BF16 decoder GGUF ran at llama.cpp b10362. The projector
derives from the vendor's own published file, downloaded from
[google/gemma-4-31B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf)
at revision `59dde24573e7e61570dba08b18a2e1fe246955ed`. The source
file's SHA-256 matches the vendor's LFS object. It carries 190
BF16 and 166 F32 tensors under the `v.` and `mm.` roots.
llama-quantize b10362 converted it with the Q4_K_M recipe,
1,145.08 MiB to 628.96 MiB, with every quantizable tensor on
fallback. The shipped type counts appear in the two-files section
above. The QAT Q4_0 comparator is the vendor's decoder GGUF from
the same repository at the same revision, SHA-256
`179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b`.

Gemma 4 is released by Google DeepMind under the Apache 2.0
license — see the
[Gemma 4 license note](https://ai.google.dev/gemma/docs/gemma_4_license)
and the `LICENSE` file in this repository. This derivative work
carries the same license. The calibration corpus is public-domain
Project Gutenberg text under the Gemma channel frame.
