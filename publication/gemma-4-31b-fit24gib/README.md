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
and solved to serve — images included — inside a 24 GiB card.
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
(`gemma-4-31B-it-mmproj.gguf`, 1.118 GiB) carries the vision tower
in BF16, byte-identical to the vendor projector. Text-only serving
needs the decoder alone. Image serving needs both files. The
sidecar stays unquantized deliberately: the literature marks vision
components as more quantization-sensitive than the decoder, and the
quality cost of quantizing this projector is unmeasured.

## The headline

The official [QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf)
of this model is 16.43 GiB. This pack is 14.92 GiB at comparable
measured text quality — four benchmark ties and one win, tables
below — and lower measured vision-conditioned divergence. The freed
1.5 GiB of weights buys context on the same card:

| Serving shape | This pack | QAT Q4_0 | Gain |
|---|---|---|---|
| Text-only, measured max load | 81,920 tokens | 61,440 tokens | +20,480 (+33.3 %) |
| One image aboard, measured max load | 61,440 tokens | 40,960 tokens | +20,480 (+50 %) |

Both rows are served results on an RTX 4090 (24 GiB), llama.cpp
b10362, `-ngl 99 -np 1`, measured at 4,096-token ladder rungs under
the same runtime buffers. Every measured number below sits beside
its baseline counterpart. The card prints the losing numbers too.

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
| BF16 reference | 57.19 GiB | 35.0668 | — | — | — |
| **This pack** | **14.92 GiB** | 37.4552 | **1.0681 ± 0.0027** | 0.0446 ± 0.0004 | 92.04 % |
| QAT Q4_0 | 16.43 GiB | 38.7227 | 1.1043 ± 0.0029 | **0.0420 ± 0.0003** | 92.32 % |

Read the split honestly: this pack holds the better PPL ratio, and
the QAT baseline holds a slightly better mean KLD (0.0420 against
0.0446) and top-token agreement (92.32 % against 92.04 %). The two
text metrics disagree at the margin, so the held-out benchmarks
arbitrate.

## Held-out benchmarks

lm-evaluation-harness 0.4.12 over llama-cpp-python 0.3.34 with the
b10362 Vulkan libraries, full splits, no sampling limit. A delta
inside the combined standard error is a tie.

| Task (shots, metric) | This pack | QAT Q4_0 | Verdict |
|---|---|---|---|
| MMLU (5, acc) | **71.36 ± 0.37** | 70.20 ± 0.38 | **win +1.15** |
| GSM8K (5, strict exact match) | 92.34 ± 0.73 | 92.42 ± 0.73 | tie |
| HellaSwag (10, acc_norm) | 58.71 ± 0.49 | 59.34 ± 0.49 | tie |
| Winogrande (5, acc) | 68.27 ± 1.31 | 68.03 ± 1.31 | tie |
| ARC-Challenge (25, acc_norm) | 61.77 ± 1.42 | 61.09 ± 1.42 | tie |

Four ties and one win, at 1.5 GiB fewer packed bytes. Per-task
records ship in the evals sidecars beside the weights.

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
teacher-forced each quantized arm on the reference sequence. All
three arms served through the same BF16 projector. Position classes
separate image-grounded `content` tokens from channel-frame policy
tokens (`markup`, `pos0`). Frame-policy positions dominate
all-position means on this channel-locked model, so the content
class carries the vision claim.

Content-class results (n = 120 positions):

| Arm | Mean KLD ↓ | p95 KLD ↓ | Top-token agreement ↑ |
|---|---|---|---|
| **This pack** | **0.0045** | **0.0239** | **99.2 %** |
| QAT Q4_0 | 0.0373 | 0.1928 | 97.5 % |

The all-position mean for this pack is 0.0489 over 178 positions —
10.9 times the content-class figure, dominated by the frame-policy
classes. The instrument noise floor is 1.07e-4 mean KLD, measured
by teacher-forcing this pack on its own greedy sequence. This pack
measures 8.3 times below the QAT baseline and 42 times above the
floor. The BF16 reference and this pack each answered 10 of 10
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
breakdown, input log hashes — ships in
`analysis/vision-campaign-kv9.json`.

## What fit24gib means

The claim: the decoder loads fully offloaded on a 24 GiB card at
81,920 tokens of context and generates, and the decoder plus the
projector load at 61,440 tokens and answer an image prompt. Both
are measured serve results under the stated configuration, not a
promise about every runtime setup.

The budget arithmetic behind the recipe:

| Quantity | Bytes | GiB |
|---|---|---|
| VRAM budget | 25,769,803,776 | 24.000 |
| KV headroom | 9,663,676,416 | 9.000 |
| Weight budget | 16,106,127,360 | 15.000 |
| Predicted pack size | 16,074,691,830 | 14.971 |
| Real packed file | 16,015,862,144 | 14.917 |

The packed file lands 86.08 MiB under the weight budget.

The serve ladders, llama.cpp b10362 Vulkan on an RTX 4090
(24,564 MiB) under desktop sharing, `-ngl 99 -np 1`, KV cache f16,
4,096-token rungs:

| Pack | Text-only max load | One image aboard |
|---|---|---|
| **This pack** | **81,920** (fails at 86,016) | **61,440** (encode fails at 65,536) |
| QAT Q4_0 | 61,440 (fails at 65,536) | 40,960 (encode fails at 45,056) |

At the 81,920-token boundary the decoder answered a completion at
7.4 tokens/s decode under desktop sharing. Serving images costs a
measured 1,600 MiB beyond text-only serving: 1,022.8 MiB of
projector weights, a 150.63 MiB CLIP compute reserve, and the
image-encode transient. One 768×768 image consumes 256 decoder
tokens, measured at the server — the checkpoint config claims 280,
and the measured cost wins.

To serve:

```
# Text only, at the measured boundary
llama-server -m gemma-4-31B-it-fit24gib.gguf -c 81920 -ngl 99 -np 1

# Images, at the measured boundary
llama-server -m gemma-4-31B-it-fit24gib.gguf \
  --mmproj gemma-4-31B-it-mmproj.gguf -c 61440 -ngl 99 -np 1
```

Two reproduction traps, stated because each cost a failed load:

- Pass `-np 1`. The b10362 server defaults to `-np 4` with unified
  KV, which adds ~2,400 MiB of SWA cache for this geometry and
  fails loads that fit at one slot.
- Keep ~200 MiB free beyond the load when serving images. The
  image-encode transient allocates at request time, and the b10362
  server crashes on the failure path instead of refusing.

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

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `gemma-4-31B-it-fit24gib.gguf` | 16,015,862,144 | `2a7bd7a7be6979c858258618ab576db573a7b671b45ee5e9785247341b8c3b1e` |
| `gemma-4-31B-it-mmproj.gguf` | 1,200,726,368 | `6bd60bdb958548b4093196d38744b0f2290c12503a3fddd7486bffa9c5eb07a4` |
| `recipe.json` | 29,951 | `2730692845959b457211c5bd23a4d67acb8744aaa15e5eda8e7f825ed1e3b320` |
| `gemma-4-31B-it-fit24gib.runlog.jsonl` | 2,036 | `8da670782e6ae96ef3cce4a2bc00c0962f91b5ab083a19f11a8c836c0ade5b6a` |
| `gemma-4-31B-it-fit24gib.gguf.evals.json` | 2,714 | `eaefcf7c6b6d40afde6ea275cd7f6b6474525d389036bdbf6a5012c61a9a62d9` |
| `baselines/gemma-4-31B_q4_0-it.gguf.evals.json` | 2,710 | `2d8561c1d9d30b5b99b586dd3b2485c51e8d49a03d58884c1bfad6efc4928f9f` |
| `analysis/vision-campaign-kv9.json` | 14,899 | `2bad5ffa6ef72ee9a680a384cebe72bd64883027345e015937755c6588673885` |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `README.md` | recorded at ship | — |

The run log ends at `pack_finished`: this pack predates the vramfit
release that records the sidecar hash in the run log, so the
projector hash above was computed directly on the shipped bytes.

## Provenance and license

The source checkpoint is
[google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
at revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`. Conversion
to a BF16 decoder GGUF and extraction of the projector ran at
llama.cpp b10362. The projector carries 190 BF16 and 166 F32
tensors under the `v.` and `mm.` roots, unmodified. The QAT Q4_0
comparator is the vendor's own GGUF of the same checkpoint family.

Gemma 4 is released by Google DeepMind under the Apache 2.0
license — see the
[Gemma 4 license note](https://ai.google.dev/gemma/docs/gemma_4_license)
and the `LICENSE` file in this repository. This derivative work
carries the same license. The calibration corpus is public-domain
Project Gutenberg text under the Gemma channel frame.
