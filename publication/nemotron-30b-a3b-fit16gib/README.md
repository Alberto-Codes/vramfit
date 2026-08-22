---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
base_model_relation: quantized
quantized_by: Alberto-Codes
pipeline_tag: text-generation
tags:
  - vramfit
  - gguf
  - imatrix
---

<!--
Authored for issue #404 under the #401 identity grammar. Upload this
file verbatim — the published card and this source must match.

Open before upload (the #404 dry run resolves each):
- The sensitivity-map dataset repo does not exist yet. The link
  below is the intended name under the #401 grammar.
- The evals sidecar needs its tier-1 and tier-2 blocks
  (make-sidecars.py re-run, #400 lane facts).
- The imatrix republish-or-link choice (bartowski's matrix, see the
  imatrix paragraph) needs the maintainer's call.
- File sha256s and the upload ledger land at the dry run (#82
  precedent).
-->

# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib-GGUF

This repository carries one mixed-precision GGUF of
[nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
solved to serve inside a 16 GiB card.
[vramfit](https://github.com/Alberto-Codes/vramfit) measured each
layer group's quantization damage — the shift in the model's output
distribution when that group quantizes — then allocated bits under
the budget and packed through llama.cpp's quantizer. The `fit16gib`
marker is a deployment claim, not a file size: a serve test ran this
exact file inside a 16 GiB VRAM boundary under the configuration
stated below. This pack has no single quantization scheme, and the
budget is the claim.

One omission to know before you download: **this pack carries no MTP
block.** The f16 conversion ran `--no-mtp`, so the multi-token
prediction layers the base checkpoint ships are not in this file and
speculative decoding (`--spec-type draft-mtp`) is unavailable. The
published comparator below carries its MTP block at Q4_0, which is
part of its larger file size.

Every measured number below sits beside its baseline counterpart.
The card prints the losing numbers too.

## Quality beside size

Held-out WikiText-2 test set, 594 chunks, measured against the f16
base on one instrument (llama.cpp b10362, same-pod f16 logits).
Lower is better for PPL and KLD. "Top-1 agree" is the share of
tokens where the quantized model and the f16 base agree on the top
token — higher is better. The comparator is bartowski's `IQ2_XXS`,
the smallest published GGUF of this model at measurement time.

| Model | File size | Bits/param | PPL ↓ | PPL / f16 ↓ | Mean KLD ↓ | Top-1 agree ↑ |
|---|---|---|---|---|---|---|
| f16 reference | 58.84 GiB | 16.007 | 6.8192 | — | — | — |
| **This pack** | **15.76 GiB** | 4.287 | **7.9177** | **1.161096** | **0.204318** | 83.13 % |
| IQ2_XXS (bartowski) | 17.54 GiB | — | 9.0075 | 1.320914 | 0.370257 | — |

This pack beats the published build on both metrics at once — 15.98
points of PPL ratio and 44.8 % lower mean KLD — at 1,915,545,664
fewer packed bytes. The comparison was ruled to read both metrics
together from one instrument before the measurement ran, so no
metric was chosen after the fact. The published build's bits/param
cell stays empty because its bytes include the MTP block this pack
omits — the division would run over different weights.

Two facts about the comparator, stated because they explain the gap:

- Both packs consume the same importance matrix (bartowski's, 185
  entries over 822 chunks). The published build quantizes 91.53 % of
  its bytes assisted. This pack quantizes 74.44 % assisted, because
  no type takes an assisted fit at 2.25 bits on expert rows of 2688
  and 1856 columns. The asymmetry runs against this pack, and it
  wins anyway.
- The `IQ2_XXS` label names 12 of that build's 417 tensors.
  llama.cpp's fallback rewrites every row 256 does not divide, which
  sends all 46 expert stacks — 93 % of the parameters — to `IQ4_NL`
  at 4.5 bits per weight. The shelf's smallest build spends 4.5-bit
  experts and loses to a recipe holding 11 stacks at 2.25.

## What fit16gib means

The claim: this file loads fully offloaded on a 16 GiB card, holds
16k context, and generates. It is a measured serve result under the
stated configuration, not a promise about every runtime setup.

The budget arithmetic:

| Quantity | Bytes | GiB |
|---|---|---|
| VRAM ceiling | 17,179,869,184 | 16.000 |
| Measured runtime buffers (KV + recurrent state + compute) | — | 0.224 (228.99 MiB) |
| Weight budget | 16,939,351,015 | 15.776 |
| Predicted pack size | 16,929,873,667 | 15.767 |
| Real packed file | 16,922,476,480 | 15.760 |

The packed file lands 16.09 MiB under the weight budget.

The serve test: llama.cpp b10326 (Vulkan), a hard ballast cap
holding the device to 16,383 MiB visible on an RTX 4090, `-ngl 99`
at 16k context. llama.cpp reported 53/53 layers offloaded with
15,774.00 MiB of weights on the device — the 357.00 MiB token
embedding stays host-mapped, as llama.cpp always keeps it. Device
buffers totaled 16,157.88 MiB of 16,383 MiB visible: KV 96.00 MiB
(16,384 cells across the six attention layers), recurrent state
190.47 MiB at four server slots, compute 97.41 MiB. `llama-server`
answered a completion request from inside that envelope. The
published build cannot take this test: its 17.54 GiB of weights
exceed the card before the first buffer allocates.

The claim's boundaries, stated plainly:

- The budget derives from a single-sequence buffer measurement. The
  recurrent state grows per sequence, and the margin absorbs that to
  8 parallel sequences. Above 8, this budget does not hold.
- The serve test is a fit bar, not a speed bar. This card publishes
  no tokens-per-second figure: the test ran on a VRAM-capped 4090,
  and a decode figure from that method would read 1.4 to 3.5 times
  optimistic against real 16 GiB silicon.
- A 16 GiB owner can also run larger builds today by offloading
  part of the weights to CPU and accepting slower decode. This pack
  is the alternative that keeps every weight on the card. The
  project has not measured that speed difference.

## The recipe

The solver — greedy damage-per-byte over the stack-keyed
sensitivity map — allocated 210 groups, under nine pins that hold
every dense class at 8-bit:

- 11 `down_proj` expert stacks at nominal 2 (Q2_0, 2.25 bits per
  weight): layers 22, 24, 27, 29, 31, 34, 43, 45, 47, 49, 51.
- The other 35 expert stacks at nominal 4 (Q4_0). The routed
  experts hold 93 % of the parameters, so these two rows are the
  budget.
- 118 dense groups at 8-bit (Q8_0): the token embedding, the
  output head, and every attention, Mamba-2, and shared-expert
  projection. Dense weights are 7 % of the parameters, so the
  8-bit spend is cheap.
- 46 groups pass through at F16: the Mamba-2 convolutions and the
  router gates, classes llama.cpp's quantizer never touches.

The trace records 11 demotion steps, and every one demotes a
`down_proj` expert stack from nominal 4 to nominal 2 in
damage-per-byte order, so the solve replays from the artifact. The
2-bit placement follows vramfit's spread placement rule: the cheap
width lands on the stacks the map prices cheapest, spread across
the depth rather than clustered. The same campaign measured eight
alternative placements of the identical width mix — blind draws,
size-matched controls, an inverted arm — and this allocation's
damage is the best of the nine, with the worst at 2.7 times this
one. Allocation decides, and the map-ranked placement wins.

One attribution bound travels with that result: a reference-frame
variant of the map derives the identical placement, so the win
credits the stack-keyed damage ranking under the placement rule,
not the imatrix-assisted repricing.

Damage records in the recipe are partial by design: the 46 expert
stacks carry measured marginals, and a pinned or passthrough group
records 0.0 at a width the map never priced. Damage values are one
scan's measurements on one measurement frame. Do not compare them
across scans or across models.

<details>
<summary>Per-group allocation (210 groups, from the recipe)</summary>

| Group | Nominal bits | GGUF type | Bytes |
|---|---|---|---|
| `model.layers.1.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.1.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.3.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.3.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.6.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.6.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.8.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.8.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.10.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.10.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.13.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.13.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.15.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.15.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.17.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.17.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.20.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.20.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.22.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.22.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.24.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.24.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.27.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.27.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.29.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.29.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.31.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.31.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.34.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.34.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.36.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.36.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.38.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.38.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.40.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.40.mixer.experts.down_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.43.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.43.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.45.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.45.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.47.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.47.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.49.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.49.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `model.layers.51.mixer.experts.up_proj` | 4 | Q4_0 | 359,921,222 |
| `model.layers.51.mixer.experts.down_proj` | 2 | Q2_0 | 179,960,611 |
| `lm_head` | 8 | Q8_0 | 375,090,316 |
| `model.embeddings` | 8 | Q8_0 | 375,090,316 |
| `model.layers.0.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.0.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.0.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.1.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.1.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.1.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.10.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.10.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.10.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.11.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.11.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.11.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.12.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.12.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.12.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.12.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.13.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.13.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.13.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.14.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.14.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.14.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.15.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.15.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.15.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.16.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.16.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.16.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.17.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.17.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.17.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.18.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.18.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.18.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.19.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.19.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.19.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.19.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.2.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.2.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.2.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.20.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.20.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.20.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.21.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.21.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.21.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.22.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.22.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.22.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.23.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.23.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.23.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.24.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.24.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.24.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.25.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.25.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.25.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.26.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.26.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.26.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.26.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.27.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.27.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.27.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.28.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.28.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.28.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.29.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.29.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.29.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.3.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.3.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.3.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.30.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.30.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.30.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.31.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.31.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.31.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.32.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.32.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.32.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.33.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.33.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.33.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.33.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.34.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.34.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.34.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.35.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.35.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.35.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.36.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.36.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.36.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.37.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.37.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.37.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.38.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.38.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.38.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.39.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.39.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.39.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.4.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.4.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.4.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.40.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.40.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.40.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.41.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.41.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.41.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.42.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.42.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.42.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.42.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.43.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.43.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.43.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.44.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.44.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.44.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.45.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.45.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.45.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.46.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.46.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.46.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.47.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.47.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.47.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.48.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.48.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.48.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.49.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.49.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.49.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.5.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.5.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.5.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.5.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.50.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.50.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.50.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.51.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.51.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.51.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.6.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.6.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.6.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.7.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.7.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.7.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.8.mixer.gate` | 16 | F16 | 689,505 |
| `model.layers.8.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.8.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.9.mixer.conv1d` | 16 | F16 | 49,251 |
| `model.layers.9.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.9.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |

</details>

## Evaluation

Three tiers. Tier 1 (perplexity) and tier 2 (whole-model KL
divergence) are the table above, measured on the ADR-0027
instrument frame: llama.cpp b10362, same-pod f16 reference logits,
594 held-out WikiText-2 chunks. Tier 3 ran lm-evaluation-harness
0.4.12 through a llama-cpp-python lane on llama.cpp b10362
(Vulkan) — full evaluation splits, no `--limit`, zero context
truncations, this pack and the comparator on the same lane.

The slice was fixed before any run: five tasks at leaderboard
few-shot settings. A delta inside the combined standard error
reports as a tie.

| Task (metric) | This pack (15.76 GiB) | IQ2_XXS (17.54 GiB) | Δ | Combined σ | Verdict |
|---|---|---|---|---|---|
| MMLU 5-shot (acc) | 0.7651 ± 0.0034 | 0.6848 ± 0.0037 | +0.0803 | 0.0050 | ahead (16.1σ) |
| GSM8K 5-shot (strict) | 0.7839 ± 0.0113 | 0.7627 ± 0.0117 | +0.0212 | 0.0163 | ahead (1.3σ) |
| HellaSwag 10-shot (acc_norm) | 0.8038 ± 0.0040 | 0.7652 ± 0.0042 | +0.0386 | 0.0058 | ahead (6.7σ) |
| Winogrande 5-shot (acc) | 0.7443 ± 0.0123 | 0.7261 ± 0.0125 | +0.0182 | 0.0175 | ahead (1.04σ) |
| ARC-Challenge 25-shot (acc_norm) | 0.6630 ± 0.0138 | 0.6715 ± 0.0137 | −0.0085 | 0.0195 | tie (0.4σ) |

Four leads and one tie, at 1.78 GiB smaller. The one nominal
deficit (ARC-Challenge) prints with its error bar. The slice was
fixed before any run, so no result selected the tasks.

Read together: tier 2 ranks (both damage metrics, same instrument),
tier 3 certifies (four leads and a tie on task benchmarks).

## Reproduce it

The repository ships the recipe, the evals sidecar, and the run
log beside the weights. One command reproduces the pack from the
base checkpoint:

```
uv run vramfit pack recipe.json --llama-cpp <llama.cpp checkout> \
  --imatrix imatrix.gguf \
  --out NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf
```

The command converts the f16 base GGUF once (`--no-mtp`), drives
the recorded types into `llama-quantize`, and reports the margin.
The pack reproduces this file's bytes on one machine — packed size
varies by tens of bytes across machines because the GGUF metadata
stores the imatrix path.

The recipe is not a magic constant. It records the full solve: the
15.776 GiB weight budget, the nine pins, the 0.002 format
overhead, and the 11-step trace. `vramfit plan` re-derives it
field-for-field from the published sensitivity map with those
recorded settings, and solves any other budget with your own
`--vram`. The map lives in the linked dataset repository:
[NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps](https://huggingface.co/datasets/Alberto-Codes/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps),
file `sensitivity-32k-q0-imx-stacks.json`.

The importance matrix both this pack and the comparator consumed is
bartowski's, published in
[bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF).

## Damage disclosure

Quantization compresses every weight tensor with one uniform lossy
procedure. It does not bypass or disable the base model's safety
training, which ships in these weights at lower precision — and it
can shift any model behavior. The tables above are the measured
bound on that shift: tier 2 measures whole-model KL divergence
against the f16 base over the full held-out set, and tier 3 holds
four leads and a tie against the published comparator.

One limit, stated plainly: damage measures the general output
distribution on held-out WikiText-2 text, not safety behavior
separately. Read this card as a damage disclosure, not a safety
certificate. Deploy this pack with the same system-prompt and
application-layer protections you would give the base model.

## License

The base model ships under
[OpenMDW 1.1](https://openmdw.ai/license/1-1/), which permits
distributing modified model materials with the license text and
origin notices retained. This repository carries both.

## Provenance

- Packed file SHA-256:
  `85ed06fac2f879ee83f83264f3b7cad9bde4947983976205ea8c2c5d6291c062`
  (16,922,476,480 B). Packed size varies by tens of bytes across
  machines because the GGUF metadata stores the imatrix path — the
  recipe, not the byte count, is the identity across machines.
- Base checkpoint:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at revision
  `ce38b6a`. Upstream `main` has moved past this revision — every
  measured number on this card derives from `ce38b6a`.
- The run log beside the weights records the pack events. The evals
  sidecar
  (`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf.evals.json`)
  records all three tiers. The comparator's sidecar sits under
  `baselines/` with its upstream file name.
- Toolchain: `convert_hf_to_gguf.py --no-mtp` for the f16 base,
  llama.cpp b10326 quantizer for the pack, llama.cpp b10362 for the
  damage instrument and the tier-3 lane, lm-evaluation-harness
  0.4.12 with llama-cpp-python 0.3.34 for tier 3.

Hashes prove identity, not quality. The evidence is the three tiers
above, and every number on this card traces to the evals sidecar
and the recipe published in this repository.
