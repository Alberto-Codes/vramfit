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
Authored for the v2 re-upload under the #401 identity grammar. Upload
this file verbatim — the published card and this source must match.

This card's tier-1 and tier-2 numbers were measured on the exact bytes
this repository carries, in one sequence on one rented pod, before the
upload and before the pod was deleted. That ordering is the point of
the revision: it closes issue #598 as a class rather than patching one
instance of it.

Known gap, stated on the card in the Evaluation section: tier 3 did
not run for this revision.
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

Two terms the card leans on. A *sensitivity map* records that
damage for every layer group. On this model the expensive groups
are *expert stacks*: the 128 routed experts of one projection in
one layer, stored as a single tensor. The 46 expert stacks hold
93 % of the parameters, so the recipe is mostly a decision about
them.

**What changed in this revision.** The allocation is unchanged. The
eleven 2-bit expert stacks are now written by vramfit's own assisted
`Q2_0` encoder (ADR-0032) instead of llama.cpp's stock `Q2_0`
quantizer, which discards the importance matrix. Same recipe, same
byte count, 65.05 % less mean KL divergence. The previous
revision's file remains addressable at its own commit, and this card
does not carry its numbers forward.

One omission to know before you download: **this pack carries no MTP
block.** The f16 conversion ran `--no-mtp`, so the multi-token
prediction layers the base checkpoint ships are not in this file and
speculative decoding (`--spec-type draft-mtp`) is unavailable. The
published comparator below carries its MTP block at Q4_0, which is
part of its larger file size.

One requirement to know too: this pack uses the `Q2_0` tensor
type, which llama.cpp merged on 2026-07-07. **Every number on this
card was measured on build b10362** (commit `4801e3c56`), and that
is the build this card states. A build older than the 2026-07-07
merge refuses the file. Builds between the merge and b10362 are not
measured here, so this card makes no claim about them.

Every measured number below sits beside its baseline counterpart.
The card prints the losing numbers too.

## Quality beside size

Held-out WikiText-2 test set, 594 chunks, measured against the f16
base on one instrument. Lower is better for PPL and KLD. "Same top"
is the share of tokens where the quantized model and the f16 base
agree on the top token — higher is better. The comparator is
bartowski's `IQ2_XXS`. The campaign ruled it the bar before the
measurement ran, after a check of five publishers' repositories
where it was the smallest file. A Hub-wide query on 2026-08-22 then
found eight other full-model GGUFs of this model below this pack's
15.76 GiB. The table below measures none of them. The project tracks
that measurement.

**Every row below was measured in one sequence, on one pod, against
that pod's own f16 reference logits, on 2026-09-17.** The first
row is the file in this repository, hashed after the measurement and
uploaded from the same pod.

| Model | File size | Bits/param | PPL ↓ | PPL / f16 ↓ | Mean KLD ↓ | Same top ↑ |
|---|---|---|---|---|---|---|
| f16 reference | 58.84 GiB | 16.007 | 6.8192 | — | — | — |
| **This pack** | **15.76 GiB** | 4.287 | **7.0689** | **1.036624** | **0.071403** | 89.418 % |
| The revision this replaces | 15.76 GiB | 4.287 | 7.9174 | 1.161055 | 0.204322 | 83.146 % |
| IQ2_XXS (bartowski) | 17.54 GiB | — | 9.0075 | 1.320914 | 0.370257 | 76.086 % |

This pack beats the published comparator on both metrics at once: a
PPL ratio 0.2843 lower (1.036624 against 1.320914) and
80.7 % lower mean KLD, at 1.78 GiB fewer packed bytes. The
comparison was ruled to read both metrics together from one
instrument before the measurement ran, so no metric was chosen after
the fact.

Against the revision it replaces — same recipe, same allocation, same
byte count — the assisted encoder cuts mean KLD by 65.05 % and
lifts top-token agreement by 6.27 points. That row is the
encoder's effect with everything else held fixed.

One comparator cell stays empty. Its bits/param cell has no
honest value because its bytes include the MTP block this pack
omits — the division would run over different weights.

Two facts about the comparator, stated because they explain the gap:

- Both packs consume the same importance matrix (bartowski's, 185
  entries over 822 chunks). The published build quantizes 91.53 % of
  its bytes assisted. This pack quantizes 74.44 % assisted through
  `llama-quantize`, and the eleven 2-bit stacks consume the same
  matrix through vramfit's own encoder before that pass. The stock
  8-bit quantizer discards the matrix outright.
- The `IQ2_XXS` label names 12 of that build's 417 tensors.
  llama.cpp's fallback rewrites every row 256 does not divide, which
  sends all 46 expert stacks to `IQ4_NL` at 4.5 bits per weight.
  The comparator spends 4.5-bit experts and loses to a recipe
  holding 11 stacks at 2.25.

## What fit16gib means

The claim: this file loads fully offloaded on a 16 GiB card, holds
16k context, and generates. It is a measured serve result under the
stated configuration, not a promise about every runtime setup — and
this revision measured it on **two** runtimes, because the result
turns out to depend on which one you use.

The budget arithmetic:

| Quantity | Bytes | GiB |
|---|---|---|
| VRAM ceiling | 17,179,869,184 | 16.000 |
| Runtime reserve | 240,518,169 | 0.224 |
| Weight budget | 16,939,351,015 | 15.776 |
| Real packed file | 16,922,476,480 | 15.760 |

The packed file lands 16.09 MiB under the weight budget.

### The serve test, on these exact bytes

Both runs used llama.cpp b10362, `--fit off`, `-ngl 99 -c 16384
-np 1`, f16 KV, one slot, on an RTX 4090 held to **16,380 MiB
visible** by a hard ballast (the #164 method). Both ran on the file
downloaded back from this repository and hashed against its published
digest first. The only difference between them is the backend.

**Both runtimes allocate the same buffers**, to within rounding:

| Buffer | Vulkan | CUDA |
|---|---:|---:|
| Model buffer on device | 15,774.00 MiB | 15,774.05 MiB |
| CPU-mapped model buffer (token embedding) | 357.00 MiB | 357.00 MiB |
| KV cache (f16, 16,384 cells, six attention layers) | 96.00 MiB | 96 MiB |
| Recurrent state | 47.62 MiB | 47.62 MiB |
| Compute buffer, device | 85.37 MiB | 85.01 MiB |
| Compute buffer, host | 26.51 MiB | 26.51 MiB |
| **Device buffers, total** | **16,002.99 MiB** | **16,002 MiB** |

**The results differ anyway:**

| Backend | Under a 16,380 MiB visible cap | Result |
|---|---|---|
| **Vulkan** | 53/53 layers offloaded, `n_ctx` 16,384, one slot | **fits** — loaded and answered a completion request |
| **CUDA** | same configuration | **does not fit** — `cudaMalloc failed` allocating the 85.01 MiB compute buffer |

The difference is the backend's own context overhead, not the file.
llama.cpp's memory breakdown on the CUDA run reports about **450 MiB
unaccounted** beside the buffers, for 16,454 MiB device-wide. Vulkan
fit the same 16,003 MiB of buffers inside the same cap, so its
overhead is under the roughly 377 MiB of headroom that leaves. The
16 GiB boundary falls between the two.

A q8_0 KV cache does not rescue the CUDA case: it was tried under the
same cap and also failed.

**So, plainly.** On a Vulkan build this pack meets the fit16gib claim
at 16k context. **On a CUDA build it does not**, and a 16 GiB NVIDIA
owner running a CUDA build should expect to lower the context or
offload part of the weights. That distinction is not on the previous
revision's card, which measured Vulkan only.

The claim's boundaries, stated plainly:

- The budget derives from a single-sequence buffer measurement. The
  recurrent state grows per sequence, and the margin absorbs that to
  8 parallel sequences. Above 8, this budget does not hold.
- The serve test is a fit bar, not a speed bar. This card publishes
  no tokens-per-second figure: the test ran on a VRAM-capped 4090,
  and a decode figure from that method would read higher than real
  16 GiB silicon delivers.
- A 16 GiB owner can also run larger builds today by offloading
  part of the weights to CPU and accepting slower decode. This pack
  keeps every weight on the card under Vulkan. The project has not
  measured that speed difference. Smaller published builds of this
  model also fit by file size, and this card does not measure them.

One honest limit on the shipped `recipe.json`. Its `assignments` are
the authoritative record of what was packed, and they reproduce this
file. Its `plan` block does not describe this allocation: the recipe
was derived by hand from the `q0-imx2` solve, returning
`model.layers.36.mixer.experts.down_proj` to 4 bits to hold the
published eleven-stack allocation, and its own `solver` field says so.
So `vramfit plan` does not re-derive this recipe field-for-field, and
this card prints no predicted size for it. The 2026-09-04 re-pricing
of the 46 passthrough rows at the 32 bits the file stores (issue #409)
still applies to the published solve, and it is why a recipe planned
today at this budget demotes a twelfth stack.

## Usage

llama.cpp serves this pack. **Every measurement on this card ran
build b10362** (commit `4801e3c56`). Settings below that carry
measurements from an earlier verification name that build where they
differ; a setting this section does not name is unmeasured.

**Ollama cannot load this pack.** The file stores 11 tensors at
ggml type 42 (`Q2_0`). Ollama's type table stops at type 41 on
`main` and at type 39 in release 0.18.0. For an unknown type Ollama
sizes the tensor at 0 B and mis-plans the offload, so the failure
is not a clean refusal. No flag works around it, and no current
update carries type 42. The same holds for any runtime without
ggml type 42.

A single-user invocation:

```
llama-server -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf \
  -ngl 99 -c 16384 -np 1 --reasoning-format none --reasoning-budget 512
```

The verification runs did not start this command as one. Each flag
was verified on its own, and each answers one measured failure. The
reasoning-behavior measurements below ran on build b10573 with an
RTX 4090 and Open WebUI 0.11.0 on 2026-08-22, against the previous
revision of this pack. The allocation and the tokenizer are
unchanged, so they describe this file's behavior too — but they were
not re-measured on these bytes, and the card says so rather than
implying otherwise.

- `--reasoning-format none`. By default llama.cpp streams the
  thoughts as `reasoning_content` and holds `content` at null. Open
  WebUI 0.11.0 reads neither field, so every reply renders empty.
  Under this flag the `<think>` tags stay inside `content`, and Open
  WebUI draws a collapsible block. Verified on the streaming and the
  non-streaming path.
- `--reasoning-budget 512`. This model reasons before it answers,
  and the reasoning length tracks the problem. Twenty trivial
  prompts drew a median of 285 generated tokens, with a maximum of
  1,207 and none past 2,000. A separate single-prompt measurement,
  "why is the sky blue", drew 2,631 tokens for a one-sentence
  answer. When a reply exceeds `max_tokens` inside the
  thoughts, `content` comes back empty with `finish_reason`
  "length". It renders blank, not truncated, and no fixed cap covers
  the tail. The server flag closes the thought and yields an answer.
  On GSM8K (60 items, temperature 1.0, top_p 0.95) the budget cut
  the mean reply from 923 to 463 tokens, ran 3.3 times faster, and
  truncated 0 replies against 2 for the default. Accuracy read 98.3 %
  against 96.7 %, which is no measurable difference. 512 suits chat
  and everyday reasoning. A hard-reasoning workload needs a larger
  budget or none: on GPQA Diamond the same pack averaged 5,819
  generated tokens, and 57 % of items exceeded an 8,000-token cap.
  This card carries no measurement of a 512 budget on that workload.
- `-np 1`. See the slot setting below.

Two controls that look like the budget and are not:

- `reasoning_budget` in a request body is accepted and ignored. Only
  the server flag works, so a front end cannot set it per
  conversation.
- `chat_template_kwargs` with `enable_thinking` false is the one
  working request-level control. It turns reasoning off rather than
  bounding it. On the same GSM8K slice it read 90.0 % at 140 mean
  tokens.

**Slots and context.** `-c 16384` alone opens four slots against one
unified pool of 16,384 cells. Each slot reports `n_ctx` 16384 over
`/slots`, and the pool is shared. Four concurrent requests contend
for it, and a request then fails inside the server:

```
W decode: failed to find free space in the KV cache, retrying with smaller batch
W decode: failed to find a memory slot for batch of size 1
```

The caller sees no error. A 198-item benchmark at 4 concurrency
logged 426 such lines, and the harness scored the failed requests
as wrong answers. Read the server log before trusting a number
measured under concurrency. Three configurations:

| Flags | KV cache | Result |
|---|---|---|
| `-c 16384` | one pool | Four slots contend. |
| `-c 16384 -np 1` | one pool | One conversation at 16k, no contention. Inside the 16 GiB claim, and the configuration the serve test above ran. |
| `-c 65536 -np 4` | 16,384 cells per slot | Four conversations at 16k. Measured on a 24 GiB card only. |

The last row reports `n_ctx_slot = 16384, kv_unified = 'false'`,
and device memory rose from 19,209 MiB to 19,340 MiB on the 24 GiB
card. This card has not run that configuration inside the 16 GiB
boundary, so the fit16gib claim does not cover it.

## The recipe

The solver — greedy damage-per-byte over the sensitivity map —
allocated 210 groups, under nine pins that hold every quantizable
dense class at 8-bit. A nominal width names the solver's
assignment, and each GGUF type spends more in practice: `Q2_0`
stores 2.25 bits per weight, `Q4_0` stores 4.5, and `Q8_0` stores
8.5, block scales included. The budget prices every group at
those real bits, and a passthrough group at the 32 bits the
converter stored it at. The file's `general.file_type` declares `Q4_0`
because that one field cannot name a mixed recipe, so vramfit
writes the type covering the most bytes (74.3 % here) and the
recipe below carries the mix. The allocation:

- 11 `down_proj` expert stacks at nominal 2 (Q2_0, 2.25 bits per
  weight): layers 22, 24, 27, 29, 31, 34, 43, 45, 47, 49, 51.
- The other 35 expert stacks at nominal 4 (Q4_0). The routed
  experts hold 93 % of the parameters, so these two rows are the
  budget.
- 118 dense groups at 8-bit (Q8_0): the token embedding, the
  output head, and every attention, Mamba-2, and shared-expert
  projection. Dense weights are 7 % of the parameters, so the
  8-bit spend is cheap.
- 46 groups pass through unquantized: the Mamba-2 convolutions and
  the router gates, classes llama.cpp's quantizer never touches. The
  recipe records them at nominal 16, and the file holds them at F32,
  the type the converter writes those classes at.

**How the eleven 2-bit stacks are written.** Stock `llama-quantize`
discards the importance matrix when it writes `Q2_0`. vramfit
pre-encodes those eleven tensors itself with an assisted `Q2_0`
encoder that consumes the matrix (ADR-0032), writes them into a
temporary mixed GGUF, and then lets stock `llama-quantize` do the
whole-file pass. Nothing non-standard reaches the output: the file
is ordinary ggml type 42 that stock llama.cpp loads.

What that costs, from this pack's own run log: the whole pack stage
took **8 m 35 s** on a 24-core machine, against about two
minutes for the same recipe with stock `Q2_0` and no pre-encoding.
It also needs about **50 GB of scratch** beside the f16 base, for the
temporary mixed GGUF. A separately instrumented pack of this shape
measured the encoder's peak resident set at 16.22 GB; this run did not
re-measure that figure.

The 2-bit placement follows vramfit's spread placement rule: the
cheap width lands on the stacks the map prices cheapest, spread
across the depth rather than clustered. The same campaign measured
eight alternative placements of the identical width mix — a
spread-map probe, three blind draws, a spread-matched control, a
measured-map arm, a class-wise arm, and a deliberately inverted arm
— and this allocation's damage is the best of the nine, with the
worst at 2.7 times this one. Allocation decides, and the map-ranked
placement wins.

One attribution bound travels with that result. A second
sensitivity map, measured without the importance matrix, agrees
with this ordering through every rank the solve reads and yields
the identical placement. The win credits the damage ranking under
the placement rule, not the importance matrix.

Damage records in the recipe are partial by design: the 46 expert
stacks carry measured marginals, and a pinned or passthrough group
records 0.0 at a width the map never priced. Damage values are one
scan's measurements on one measurement frame. Do not compare them
across scans or across models.

<details>
<summary>Per-group allocation (210 groups, from the recipe)</summary>

| Group | Precision | GGUF type | Bytes |
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
| `model.layers.0.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.0.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.0.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.1.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.1.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.1.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.10.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.10.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.10.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.11.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.11.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.11.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.12.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.12.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.12.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.12.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.13.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.13.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.13.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.14.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.14.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.14.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.15.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.15.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.15.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.16.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.16.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.16.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.17.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.17.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.17.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.18.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.18.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.18.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.19.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.19.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.19.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.19.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.2.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.2.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.2.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.20.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.20.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.20.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.21.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.21.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.21.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.22.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.22.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.22.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.23.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.23.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.23.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.24.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.24.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.24.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.25.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.25.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.25.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.26.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.26.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.26.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.26.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.27.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.27.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.27.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.28.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.28.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.28.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.29.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.29.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.29.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.3.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.3.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.3.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.30.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.30.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.30.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.31.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.31.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.31.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.32.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.32.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.32.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.33.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.33.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.33.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.33.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.34.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.34.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.34.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.35.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.35.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.35.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.36.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.36.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.36.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.37.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.37.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.37.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.38.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.38.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.38.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.39.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.39.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.39.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.4.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.4.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.4.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.40.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.40.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.40.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.41.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.41.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.41.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.42.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.42.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.42.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.42.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.43.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.43.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.43.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.44.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.44.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.44.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.45.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.45.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.45.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.46.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.46.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.46.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.47.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.47.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.47.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.48.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.48.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.48.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.49.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.49.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.49.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.5.mixer.k_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.5.mixer.o_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.5.mixer.q_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.5.mixer.v_proj` | 8 | Q8_0 | 732,599 |
| `model.layers.50.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.50.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.50.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.51.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.51.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.51.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.6.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.6.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.6.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.7.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.7.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.7.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |
| `model.layers.8.mixer.gate` | 16 | F32 | 1,379,009 |
| `model.layers.8.mixer.shared_experts.down_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.8.mixer.shared_experts.up_proj` | 8 | Q8_0 | 10,622,675 |
| `model.layers.9.mixer.conv1d` | 16 | F32 | 98,501 |
| `model.layers.9.mixer.in_proj` | 8 | Q8_0 | 29,487,081 |
| `model.layers.9.mixer.out_proj` | 8 | Q8_0 | 11,721,573 |

</details>

## Evaluation

**Tier 1 (perplexity) and tier 2 (whole-model KL divergence) are the
table above, and both were measured on the exact bytes this
repository carries.** The file was packed, hashed, evaluated, and
uploaded in one sequence on one rented H100 pod on 2026-09-17, and
the pod was deleted afterwards. The reference logits were recorded on
that same pod and build from the same f16 conversion, over 594
held-out WikiText-2 chunks at `n_ctx` 512. The previous revision's
file and bartowski's `IQ2_XXS` were evaluated in the same sequence
against the same logits, which is why the three rows compare.

**Tier 3 did not run for this revision, and this card carries no
tier-3 numbers.** The fixed five-task slice (ADR-0024) is measured
through an lm-evaluation-harness lane, and ADR-0027 forbids reading
a new column against a column from a different instrument — so a
tier-3 table for this file needs both its own column and a re-run
comparator column, about 4.7 hours of rented H100 time, roughly
16 USD. That did not fit this run's budget. The previous revision's
tier-3 result was measured on the previous revision's bytes and
stays with that revision; it is not reproduced here and it does not
describe this file. Read this card's quality claim as tiers 1 and 2
only.

What that costs a reader: tier 2 ranks these three files against each
other on whole-model KL divergence and perplexity over held-out text,
measured on one instrument. It does not certify task accuracy. If
your decision needs task benchmarks, the number this card can give
you is the KL divergence, and the previous revision's tier-3 table —
measured on a file whose mean KLD was 0.204322 against this file's
0.071403 — is the nearest published evidence, on different bytes.

## Reproduce it

The repository ships the recipe, the evals sidecar, and the run
log beside the weights. The importance matrix is bartowski's and
stays in bartowski's repository — the paragraph below the commands
says where.

**Reproduce the pack.** Install the pack extra, then:

```
pip install "vramfit[pack]"
python convert_hf_to_gguf.py <checkpoint dir> \
  --outfile nemotron-30b-a3b-f16.gguf --outtype f16 --no-mtp
vramfit pack recipe.json --llama-cpp <llama.cpp b10362 checkout> \
  --model <checkpoint dir> --base-gguf nemotron-30b-a3b-f16.gguf \
  --imatrix NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf \
  --out NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf \
  --threads <cores>
```

The convert step must pass `--no-mtp`: current converters fold the
MTP block in by default, and this pack's base is the 401-tensor
no-MTP form. The pack step reuses that base, pre-encodes the eleven
2-bit stacks, drives the recorded types into `llama-quantize`, and
reports the margin.

**Pin the quantizer build.** On b10362 this recipe is bit-exact: two
independent runs, on two different rented machines on different days,
produced the identical SHA-256 from the same inputs. A *different*
build gives the same byte count with a different hash and a slightly
different KL divergence, so the build is part of the file's identity,
not an implementation detail. The absolute path you pass to
`--imatrix` also moves the bytes, because the packer stores it in the
GGUF metadata.

**Reproduce the evaluation.** The corpus is the WikiText-2 **raw**
test split, as the single file `wiki.test.raw` from
`wikitext-2-raw-v1` — 1,290,590 B, SHA-256
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
That exact file produced every PPL and KLD on this card. A different
WikiText variant, or the tokenized rather than the raw split, gives
different numbers, so check the digest before comparing:

```
llama-perplexity -m nemotron-30b-a3b-f16.gguf -f wiki.test.raw \
  --kl-divergence-base base-logits.bin
llama-perplexity -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf \
  -f wiki.test.raw --kl-divergence-base base-logits.bin --kl-divergence
```

The reference logits must come from your own f16 conversion of the
same checkpoint revision on the same build. A KL divergence read
against someone else's logits is not this card's number.

The recipe records the allocation that was packed: the 15.776 GiB
weight budget, the nine pins, the 0.002 format overhead, and the
per-group assignments. Its `plan` block is hand-derived and says so
in its own `solver` field, so `vramfit plan` does not re-derive this
file field-for-field — the fit16gib section above states that bound.
`vramfit plan` solves any other budget from the published sensitivity
map with your own `--vram`. The map lives in the linked dataset
repository:
[NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps](https://huggingface.co/datasets/Alberto-Codes/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps).

The importance matrix both this pack and the comparator consumed
is bartowski's, published in
[bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF).
This repository does not carry a copy. Download
[`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf` at the pinned revision](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/f0eec2267ae843d9eb21ea3926ab0046da0a8628/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf)
and keep that filename — the pack stores the imatrix path in the
GGUF metadata, so the name you pass moves the packed bytes.
SHA-256
`fbd36e4fa9be8324062a041ba5cb6247e9f68594168596257a85deb86438aac5`,
55,314,688 B.

## Damage disclosure

Quantization compresses every weight tensor with one uniform lossy
procedure. It does not bypass or disable the base model's safety
training, which ships in these weights at lower precision — and it
can shift any model behavior. The tier-2 table above is the measured
bound on that shift for this file: whole-model KL divergence against
the f16 base over the full held-out set, measured on these exact
bytes. No task-benchmark evidence was measured for this revision.

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

**These numbers were measured on these bytes.** The pack, its
hashing, its evaluation, and its upload ran as one sequence on one
rented pod on 2026-09-17, and the pod was deleted afterwards. The
digest below was computed on that pod after the evaluation and before
the upload, and it equals the SHA-256 the Hub reports for the stored
object. Published equals measured, and you can check both halves
yourself.

**This file is bit-reproducible from the published recipe.** Two
independent runs, on two different rented machines on different days,
built llama.cpp b10362 from source and packed this recipe — and
produced the *identical* SHA-256 at the identical byte count. The two
quantizer binaries themselves hash identically. So the recipe plus the
pinned build plus the pinned inputs determine these bytes exactly, and
a reader who follows the reproduce block can confirm that rather than
take it on trust. The one caveat is that the build is part of the
identity: a *different* quantizer build yields the same byte count
with a different hash and a slightly different KL divergence.

- Packed file SHA-256:
  `187858b04dccae82a8c6fbf8bc5f0a62cfedb21d2f5aef3b4589456b09b6cd75`
  (16,922,476,480 B). Hashed on the pod after the evaluation and
  before the upload; the uploaded object's digest was read back from
  the Hub and matched.
- Base checkpoint:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at revision
  `ce38b6ab8b252b4b8ee7165b4605e93191cafd73`. Upstream `main` has
  moved past this revision — every measured number on this card
  derives from it.
- Instrument: one NVIDIA H100 80GB HBM3, llama.cpp b10362
  (`4801e3c567d5131dd41b387df5f2d4b1370d92be`) built with CUDA,
  `llama-quantize` SHA-256 `bc9c071dfc682f7b05f6f9f6c91e19dd25845ee69197d55237800d79557684f9`,
  `llama-perplexity` SHA-256 `32e3dc1accad67d56c868bd0d827d24eaeab2a1005a5d863f418b16c5d9f70d5`.
- Packer: vramfit `cfda9a2e6056fb873270ddf9656ab10cfd257082` with the
  pack extra.
- The run log beside the weights records the pack events. The evals
  sidecar
  (`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib.gguf.evals.json`)
  records tiers 1 and 2 with the corpus digest. The comparator's
  sidecar sits under `baselines/` with its upstream file name.
- The previous revision of this file, its own numbers, and its
  tier-3 table remain in this repository's commit history at their
  own revision.

Hashes prove identity, not quality. The evidence is the tier-1 and
tier-2 tables above, every number of which traces to the evals
sidecar and the recipe published in this repository.
