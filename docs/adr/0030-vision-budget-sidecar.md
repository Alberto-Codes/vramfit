# ADR-0030: Vision serves through a projector sidecar and a measured budget line

- **Status:** Accepted
- **Date:** 2026-08-29 (accepted 2026-08-29)
- **Amendment (2026-08-29, issue #419):** the falsification campaign
  delivered the measured vision-quality bound. Decision 5's interim
  clause now applies to unmeasured targets only. A measured target's
  card states its bound. Maintainer ruling 2026-08-29. See
  "Amendment: the measured vision bound" below.
- **Amendment (2026-08-31, chart #441):** the published sidecar
  swaps to Q4_K_M. The vision line re-measured at 960 MiB with the
  Q4_K_M sidecar aboard. Maintainer ruling 2026-08-31. See
  "Amendment: the sidecar swaps to Q4_K_M" below.
- **Origin:** Maintainer ruling 2026-08-29 on #236, with #419's
  interim clause folded in. Chart #441 indexes the image lane.
  Evidence lives in #236's checks comments (2026-08-28, corrected
  2026-08-29) and #419's checks, literature, campaign-result, and
  ruling comments.
- **Instrument:** llama-server b10362 Vulkan prebuilt, RTX 4090
  24 GiB, `-ngl 99 -np 1`, 4,096-token ladder rungs, serve-one-image
  bar. Every number below carries that frame unless it names another.

## Context

A multimodal checkpoint carries a vision tower beside the decoder. No
record said whether the tower enters the weight budget or the pack
map (#236). The gap had a live symptom: #367 measured a recipe's
protections landing on the wrong tensors under a second root.

The toolchain already splits the towers. The converter emits a
decoder-only GGUF unless the caller asks for the projector file. The
tower ships as its own `mmproj` file, and `llama-server` serves
images through it at b10362. The Gemma 4 31B mmproj is 1.118 GiB —
190 BF16 and 166 F32 tensors, roots `v.` and `mm.`.

Refusal is falsified as the default. The kv9 pack served an image
correctly with zero new code. The decoder pack and the vendor mmproj
compose at the runtime.

Quantizing the mmproj is possible but degraded at b10362. The session
measured it on 2026-08-29 with the instrument's own `llama-quantize`.
Q8_0 refuses: `v.blk.0.ffn_down.weight` has 4,304 columns, Q8_0 needs
a multiple of 32, and Q8_0 defines no fallback. Q4_K_M succeeds at
628.96 MiB (9.16 effective bits) with fallback on 190 of 356 tensors.
The vision-quality cost is unmeasured, and the literature marks
vision components as more quantization-sensitive than the decoder
(#419's 2026-08-29 literature comment).

The vision cost is measured, not estimated. Serving with the mmproj
aboard costs 1,022.8 MiB of weights, a 150.63 MiB CLIP compute
reserve, and an image-encode transient (2026-08-28 frame). The
encode allocation fails
at 65,536 context and passes at 61,440. On the serve ladder that
total displaces 20,480 context tokens of global KV, five ladder rungs
at the ruled 81,920 bytes/token: **1,600 MiB**. One 768×768 image
consumes 256 decoder tokens, measured at the server. The checkpoint
config claims 280 (`vision_soft_tokens_per_image`).

## Decision

1. **The scan and the pack quantize one decoder stack.** The
   sensitivity map prices decoder groups only, and the recipe
   addresses decoder tensors only. The #367 refusal and ADR-0029's
   root table already practice this on the pack and plan paths. This
   record makes it policy.

2. **The mmproj ships as a projector sidecar, unquantized by
   default.** The
   artifact ships the vendor mmproj beside the decoder GGUF,
   byte-identical. This defers rather than refuses. `llama-quantize`
   takes an mmproj at b10362. On this file Q4_K_M frees ~0.50 GiB
   (1,145.08 → 628.96 MiB), with fallback on every BF16 tensor, at
   an unmeasured vision-quality cost. The sidecar stays unquantized until #419
   delivers a vision-quality instrument that prices the trade. #442
   carries the build. **The 2026-08-29 amendment below proves the
   instrument. #451 priced the mmproj trade on 2026-08-30 — see
   open question 2. Amended 2026-08-31: the swap amendment below
   restates this decision and swaps the published sidecar.**

3. **The weight budget subtracts the measured vision line when the
   model card claims vision.** The line is 1,600 MiB on this target
   with the BF16 sidecar (2026-08-28 frame). It comes from the
   serve ladder, never from
   the mmproj file size —
   the file is 1.118 GiB and the serving cost is larger. A card that
   claims no vision subtracts nothing and states the absence. #442
   carries the build. **The 2026-08-31 amendment below re-measures
   the line at 960 MiB with the Q4_K_M sidecar aboard.**

4. **The caller supplies `--tokens-per-image` from the measured image
   token cost.** 256 tokens on this target. The measured cost wins
   over the config's claim of 280. Capacity arithmetic uses the
   instrument's number (#419's checks comment, and the precedent of
   pricing KV growth at the runtime's measured allocation). The
   capacity readout keeps its contract: the caller supplies the cost,
   and the readout divides by it.

5. **A text-measured sensitivity map licenses no vision-quality
   claim.** A vision-capable model card states that the map measured
   text damage only, and states the vision line the budget reserved.
   This is #419's interim clause, practiced on #423 and codified
   here. The clause stands until #419's falsification campaign
   delivers a measured bound. **Amended 2026-08-29 by the #419
   amendment below.**

## Open questions

- **The vision-quality bound itself.** Answered 2026-08-29. The
  campaign delivered the bound (#419's campaign-result comment). See
  "Amendment: the measured vision bound" below.
- **The mmproj's own precision.** Answered 2026-08-30. The
  maintainer authorized the arm on 2026-08-30, recorded on #451.
  The measurement ran at the amendment's frame with the kv9 decoder
  held fixed. Both arms measure divergence from the amendment's
  BF16 reference. The Q4_K_M mmproj (628.96 MiB) reaches 0.0050
  mean content-class truncated top-20 KLD (n = 120). The same kv9
  decoder behind the BF16 mmproj reaches 0.0045. Both arms hold
  99.2 % top-token agreement,
  and the all-position means sit at 0.0483 against 0.0489, over a
  1.07e-4 noise floor. It frees 482 MiB of VRAM at load. At `-np 3`
  with 4,096-token slots it served three of three on 2026-08-30's
  ceiling (#450, #451). The maintainer ruled the swap on
  2026-08-31 — see "Amendment: the sidecar swaps to Q4_K_M" below.
- **The vision line on an unmeasured target.** The measured line is
  this target's alone (960 MiB with the shipped sidecar, 2026-08-31
  frame). No clause says whether the budget warns
  or refuses on a vision-capable target with no measured line.
- **The swap default on a future measured target.** The 2026-08-31
  ruling covers this artifact. No clause says whether a future
  target's measured trade swaps without a ruling.
- **The image token cost is resolution-dependent.** Measured
  2026-08-30 on #450. The cost is 49 tokens at 256×256, 121 at
  512×512, and saturates at 256 for square inputs at and above
  768×768. A 1280×720 input costs 264. The measured inputs span
  65,536 to 921,600 pixels, and the server accepted the inputs
  outside the header's 92,160–645,120 pixel bounds.

## Consequences

- **The image-capable headline becomes stateable.** kv9 serves one
  image at 61,440 tokens — the count QAT reaches text-only at the
  load bar. Against QAT serving one image at 40,960, kv9 gains
  20,480 tokens (+50 %). Both figures carry the 2026-08-28 frame.
- **A budget that honors the vision line covers the encode
  transient.** An under-budgeted encode fails: the QAT arm returned
  an HTTP 500 at 45,056 context, and one kv9 run at 65,536 died
  without writing a response.
- **The pack gains a second output artifact.** The sidecar reaches
  publication, hashing, and upload beside the decoder GGUF. It cost
  its full 1.118 GiB on disk and in distribution until the
  2026-08-31 swap ruling. The Q4_K_M sidecar costs 629 MiB.
- **The sidecar seam stays as built.** `ship_sidecar` and
  `config_claims_vision` stay bare outbound functions. The sidecar
  ships after the reconstruction gate and before the smoke test
  (maintainer ruling 2026-08-29 on #444). A smoke-failed run
  therefore keeps a shipped sidecar beside the kept decoder. #445
  explores folding the seam into the port architecture.
- **#208 builds to decision 1.** A sole-foreign-root recipe is
  outside the decoder-only scope. The mechanism that recognizes one
  stays #208's to design, per #236's 2026-08-16 correction.

## Amendment: the measured vision bound (2026-08-29, issue #419)

### Context

Decision 5 held a claim gate: no vision-quality claim without an
image-conditioned measurement. #419's falsification campaign ran on
2026-08-29 and measured the vision-quality divergence between two
packed arms and a BF16 reference (#419's campaign-result comment).
Three decoder arms served through the same BF16 mmproj. The BF16
reference decoder (61.4 GB) ran on CPU at `-ngl 0`. Both quantized
arms ran on the RTX 4090 at `-ngl 99`. Decoder quantization is the
only variable between the two quantized arms.

The reference arm is a second instrument under ADR-0027 decision 1.
ADR-0027 decision 3 bars magnitudes from crossing instruments. The
numbers below therefore stand as divergences from the reference
distribution, never as same-instrument damage numbers. The
kv9-against-QAT ratio compares two arms on one instrument.

The metric is a truncated top-20 KLD in nats at the serve boundary.
The server caps `n_probs` at 20, so this is not ADR-0021's full-set
KLD. The BF16 reference generated greedily over 10 held-out 768×768
images. The harness teacher-forced each quantized arm on the
reference sequence. Position classes separate image-grounded
`content` tokens from channel-frame policy (`markup`, `pos0`).
Frame-policy positions dominate the all-position means on this
channel-locked target. kv9's all-position mean is 0.0489 over 178
positions, 10.9x the content-class figure. The content class
carries the vision claim.

Content-class results (n = 120):

| Arm | Mean KLD | p95 KLD | Top-token agreement |
|---|---|---|---|
| kv9 (text-solved, 14.92 GiB) | 0.0045 | 0.0239 | 99.2 % |
| Vendor QAT Q4_0 | 0.0373 | 0.1928 | 97.5 % |

The instrument noise floor is 1.07e-4 mean KLD, measured by
teacher-forcing kv9 on its own greedy sequence (99.4 % agreement).
kv9 measures 8.3x below QAT and 42x above the floor. The BF16
reference and kv9 each answered 10 of 10 held-out questions
correctly on their own paths.

### Decision

Maintainer ruling 2026-08-29 on #419 (ruling comment, live
exchange): the campaign delivers the bound decision 5 waited on.
Decision 5 now reads:

- A text-measured sensitivity map licenses no vision-quality claim
  on an **unmeasured** artifact. That card states that the map
  measured text damage only, and states the vision line the budget
  reserved.
- A **measured** artifact's card states its own bound: the metric,
  the position class, the content-class numbers beside the
  all-position mean, and the noise floor. The kv9 pack (14.92 GiB)
  on Gemma 4 31B measures content-class truncated top-20 KLD 0.0045
  mean and 0.0239 p95, at 99.2 % top-token agreement, against a
  1.07e-4 floor and a 0.0489 all-position mean.
- The bound belongs to the measured artifact and metric. A
  different recipe on the same checkpoint measures its own bound or
  states none. The literature reports a sensitivity gap between
  visual and language tokens (#419's literature comment), and no
  measurement covers transfer between artifacts.

### Consequences

- The kv9 card may state the measured bound. #446 carries the
  publication decision this unblocks.
- The #451 session measured the mmproj arm on 2026-08-30, under
  the authorization open question 2 reserved. Open question 2
  carries the numbers.
- `content class` and `position class` enter the glossary with this
  amendment.

## Amendment: the sidecar swaps to Q4_K_M (2026-08-31, chart #441)

### Context

#451 priced the mmproj precision trade on 2026-08-30 — see open
question 2. The maintainer ruled the swap on 2026-08-31 on chart
#441 in live exchange. The #451 deferral bound the execution:
decision 3's line was measured with the BF16 sidecar aboard, so the
line re-measures with the Q4_K_M sidecar before any record reuses
it.

An instrument fault delayed the measurement one day. Under the
2026-08-30 OS image (bazzite 44.20260827) the text ladder failed
16,384 tokens below the published boundary, and a 1 GiB KV-cache
buffer refused with the device mostly free. A reboot did not clear
the fault. The box rolled back to image 44.20260820 on 2026-08-31.
The published 81,920 text rung reproduced before any new number
counted (465 MiB free at load). Kernel BAR1 mapping-reuse asserts
(`NVRM: dmaAllocMapping`) accompany failed Vulkan allocations in
both frames, so the assert does not identify the fault. The
depressed ceiling does.

The ladder boundary also moves with the box's idle VRAM share. The
2026-08-31 frame recorded 23,629–23,631 MiB free before each load
(device 24,564 MiB), and every boundary below carries that frame.
The 2026-08-28 frame recorded no idle figure. Its text boundary
sits one rung lower, and its image boundaries sit further below.
The bullets under the table decompose the image movement.

### The re-measured serve boundaries (2026-08-31 frame)

Five ladders ran on 2026-08-31 in one frame: 4,096-token rungs,
`-ngl 99 -np 1`, KV cache f16, 23,629–23,631 MiB free before each
load.

| Arm | Boundary | Fail rung (mode) |
|---|---|---|
| kv9 text-only | 86,016 | 90,112 (load) |
| kv9 + Q4_K_M mmproj, one image | 73,728 | 77,824 (encode, 19 MiB free) |
| kv9 + BF16 mmproj, one image | 69,632 | 73,728 (load, two tries 20 s apart) |
| QAT Q4_0 text-only | 65,536 | 69,632 (load) |
| QAT Q4_0 + BF16 mmproj, one image | 49,152 | 53,248 (encode, 44 MiB free) |

- The swap buys 4,096 tokens on the same decoder and day: 73,728
  against 69,632.
- The Q4_K_M line is 12,288 displaced tokens: **960 MiB** at the
  ruled 81,920 bytes/token. The same-context load delta at 73,728
  is 772 MiB, and the encode transient covers the rest.
- The BF16 line measures 1,280 MiB on both decoders in this frame
  (16,384 displaced tokens). Decision 3's 1,600 MiB carried the
  2026-08-28 frame. The line quantizes to rung granularity and
  moves with the frame.
- This pack's one-image boundary moved three rungs from the
  published 61,440. One rung came from the frame, one from the
  BF16 line's move, and one from the sidecar conversion.
- The boundary generation check passed at 86,016 with five decoded
  tokens.
- The 77,824 encode failure at 19 MiB free re-confirms the
  ~200 MiB encode headroom rule.

### Decision

Maintainer ruling 2026-08-31 (chart #441, live exchange): the
published sidecar swaps to Q4_K_M.

- The artifact ships the kv9 decoder beside the converted mmproj:
  `gemma-4-31B-it-mmproj-q4km.gguf`, 628.96 MiB, llama-quantize
  b10362 from the vendor BF16 projector. The recipe name labels
  the command, not the contents. Every quantizable tensor falls
  back on this geometry. The file's header reads 150 Q5_0, 13
  Q8_0, 27 F16, and 166 F32 tensors at 9.16 effective bits per
  parameter.
- Decision 2 now reads: the mmproj ships as a projector sidecar.
  This artifact's sidecar ships at the measured Q4_K_M conversion.
  An unmeasured target's sidecar ships unquantized until a
  measurement prices its trade.
- Decision 3's line on this target becomes 960 MiB with the
  shipped sidecar aboard. A record that reuses a vision line names
  the sidecar and the frame.
- The shipped configuration's quality bound comes from #451:
  content-class mean KLD 0.0050, p95 0.0193, 99.2 % top-token
  agreement against the campaign's BF16 reference.

### Consequences

- The card restates its serve table in the 2026-08-31 frame and
  names both frames.
- The model repo replaces the BF16 sidecar with the Q4_K_M file.
  The vendor repository carries the identical BF16 bytes — the
  shipped file was vendor-verbatim and hash-equal.
- The sidecar's distribution cost drops from 1.118 GiB to 629 MiB.
- The mmproj file name gains a precision marker and the serve
  commands change.
