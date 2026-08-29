# ADR-0030: Vision serves through a projector sidecar and a measured budget line

- **Status:** Accepted
- **Date:** 2026-08-29 (accepted 2026-08-29)
- **Amendment (2026-08-29, issue #419):** the falsification campaign
  delivered the measured vision-quality bound. Decision 5's interim
  clause now applies to unmeasured targets only. A measured target's
  card states its bound. Maintainer ruling 2026-08-29. See
  "Amendment: the measured vision bound" below.
- **Origin:** Maintainer ruling 2026-08-29 on #236, with #419's
  interim clause folded in. Chart #441 indexes the image lane.
  Evidence lives in #236's checks comments (2026-08-28, corrected
  2026-08-29) and #419's checks, literature, and ruling comments.
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
reserve, and an image-encode transient. The encode allocation fails
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

2. **The mmproj ships as an unquantized projector sidecar.** The
   artifact ships the vendor mmproj beside the decoder GGUF,
   byte-identical. This defers rather than refuses. `llama-quantize`
   takes an mmproj at b10362. On this file Q4_K_M frees ~0.50 GiB
   (1,145.08 → 628.96 MiB), with fallback on every BF16 tensor, at
   an unmeasured vision-quality cost. The sidecar stays unquantized until #419
   delivers a vision-quality instrument that prices the trade. #442
   carries the build.

3. **The weight budget subtracts the measured vision line when the
   model card claims vision.** The line is 1,600 MiB on this target.
   It comes from the serve ladder, never from the mmproj file size —
   the file is 1.118 GiB and the serving cost is larger. A card that
   claims no vision subtracts nothing and states the absence. #442
   carries the build.

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
   delivers a measured bound.

## Open questions

- **The vision-quality bound itself.** Answered 2026-08-29. The
  campaign delivered the bound (#419's checks comment). See
  "Amendment: the measured vision bound" below.
- **The mmproj's own precision.** Q4_K_M frees ~0.50 GiB, less than
  the vision line, at unmeasured quality. Measure it once #419's
  instrument works.
- **The vision line on an unmeasured target.** The 1,600 MiB line is
  this target's measurement. No clause says whether the budget warns
  or refuses on a vision-capable target with no measured line.
- **The image token cost is resolution-dependent.** The mmproj header
  bounds pixels at 92,160–645,120. One point is measured: 256 tokens
  at 768×768. The cost curve across resolutions is unmeasured.

## Consequences

- **The image-capable headline becomes stateable.** kv9 serves one
  image at 61,440 tokens — the count QAT reaches text-only at the
  load bar. Against QAT serving one image at 40,960, kv9 gains
  20,480 tokens (+50 %).
- **A budget that honors the vision line covers the encode
  transient.** An under-budgeted encode fails: the QAT arm returned
  an HTTP 500 at 45,056 context, and one kv9 run at 65,536 died
  without writing a response.
- **The pack gains a second output artifact.** The sidecar reaches
  publication, hashing, and upload beside the decoder GGUF. It costs
  its full 1.118 GiB on disk and in distribution until #419 prices
  the alternative.
- **The sidecar seam stays as built.** `ship_sidecar` and
  `config_claims_vision` stay bare outbound functions, and the
  sidecar ships before the smoke test (maintainer ruling 2026-08-29
  on #444). #445 explores folding the seam into the port
  architecture.
- **#208 builds to decision 1.** A sole-foreign-root recipe is
  outside the decoder-only scope. The mechanism that recognizes one
  stays #208's to design, per #236's 2026-08-16 correction.

## Amendment: the measured vision bound (2026-08-29, issue #419)

### Context

Decision 5 held a claim gate: no vision-quality claim without an
image-conditioned measurement. #419's falsification campaign ran on
2026-08-29 and measured the gap (#419's checks comment). Three
decoder arms served through the same BF16 mmproj, so decoder
quantization was the only variable. The bf16 decoder generated
greedily over 10 held-out 768×768 images. Each quantized arm was
teacher-forced on the reference sequence.

The metric is a truncated top-20 KLD in nats at the serve boundary.
The server caps `n_probs` at 20, so this is not ADR-0021's full-set
KLD. Position classes separate image-grounded `content` tokens from
channel-frame policy (`markup`, `pos0`). Frame-policy positions
dominate all-position means on this channel-locked target, so the
content class carries the vision claim.

Content-class results (n = 120):

| Arm | Mean KLD | p95 KLD | Top-token agreement |
|---|---|---|---|
| kv9 (text-solved, 14.92 GiB) | 0.0045 | 0.0239 | 99.2 % |
| Vendor QAT Q4_0 | 0.0373 | 0.1928 | 97.5 % |

The instrument noise floor is 1.07e-4 mean KLD, measured by
teacher-forcing kv9 on its own greedy sequence (99.4 % agreement).
kv9 measures 8.3× below QAT and 42× above the floor. The bf16
reference and kv9 each answered 10 of 10 held-out questions on
their own paths.

### Ruling

Maintainer ruling 2026-08-29 on #419: the campaign delivers the
bound decision 5 waited on. Decision 5 now reads:

- A text-measured sensitivity map licenses no vision-quality claim
  on an **unmeasured** target. That card states that the map
  measured text damage only, and states the vision line the budget
  reserved.
- A **measured** target's card states its bound: the metric, the
  content-class numbers, and the noise floor. On this target the
  bound is content-class truncated top-20 KLD 0.0045 mean and
  0.0239 p95, with 99.2 % top-token agreement against a 1.07e-4
  floor.
- The bound is scoped to the measured target and metric. No target
  inherits another's bound. Visual and text token sensitivities
  diverge across targets (#419's literature comment).
