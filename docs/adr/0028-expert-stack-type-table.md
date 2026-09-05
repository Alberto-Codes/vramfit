# ADR-0028: Expert stacks map through their own GGUF type table

- **Status:** Accepted, amended by
  [ADR-0029](0029-plan-independent-size-source.md)
- **Date:** 2026-08-14
- **Note (2026-09-04, issue #362):**
  [ADR-0029](0029-plan-independent-size-source.md) decision 4 adds a
  16 row to `EXPERT_STACK_EFFECTIVE_BITS` (16.0) and
  `EXPERT_STACK_TYPE_BY_BITS` (`f16`). It is the F16 passthrough for
  an uncovered expert-stack group. No such group exists on the #328
  map.
- **Amends:** [ADR-0012](0012-gguf-type-mapping.md) decisions 1
  and 5. Decision 1's table keeps every group that is not a
  routed-expert stack. Decision 5's halt stages gain
  `type_fallback`.
- **Note (2026-08-14, issue #248):** the #229 gate measured both
  open questions on Nemotron 3.5 Lightning 30B-A3B, and both
  resolve below. `Q2_0` does not hold quality post-hoc. MXFP4 does
  not take the 4-bit row. Maintainer ruling (2026-08-14): decision
  1's table stands unchanged, including its nominal-2 row. The
  table states what a nominal width **means** on a stack.
  [ADR-0021](0021-runtime-frame-measurement.md) decision 4 rules
  whether a solver may buy that width, and after its own
  2026-08-14 amendment it bars 2-bit on this target. Record:
  [#229 closing comment](https://github.com/Alberto-Codes/vramfit/issues/229#issuecomment-5300460635).
  **Amended 2026-08-22 (#301): the bar lifts for expert-stack
  groups on this target — ADR-0021's 2026-08-22 amendment.**
- **Amendment (2026-08-20, issue #183):** decision 1's table also
  reaches a layer-class group whose rows refuse the 256
  super-block. The Nemotron-H dense classes qualify at 2688. (The
  2026-09-05 amendment below replaces that class list with the
  measured width, #515.) The
  2026-08-20 amendment to [ADR-0012](0012-gguf-type-mapping.md)
  carries the class table and the F16 pin, and #368 lands the
  build.
- **Amendment (2026-09-05, issue #515):** the measured row width
  decides which groups decision 1's table reaches, and no class
  name does. Maintainer ruling 2026-09-05 on #515, option A. A
  group whose measured rows the 256 super-block does not divide
  takes this table. Every other group takes ADR-0012 decision 1's
  k-quant table, **a routed-expert stack included**. Decision 1
  read "the backend already recognizes a routed-expert-stack group
  from its name", and that name was a proxy for the 30B target's
  2688- and 1856-wide rows. The proxy holds on that target and
  fails elsewhere: Qwen3-Coder-30B-A3B's routed-expert rows are
  2048 and 768, both of which the super-block divides, so the name
  sent 94.95 % of its parameters to this table and banned nominal 3
  there. The name list `SUPER_BLOCK_REFUSED_CLASSES` is deleted.
  The plan reads each group's row width from the size source
  (ADR-0029), and the pack reads the same widths, so the predicted
  bits per weight and the emitted type come from one table. A group
  the routing reaches with no measured width refuses and names
  ``--checkpoint``. It never defaults. The refusal reaches a
  runtime carrying both tables the width routes between, which is
  `llama.cpp` today. A runtime with no table prices at nominal
  bits, where the width selects nothing.
- **Amendment (2026-09-04, issue #232):** decision 1's table gains
  5- and 6-bit rows (5→`Q5_0` at 5.50, 6→`Q5_1` at 6.00 bits per
  weight). Maintainer ruling 2026-09-04. Both block 32, so both
  divide the stack rows of 2688 and 1856. #232 landed the build in the
  backend's expert-stack table and in the domain's effective-bits
  table. Pack emits `Q5_1` and `Q5_0` on a stack, and the plan step
  prices nominal 6 at 6.00 and nominal 5 at 5.50 bits per weight.

## Context

ADR-0012 decision 1 maps nominal bits to k-quant types. Every
k-quant packs 256-element super-blocks, so `llama-quantize` accepts
one only on rows divisible by 256. `tensor_type_fallback` enforces
this after a manual `--tensor-type` override
(`src/llama-quant.cpp:374`, applied at `src/llama-quant.cpp:712`,
checkout `e9fa078`). A rejected override degrades to another type
on a zero exit.

The 30B target's routed-expert stacks carry rows of 2688 and 1856
(#159). Neither divides by 256. The stacks hold 93.0 % of the
parameters. So every k-quant the ADR-0012 table emits is
unreachable there, and the quantizer rewrites it without failing.
Only the table's Q8_0 row survives on the stacks. The community
Q4_K_M build shows the result: zero k-quant expert tensors, Q5_0
on 13 layers, Q8_0 on 11. That breaks ADR-0012 decision 3 (the
packed file is recipe-driven, never heuristic-driven) and moves
packed bytes away from predicted bytes.

Facts verified upstream on 2026-08-14 (#189):

- The fallback warns. `tensor_type_fallback` logs an
  `ncols … not divisible` line and a `falling back to …` line before
  it substitutes. Pack scans the quantizer's output only for
  ADR-0016's imatrix-miss warning and persists none of it, so the
  rewrite is silent at the vramfit boundary only. `run_tool`
  already returns the merged output.
- The types the stack rows accept, in bits per weight: Q2_0 at
  2.25 (block 64), MXFP4 at 4.25 (block 32), Q4_0 and IQ4_NL and
  NVFP4 at 4.5, Q4_1 at 5.0, Q5_0 at 5.5, Q5_1 at 6.0, Q8_0 at 8.5.
  No type lands between 2.25 and 4.25. Q1_0 at 1.125 uses block
  128, which divides 2688 but not 1856, so it reaches half the
  stacks only.
- `quantize_q2_0`, `quantize_mxfp4`, and `quantize_nvfp4` ignore
  the importance matrix. `quantize_q4_0` consumes it, and the
  quantizer slices the matrix per expert inside a stack (#159).
- Q2_0 landed upstream on 2026-07-07
  ([llama.cpp #24448](https://github.com/ggml-org/llama.cpp/pull/24448))
  as a carrier for ternary QAT models. Its quality numbers come
  from weights that were already ternary. No upstream measurement
  covers post-hoc quantization of bf16 weights to Q2_0.
- `--tensor-type` accepts any ggml type name, matched
  case-insensitively, so every palette entry is drivable today.

## Decision

1. **A routed-expert-stack group maps through its own type table.**

   | Nominal bits | Tensor type | Effective bits/weight | Drift |
   |--------------|-------------|----------------------|-------|
   | 8 | `Q8_0` | 8.50 | +6.25 % |
   | 6 | `Q5_1` | 6.00 | +0 % |
   | 5 | `Q5_0` | 5.50 | +10.0 % |
   | 4 | `Q4_0` | 4.50 | +12.5 % |
   | 2 | `Q2_0` | 2.25 | +12.5 % |

   The 6- and 5-bit rows date from the 2026-09-04 amendment (#232).
   The drift column states each type's cost over its nominal width.

   ~~The backend already recognizes a routed-expert-stack group from
   its name (ADR-0012, 2026-08-12 amendment).~~ **Superseded
   2026-09-05 (#515): the measured row width selects this table,
   and a stack whose rows divide 256 takes the k-quant table
   instead.** Every entry's block size divides both 2688 and 1856,
   so the fallback never fires on these rows. Q4_0 takes the 4-bit row over MXFP4 because
   `quantize_q4_0` consumes the importance matrix per expert and
   MXFP4 ignores it.

2. **The backend refuses nominal 3 on an expert stack at pack
   time.** The stack rows accept no type between 2.25 and 4.25
   bits per weight. The refusal names the group, the empty
   2.25–4.25 gap, and both neighboring table entries.

3. **A type-fallback warning halts the pack.** Pack scans the
   quantizer's merged output for the `tensor_type_fallback` warning
   pair. A match halts with exit 1 and keeps the file for
   inspection, matching ADR-0012 decision 4's shape. The
   `pack_halted` event carries stage `type_fallback`, every
   rewritten tensor, and the substituted types. The ADR-0016
   imatrix-miss scan records and continues. This scan halts,
   because a rewritten type breaks the recipe the artifact claims
   to carry.

## Open questions

- ~~Whether Q2_0 holds quality on a bf16 expert stack. Upstream
  validated the type on ternary QAT models only. The runtime-frame
  lane (#40, ADR-0021) measures this before the solver may buy
  it.~~ **Measured (2026-08-14, #229): it does not.** A
  whole-frontier `Q2_0` pack scores **27.9380 PPL** against the f16
  reference's 6.8192, which is **4.097 times**. Mean KLD is
  1.604130, the 99.9th percentile is 12.3312, and the maximum is
  22.206. The pack agrees with the reference's top token on
  **51.65 %** of positions, against 94.42 % for `Q4_0`. That is
  **90.0 times** `Q4_0`'s mean KLD, on one text and one instrument.
  The measurement covers post-hoc bf16 use, which no upstream
  number reached.
- ~~Whether MXFP4 replaces Q4_0 in the 4-bit row. MXFP4 saves 0.25
  bits per weight and drops the per-expert importance weighting.
  Upstream's `MXFP4_MOE` ftype targets exactly the 3D expert
  tensors, so precedent exists. A runtime-frame comparison on this
  target decides the swap.~~ **Measured (2026-08-14, #229): it does
  not. `Q4_0` keeps the 4-bit row.** MXFP4 packs 17,980,129,184
  bytes against `Q4_0`'s 18,898,091,936, which is **4.86 % fewer**.
  Its mean KLD is 0.030277 against 0.017825, which is **1.699
  times**, and its 99.9th percentile is 0.9003 against 0.5416.
  Decision 1's stated rationale survives measurement, because
  `quantize_q4_0` consumed the per-expert importance matrix and
  `quantize_mxfp4` ignored it. One number runs the other way.
  MXFP4's maximum KLD is 3.183 against `Q4_0`'s 4.703, so its worst
  token is better while its bulk is worse.
- Whether the decision 2 refusal also lands at plan time. Today the
  plan step knows no type table, and pack refuses first.

## Consequences

- The routing now reads a measured width, so the plan needs the
  size source for every layer-class and routed-expert-stack group
  it prices under a runtime carrying this table (noted 2026-09-05,
  #515). `plan` without ``--checkpoint`` refuses such a map and
  names the flag. A map of whole-layer groups alone is unaffected,
  because a layer group holds several row widths and never took
  this table. A runtime with no effective-bits table is unaffected
  too, because it prices at nominal bits.
- `TensorSize` gains a `rows` field (2026-09-05, #515). ADR-0029
  decision 5 named `dtype` and `bytes` only, so the shapes the
  shard headers carry reached no caller. The safetensors adapter
  already parsed the shape and dropped it.
- The plan step prices an expert-stack group at this table's
  effective bits (ADR-0014): 2.25 at nominal 2, not Q2_K's 2.625.
  Without that entry the size prediction drifts and the ADR-0012
  decision 4 re-check fails late. #228 carries the build.
- **Superseded 2026-08-14 (#248):** ADR-0021 decision 4 now buys a
  width against a measured bar. This table still states what nominal
  2 means on a stack. The runtime-frame price arrived and refused the
  width on this target (#229). **The #249 campaign then priced the
  mixed use and the bar lifted for expert-stack groups, 2026-08-22
  (#301).**
- On this target the decision 3 scan detects nothing, because
  every table entry's block size divides the stack rows. The scan
  guards every other tensor class and every future target.
  **Confirmed 2026-08-14 (#229) by reading the packed files rather
  than the log.** All 46 expert stacks carried the recipe's type in
  each of the three gate packs, at `Q2_0`, `Q4_0`, and MXFP4.
  MXFP4's block of 32 divides both 2688 and 1856, so it fires no
  fallback either. The packs stayed pod-side and the pod is
  terminated, so the #229 record carries that check and no file on
  the box repeats it.
- The decision 3 scan did not run during the #229 gate, and #247
  carries the reason (noted 2026-08-14). `run_tool` decoded the
  quantizer's merged output as strict UTF-8, and llama.cpp
  truncates its `tokenizer.ggml.merges` preview inside a character.
  The decode raised before any scan read the output. **#247 landed
  (PR #250), so the scan now runs and the workaround is retired
  (noted 2026-08-15).** `run_tool` replaces an undecodable byte
  instead of refusing it.
- A replaced byte can delete a decision 3 match, and #252 measured
  how far that reaches (2026-08-15). It reaches no scanned line.
  llama.cpp truncates only inside its `- kv` dump loop, and a
  warning line never passes through it. Splitting a warning would
  need a second writer. Pipe-write atomicity does not supply that
  guarantee, because the warning spans two or three separate log
  writes (`src/llama-quant.cpp:381`, `:415`, `:418`).
  Single-threadedness supplies it. `llama-quantize` installs no
  threaded log callback, and it picks every tensor type in a
  metadata pass that ends before the first quantize worker starts.
  The #252 measurement record carries the method and the trial
  counts.
- The nominal-2 row now means a width the solver may not buy on
  this target (noted 2026-08-14). The empty band from 2.25 to 4.25
  bits per weight therefore separates the only width that fits a
  10.5 GiB budget from the cheapest width that holds quality. That
  width packs at 17.600 GiB. #249 carries what the campaign does
  about the gap. The 2026-09-04 amendment (#232) adds the 5- and
  6-bit rows, and neither falls inside the band.
- At nominal 2 and 8 the importance matrix does not shape the
  stack quantization. At nominal 4 it does, per expert, through
  `quantize_q4_0`. The imatrix counts keep their provenance role
  (ADR-0026).
- Pack needs a llama.cpp build that carries Q2_0, merged upstream
  2026-07-07. The pinned checkout `e9fa078` carries it (#159).
- The Nemotron-H tensor classes outside this mapping stay with
  #183. `ssm_in` shares the 2688-row k-quant exclusion and waits
  there. **Resolved 2026-08-20 (#183): decision 1's table also
  reaches a layer-class group whose rows refuse the 256
  super-block. The Nemotron-H classes qualify at 2688, so the
  wait ends. The 2026-08-20 amendment to
  [ADR-0012](0012-gguf-type-mapping.md) carries the class table.**
- ADR-0012's open question on persisting toolchain output narrows.
  The fallback warnings now feed a gate. Whether the full output
  persists as a sidecar artifact stays open there.
