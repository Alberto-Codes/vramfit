# ADR-0028: Expert stacks map through their own GGUF type table

- **Status:** Accepted
- **Date:** 2026-08-14
- **Amends:** [ADR-0012](0012-gguf-type-mapping.md) decisions 1
  and 5. Decision 1's table keeps every group that is not a
  routed-expert stack. Decision 5's halt stages gain
  `type_fallback`.

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
   | 4 | `Q4_0` | 4.50 | +12.5 % |
   | 2 | `Q2_0` | 2.25 | +12.5 % |

   The backend already recognizes a routed-expert-stack group from
   its name (ADR-0012, 2026-08-12 amendment). Every entry's block
   size divides both 2688 and 1856, so the fallback never fires on
   these rows. Q4_0 takes the 4-bit row over MXFP4 because
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

- Whether Q2_0 holds quality on a bf16 expert stack. Upstream
  validated the type on ternary QAT models only. The runtime-frame
  lane (#40, ADR-0021) measures this before the solver may buy it.
- Whether MXFP4 replaces Q4_0 in the 4-bit row. MXFP4 saves 0.25
  bits per weight and drops the per-expert importance weighting.
  Upstream's `MXFP4_MOE` ftype targets exactly the 3D expert
  tensors, so precedent exists. A runtime-frame comparison on this
  target decides the swap.
- Whether the decision 2 refusal also lands at plan time. Today the
  plan step knows no type table, and pack refuses first.

## Consequences

- The plan step prices an expert-stack group at this table's
  effective bits (ADR-0014): 2.25 at nominal 2, not Q2_K's 2.625.
  Without that entry the size prediction drifts and the ADR-0012
  decision 4 re-check fails late. #228 carries the build.
- ADR-0021 decision 4 stands. This table states what nominal 2
  means on a stack. The solver still buys no 2-bit until a
  runtime-frame price exists.
- On this target the decision 3 scan detects nothing, because
  every table entry's block size divides the stack rows. The scan
  guards every other tensor class and every future target.
- At nominal 2 and 8 the importance matrix does not shape the
  stack quantization. At nominal 4 it does, per expert, through
  `quantize_q4_0`. The imatrix counts keep their provenance role
  (ADR-0026).
- Pack needs a llama.cpp build that carries Q2_0, merged upstream
  2026-07-07. The pinned checkout `e9fa078` carries it (#159).
- The Nemotron-H tensor classes outside this mapping stay with
  #183. `ssm_in` shares the 2688-row k-quant exclusion and waits
  there.
- ADR-0012's open question on persisting toolchain output narrows.
  The fallback warnings now feed a gate. Whether the full output
  persists as a sidecar artifact stays open there.
