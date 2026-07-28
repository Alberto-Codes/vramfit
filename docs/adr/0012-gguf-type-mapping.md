# ADR-0012: The GGUF backend maps nominal bits to K-quant types

- **Status:** Proposed
- **Date:** 2026-07-28

## Context

ADR-0010 routes sub-4-bit serving through llama.cpp and leaves one
question open: which GGUF types map to the recipe's nominal bits.
Recipes assign nominal precisions {8, 4, 3, 2} per group. GGUF has no
exact 4-, 3-, or 2-bit type. Every quantization type spends a
fractional number of effective bits per weight on block scales.

`llama-quantize` accepts `--tensor-type PATTERN=TYPE` overrides.
Verified against llama.cpp source (commit e9fa078, 2026-07-28):

- The pattern is a lower-cased ECMAScript regex. The tool matches it
  against GGUF tensor names with `regex_search`. The first matching
  override wins.
- A matched override replaces the built-in per-tensor heuristics.
- `--token-embedding-type` binds the embedding tensor before any
  pattern.
- `--pure` disables the heuristic type mixing for tensors no override
  covers.

GGUF tensor names differ from Hugging Face names. Layer tensors live
under `blk.<n>.` and the embedding under `token_embd`. An unescaped
dot makes `blk.1.` also match `blk.11.` — patterns must escape dots.

I-quants (the IQ families) need an importance matrix as input. The
scan does not produce one today. K-quants need no extra input.

## Decision

1. **A fixed table maps nominal bits to K-quant types.**

   | Nominal bits | GGUF type | Effective bits/weight | Drift |
   |--------------|-----------|----------------------|-------|
   | 8 | `Q8_0` | 8.50 | +6.25 % |
   | 4 | `Q4_K` | 4.50 | +12.5 % |
   | 3 | `Q3_K` | 3.44 | +14.6 % |
   | 2 | `Q2_K` | 2.63 | +31.25 % |

   K-quants only in v1 — an i-quant table waits for the scan to emit
   the importance matrix as a calibration byproduct. That part of
   ADR-0010's open question stays open.
2. **One override per layer group.** The group `model.layers.<n>`
   becomes the override `blk\.<n>\.` = type, dots escaped. The group
   `model.embed_tokens` becomes `--token-embedding-type` **and**
   `--output-tensor-type`. On a model that ties embeddings
   (Qwen2.5-3B does), the output flag never applies. On a model with
   an untied output head (the 49B target), the head takes the
   embedding's precision — without the flag, `--pure` would drop it
   to the recipe's floor with no warning. The v1 backend rejects
   tensor-level groups with a clear error.
3. **The base type is the recipe's floor, applied with `--pure`.**
   The quantizer's positional type argument speaks ftype names, not
   tensor-type names, so the floor maps through a second table:
   8→`Q8_0`, 4→`Q4_K_S`, 3→`Q3_K_S`, 2→`Q2_K` (`Q4_K` as an ftype
   aliases `Q4_K_M`, and `Q3_K` is not an ftype). `--pure` stops the
   heuristic mixing, so a tensor no override covers gets exactly the
   base type — the packed file is recipe-driven, never
   heuristic-driven.
4. **Pack re-checks real bytes.** Effective bits exceed nominal bits
   by 6-31 %, so predicted sizes undershoot. Pack stats the output
   file and reports the margin against `plan.weight_budget_bytes`.
   An over-budget result exits 1 and keeps the file for inspection.
5. **Pack emits six run-log events**, resolving the pack part of
   ADR-0011's open question: `pack_started`, `gguf_converted`,
   `model_packed`, `size_checked`, `pack_finished`, `pack_halted`.
   Halt events carry the stage (`convert`, `quantize`, `size_check`)
   and the error.

## Consequences

- The plan step's `--format-overhead 0.05` under-predicts a 4-bit
  group's real size by ~7 %. The re-check catches the drift. If the
  drift starts breaking budgets, the sensitivity map grows measured
  per-precision byte counts (ADR-0010's second open question).
- K-quant super-blocks need row sizes divisible by 256. When a row is
  not, `llama-quantize` falls back per tensor and warns. Every
  Qwen2.5-3B weight row divides by 256.
- The pack step needs a llama.cpp checkout with built tools and a
  Python able to import torch for `convert_hf_to_gguf.py`. Both stay
  outside the package — the adapter drives them as subprocesses, and
  the base install stays torch-free (ADR-0005).

## Open questions

- The i-quant table, once the scan emits an importance matrix.
- Whether pack persists the toolchain's own output as a sidecar
  artifact. Today a zero-exit tool's warnings (for example an
  override pattern that matched no tensor) are discarded.
- Whether the solver should consume per-type effective-bit tables
  instead of one `format_overhead` fraction (ties to the
  runtime-capability milestone from ADR-0010).
- Whether pack should verify the base GGUF matches the recipe's
  `model_id` (today the caller vouches for the pairing).
