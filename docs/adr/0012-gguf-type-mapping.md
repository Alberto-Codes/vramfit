# ADR-0012: The GGUF backend maps nominal bits to K-quant types

- **Status:** Accepted, amended by
  [ADR-0013](0013-runtime-capability-in-recipes.md),
  [ADR-0022](0022-within-layer-protections.md), and
  [ADR-0028](0028-expert-stack-type-table.md)
- **Date:** 2026-07-28 (accepted 2026-07-28)
- **Amendment (2026-07-28):** the type tables in decisions 1 and 3
  gain 6- and 5-bit rows (6→`Q6_K`, 5→`Q5_K`, base ftype
  5→`Q5_K_S`). Everything else stands.
- **Amendment (2026-07-29):** decision 2 changes. An `lm_head`
  group drives `--output-tensor-type` with its own assignment. The
  embedding assignment stands in only when the scan measured no
  head. The first untied-head pack (the 49B target) forced this.
- **Amendment (2026-07-29, second):** the size re-check in decision 4
  is not a sufficient acceptance gate. A 3-bit-heavy 49B recipe
  passed the size check and produced a destroyed artifact (PPL ~10⁶
  on two backends, payloads finite). Pack output needs a smoke test —
  a few perplexity chunks — before anything downstream trusts it.
  Implementation is an open question below.
- **Amendment (2026-08-08):** decision 2 gains within-layer
  protections ([ADR-0022](0022-within-layer-protections.md)). A
  recipe's resolved (tensor, precision) pairs become per-tensor
  overrides, placed before the group overrides — the quantizer
  applies the first matching pattern. The backend still rejects
  tensor-level *groups*: the boundary moved for protections only.
- **Amendment (2026-08-12, issue #180):** decision 2 gains a second
  group shape, drops a fixed prefix, and gains one refusal.

    The backend derives the layer index from the group name. Any
    layer group ending in `.layers.<n>`, `.h.<n>`, or `.blocks.<n>`
    becomes `blk\.<n>\.`. GGUF numbers every layer `blk.<n>.`
    whatever the checkpoint calls it. Matching `model.layers.<n>`
    alone refused the Nemotron 3.5 Lightning target at
    `backbone.layers.<n>` (#160).

    The embedding group gains the same treatment. Decision 2 fixed
    it at `model.embed_tokens`, and the target names it
    `backbone.embeddings`. The backend now carries both names. The
    output head stays the literal `lm_head`, which the target
    carries verbatim.

    A routed-expert stack group becomes its fused tensor:
    `blk\.<n>\.ffn_up_exps\.`, `blk\.<n>\.ffn_down_exps\.`, or
    `blk\.<n>\.ffn_gate_exps\.`. llama.cpp fuses one layer's routed
    experts into a single 3D tensor. That tensor carries one
    quantization type, so the expert stack is the unit a pack
    addresses (#159, ADR-0001 as amended by #161).

    Expert-stack overrides go before layer overrides, and both go
    after the protection overrides. The quantizer applies the first
    matching pattern, and `blk\.1\.` also matches
    `blk.1.ffn_up_exps.weight`. Order is priority: per-tensor, then
    expert stack, then layer.

    **Every mapped group must hang from one parameter-tree root.**
    A free prefix cannot tell a decoder layer from any other layer
    stack. The target carries `mtp.layers.<n>` beside
    `backbone.layers.<n>`, and a multimodal checkpoint carries a
    vision tower that GGUF names `v.blk.<n>.`. Both would map onto
    `blk.<n>.` and lose one assignment to the first-match rule. The
    backend refuses a recipe naming two roots, and names both.

    The backend still refuses every other group, and it names the
    group. It refuses tensor-level groups. It refuses any tensor
    class outside this mapping — the Mamba `in_proj`, `out_proj`,
    and `conv1d`, the attention projections, the router, the shared
    experts. Issue #183 carries those.

- **Amendment (2026-08-14, ADR-0028):** decision 1's table does not
  reach a routed-expert stack. The quantizer rejects every k-quant
  on the expert rows and substitutes on a zero exit. Expert-stack
  groups map through [ADR-0028](0028-expert-stack-type-table.md)'s
  table, pack halts on the quantizer's type-fallback warning, and
  decision 5's halt stages gain `type_fallback`.

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

   | Nominal bits | Tensor type | Effective bits/weight | Drift |
   |--------------|-------------|----------------------|-------|
   | 8 | `Q8_0` | 8.50 | +6.25 % |
   | 4 | `Q4_K` | 4.50 | +12.5 % |
   | 3 | `Q3_K` | 3.44 | +14.6 % |
   | 2 | `Q2_K` | 2.63 | +31.25 % |

   These are tensor-type names, driven through `--tensor-type` and
   `--token-embedding-type` (the quantizer matches them
   case-insensitively). K-quants only in v1 — an i-quant table waits
   for the scan to emit the importance matrix as a calibration
   byproduct. That part of ADR-0010's open question stays open.
2. **One override per layer group.** The group `model.layers.<n>`
   becomes the override `blk\.<n>\.` = type, dots escaped. The group
   `model.embed_tokens` becomes `--token-embedding-type`. The group
   `lm_head`, scanned on models with an untied head, becomes
   `--output-tensor-type` with its own assignment (amendment
   2026-07-29). Without an `lm_head` group the embedding assignment
   drives the output flag. On a model that ties embeddings
   (Qwen2.5-3B does) the flag never applies. On an untied model
   with no scanned head the flag stops `--pure` from dropping the
   head to the recipe's floor. The v1 backend rejects tensor-level
   groups with a clear error.
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
  Python able to run `convert_hf_to_gguf.py`. Both stay outside the
  package — the adapter drives them as subprocesses, and the base
  install stays torch-free (ADR-0005). The `pack` extra provisions
  the convert interpreter (torch, transformers, sentencepiece)
  without adding a single import to vramfit.

## Open questions

- ~~The i-quant table, once the scan emits an importance matrix.
  **Escalated 2026-07-29:** the control experiment traced ~81 % of
  the 49B head-to-head perplexity gap to the baseline's importance
  matrix (see
  [evaluating packed models](../explanation/evaluating-packed-models.md)).
  This question now gates the north-star claim.~~ The imatrix half
  is resolved by [ADR-0016](0016-imatrix-in-the-pack-path.md): pack
  consumes one via `--imatrix`, K-quants first. The i-quant type
  table itself stays open there.
- Whether pack persists the toolchain's own output as a sidecar
  artifact. Today a zero-exit tool's warnings (for example an
  override pattern that matched no tensor) are discarded. Narrowed
  by [ADR-0028](0028-expert-stack-type-table.md): the type-fallback
  warning now halts the pack, and the imatrix-miss warning was
  already recorded (ADR-0016). The sidecar question itself stays
  open.
- ~~Whether the solver should consume per-type effective-bit tables
  instead of one `format_overhead` fraction (ties to the
  runtime-capability milestone from ADR-0010).~~ Resolved by
  [ADR-0014](0014-per-type-effective-bits.md): it should, and does.
- Whether pack should verify the base GGUF matches the recipe's
  `model_id` (today the caller vouches for the pairing).
- ~~Where the post-pack smoke test lives (second 2026-07-29 amendment):
  inside `vramfit pack` behind a flag, or as the first step of the
  evaluation tier. The destroyed-artifact evidence is in
  [evaluating packed models](../explanation/evaluating-packed-models.md).~~
  Resolved by [ADR-0017](0017-post-pack-smoke-test.md): inside
  `vramfit pack`, behind `--smoke-text`.
