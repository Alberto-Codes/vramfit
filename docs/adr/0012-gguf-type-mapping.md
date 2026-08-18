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
- **Amendment (2026-08-16, issue #303):** decision 2 gains a
  pre-run check. Every override must match a tensor the base GGUF
  carries. The backend reads that file's tensor names before it runs
  the quantizer, and it refuses an override matching none. The
  refusal names each unmatched pattern.

    `llama-quantize` compiles each pattern and searches it per
    tensor. A pattern that matches nothing changes no type. The tool
    then exits 0 and reports nothing, so the packed file drops that
    part of the recipe with no signal. Checked against
    `src/llama-quant.cpp` at commit `3653e6d6d` (b10326, the pinned
    instrument) and at `e9fa0781f`: the match loop runs per tensor,
    it never records an unused pattern, and the file carries no
    unused-pattern report. The `--override-tensor` runtime flag
    reports nothing either. A keyword sweep of ggml-org issues and
    pull requests on 2026-08-16 found nothing tracking the gap. That
    sweep read titles and open issues, so it bounds the claim rather
    than settling it.

    The check holds the pattern and never the group's root. Naming
    the roots a pack accepts would contradict the 2026-08-12
    amendment's any-family clause, which is why the check reads the
    file instead. Issue #236 records that reasoning.

    **The check refuses an unmatched override and not a foreign
    root.** A recipe whose groups hang from a root the base GGUF
    carries no tensor for refuses here. A multimodal base GGUF is the
    other case: it carries the decoder at `blk.<n>.` and the vision
    tower at `v.blk.<n>.`. `regex_search` searches a substring, so a
    vision group's `blk.<n>.` pattern matches the decoder's layer
    `<n>` and applies to the wrong tensors. That pack is wrong and
    this check passes it. #236 still owns the root question.

    **The check is a superset of the tool's match set, on purpose.**
    It never refuses a pack the tool would honour. Three upstream
    filters it does not model each let a pattern pass and still
    apply nothing: the first-match-wins order at `:694`, the
    `tensor_allows_quantization` gate at `:675`, and the early
    returns for the embedding and the output head at `:678-683`.
    Re-deriving any of them would make the backend a second source
    of truth for upstream (#190). #305 carries the residual. The two
    flags gained their own check on 2026-08-17, in the #306
    amendment below. No pattern this backend builds reaches any of
    the three today.

    The read needs gguf-py, so every pack carrying at least one
    override requires it. A recipe yielding no override skipped the
    read until 2026-08-17. The #306 amendment below widened that:
    such a recipe now reads whenever it emits a dedicated flag. An
    `--imatrix` pack already required gguf-py (ADR-0026, the #198
    amendment), and #310 carries giving pack a thin extra that does
    not also install torch.

    The refusal reports halt stage `quantize`. Decision 5 lists
    `convert`, `quantize`, and `size_check`, and `convert` does run
    before the quantizer — so the stage list is not the reason. The
    reason is that a new stage amends decision 5, which #275 owns.
    The nearest precedent went the other way: `_read_zero_count_experts`
    is also a pre-quantizer refusal and it reports stage
    `imatrix_counts`, which decision 5 does not list either. #275
    should rule both together.
- **Amendment (2026-08-16, issue #307):** decision 3 gains a report.
  The pack step names every layer the base GGUF numbers that no
  override reaches, in `PackResult.floored_layers`, on the
  `model_packed` run-log event, and as one `warning:` line. The same
  header read serves this and the #303 refusal.

    **Decision 3 supplies the mechanism, not an acceptance policy.**
    It states that a tensor no override covers gets the base type,
    and that `--pure` keeps the file recipe-driven. Its contrast is
    recipe-driven against heuristic-driven. It does not say vramfit
    must accept a recipe that leaves a whole layer unaddressed.
    Decision 3 also predates the case. It dates to 2026-07-28 against
    a dense model the scan priced whole, where decision 2's one
    override per layer group left no layer unreached.

    **So this amendment records the narrow action and leaves the
    wider one open.** The report is strictly additive. It refuses no
    pack the tool would honour, and it ends the silence. Whether an
    unreached layer should instead refuse stays open, and #320
    carries it. Two clauses pull toward refusal. Decision 2 gives an
    untied head its own flag, which "stops `--pure` from dropping the
    head to the recipe's floor" — the record's answer to one unscanned
    unit silently taking the floor was to prevent it. The 2026-08-12
    (#180) amendment refuses a recipe naming two roots, because
    `mtp.layers.<n>` and `backbone.layers.<n>` both map onto
    `blk.<n>.`. That is this defect's mirror in the same namespace.

    The report itself is the ADR-0026 decision 5 shape: a report,
    never a gate.

    **The unit is the layer index and not the tensor.** A layer counts
    as covered when at least one override reaches at least one tensor
    under it. An expert-stack recipe addresses one tensor class per
    layer on purpose, so a per-tensor report would name every
    attention and dense tensor in the model.

    **The packed file grows.** A layer reaches no override only when
    the recipe holds no assignment for it, because `tensor_overrides`
    emits one override per mapped assignment.
    `plan.predicted_total_bytes` sums the assignment sizes, so it
    never counted that layer. The quantizer still writes its tensors
    at the floor. Decision 4's size re-check therefore grows more
    likely to refuse, not less.

    The reach is a base GGUF that numbers more layers than the recipe
    addresses. #256 measured the published 30B builds carrying 48
    expert stacks: 46 backbone plus 2 under `blk.52`. **Our own
    converter drops that block**, so the MTP case reaches vramfit only
    through a base GGUF from another source. A stale or wrong-variant
    base is the trigger with no other gate, because `vramfit pack
    --base-gguf` takes any file and `convert` reuses any existing one.
    That lands on this ADR's open question about verifying the base
    GGUF against the recipe's `model_id`, which stays open.

    The prefix match is anchored at `blk.`, so a vision tower's
    `v.blk.<n>.` reports nothing here. #236 still owns the root
    question, and this report does not pre-empt it.

- **Amendment (2026-08-17, issue #306):** decision 2 gains a second
  pre-run check. A dedicated flag must reach the tensor it binds. The
  backend refuses a scanned `lm_head` group when the base GGUF
  declares no `output.weight`, and an embedding group when it declares
  neither `token_embd.weight` nor `per_layer_token_embd.weight`. The
  refusal names each flag and its targets, on the #303 read. A recipe
  that drives no override reads the file for the flags alone, which
  widens the #303 amendment's gguf-py clause above.

    The two flags carry no pattern. Each binds an exact tensor name
    through `tensor_get_category`, which delegates to two helpers that
    compare with `std::strcmp` (`src/llama-quant.cpp:101-108`). The
    embedding flag accepts either of two names and the output flag
    accepts one. The base GGUF may carry none of a flag's targets.
    That flag then binds nothing, so the early return at `:678-683`
    never fires. The quantizer applies
    nothing, prints nothing, and exits 0. The tensor takes the
    `--pure` floor while `PackResult` records the recipe's type as
    fact. Checked at commit `3653e6d6d` (b10326, the pinned
    instrument) and at `e9fa0781f`. This closes the half the #303
    amendment left open, where it named #306 beside #305.

    **The refusal depends on `--pure`, and decision 3 supplies it.**
    `llama_tensor_get_type_impl` carries a tied-embedding branch at
    `:452`. It applies `--output-tensor-type` to `token_embd.weight`
    whenever `has_tied_embeddings` holds, which is the exact condition
    this refusal fires on. That branch is dead here. `:708` calls the
    function only under `if (!manual && !params->pure)`, and
    `LlamaCppPacker.pack` passes `--pure` on every command. So the
    flag really does apply nothing. A future pack that drops `--pure`
    makes `:452` live and this refusal wrong. Decision 3 makes
    `--pure` the mechanism that keeps a packed file recipe-driven, so
    the dependency is on a ruled property rather than an accident.

    **The tied fallback is exempt, and that is this amendment's
    load-bearing clause.** Decision 2 states that without an `lm_head`
    group the embedding assignment drives the output flag, and that on
    a model that ties embeddings "the flag never applies". That is a
    ruled outcome, so refusing it would refuse a pack this record
    sanctions. It would also refuse every tied model, because such a
    conversion writes no `output.weight` at all. The recorded type
    stays true there: a tied model's head *is* the embedding tensor,
    which took `--token-embedding-type` at that same type. So the
    refusal reads the recipe's groups and never the flag's value.

    **The exemption's test is narrower than its reason, and it still
    holds.** `output_group_type` returns None when the scan measured
    no head group, which is not the same proposition as "the model
    ties embeddings". The two coincide on every reachable pack. On the
    exempt path `output_tensor_type` is non-None only when
    `token_embedding_type` is, so the pack always emits
    `--token-embedding-type` beside the output flag. The early return
    at `:678` then fires for `token_embd` before the output flag is
    read. An untied file reached by the same path carries
    `output.weight`, so the flag binds and nothing refuses.

    **What makes the scanned case different.** The 2026-08-16 (#307)
    amendment reads decision 2's untied-head clause as the record's
    answer to one unscanned unit silently taking the floor: prevent
    it. A base GGUF with no `output.weight` defeats exactly that
    prevention, and no clause addresses the pairing. So it is a
    malformed input rather than a ruled outcome, which is the line
    #309 drew for the exclusion refusal.

    The refusal reports halt stage `quantize`, matching #303. #275
    still owns whether a zero-exit refusal earns its own stage.
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
  artifact. Today a zero-exit tool's warnings are discarded. Narrowed
  by [ADR-0028](0028-expert-stack-type-table.md): the type-fallback
  warning now halts the pack, and the imatrix-miss warning was
  already recorded (ADR-0016). The sidecar question itself stays
  open.

    **Correction (2026-08-16, issue #303):** this question read "for
    example an override pattern that matched no tensor". No such
    warning exists. `llama-quantize` emits nothing for an unused
    pattern, so that case was never a discarded warning. The
    2026-08-16 amendment closes it by reading the base GGUF instead.
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
