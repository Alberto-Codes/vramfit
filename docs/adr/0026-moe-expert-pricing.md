# ADR-0026: Expert pricing trusts any nonzero imatrix count

- **Status:** Accepted, except decision 2
- **Date:** 2026-08-11 (accepted 2026-08-11)
- **Note (2026-08-11):** decision 2 stays **Proposed**. It weights a
  stack's damage by routing frequency, and no packed data point tests
  it. ADR-0019 and ADR-0020 waited under the same bar and both lost
  packed. Decisions 1, 3, 4, and 5 are Accepted. Do not build
  decision 2 against this record. The 2026-08-12 (#167) amendment
  fixes the clause's reach: frequency ranks experts inside a layer,
  and never one layer against another.
- **Extends:** [ADR-0023](0023-imatrix-exclusions.md) decisions 1 and
  4. The amendment bullet lands there when this record is accepted.
- **Note:** [ADR-0020](0020-imatrix-assisted-pricing.md) is superseded.
  This record does not amend it. ADR-0021 decision 1 keeps ADR-0020's
  port in the codebase, and decision 3 below cites it on that basis.

- **Amendment (2026-08-12, issue #167):** decision 2 weights an expert's
  damage by routing frequency. It never said whether that weight ranks
  one layer against another. #167 asked whether a converter that splits
  a layer's experts into two expert stacks pays for itself, and the
  question turned on that reach. Maintainer ruling (2026-08-12):
  **vramfit builds no split until #178 reports a measured damage
  number, and any split lands upstream before vramfit packs against
  it.**

    Decision 2's reach stops at the layer boundary. Each of the 23 MoE
    layers counts exactly 2,528,256 expert firings. Top-6-of-128
    routing fires 6 experts per token in every layer, so the totals
    match by construction. Measured 2026-08-12 from bartowski's
    published GGUF imatrix, by the range-request method in Context
    below. Frequency alone therefore ranks experts inside a layer. It
    never ranks one layer against another. Decision 2 still moves bits
    on one assumption: an expert's damage varies with its count. Under
    that assumption a weighted price reorders the 46 expert stacks.
    #178 tests the assumption.

    A split reaches 1.73 times the routing mass at the same budget.
    The 10.5 GiB budget puts 18.5 % of expert parameters at `MXFP4`.
    Without a split those bits reach 18.5 % of routing mass. A
    per-layer split at the same budget reaches 31.9 %, at a 23.2 %
    floor and a 41.0 % peak. Read the other way, a split needs 9.0 %
    at `MXFP4` for the same routing mass. The expert average then
    falls to 2.43 bits per weight, which returns 0.65 GiB of the
    10.5 GiB budget.

    Those ratios are derived, not measured. The counts are measured.
    The ratios apply an assumed top-k allocation to them. The damage
    cut that follows runs from 7.5 % to 14.8 %, across assumed
    `MXFP4`-to-`Q2_0` damage ratios of 0.5 down to 0.1. That cut is a
    model output. It applies decision 2, which stays Proposed. #178
    carries the measurement.

    Stock llama.cpp quantizes a split file and refuses to serve it.
    `llama_model_quantize_impl` calls `load_hparams` and `load_stats`
    only (`src/llama-quant.cpp:871-908`, checkout `e9fa078`). It
    validates no tensor set, so `--tensor-type` reaches the new names.
    Inference builds a fixed tensor list in `nemotron-h`'s
    `load_arch_tensors` (`src/models/nemotron-h.cpp:33`). The extra
    tensors leave `n_created` below `n_tensors`, and
    `done_getting_tensors` throws (`src/llama-model-loader.cpp:1315`,
    called at `src/llama-model.cpp:1514`). A split forks the model
    definition. It never forks the GGUF format.

    A split also splits its imatrix, or it loses the assistance this
    record prices. `llama-quantize` throws when an imatrix entry's
    size does not match `ne[0]*ne[2]` (`src/llama-quant.cpp:1211`), so
    a stock imatrix aborts a same-named stack. Under a new name the
    lookup logs "did not find weights" at INFO level, then quantizes
    with no imatrix at all (`src/llama-quant.cpp:1198`). `Q2_0` and
    `MXFP4` both return false from `tensor_requires_imatrix`
    (`src/llama-quant.cpp:780-799`), so nothing stops the silent case.

    llama.cpp already carries a two-stack layer. GroveMoE declares
    `ffn_up_chexps`, `ffn_gate_chexps`, and `ffn_down_chexps`
    (`src/models/grovemoe.cpp:57-59`). It then calls `build_moe_ffn`
    twice against one `probs` (`:136-148` and `:153-164`). A graph
    change to `src/models/nemotron-h.cpp` copies that pattern.
    GroveMoE's second stack is no partition of its first. It holds
    fewer and narrower chunk experts, so it prices differently.

    A split costs about 2.5 times the expert traffic per decoded
    token. `ggml_mul_mat_id` gathers `n_expert_used` rows per call and
    carries no per-token mask (`ggml/src/ggml.c:3329-3354`). No
    backend skips a gather by routing weight. Each backend skips only
    an expert that no token selects. A token's 6 selections fall
    across both stacks, so both calls size at 6. Decode then reads 6
    rows at 4.25 bits plus 6 rows at 2.25 bits. Today it reads 6 rows
    at a model-wide average of 2.62 bits. Prefill pays nothing,
    because it reads every expert once at 2.43 bits per weight. The
    #164 pass bar states no throughput clause, so this cost fails no
    gate.

    A split opens no new width. The cut runs along `ne[2]`. Row length
    stays 2688 and 1856, so `tensor_type_fallback` keeps the k-quant
    family unreachable (`src/llama-quant.cpp:373-420`). A split
    targets experts inside the existing palette. vramfit calls
    upstream `convert_hf_to_gguf.py`, so a split patches the converter
    as well as the graph.

    Two is the right expert-stack count, measured and not proved. An
    earlier draft argued that one budget constraint admits at most two
    widths. That argument is false. Each expert takes the width that
    minimizes its own frequency-weighted damage plus a price on bits,
    so the chosen widths spread across the palette. Under a halving
    damage curve at this budget, the unrestricted optimum on `blk.20`
    uses 4 widths, and 2 of them cover 1 expert each. Restricting a
    layer to two widths costs at most 1.1 % of the achievable damage
    cut. Measured 2026-08-12 across three damage curves on all 23
    layers. Width also stays monotone in frequency, so each stack
    holds one contiguous frequency band.

    The frequency distribution is not bimodal. Its bimodality
    coefficient runs from 0.19 to 0.38 on log counts, under a 0.555
    threshold. An exhaustive 2-means fit puts the smaller cluster at 6
    to 64 of 128 experts. The centroid gap measures 2.0 to 4.2 pooled
    within-cluster standard deviations. A 2-means fit always returns
    two clusters, so read that gap against a unimodal null near 2.7.
    The split point is a budget parameter, not a cluster boundary.

    ADR-0019 and ADR-0020 both built on a modeled prior, and both lost
    packed. #166 chose llama.cpp because it is the finest-grained
    target available. A private fork forfeits that property.

- **Amendment (2026-08-12, issue #202):** decision 2 has no input on
  this model. The consequence below recorded #187's read as the input
  the clause needs. That read resolves against checkpoint parameter
  names. transformers fuses this model's routed experts at load, so
  the loaded module reports none of those names.

    `transformers/conversion_mapping.py:1266-1283` (version 5.14.1)
    applies `MergeModulelist(dim=0)` to
    `mixer.experts.*.up_proj.weight` and to the `down_proj` twin.
    `NemotronHExperts` declares both as 3D parameters. The checkpoint
    holds 6,144 indexed expert tensors: 128 experts times 2
    projections across the 23 MoE layers and the MTP block. The
    causal-LM load drops the MTP block. The loaded module holds 46
    fused parameters, named `model.layers.<n>.mixer.experts.up_proj`
    and the `down_proj` twin, shaped (128, 1856, 2688) and
    (128, 2688, 1856).

    Measured 2026-08-12 under transformers 5.14.1. The measurement
    instantiated the model on the meta device from the published
    model config. vramfit resolved the reported names at checkout
    `cea5c4d`.
    `gguf_tensor_name` maps 0 of the 46 routed-expert parameters. It
    maps 140 of 164 dense parameters, so #186's mapping holds.
    `resolve_imatrix_counts` reads a routing frequency for no expert
    on this model.

    `_EXPERT_INDEX` requires an `.experts.<n>.` segment. `_LAYER_PARAM`
    requires a `.weight` suffix. A fused parameter carries neither.

    #177 verified 128 names against the published imatrix. It resolved
    names it constructed, not names the model reports. #163's lane has
    not run, so no session has loaded this checkpoint.

    The same fusion makes `group_key`'s `stack` rule inert here.
    `group_key(name, "stack")` equals `group_key(name, "tensor")` for
    every parameter, at 210 groups each. ADR-0001's #161 amendment
    keys the map on the pack-addressable stack, and the map still
    holds 46 expert stacks on this model. transformers produces those
    46 stacks, not the rule.

    `transformers>=4.56` is an open floor, so the expert layout depends
    on the resolved version. Nothing in vramfit pins it or asserts it.
    #202 carries the version fork.

    Decision 4 stores a stack's count minimum, median, and maximum.
    Those fields read the same counts, so they have no input on this
    model either. #179 builds decision 4 and must read #202 first.

    #202 carries the question of how the scan reads a fused expert
    stack's counts. It blocks #200, which blocks #178. Decision 2
    stays Proposed.

- **Amendment (2026-08-13, issue #202):** the scan reads a fused
  expert stack's counts as one vector, keyed by the loaded parameter
  name. Element `i` is expert `i`'s routing frequency. A dense name
  keeps its scalar chunk tally. The return shape separates the two
  quantities, which rules #193's resolver fork. #193 builds the
  contract, and #179 reads it for decision 4. This amendment also
  resolves the version fork the 2026-08-12 amendment left with #202.

    The read accepts both expert layouts. An indexed 2D parameter
    reads its own row by expert index, as #187 built. A fused 3D
    parameter reads its whole entry as one stack. The read asserts
    that the entry's count length equals the parameter's first
    dimension, and it refuses a mismatch. The meter never constructs
    indexed names for a fused parameter. Constructed names are how
    #177's verification missed the fusion.

    No version pin decides the layout, so vramfit pins none. The
    conversion registry decides it per model class.
    `get_model_conversion_mapping` skips a custom-code model unless
    the user registers it (`conversion_mapping.py:1924-1930`, version
    5.14.1), so such a model loads unfused under transformers 5.
    `save_pretrained` reverses the merge, so checkpoints stay
    indexed. Both layouts persist, and the shape assertion is the
    vouching mechanism, not the floor.

    The read assumes stack order. transformers sorts checkpoint keys
    numerically before `MergeModulelist` stacks them
    (`core_model_loading.py`, `dot_natural_key`, version 5.14.1).
    `convert_hf_to_gguf.py` stacks experts by index
    (`conversion/nemotron.py:406-408`, checkout `e9fa078`). So slice
    `i`, imatrix row `i`, and checkpoint expert `i` name one expert.
    No runtime check detects a permutation, because all 128 slices
    share one shape. #163's lane verifies one loaded slice against
    its checkpoint tensor, tracked on #163.

    Assisted weights refuse a fused stack by rule.
    `resolve_assisted_weights` reports the fused name uncovered, and
    the stacks stay unassisted until a non-k-quant assisted fit
    exists. Today the super-block gate reaches the same outcome by
    arithmetic, because expert rows are 2688 and 1856. The rule makes
    the refusal deliberate. A future fused stack with 256-divisible
    rows must not reach `_matrix_row`'s dense branch.

    `group_key` keeps the `stack` rule's expert-index form. The rule
    is inert on a fused layout and still groups an unfused one. A
    custom-code MoE model reports indexed names under transformers 5,
    so the unfused case stays reachable.

## Context

Issue #162 asked how the solver prices an expert that the imatrix
barely fires. Top-6-of-128 routing gives each expert 4.7 % of tokens
on average. Chart #158 expected skewed routing to starve the tail
below any usable statistic. ADR-0020 priced maps from imatrix
statistics and ADR-0023 excludes tensors from imatrix use. Neither
record anticipated a tensor with thin statistics rather than none.

Two facts settle the question.

**llama.cpp already prices per expert, with a hard rule and no
threshold.** The loader normalizes each expert's row by that expert's
own count (`tools/quantize/quantize.cpp:196-212`, checkout `e9fa078`).
A count above zero divides the sums. A count of zero fills the row
with `1`, which is the unassisted fit. Nothing sits between. An expert
fired five times carries the same trust as one fired 400,000 times.
The quant loop then hands each expert its own imatrix slice and the
same type (`src/llama-quant.cpp:1254`). vramfit's loader already
copies this rule (`src/vramfit/adapters/outbound/scan/imatrix.py:138`).

**The starved tail does not exist at calibration scale.** Measured
2026-08-11 from bartowski's published GGUF imatrix for
NVIDIA-Nemotron-3.5-Lightning-30B-A3B. The read covers the GGUF header
and the file's 185 `.counts` tensors, by HTTP range request. Of those,
46 carry one count per expert and 139 carry one count per dense tensor.

`ffn_down_exps` and `ffn_up_exps` carry identical counts. The 46 stacks
therefore hold 23 distinct routing vectors. That is 23 times 128, which
is 2,944 layer-expert cells rather than 5,888.

Three token totals appear in the file and they differ by under 0.2 %.
The metadata records 822 chunks at a chunk size of 512, which is
420,864 tokens. Each MoE layer's counts sum to 2,528,256, which is 6
times 421,376. The dense tensors count 421,370 rows. The table below
reports the raw counts, not a derived token total.

| Quantity | Count |
|---|---|
| Dense tensor (`ssm_out`, `shexp`, router) | 421,370 |
| Routed expert, mean | 19,752 |
| Routed expert, median | 18,114 |
| Routed expert, global minimum | 426 |
| Routed expert, global maximum | 192,191 |
| Cells with a zero count | 0 of 2,944 |

Two cells of 2,944 fall below 10 % of the mean, at 426 and 823. Both
sit in `blk.20`, the most skewed layer at a 193x spread. Twenty-two
cells fall below 25 %. The heavy tail runs upward, toward hot experts,
not downward. The thinnest expert still holds 426 samples per column,
which is 1/989 of a dense tensor's coverage.

A rule that guards the starved case would guard an empty set. A rule
that shrinks thin statistics toward the unassisted fit would price a
fit the packer does not ship. ADR-0021 recorded that failure.

## Decision

1. **vramfit adds no coverage threshold.** The solver trusts an
   expert's imatrix statistic whenever its count exceeds zero. At a
   count of zero the solver uses the unassisted fit. This copies
   `tools/quantize/quantize.cpp:196-212` exactly. The scan frame and
   the pack apply the same weights to the same columns.
2. **Routing frequency weights an expert's damage inside its stack.**
   *(Proposed. See the header note.)* A stack carries one type, proved
   by #159 and by `src/llama-quant.cpp:1256-1262`, where `new_type`
   sits outside the per-expert loop. The meter prices the stack as one
   cell. Inside that cell, each expert's damage contributes
   in proportion to its imatrix count. The frequency term enters the
   damage total only. It never enters the fit, which stays identical
   to the packer's.
3. **The counts come from the recipe's own imatrix file.** The weights
   in decision 2 read the `.counts` tensors of the file the pack
   consumes. A recipe that packs against a different matrix carries a
   stale weighting. ADR-0020's path-identity warning already reports
   the mismatch, and ADR-0021 decision 1 keeps that port in the
   codebase.
4. **The map records per-stack coverage.** An assisted map over a
   fused stack stores the stack's count minimum, median, and maximum.
   The numbers are provenance, not a gate. They let a later data point
   challenge decision 1 with evidence instead of re-deriving the
   distribution. The fields are additive and optional, so a reader
   that drops them loses provenance and no assignment. ADR-0013's
   silent-drop test does not fire, and `vramfit_schema` holds.
5. **The pack path reports a zero-count expert and never flattens it
   silently.** The report names the stack and the expert index, in a
   field separate from ADR-0016's `imatrix_uncovered`. That field
   names whole tensors, and ADR-0023 fenced it to unintentional gaps.
   A zero-count expert inside a covered stack is a third case. On this
   model and this corpus no such expert exists.

## Open questions

- Does routing-frequency weighting beat an unweighted mean, packed?
  No data point tests decision 2. The bar mirrors ADR-0019's and
  ADR-0020's. A frequency-weighted recipe must beat an unweighted one
  through the runtime frame, at the same size.
- How does the meter attribute damage to one expert inside a stack
  cell? The meter emits one damage number per group. Decision 2 needs
  a per-expert decomposition that does not exist. The chart ruled a
  full per-expert scan out of scope, so the mechanism must come from
  somewhere cheaper. #200 carries the question. Maintainer ruling
  (2026-08-12): **settle the read before ruling the decomposition.**
  The 2026-08-13 (#202) amendment settles the read, so #200 is open
  to rule.
- Does a split of a layer's experts into two expert stacks pay for
  itself? The 2026-08-12 measurement reports 1.73 times the routing
  mass at the same budget. It reports no damage number. #167 carries
  the question and waits on #178.
- What does a two-stack pack cost at decode? The amendment derives
  about 2.5 times the expert traffic per decoded token from
  `ggml_mul_mat_id`'s row count. No run measures it. #167 carries the
  item.
- Does per-expert damage vary apart from routing frequency? Decision 2
  assumes the two move together. A third expert stack pays only when
  they separate. #167 carries the test.
- What count floor makes a statistic worthless? The measurement bounds
  the question from one side only. It shows 426 samples is the
  thinnest this model and corpus produce. It does not show 426 is
  enough.
- Does calibration routing match serving routing? Decision 2 assumes a
  stable routing distribution across texts. The counts above come from
  one calibration corpus.
- ADR-0021 decision 4 blocks the chart's destination. That decision
  bars the solver from buying 2-bit until a runtime-frame price
  exists. Chart #158 needs about 82 % of stacks at `Q2_0`, which is
  2.25 bits. The chart cannot reach 10.5 GiB until the runtime-frame
  lane reports.

## Consequences

- Decision 1 needs no new machinery. The loader already implements it.
- Decisions 4 and 5 add fields to the map and to the pack report.
  Decision 2 adds the frequency term.
- Decision 2 needs the per-expert rows of a fused stack. #187 built the
  read on 2026-08-12. `load_imatrix` reshapes a stack's sums per
  matrix, and `resolve_imatrix_counts` serves the counts against an
  indexed parameter name. **That read reaches no expert on this
  model.** The 2026-08-12 (#202) amendment above measures it, and the
  2026-08-13 (#202) amendment rules the fused read. #193 builds it.
  The clause then waits on #178 for its data point.
- **ADR-0023 cannot reach inside a stack.** `--exclude-weights` matches
  by substring against imatrix entry names
  (`tools/quantize/quantize.cpp:274`), and a fused expert stack is one
  entry. An exclusion on a stack drops all 128 rows. The fit-collapse
  remedy is all-or-nothing on 93.0 % of this model's parameters.
- The measurement method is cheap and repeats. The GGUF header and
  tensor index cost 24 KB, and all 185 `.counts` tensors cost a
  further 24 KB. The method needs no checkpoint download to learn a
  model's routing distribution.
