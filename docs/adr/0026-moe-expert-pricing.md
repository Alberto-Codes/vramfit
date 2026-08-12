# ADR-0026: Expert pricing trusts any nonzero imatrix count

- **Status:** Accepted, except decision 2
- **Date:** 2026-08-11 (accepted 2026-08-11)
- **Note (2026-08-11):** decision 2 stays **Proposed**. It weights a
  stack's damage by routing frequency, and no packed data point tests
  it. ADR-0019 and ADR-0020 waited under the same bar and both lost
  packed. Decisions 1, 3, 4, and 5 are Accepted. Do not build
  decision 2 against this record.
- **Extends:** [ADR-0023](0023-imatrix-exclusions.md) decisions 1 and
  4. The amendment bullet lands there when this record is accepted.
- **Note:** [ADR-0020](0020-imatrix-assisted-pricing.md) is superseded.
  This record does not amend it. ADR-0021 decision 1 keeps ADR-0020's
  port in the codebase, and decision 3 below cites it on that basis.

- **Amendment (2026-08-12, issue #167):** #167 asked whether a
  converter that splits a layer's experts into two stacks pays for
  itself. This amendment corrects decision 2's premise, records the
  measurement, and holds the build.

    **Every MoE layer draws the same total routing.** Each of the 23
    layers counts exactly 2,528,256. Top-6-of-128 routing fires 6
    experts per token in every layer, so the totals match by
    construction. Measured 2026-08-12 from the same published
    imatrix, by the method this record describes. Frequency alone
    therefore cannot rank one layer against another. Decision 2 keeps
    its lever, because an expert's damage varies with its count, and
    that correlation still reorders the 46 stacks. The lever acts
    through the correlation, never through a layer total.

    **A split raises the lever 1.74 times.** The 10.5 GiB budget puts
    18.5 % of expert parameters at `MXFP4`. Without a split those
    bits reach 18.5 % of routing mass. A per-layer split at the same
    budget reaches 32.2 % on average, at a 23.5 % floor and a 41.3 %
    peak. Read the other way, a split needs 9.0 % at `MXFP4` to reach
    the same mass. That is 2.43 bits per weight, which returns
    0.65 GiB of the 10.5 GiB budget. Under decision 2's own model the
    damage cut runs from 8 % to 15 %, across assumed `MXFP4`-to-`Q2_0`
    damage ratios of 0.5 down to 0.1.

    **That number is a model output, not a measurement.** It applies
    decision 2, which stays Proposed. #178 carries the measurement.

    **Stock llama.cpp quantizes a split file and refuses to serve
    it.** `llama_model_quantize_impl` calls `load_hparams` and
    `load_stats` only (`src/llama-quant.cpp:871-908`, checkout
    `e9fa078`). It validates no tensor set, so `--tensor-type` reaches
    the new names. Inference builds a fixed tensor list in
    `nemotron-h`'s `load_arch_tensors`. The extra tensors leave
    `n_created` below `n_tensors`, and `done_getting_tensors` throws
    (`src/llama-model-loader.cpp:1315`, called at
    `src/llama-model.cpp:1514`). A split forks the model definition.
    It never forks the GGUF format.

    **llama.cpp already carries a two-stack layer.** GroveMoE declares
    `ffn_up_chexps`, `ffn_gate_chexps`, and `ffn_down_chexps`, then
    calls `build_moe_ffn` twice against one `probs`
    (`src/models/grovemoe.cpp`). `build_moe_ffn` accepts `probs_in`
    and `selected_experts_in` for that purpose. A graph change to
    `src/models/nemotron-h.cpp` copies a working pattern.

    **A split costs about 2.5 times the expert traffic per token.**
    `ggml_mul_mat_id` gathers `n_expert_used` rows per call and
    carries no per-token mask. A token's 6 selections fall across both
    stacks, so both calls size at 6. Decode then reads 6 rows at
    4.25 bits plus 6 rows at 2.25 bits. Today it reads 6 rows at the
    layer's single width, which averages 2.62 bits. The #164 pass bar
    states no throughput clause, so this cost fails no gate. GroveMoE
    pays the same shape upstream.

    **A split opens no new width.** The cut runs along `ne[2]`. Row
    length stays 2688 and 1856, so the k-quant family stays
    unreachable. A split buys targeting inside the existing palette.

    **Two stacks is the right count.** The frequency distribution is
    not bimodal. A 2-means fit on log counts puts the smaller cluster
    at 38 to 64 of 128 experts. The centroid gap measures 1.8 to 2.9
    within-cluster standard deviations. That is the shape of one
    heavy-tailed mode. The split point is a budget parameter, not a
    cluster boundary. One budget constraint gives an optimum that uses
    at most two widths, and the chosen experts form a top-k prefix by
    frequency. A third stack pays only when per-expert damage varies
    apart from frequency, which no measurement reports.

    **Ruling.** vramfit builds no split until #178 reports a measured
    damage number. ADR-0019 and ADR-0020 both built on a modeled prior
    and both lost packed. If #178 upholds decision 2, the split lands
    upstream before vramfit packs against it. #166 chose llama.cpp
    because it is the finest-grained target available, and a private
    fork forfeits that property.

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
  somewhere cheaper.
- Does a split of a layer's experts into two stacks pay for itself?
  The 2026-08-12 measurement reports 1.74 times the routing mass at
  the same budget. It reports no damage number. #167 carries the
  question and waits on #178.
- What does a two-stack pack cost at decode? The amendment derives
  about 2.5 times the expert traffic per token from
  `ggml_mul_mat_id`'s row count. No run measures it. #167 carries the
  item.
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
- Decision 2 needs the per-expert rows the loader currently flattens.
  `load_imatrix` reshapes a stack's weights to one vector, and
  `resolve_assisted_weights` matches by a single row length. Reading a
  fused stack against 128 HF parameters is unbuilt.
- **ADR-0023 cannot reach inside a stack.** `--exclude-weights` matches
  by substring against imatrix entry names
  (`tools/quantize/quantize.cpp:274`), and a fused expert stack is one
  entry. An exclusion on a stack drops all 128 rows. The fit-collapse
  remedy is all-or-nothing on 93.0 % of this model's parameters.
- The measurement method is cheap and repeats. The GGUF header and
  tensor index cost 24 KB, and all 185 `.counts` tensors cost a
  further 24 KB. The method needs no checkpoint download to learn a
  model's routing distribution.
