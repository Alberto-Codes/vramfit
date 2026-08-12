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
