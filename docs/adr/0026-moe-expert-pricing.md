# ADR-0026: Routing frequency prices a fused expert stack

- **Status:** Accepted
- **Date:** 2026-08-11 (accepted 2026-08-11)
- **Extends:** [ADR-0016](0016-imatrix-in-the-pack-path.md) (the pack's
  imatrix gains a per-expert reading) and
  [ADR-0021](0021-runtime-frame-measurement.md) decision 4 (an expert
  stack is a group like any other).
- **Note:** [ADR-0020](0020-imatrix-assisted-pricing.md) is superseded.
  This record does not amend it. The rules that survive are ADR-0018's
  scan methods and ADR-0016's pack-side matrix.

## Context

Issue #162 asked how the solver prices an expert that the imatrix
barely fires. Top-6-of-128 routing gives each expert 4.7 % of tokens
on average. Skewed routing was expected to starve the tail below any
usable statistic. ADR-0020 priced maps from imatrix statistics and
ADR-0023 excluded tensors from imatrix use. Neither record anticipated
a tensor with thin statistics rather than none.

Two facts settle the question.

**llama.cpp already prices per expert, with a hard rule and no
threshold.** The loader normalizes each expert's row by that expert's
own count (`tools/quantize/quantize.cpp:196-212`, checkout `e9fa0781`).
A count above zero divides the sums. A count of zero fills the row with
`1`, which is the unweighted fit. Nothing sits between. An expert fired
five times carries the same trust as one fired 400,000 times. The quant
loop then hands each expert its own imatrix slice and the same type
(`src/llama-quant.cpp:1254`). vramfit's loader already copies this rule
(`adapters/outbound/scan/imatrix.py:138`).

**The starved tail does not exist at calibration scale.** Measured
2026-08-11 from bartowski's published GGUF imatrix for
NVIDIA-Nemotron-3.5-Lightning-30B-A3B, read by HTTP range request over
the header and the 46 `.counts` tensors. The run covers 822 chunks of
512 tokens, which is 421,376 tokens.

| Quantity | Count |
|---|---|
| Dense tensor (`ssm_out`, `shexp`, router) | 421,370 |
| Routed expert, mean | 19,752 |
| Routed expert, global minimum | 426 |
| Routed expert, global maximum | 192,191 |
| Cells with a zero count | 0 of 2,944 |

Two cells of 2,944 fall below 10 % of the mean. Both sit in `blk.20`,
the most skewed layer at a 193x spread. Twenty-two cells fall below
25 %. The heavy tail runs upward, toward hot experts, not downward.
The thinnest expert still holds 426 samples per column, which is 1/989
of a dense tensor's coverage. `ffn_down_exps` and `ffn_up_exps` carry
identical counts, so 46 stacks share 23 routing vectors.

A rule that guards the starved case would guard an empty set. A rule
that shrinks thin statistics toward a prior would price a fit the
packer does not ship — the failure ADR-0021 recorded.

## Decision

1. **vramfit adds no coverage threshold.** The solver trusts an
   expert's imatrix statistic whenever its count exceeds zero, and
   falls back to the flat prior at zero. This copies
   `tools/quantize/quantize.cpp:196-212` exactly. The scan frame and
   the pack apply the same weights to the same columns.
2. **Routing frequency weights the stack price.** A stack carries one
   type (ADR-0012, proved for fused experts by #159). The solver
   combines its 128 per-expert prices into that one price in
   proportion to each expert's imatrix count. A rarely-fired expert
   moves the stack's damage in proportion to the tokens it serves.
3. **The counts come from the recipe's own imatrix file.** The weights
   in decision 2 read the `.counts` tensors of the file the pack
   consumes. A recipe that packs against a different matrix than it
   priced against carries a stale weighting, and ADR-0020's
   path-identity warning already reports the mismatch.
4. **The map records per-stack coverage.** An assisted map over a
   fused stack stores the stack's count minimum, median, and maximum.
   The numbers are provenance, not a gate. They let a later data point
   challenge decision 1 with evidence instead of re-deriving the
   distribution.
5. **A zero-count expert is reported, never silently flattened.** The
   pack path files it beside ADR-0016's `imatrix_uncovered` names.
   Today no such expert exists. A shorter calibration run or a
   narrower corpus can produce one, and the user learns before the
   pack, not after.

## Open questions

- Does routing-frequency weighting beat an unweighted mean, packed?
  No data point tests decision 2. The bar mirrors ADR-0019's and
  ADR-0020's: a frequency-weighted recipe must beat an unweighted one
  through the runtime frame, at the same size. Until then decision 2
  is a modeling choice with a mechanism behind it, not a measured win.
- What count floor makes a statistic worthless? The measurement bounds
  the question from one side only. It shows 426 samples is the
  thinnest this model and corpus produce. It does not show 426 is
  enough.
- Does calibration routing match serving routing? Decision 2 assumes a
  stable routing distribution across texts. The counts above come from
  one calibration corpus.

## Consequences

- The pricing rule needs no new machinery. The loader already
  implements decision 1, and decision 2 reads counts the imatrix
  already stores.
- Per-expert statistics stay an input to a per-stack decision.
  ADR-0021 decision 4 still binds: the 12 GiB budget puts about 82 %
  of stacks at `Q2_0`, which is sub-4-bit and needs a runtime-frame
  price.
- Decision 2 needs the per-expert rows the loader currently flattens.
  `load_imatrix` reshapes a stack's weights to one vector, and
  `resolve_assisted_weights` matches by a single row length. Reading a
  fused stack against 128 HF parameters is unbuilt.
- **ADR-0023 cannot reach inside a stack.** `--exclude-weights` matches
  by substring against imatrix entry names
  (`tools/quantize/quantize.cpp:274`), and a 128-expert stack is one
  entry. An exclusion on a fused stack drops all 128 rows. The
  fit-collapse remedy is all-or-nothing on 93.0 % of this model's
  parameters.
- The measurement method is cheap and repeats. Reading a published
  imatrix costs about 200 KB of range requests over the GGUF header
  and its `.counts` tensors. No checkpoint download is needed to learn
  a model's routing distribution.
