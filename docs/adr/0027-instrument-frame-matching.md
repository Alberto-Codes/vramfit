# ADR-0027: Damage numbers compare only within one instrument

- **Status:** Accepted
- **Date:** 2026-08-14
- **Extends:** [ADR-0021](0021-runtime-frame-measurement.md)
  decision 3. This record resolves that ADR's open question on where
  the lane runs.

## Context

The #163 instrument check scanned the 30B target on a rented H100 and
re-measured 10 hold-out cells on the reference 4090. The cells span
all four precisions and every group kind the map holds — embedding,
Mamba-2, attention, MoE, and the head. Three results
([#40 evidence](https://github.com/Alberto-Codes/vramfit/issues/40#issuecomment-5291714600)):

- The 4090 reproduces itself bit-exactly across two runs. Its
  same-instrument noise is zero.
- The H100 differs from the 4090 by 0.3 % to 10.6 % relative per
  cell. The worst cell is the shallow MoE layer group
  `model.layers.1` at 4-bit. The four cells with damage below 0.04
  agree within 2.1 %.
- Sorting both columns by damage gives the same order on all 10
  cells. The instruments disagree on magnitude, not on rank.

The largest gaps sit on the high-damage cells. That is error
amplification through a perturbed forward pass, not random noise.

Vendor documentation says this disagreement is expected, not a
defect. cuBLAS guarantees bitwise repetition only on GPUs with one
architecture and one streaming-multiprocessor count, under one
toolkit version with a single stream
([cuBLAS introduction](https://docs.nvidia.com/cuda/cublas/)). The
H100 is Hopper at 132 SMs. The 4090 is Ada at 128 SMs. The pair sits
outside the guarantee. PyTorch states results are not reproducible
across platforms, and may differ between CPU and GPU execution
([reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)).
The 4090 run held most groups on host RAM under its 6 GiB ballast
cap, so its numerics mix CPU and GPU paths. llama.cpp's own GPU
perplexity moves with batch size while CPU holds constant
([llama.cpp #3014](https://github.com/ggml-org/llama.cpp/issues/3014)),
so the runtime frame shares the property.

One cross-hardware acceptance precedent exists. MLPerf's closed
division accepts 99 % or 99.9 % of reference accuracy
([submission rules](https://github.com/mlcommons/inference/blob/master/Submission_Guidelines.md)).
That band governs end-task accuracy. Applied per cell to damage, it
would reject the #163 map at its 10.6 % worst cell. MLPerf never
compares intermediate metrics across hardware — each submitter
measures on its own silicon. The #220 search found no published
sensitivity method that states a cross-hardware tolerance. A
numeric band here would be an invented number.

The glossary already commits the shape: damage values compare only
within one measurement frame. This record adds silicon to the
frame's identity and rules how a rented map earns trust.

## Decision

1. **The instrument joins the frame.** An instrument is the GPU
   model, its streaming-multiprocessor count, the torch build, and
   the offload split. The torch build string carries the CUDA
   toolkit, as `2.13.0+cu130` does. Two instruments differ when any
   part differs. A resident run's offload split is empty. So two
   resident pods on one GPU model, SM count, and torch build are
   one instrument, and decision 5 confirms that identity by
   measurement.
2. **A map's instrument governs the map's comparisons.** The
   solver, the validation pass, and any within-map ranking read one
   map measured on one instrument. Validation runs frame-matched to
   its map, as ADR-0021 already requires.
3. **Magnitudes never cross instruments. Orderings may.** A
   decision never mixes two instruments' damage magnitudes. The
   #163 hold-out measured the same shape between these two
   instruments that the #200 amendment records for slice
   overstatement: a monotone transform preserves ordering and
   loses absolutes.
4. **A map measured on a rented instrument passes an ordering bar
   before it prices anything.** The reference box re-measures about
   10 hold-out cells, spanning every precision and every group
   kind the map holds. The box ordering must reproduce the map
   ordering. Decision 3 licenses the comparison, because the bar
   reads orderings and never magnitudes. Two cells count as one
   rank when their map damages differ by less than 2 % of the
   larger damage. The band rounds the 2.1 % low-damage agreement
   the #163 check measured. No absolute tolerance exists. The check's
   result lands on the owning ticket.
5. **An instrument measures its own noise floor before its numbers
   price anything.** This generalizes ADR-0021 decision 3 beyond
   the runtime frame. The 4090 measured zero on 2026-08-14 (#163).
   The H100 measured zero on 2026-08-14 (#220). A second rented
   pod re-ran the 10 hold-out cells by checkpoint resume. All 10
   damage values reproduced bit-exactly. Both pods ran torch
   2.13.0+cu130 on an H100 SXM at 132 SMs, resident. The repeat
   pod's driver is 580.126.09, and the #163 pod's driver went
   unrecorded. The two pods measured as one instrument, which
   confirms decision 1's identity across rentals.

## Open questions

- The 4090's offload confound. The #163 hold-out ran under a 6 GiB
  ballast cap with most groups on host RAM. One 10-cell run at a
  second cap separates the offload split from the silicon. A
  bit-identical second-cap result would drop the split from
  decision 1's list, by amendment. Until it runs, the 4090
  instrument's identity includes its cap.
- Whether a later rental needs its own floor run. Decision 5
  measured two pods as one instrument, a sample of two. Decision
  4's bar already re-measures every rented map on the box, so the
  floor question costs nothing while the bar stands.
- The ordering bar's size. Ten cells spanning the precisions and
  group kinds come from #163's selection. No rule sizes the
  hold-out.
- Whether the map records its instrument. No artifact field names
  the GPU, the torch build, or the offload split today. Adding one
  is a schema decision this record does not take.

## Consequences

- The #163 H100 map prices the 30B scan-frame decisions. ADR-0021
  decision 4 still bars 2-bit purchases until the runtime frame
  reports.
- The #210 probe runs on the rented instrument. A probe produces a
  ranking, and decision 3 lets orderings cross.
- The serve test's measured damage stays a box number and stays
  the final bar (#164). It compares against its own frame's
  reference, never against a scan-frame prediction from another
  instrument.
- ADR-0021's open question on where the lane runs closes: the lane
  runs rented, and rented magnitudes never compare with box
  magnitudes.
- The glossary's measurement-frame entry gains the instrument, and
  an **instrument** entry lands beside it.
