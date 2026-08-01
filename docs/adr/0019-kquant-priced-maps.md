# ADR-0019: Sub-4-bit recipes solve on kquant-priced maps

- **Status:** Proposed
- **Date:** 2026-07-31

## Context

The seventh evidence data point is a two-probe duel over the 49B
target's residual 0.62 PPL gap to the size-matched baseline
(bartowski Q3_K_S, 8.532 PPL / 0.1584 mean KLD at the 20.47 GiB
weight budget).

**The granularity probe lost.** The baseline's only within-layer
protection is `attn_v` at `Q5_K` over flat `Q3_K` — and only 10 of
the recipe's 35 2-bit layers have attention tensors at all (the NAS
architecture makes layers 42–70 FFN-only). Holding `attn_v` at
`q4_k` in those 10 layers recovered 0.05 PPL for 18.75 MiB. Adding
`attn_output` at `q3_k` recovered 0.088 PPL total (9.068 vs 9.156)
and 27 % of the KLD gap (0.2367 vs 0.2653) for 87 MiB. The
within-layer lever saturates far below the 0.62 PPL gap.

**The frame-honesty probe hit, with the sign inverted.** Sixteen
cells re-measured on the 65,536-token frame with the K-quant-faithful
method (ADR-0018) show RTN *over-prices* low-bit damage: 2.0–3.9x at
2-bit, 1.05–1.7x at 3-bit, per-cell and non-uniform. The solver
bought every sub-4-bit assignment at those wrong prices. Distorted
2-bit prices also feed the 2-bit membership problem — the additive
model's failure mode (ADR-0006, fourth measurement).

## Decision (proposed)

1. **A scan that can feed sub-4-bit assignments runs with
   `--within-group kquant`.** RTN remains the default for
   exploratory scans and stays valid for recipes at 4 bits and
   above, where its distortion is small.
2. **The 49B target re-scans at 65,536 tokens with the kquant
   method**, precisions {8, 4, 3, 2} — cell-for-cell comparable
   with the RTN map of the same frame.
3. **The solver changes nothing.** Honest prices arrive through the
   map. The plan step already consumes whatever damages the map
   records.
4. **The re-priced recipe walks the full loop** — plan, validate
   (a super-additive result is a solve-again signal, ADR-0006),
   pack with the importance matrix, smoke, and the same two
   evaluation tiers against the same baselines.

## Open questions

- Does honest 2-bit pricing move the recipe toward the baseline's
  flat-3-bit region, and does the packed result close the 0.62 PPL
  gap? The re-scan and re-plan answer this.
- Does the validation pass with kquant perturbation stay
  sub-additive on the re-planned recipe? The pass must run with the
  same within-group method as the map that priced it —
  `quantfit validate --within-group kquant`. The recipe does not
  record its map's method, so the pairing is the caller's duty
  today.
- Whether RTN should ever price a 3-bit cell again (1.05–1.7x
  distortion straddles the harmless-to-material range).

## Consequences

- Scan wall-clock rises: K-quant fitting costs more than RTN
  (~5.5–7.5 minutes per 65,536-token cell against ~4 for RTN on the
  reference box). The full 328-cell re-scan prices at ~30–35 hours.
- Maps must not mix methods across cells. `scan.within_group` is
  map-level, and the fingerprint separates checkpoints (ADR-0018).
- The RTN 65,536-token map stays on disk as the comparison
  artifact for the eighth data point.
