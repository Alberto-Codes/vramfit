# ADR-0019: Sub-4-bit recipes solve on kquant-priced maps

- **Status:** Proposed
- **Date:** 2026-07-31
- **Note (2026-08-02):** the first full measurement contradicts
  the decision as stated. The kquant-priced recipe validated
  sub-additive by 2.0x. Packed, it lost to the RTN recipe (9.251
  against 9.156 PPL) and widened the baseline gap from 0.62 to
  0.72 PPL (the eighth data point). Matching the pack's
  super-block structure did not close the frame leak. The likely
  amendment is imatrix-aware pricing (ADR-0018's first open
  question) — this record stays Proposed until a kquant-priced
  recipe beats an RTN-priced one packed.
- **Note (2026-08-06):** the likely amendment ran and failed too.
  The imatrix-assisted map's recipe packed worse still (9.607 PPL,
  the tenth data point). The elimination ledger now covers
  granularity, super-block structure, the evaluation set, and
  imatrix assistance. K-quant-faithful pricing improved the frame's
  arithmetic and worsened the packed artifact — the leak is the
  frame's transfer to the runtime. No measurement has met this
  record's bar (a kquant-priced recipe beating an RTN-priced one
  packed). Status disposition belongs with ADR-0020's, in the
  same decision.

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

- ~~Does honest 2-bit pricing move the recipe toward the baseline's
  flat-3-bit region?~~ **No — the prediction was wrong in
  direction (2026-08-02).** The full kquant map re-prices 2-bit
  ~40 % cheaper relative to 3-bit than RTN priced it. The raw
  cross-process medians (0.74 at 2-bit, 1.28–1.43x at 8/4/3-bit)
  carry the ~20 % frame offset, which cancels in relative prices.
  The re-plan moves 52 of 82 groups to 2-bit (was 35) and buys
  four 8-bit groups up front. The packed result did not close the
  0.62 PPL gap — it widened to 0.72 (the eighth data point).
- ~~Does the validation pass with kquant perturbation stay
  sub-additive on the re-planned recipe?~~ **Yes — sub-additive by
  2.0x** (measured 0.0610 against predicted 0.1221, ADR-0006 fifth
  measurement). The breadth was wider than the recipe that went
  super-additive 11.9x on RTN prices. The pass must run with the
  map's method — `quantfit validate --within-group kquant`. The
  recipe does not record its map's method, so the pairing is the
  caller's duty today. **Closed (2026-08-04):** the recipe now
  records its map's `within_group` token, `validate` resolves the
  frame from it, and a contradicting flag is refused. Recipes
  written before the field carry no record — the pass warns and
  the pairing stays the caller's duty for those only.
- Whether RTN should ever price a 3-bit cell again. The probe's
  in-frame 3-bit distortion (1.05–1.7x) and the full map's raw
  cross-process ratio (1.28x median, opposite side of 1) bracket
  the answer inside the frame-noise band.

## Consequences

- Scan wall-clock rises: K-quant fitting costs more than RTN
  (~5.5–7.5 minutes per 65,536-token cell against ~4 for RTN on the
  reference box). The full 328-cell re-scan prices at ~30–35 hours.
- Maps must not mix methods across cells. `scan.within_group` is
  map-level, and the fingerprint separates checkpoints (ADR-0018).
- The RTN 65,536-token map stays on disk as the comparison
  artifact for the eighth data point.
