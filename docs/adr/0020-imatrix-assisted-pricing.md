# ADR-0020: The meter prices K-quant cells with the pack's imatrix

- **Status:** Proposed
- **Date:** 2026-08-03
- **Note (2026-08-03):** the 16-cell probe reported — the
  re-pricing clears the 2x gate, per-cell and structure-dependent
  (decision item 5). The full assisted re-scan is justified. The
  record stays Proposed until an assisted-priced recipe beats an
  unassisted one packed, the same bar ADR-0019 waits under.
- **Note (2026-08-04):** the CLI wiring landed — `scan --imatrix`
  and `validate --imatrix`, the `kquant-imx` token, the map's
  `scan.imatrix` field, and the run-log coverage split. The recipe
  carries its map's token and imatrix path, `validate` refuses a
  contradicting frame and warns on a different imatrix file, and
  `pack` warns when an assisted recipe packs without the map's
  file. File identity compares by resolved path — evidence, not
  proof of content. The fingerprint gained a trailing imatrix
  field, so checkpoints written before this change do not resume —
  none were in flight.

## Context

The eighth evidence data point eliminated the within-group method as
the frame leak. The kquant-priced recipe matched the pack's
super-block structure cell for cell. Packed, it still lost to the
RTN-priced recipe (9.251 against 9.156 PPL) and widened the baseline
gap to 0.72 PPL.

One structural difference remains between the scan frame and the
packed artifact's quantizer. `llama-quantize --imatrix` fits every
covered tensor with per-column activation weights
(`quantize_row_q2_K_impl` and kin, llama.cpp checkout e9fa078). The
kquant method (ADR-0018) ports only the unassisted reference path.
The meter therefore prices a format the pack never ships.

The imatrix factor is violently allocation-dependent — 0.86 PPL on
one packed recipe, 0.07 on another (the fifth data point). An
unassisted map plausibly overvalues 2-bit breadth: assistance
recovers more damage exactly where the unassisted fit is worst.

## Decision (proposed)

1. **The kquant method gains the weighted fit paths.** A new scan
   module ports `quantize_row_q2_K_impl`, `quantize_row_q3_K_impl`,
   and `quantize_row_q4_K_impl` (checkout e9fa078) to torch,
   including `make_qkx3_quants`, `make_qx_quants`, and
   `make_qp_quants`. The element weight is
   `qw[i] * sqrt(sigma2 + x[i]^2)`, per the C. `Q8_0` stays
   unweighted — `quantize_q8_0` discards the imatrix.
2. **An imatrix adapter feeds the meter.** It reads the GGUF imatrix
   (`in_sum2 / counts` per column, the `llama-quantize` load
   formula) and maps GGUF tensor names to HF parameter names.
   A parameter without imatrix coverage takes the unassisted path —
   the C behavior for a NULL imatrix row. `token_embd` is never
   covered and always falls back.
3. **The meter takes optional per-tensor imatrix weights.** The
   argument is valid only with the kquant method. RTN with an
   imatrix has no C counterpart and is refused.
4. **The port verifies against the C with a non-NULL imatrix.** A
   committed harness drives `ggml_quantize_chunk` through ctypes
   with real weight vectors and records golden fixtures. The
   unweighted fixtures from ADR-0018 stay untouched.
5. **A 16-cell probe gates the full re-scan.** The probe re-measures
   cells spanning the kquant recipe's 52-group 2-bit set on the
   65,536-token frame, assisted against unassisted, in one process.
   A per-cell re-pricing above 2x justifies the ~30-35 h assisted
   re-scan. This record stays Proposed until the probe reports.

## Open questions

- ~~Does imatrix assistance change *relative* prices across cells,
  and by how much at 2 against 3 bits?~~ **Measured (2026-08-03,
  16 in-process pairs on the 65,536-token frame).** Assistance
  cuts in-frame damage to 0.47–0.67 of unassisted on
  attention-bearing 2-bit cells (median 0.55) and 0.83–0.99 on
  FFN-only 2-bit cells (median 0.90) — a 1.65x relative tilt the
  unassisted map bakes into every 2-bit membership decision. At
  3 and 4 bits the factor spans 0.40–0.95 per cell. No scalar
  rescale reproduces assisted prices. The two RTN sanity cells
  read 2.7x and 4.1x their stored map values (previous bound
  ~20 %) — cross-process absolute damages do not transfer, and
  only in-process comparisons are load-bearing. Full numbers in
  the ninth data point
  (../explanation/evaluating-packed-models.md).
- ~~The eval-set control: the meter scores damage on calibration
  text, the tiers score wiki.test. If the kquant recipe wins in-set
  and loses on wiki, the meter's objective needs held-out text and
  no pricing change can show up in the tiers.~~ **Measured
  (2026-08-03, tier-2 KLD on calibration text, fresh f16 base): the
  mismatch distorts PPL but not the ranking.** By PPL the kquant
  recipe wins in-set (8.673 against 8.829) and loses on wiki — PPL
  orderings do not transfer across texts. By mean KLD — the meter's
  own damage metric — the kquant recipe loses on both texts (0.611
  against 0.561 in-set) and the baseline wins on both (0.454
  in-set, with an imatrix from a different dataset). The frame leak
  survives text matching, so this probe is unconfounded.
- ~~The method token and map field for assisted scans (`kquant-imx`
  in the fingerprint, an imatrix provenance field beside
  `scan.within_group`), and the `scan --imatrix` flag. Deferred to
  the full-loop change — a probe does not need a CLI.~~
  **Resolved (2026-08-04).** `scan --imatrix` selects assisted
  pricing (kquant only), the map and the fingerprint record the
  `kquant-imx` token with the imatrix path, and the checkpoint
  cannot mix with an unassisted scan's (ADR-0006's rule). The
  recipe records its map's method token and imatrix path, and
  `validate --imatrix` refuses a frame that contradicts the token —
  the pairing ADR-0019 left to the caller is now enforced. Imatrix
  file identity compares by resolved path and surfaces as a
  warning in `validate` and `pack`: path equality is evidence, not
  proof of content.
- Whether the solver should ever consume an unassisted kquant map
  again once assisted maps exist.

## Consequences

- The scan frame can price the format the pack actually ships —
  assisted `Q2_K`/`Q3_K`/`Q4_K` — closing the last structural gap
  short of runtime numerics.
- The scan grows a soft dependency on the imatrix artifact: an
  assisted map is only comparable to the pack it predicts when both
  consume the same imatrix file. Provenance must ride in the map.
- The weighted fit costs more than the unassisted one
  (37 candidate steps against 16 for `Q2_K` sub-blocks). Forward
  passes still dominate cell time at 49B scale.
- The `gguf` package joins the scan extra to read the imatrix
  artifact.
