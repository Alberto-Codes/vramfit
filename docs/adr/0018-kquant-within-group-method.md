# ADR-0018: A K-quant-faithful within-group method behind a scan flag

- **Status:** Accepted
- **Date:** 2026-07-31 (accepted 2026-07-31)

## Context

ADR-0006 fixed round-to-nearest with 32-element scale blocks as the
v1 within-group method. It also ruled that a method change is a new
scan, not a new schema. The fingerprint carries a method token
(`rtn-block32`) for exactly this case.

The sixth evidence data point exposes a frame-transfer leak at low
bits. The 65,536-token recipe measures 0.0589 joint damage in the
scan frame. Packed with the same assignments, it loses to the
size-matched baseline by 0.62 PPL and 0.11 mean KLD. The scan frame
and the packed artifact quantize differently:

- The scan's RTN spends four symmetric levels per 32-element block
  at 2-bit: {-2s, -s, 0, +s}. No offset, one absmax scale.
- The pack's `Q2_K` spends four unsigned levels per 16-element
  sub-block, with a fitted scale **and** a fitted minimum, both
  re-quantized to 4 bits against super-block fp16 constants
  (llama.cpp `ggml-quants.c`, checkout e9fa078).
- `Q3_K` differs the same way: RTN uses eight symmetric levels per
  32 elements, `Q3_K` fits eight levels per 16 with 6-bit
  super-block scale quantization and iterative refinement.

The solver buys 2-bit groups on RTN prices. If RTN under-prices
2-bit damage, every 2-bit assignment is mispriced, and the additive
model's 2-bit membership problem (ADR-0006, fourth measurement)
compounds on wrong marginals.

## Decision

1. **A new scan module ports the K-quant round trip to torch.** It
   reimplements `quantize_row_q2_K_ref`, `quantize_row_q3_K_ref`,
   `quantize_row_q4_K_ref`, and `quantize_row_q8_0_ref` from
   llama.cpp (checkout e9fa078), including sub-block scale/min
   fitting, super-block scale re-quantization, and fp16 storage
   rounding. The port returns dequantized values, like the RTN
   round trip. It imports torch only — no llama.cpp dependency at
   scan time (ADR-0005).
2. **The meter takes a within-group method argument.** `rtn` stays
   the default and the behavior is unchanged. `kquant` routes
   2-bit cells through `Q2_K`, 3-bit through `Q3_K`, 4-bit through
   `Q4_K`, and 8-bit through `Q8_0`. The method refuses 5- and
   6-bit cells until `Q5_K`/`Q6_K` ports exist.
3. **The CLI exposes `--within-group {rtn,kquant}` on `scan`.** The
   fingerprint method token becomes `kquant-ref` under the flag, so
   checkpoints never mix methods (ADR-0006's rule, mechanized). The
   sensitivity map records the method in a new optional
   `scan.within_group` field. Absent means `rtn-block32`. The field
   is additive — `vramfit_schema` stays at 1. (Since bumped to 2 with
   the rename, #118.)
4. **The port verifies against the C reference.** A local harness
   drives `ggml_quantize_chunk` through ctypes and records small
   golden fixtures (random, outlier-heavy, constant, zero, and
   subnormal-scale blocks). A unit-tier golden-fixture suite asserts
   the torch round trip matches the fixtures' dequantized values —
   bit-exact for `Q3_K`/`Q8_0`, error parity within representation
   ties for `Q2_K`/`Q4_K`.

## Open questions

- The v1 port takes the no-imatrix reference path.
  `llama-quantize --imatrix` fits with activation weights instead
  (`quantize_row_q2_K_impl`). A kquant cell therefore prices the
  *unassisted* format — expect it to sit at or above the packed
  artifact's damage for imatrix-covered tensors. Whether the meter
  should consume the imatrix itself is a follow-up decision.
- ~~Coverage of {4, 8} nominal bits, required before a full kquant
  scan can feed the solver.~~ Landed with this ADR: `Q4_K` reuses
  the `make_qkx2_quants` machinery (squared error, 32-element
  sub-blocks, 6-bit scales) and `Q8_0` is a 32-block absmax code.
  `Q5_K`/`Q6_K` stay open — no recipe has assigned 5 or 6 bits yet.
- ~~What per-precision inflation factor (kquant damage over RTN
  damage, same cell) invalidates RTN pricing.~~ **Measured
  (2026-07-31, 16 cells on the 65,536-token frame): the question
  inverted.** RTN does not under-price low bits — it over-prices
  them. RTN 2-bit damage runs 2.0–3.9x the `Q2_K` damage in-frame
  (attention-bearing layers worst, deep FFN-only layers ~2.0x).
  RTN 3-bit runs 1.05–1.7x `Q3_K`. The distortion is per-cell, not
  a per-precision scalar — no rescale fixes the ranking. Mechanism:
  RTN's symmetric absmax grid cannot reach its lowest level at
  2-bit, so it spends three levels where `Q2_K` fits four plus a
  minimum. Two RTN re-measurements bounded cross-process frame
  noise at ~20 % (0.79–0.81 of the stored map values) — the
  in-frame ratios above are corrected for it. Consequence drawn in
  [ADR-0019](0019-kquant-priced-maps.md).

## Consequences

- The scan can price 2/3-bit cells in the structure the pack
  actually ships. The 2-bit membership decisions (ADR-0006, fourth
  measurement) then rest on honest marginals.
- K-quant fitting costs more than RTN: a 17-candidate grid search
  per 16-element sub-block for `Q2_K`, five refinement sweeps for
  `Q3_K`. Both vectorize across sub-blocks in torch. Forward
  passes still dominate cell time at 49B scale.
- Two maps of the same model are now comparable only when their
  `scan.within_group` matches — the same rule calibration
  provenance already imposes.
