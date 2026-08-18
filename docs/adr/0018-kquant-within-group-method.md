# ADR-0018: A K-quant-faithful within-group method behind a scan flag

- **Status:** Accepted
- **Date:** 2026-07-31 (accepted 2026-07-31)
- **Amendment (2026-08-17, issue #319):** a third method, `gguf`, ports
  the block quantizers `llama-quantize` applies where no K-quant
  reaches. Its token `gguf-ref` is the fourth, because `kquant` carries
  two. Maintainer ruling 2026-08-17. See "Amendment: the `gguf-ref`
  method" below.
- **Amendment (2026-08-18, issue #332):** decision 2 of the 2026-08-17
  amendment changes the token. The method is `q0` and the token is
  `q0-ref`. Maintainer ruling 2026-08-18. See "Amendment: the token
  becomes `q0-ref`" below.

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

## Amendment: the `gguf-ref` method (2026-08-17, issue #319)

### Context

This record ported the K-quants because the pack applied them. On the
Nemotron 3.5 Lightning 30B-A3B target the pack cannot apply them at all.
`tensor_type_fallback` rejects any type whose block does not divide
`ne[0]`, and the 46 routed-expert stacks hold rows of 2688 and 1856
against `QK_K` 256 (#159, #189). ADR-0028 packs those stacks at `Q8_0`,
`Q4_0`, and `Q2_0`. The stacks hold 93.0 % of the parameters.

`kquant` priced them anyway. A box-side spike measured the gap on real
bf16 weights, over 32 of the 128 experts in each of the 46 stacks.

| nominal | the method applied | the pack applies | under-priced by |
|---|---|---|---|
| 2 | `Q2_K` 0.088908 | `Q2_0` 0.518557 | 5.83x |
| 4 | `Q4_K` 0.005163 | `Q4_0` 0.007526 | 1.46x |

The numbers are weight-space reconstruction error at
`||q - w||² / ||w||²`. They are not damage and they set no price.

A solver spends the ratio between two widths. The method priced nominal
2 at 17.22 times nominal 4, against the pack's 68.90 times. **So the map
made 2 bits look 4.00 times cheaper against 4 bits than the artifact
delivers.** The frames also order the stacks apart, at Spearman rho
+0.4638 for `Q2_K` against `Q2_0` and +0.7853 for `Q4_K` against
`Q4_0`.

The `kquant` port also states a straddle assumption its docstring never
checks (`scan/kquant.py:405`). 256 divides neither 2688 nor 1856, so a
flat super-block spans two rows. That defect is real and inert. Per-row
padding moves the fit 0.05 %. It leaves the layer ordering unchanged at
rho +1.0000, and the stack ordering at rho +0.9974.

ADR-0021 decision 1 closed the scan-frame refinement lane. This
amendment does not reopen it. That lane refined a frame toward a type
the pack applied. This one replaces a frame the pack cannot apply.

### Decision

1. **A third within-group method ports the block quantizers.** The
   methods are `rtn`, `kquant`, and now `gguf`. The tokens number four,
   because `kquant` carries `kquant-ref` and `kquant-imx`. It
   reimplements `quantize_row_q2_0_ref` and `quantize_row_q4_0_ref` from
   llama.cpp b10326, commit `3653e6d6d`, including the fp16 scale
   rounding. The two types round differently and the port must keep them
   apart. `Q2_0` calls `roundf`, which rounds half away from zero.
   `Q4_0` truncates through `(int8_t)(x*id + 8.5f)`, which rounds half
   up. The port returns dequantized values, like the other two methods.
   #327 builds it.
   **The method covers nominal 8, 4, and 2.** Nominal 8 reuses this
   record's `_q8_0_round_trip`, because `Q8_0` blocks 32 elements and
   ADR-0028 decision 1 packs it on these rows. `ggml-quants.c` and
   `ggml-common.h` are byte-identical between `e9fa078` and `3653e6d6d`,
   so the two checkouts do not conflict. The method refuses nominal 3,
   which ADR-0028 decision 2 refuses at pack. It refuses 5 and 6 until
   ports exist.
2. **The method token is `gguf-ref`.** The CLI accepts
   `--within-group gguf`. `gguf-imx` stays reserved for the assisted
   path, because `quantize_row_q4_0_impl` fits with imatrix weights.
   **Superseded 2026-08-18 by #332.** The token is `q0-ref` and the
   reserved token is `q0-imx`. See the amendment below.
3. **The port verifies against the C reference.** A golden-fixture suite
   asserts the torch round trip matches recorded dequantized values, on
   decision 4's pattern above. It covers random, outlier-heavy,
   constant, zero, and subnormal-scale blocks, and it adds exact ties.
   **The bar is bit-exact for `Q2_0` and `Q4_0`.** Neither type fits a
   candidate grid, so neither carries `Q2_K`'s representation ties.
4. **`kquant` refuses a cell whose mapped type's block size does not
   divide the tensor's row length, and names both.** The refusal fires
   for `Q2_K`, `Q3_K`, and `Q4_K` at `QK_K` 256. **It does not fire for
   `Q8_0` at block 32**, which divides 2688 and 1856 and which ADR-0028
   decision 1 packs on these rows. A refusal keyed to 256 alone would
   refuse a cell the pack realizes. The refusal mirrors ADR-0028's
   pack-side halt on the type fallback. A silent substitution would make
   one map record two frames under one token.
5. **The 30B target re-scans its 46 expert stacks under `gguf-ref`**, at
   precisions 2 and 4, for 92 cells. **The run writes its own
   stack-keyed map and mixes no token**, because `scan.within_group` is
   map-level. It runs on #163's instrument, an H100 SXM 80 GB, which
   ADR-0027 decision 1 makes part of the frame. The run needs #282's
   group-subset flag first, at $4.04 against $18.42 for the whole
   210-group scan. #328 carries it.
   **The record does not say what the solver consumes meanwhile.** The
   new map covers 46 of 210 stack groups and #163's map is layer-keyed,
   so no map yet prices the whole model in one frame. #328 closes with
   that gap named and a ticket for it.

### Consequences

- `rtn` is not the fallback for these stacks. It reaches `Q2_0`'s grid
  at block 64 and the CLI hardcodes block 32. At block 32 it
  **under-prices** `Q2_0`'s level by 16.1 %, at 0.434794 against
  0.518557 and rho +0.9888. At nominal 4 it over-prices `Q4_0` by 1.28
  times, at rho +0.9967.
- `Q2_0` reaches three levels, not four. The reference clamps
  `round(w/amax)` to `[-1, 2]` and `|w| <= amax` caps it at 1. Upstream
  built the type for ternary QAT checkpoints and packed 2 bits for
  acceleration (ggml-org/llama.cpp#24448).
- **`quantize_q2_0` accepts an importance matrix and ignores it**
  (`ggml/src/ggml-quants.c:2113-2126`, b10326). So `gguf-imx` can only ever
  differ from `gguf-ref` at nominal 4. #278 carries the consequence for
  the published-build comparison.
- Maps priced under `kquant` on this target do not compare with maps
  priced under `gguf-ref`. #163's map is the campaign's input until the
  re-scan lands.
- No published work prices a sensitivity map against the exact type its
  artifact ships. SPEAR (arXiv 2606.11244) appendix C.3 measures the
  nearest thing. It compares RTN, GPTQ, and AWQ at 4-bit per channel, at
  Spearman rho 0.77 to 0.98 against a top-30 % set mismatch of 7 % to
  32 %. **Read at the matched width, this target agrees with that band**
  — `Q4_K` against `Q4_0` reads +0.7853. At 2 bits it reads +0.4638,
  below every cell SPEAR reports. SPEAR's own finding is that ordering
  survives a quantizer swap and set membership does not, which is what
  #300 measured directly.
- **The campaign's measured arm lost to a random control under this
  defect (#300).** The map ordered the stacks correctly and the arm
  built from it still lost. #319 carries the confound and #321 asks
  whether the allocation policy spends a correct ordering badly.
- **This amendment states no falsifier, and ADR-0019 is the warning.**
  Three scan-frame refinements each looked principled and each packed
  worse. A `gguf-ref` map earns nothing until an arm built from it beats
  #300's blind draws at 1.184126. #321 and #328 carry that test.

## Amendment: the token becomes `q0-ref` (2026-08-18, issue #332)

### Context

The 2026-08-17 amendment named the method `gguf` and its token
`gguf-ref`. Review argued against both after that ruling. The argument
reached PR #326's body and a comment on #327, and neither survives a
merge. #332 carried it to the tracker.

Three facts drive the change.

- The other tokens name a quantizer family. `rtn` names an algorithm.
  `kquant` names a type family. `gguf` names the container.
- `Q2_K` and `Q4_K` are GGUF types too. So the `gguf-` prefix separates
  this method from `kquant` by nothing a reader can see.
- The 30B target's usable expert palette holds `MXFP4`, `NVFP4`,
  `IQ4_NL`, `Q4_1`, `Q5_0`, and `Q5_1`. Each is a GGUF type. A `gguf-`
  token spends the prefix every later port needs. #288 carries an
  `MXFP4` mix that reaches 46 of 46 stacks.

`legacy-ref` is the community word for this family. It is wrong here,
because `Q2_0` merged 2026-07-07.

The change is time-boxed. PR #330 writes the token into the scan
fingerprint and into `scan.within_group`. A token change after a map
carries it invalidates that map. No map carries `gguf-ref` today. #328
scans 92 cells at $4.04, and its map is the first that would.

### Decision

1. **The method is `q0` and its token is `q0-ref`.** The name states the
   `_0` family: one fp16 scale per block, no minimum, and no
   super-block. `Q2_0`, `Q4_0`, and `Q8_0` are that set. This supersedes
   decision 2 of the 2026-08-17 amendment. Maintainer ruling 2026-08-18.
2. **`q0-imx` replaces `gguf-imx` as the reserved assisted token.** The
   reservation does not change. `quantize_q2_0` still ignores an
   importance matrix, so the reserved path can differ at nominal 4 only.
3. **A later GGUF-type port takes its own family token.** `mxfp4-ref`
   stays free. The `gguf-` prefix names no method.
4. **The rename reaches the code, the tests, and the reference pages.**
   The module is `scan/q0_ref.py`, the constant is `Q0_REF_METHOD`, and
   the CLI accepts `--within-group q0`.

### Consequences

- No artifact moves. No map, recipe, or run log carries `gguf-ref`.
- The 2026-08-17 amendment stays as written above. Read its `gguf-ref`
  and `gguf-imx` as `q0-ref` and `q0-imx`.
- The golden-fixture suite still asserts all three types bit-exact
  against libggml. The rename moves names and no arithmetic.
