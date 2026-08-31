# ADR-0016: Pack consumes an importance matrix

- **Status:** Accepted
- **Date:** 2026-07-29 (accepted 2026-07-29)
- **Amendment (2026-08-21, issue #278):** this amendment fences the
  first consequence to the 49B target. That consequence reads that the
  rematch becomes fair at matched size, and both clauses fail on chart
  #158's 30B target. The comparison states both assisted shares and
  buys no control. Maintainer ruling 2026-08-21. See "Amendment: the
  assisted shares differ" below.
- **Correction (2026-08-31, issue #415):** the 2026-08-21 amendment
  calls the `IQ2_XXS` build the smallest published build. It was not.
  A 2026-08-22 Hub-wide query found nine full-model builds at or
  below the pack's 15.76 GiB. The fallback analysis stands. See the
  correction note in the amendment below.

Acceptance evidence (the same night, PR #38): the matrix generated
in 67 minutes (345 chunks, `--process-output`, 18.3 MB). Both
imatrix packs fit first try and differ from their blind twins by
~350 bytes of embedded provenance — the size tables held. The
coverage scan reported one uncovered tensor (`token_embd`,
expected). Measured effect on the 8k recipe: PPL 9.917 → 9.061.
The rematch numbers live in
[the fifth data point](../explanation/evaluating-packed-models.md).


## Context

The v1 pack path rounds blind. `llama-quantize` fits each block by
plain squared error, with every weight equal. An importance matrix
weights that fit by mean squared activation per weight, collected
over a calibration run. K-quants accept one — the community
baselines use exactly this.

The first 49B head-to-head measured the cost of packing without one
([the fourth data point](../explanation/evaluating-packed-models.md)).
The recipe scored PPL 9.917 against the size-matched imatrix Q3_K_S
at 8.532. The control — our own Q3_K_S from the same f16 base, no
imatrix — scored 9.655. The importance matrix alone accounts for
~81 % of the gap. ADR-0012 carried this as an open question and
escalated it to north-star-gating. Issue #35 tracks it.

Facts verified on the reference box (llama.cpp e9fa078, Vulkan
b10172):

- `llama-quantize --imatrix <file>` applies the matrix to K-quant
  types and embeds provenance keys (`quantize.imatrix.file`,
  dataset, entry and chunk counts) in the output GGUF.
- `llama-imatrix` computes the matrix from a model plus a text
  file. The Vulkan build ships it.
- The f16 base GGUF of the 49B target (93 GB) is on disk, kept for
  this.
- ADR-0010 recorded a design intent: the scan emits the matrix as a
  calibration byproduct. That requires vramfit to write llama.cpp's
  imatrix format from the torch frame — a real subproject, not a
  flag.

## Decision

1. **`vramfit pack` accepts `--imatrix <file>` and forwards it to
   `llama-quantize --imatrix`.** The type tables (ADR-0012) and the
   effective-bits table (ADR-0014) do not change. The matrix changes
   how values round inside each block, not the block layout or the
   size. `PackResult` and the `model_packed` event record the path —
   an imatrix-assisted artifact must say so in its provenance.
2. **v1 generates the matrix with `llama-imatrix` against the kept
   f16 base GGUF.** Generation stays outside `vramfit pack`: it is
   a GPU-scale forward pass with its own runtime flags, and the pack
   step must stay a CPU subprocess driver. The scan-emits-it design
   intent stays open below.
3. **The imatrix text is the scan's calibration set.** One text
   source feeds the whole measured pipeline: scan, validation pass,
   importance matrix. The baseline's matrix (bartowski) was computed
   on their own calibration set. The head-to-head therefore compares
   whole pipelines, calibration text included. This difference is
   recorded, not hidden.
4. **I-quant types stay deferred.** The size-matched baseline is a
   K-quant with an imatrix, so K-quants-plus-imatrix isolates the
   measured dominant factor first. The i-quant table remains ADR-0012's
   open question.

## Consequences

- The rematch becomes fair: recipe and baseline both quantize
  imatrix-assisted, at matched size. Whatever gap remains belongs to
  the recipe, not the toolchain handicap.
  **Fenced to this target 2026-08-21 by the #278 amendment below.** A
  target whose palette holds no assisted type at the recipe's floor
  reaches no matched rematch.
- One more input artifact exists: the imatrix file. It carries its
  own provenance (model, text, chunk count) inside the packed GGUF's
  metadata.
- Generation costs one calibration-text pass through the 93 GB f16
  base — GPU-assisted, tens of minutes to hours at 49B scale.
- Our calibration text (~770 KB) is smaller than the community
  calibration sets. If the imatrix rematch still loses, text size
  and diversity join the candidate list.

## Open questions

- The scan emitting the matrix natively as a calibration byproduct
  (ADR-0010's design intent). This would remove the llama-imatrix
  dependency and tie the matrix to the scan's fingerprint.
- Whether the calibration text needs more tokens or more diversity
  than the scan's set for a competitive matrix.
- The i-quant type table, carried from ADR-0012.

## Amendment: the assisted shares differ (2026-08-21, issue #278)

### Context

Decision 1 forwards `--imatrix` to `llama-quantize`. The first
consequence reads that the rematch becomes fair, because recipe and
baseline both quantize imatrix-assisted at matched size. That sentence
describes the 49B acceptance test. It does not describe chart #158's
Nemotron 3.5 Lightning 30B-A3B target.

Four measurements settle the difference. All ran 2026-08-21.

**The smallest published build is a fallback build.** bartowski's
`IQ2_XXS` at 18,838,022,112 B declares 417 tensors. Its header holds
`F32` 243, `IQ4_NL` 69, `Q5_0` 40, `Q8_0` 25, `Q4_K` 17, `IQ2_XXS` 12,
`Q4_0` 10, and `BF16` 1. `tensor_type_fallback` rewrites `IQ2_XXS` to
`IQ4_NL` on every row that 256 does not divide. The 46 backbone expert
stacks hold rows of 2688 and 1856, so each packs at `IQ4_NL` and 4.5
bits per weight. The `blk.52` MTP block holds two further expert stacks,
at `Q4_0` and the same 4.5 bits per weight. The label reaches 12 of 417
tensors, and all 12 are `ssm_out` at rows of 4096. A range request over
the first 12 MB of the file reads the header. The fallback rule is
`src/llama-quant.cpp:374` at llama.cpp commit `4801e3c56` (b10362).

> Correction 2026-08-31 (issue #415): the first sentence above
> overstates the build's position. The build was the smallest in the
> five repositories the campaign had checked, not the smallest
> published. A 2026-08-22 Hub-wide query found nine full-model builds
> at or below the pack's 15.76 GiB, and the 17.544 GiB comparator is
> the largest of that set. The fallback analysis in this paragraph is
> unaffected. Issue #416 measures the shelf.

**Both packs consume the same matrix.** The published build records
`quantize.imatrix.file` at 185 entries and 822 chunks. #300 confirmed
that every campaign arm passes that same file, at 55,314,688 B.

**The matrix covers the backbone only.** Its 185 logical entries name no
tensor in `blk.52`, and its highest block index is 51. So nine quantized
MTP tensors packed unassisted, at 750,919,680 B. With `token_embd` and
`output` the published build leaves 11 tensors uncovered.

**The assisted shares differ by 17.09 % of bytes, against the recipe.**
The published build quantizes 91.53 % of its bytes assisted. It leaves
1.25 % at `Q8_0`, which discards the matrix, 7.03 % uncovered, and
0.19 % unquantized. A campaign arm quantizes 74.44 % of its bytes
assisted, at the 35 expert stacks on `Q4_0`. It leaves 11.70 % at
`Q2_0` and 13.86 % at nominal 8. A read of the published header and of
the recipe's own byte fields measures these shares.

[ADR-0028](0028-expert-stack-type-table.md) already records that the
matrix does not shape a stack at nominal 2 or nominal 8, and that
`quantize_mxfp4` and `quantize_nvfp4` ignore it. This amendment adds the
line numbers at `4801e3c56` and the types ADR-0028 does not name.

| call | line | reads the matrix |
| --- | --- | --- |
| `quantize_q4_0` | 2128 | yes |
| `quantize_q4_1` | 2173 | yes |
| `quantize_q5_0` | 2227 | yes |
| `quantize_q5_1` | 2280 | yes |
| `quantize_iq4_nl` | 5077 | yes |
| `quantize_q2_0` | 2113 | no |
| `quantize_q8_0` | 2295 | no |
| `quantize_mxfp4` | 2302 | no |
| `quantize_nvfp4` | 2308 | no |

### Decision

1. **The comparison states both assisted shares beside the damage
   numbers.** It buys no same-conditions control. Maintainer ruling
   2026-08-21.
2. **The asymmetry is a cost of the width, not a toolchain handicap.**
   No type reaches an assisted fit at 2.25 bits on rows of 2688 and
   1856. No 8-bit type consumes a matrix at all. The published build
   earns 91.53 % assistance by spending 18,838,022,112 B, which the
   16 GiB card refuses.
3. **This amendment fences the first consequence to the 49B target.** A
   target whose palette holds no assisted type at the recipe's floor
   reaches no matched rematch.

### Consequences

- **The confound runs against the recipe, and the recipe still wins.**
  The arm quantizes 17.09 % fewer of its bytes assisted than the build
  it beats. A reader who prices the matrix as the recipe's advantage
  reads the sign backwards.
- **The arm's own matrix does not explain its win.** Dropping
  `--imatrix` from an arm changes only the 35 `Q4_0` stacks, because
  `Q2_0` and `Q8_0` discard the matrix. llama.cpp PR #4969 added the
  matrix to `quantize_row_q4_0_impl` and reports a `Q4_0` `QError` ratio
  of 0.507 to 0.877 across five dense models. `QError` is that PR's own
  name for `PPL(Q)/PPL(fp16)-1`, and it is not this project's damage.
  #229 measured a whole-frontier `Q4_0` pack of this target at
  1.008684, which is 0.8684 % `QError` at plus or minus 0.064. An
  unassisted twin reads 0.990 % to 1.713 %. So the matrix is worth at
  most 0.12 to 0.84 points of PPL ratio on a pack whose every byte is
  `Q4_0`, and an arm holds 74.44 % of its bytes there. The probe arm
  beats the published build by 14.23 points of PPL ratio, at 1.178594
  against 1.320914.
- **Three transfers sit under that bound, and a reviewer may challenge
  each.** PR #4969 measured five dense models at 7B and 13B, against a
  30B MoE here. It measured at 4 bits, and a matrix matters more as the
  width falls. The decomposition is linear, and ADR-0006 carries one
  super-additive measurement at 11.9 times. Closing 14.23 points from
  0.84 needs about 17 times compounding across the 35 stacks at nominal
  4, which span 23 layers. No measurement on record reports that.
- **The two figures come from different instruments.** #229 measured at
  b10326 and the campaign arms at b10362, and ADR-0027 decision 3 bars a
  magnitude comparison across instruments. #372 re-ran every campaign
  arm at b10362 and reproduced each b10326 figure to every printed
  decimal, so the instruments agree where they overlap. #229's own point
  never re-ran.
- A same-conditions control stays buyable at about $2.03 and 37 minutes,
  on the #321 fresh-pod precedent. It prices the confound and it never
  removes it, because the 15.776 GiB budget reaches no matched assisted
  share. #350 carries the same repack as its own third candidate, so a
  later session buys it once or not at all.
- The 49B control's figure does not transfer. It priced the matrix at
  about 81 % of that recipe's gap, against a baseline our pack did not
  match on the matrix at all.
