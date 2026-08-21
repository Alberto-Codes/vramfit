# ADR-0016: Pack consumes an importance matrix

- **Status:** Accepted
- **Date:** 2026-07-29 (accepted 2026-07-29)

Acceptance evidence (the same night, PR #38): the matrix generated
in 67 minutes (345 chunks, `--process-output`, 18.3 MB). Both
imatrix packs fit first try and differ from their blind twins by
~350 bytes of embedded provenance — the size tables held. The
coverage scan reported one uncovered tensor (`token_embd`,
expected). Measured effect on the 8k recipe: PPL 9.917 → 9.061.
The rematch numbers live in
[the fifth data point](../explanation/evaluating-packed-models.md).

- **Amendment (2026-08-21, issue #278):** the first consequence below
  is fenced to the 49B target. It reads that the rematch becomes fair
  at matched size, and both clauses fail on the 30B target. The
  comparison states both assisted shares and buys no control.
  Maintainer ruling 2026-08-21. See "Amendment: the assisted shares
  differ" below.

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

Three measurements settle the difference. All ran 2026-08-21.

**The smallest published build is a fallback build.** bartowski's
`IQ2_XXS` at 18,838,022,112 B declares 417 tensors. Its header holds
`F32` 243, `IQ4_NL` 69, `Q5_0` 40, `Q8_0` 25, `Q4_K` 17, `IQ2_XXS` 12,
`Q4_0` 10, and `BF16` 1. `tensor_type_fallback` rewrites `IQ2_XXS` to
`IQ4_NL` on every row that 256 does not divide. So all 46 routed-expert
stacks pack at `IQ4_NL` and 4.5 bits per weight. The label reaches 12 of
417 tensors. The header read is a range request over the first 12 MB of
the file. The fallback rule is `src/llama-quant.cpp:374` at llama.cpp
commit `4801e3c56` (b10362).

**Both packs consume the same matrix.** The published build records
`quantize.imatrix.file` at 185 entries and 822 chunks. #300 confirmed
that every campaign arm passes that same file, at 55,314,688 B.

**The assisted shares differ by 21.08 points.** `quantize_iq4_nl`
consumes the matrix, so the published build quantizes 95.52 % of its
bytes assisted. It leaves 1.25 % at `Q8_0`, which discards the matrix,
and 3.04 % at the two uncovered tensors. A campaign arm quantizes
74.44 % of its bytes assisted, at the 35 expert stacks on `Q4_0`. It
leaves 11.70 % at `Q2_0` and 13.86 % at nominal 8. Measured from the
recipe's own byte fields.

The pack-side split runs wider than ADR-0018 records. At `4801e3c56`,
`quantize_q8_0` discards the matrix at line 2295, and `quantize_mxfp4`
and `quantize_nvfp4` discard it at lines 2302 and 2308. `quantize_q4_0`,
`quantize_q4_1`, `quantize_q5_0`, `quantize_q5_1`, and
`quantize_iq4_nl` all consume it.

### Decision

1. **The comparison states both assisted shares beside the damage
   numbers.** It buys no same-conditions control. Maintainer ruling
   2026-08-21.
2. **The asymmetry is a cost of the width, not a toolchain handicap.**
   No type reaches an assisted fit at 2.25 bits on rows of 2688 and
   1856. No 8-bit type consumes a matrix at all. The published build
   earns 95.52 % assistance by spending 18,838,022,112 B, which the
   16 GiB card refuses.
3. **The first consequence is fenced to the 49B target.** A target
   whose palette holds no assisted type at the recipe's floor reaches
   no matched rematch.

### Consequences

- **The confound runs against the recipe, and the recipe still wins.**
  The arm carries 21.08 points less assistance than the build it beats.
  A reader who prices the matrix as the recipe's advantage reads the
  sign backwards.
- **The confound is bounded under 1 point of PPL ratio.** llama.cpp
  PR #4969 measures `Q4_0` with and without a matrix across five dense
  models. Its `QError` is `PPL(Q)/PPL(fp16)-1`, and the with-over-without
  ratio spans 0.507 to 0.877. #229 measured a whole-frontier `Q4_0` pack
  of this target at 1.008684, which is 0.8684 % `QError`. An unassisted
  twin reads 0.990 % to 1.713 %. So the matrix is worth 0.12 to 0.84
  points of PPL ratio on a pack that is 100 % `Q4_0`. A campaign arm is
  74.44 % `Q4_0`. The probe arm beats the published build by 14.23
  points, at 1.178594 against 1.320914.
- **The bound is linear and a reviewer may challenge it.** ADR-0006
  carries one super-additive measurement at 11.9 times. Closing 14.23
  points from 0.84 needs about 17 times compounding across 11 layers at
  nominal 4. No measurement on record reports that.
- A same-conditions control stays buyable at about $2.03 and 37 minutes,
  on the #321 fresh-pod precedent. It prices the confound and it never
  removes it. The 15.776 GiB budget reaches no matched assisted share.
- The 49B control's figure does not transfer. It priced the matrix at
  about 81 % of that recipe's gap, against a baseline our pack did not
  match on the matrix at all.
