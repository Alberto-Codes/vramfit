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
  calibration byproduct. That requires quantfit to write llama.cpp's
  imatrix format from the torch frame — a real subproject, not a
  flag.

## Decision

1. **`quantfit pack` accepts `--imatrix <file>` and forwards it to
   `llama-quantize --imatrix`.** The type tables (ADR-0012) and the
   effective-bits table (ADR-0014) do not change. The matrix changes
   how values round inside each block, not the block layout or the
   size. `PackResult` and the `model_packed` event record the path —
   an imatrix-assisted artifact must say so in its provenance.
2. **v1 generates the matrix with `llama-imatrix` against the kept
   f16 base GGUF.** Generation stays outside `quantfit pack`: it is
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
