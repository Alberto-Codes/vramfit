# ADR-0032: Pre-encode assisted Q2_0 tensors before stock whole-file quantization

- **Status:** Proposed
- **Date:** 2026-09-14
- **Origin:** [Issue #597](https://github.com/Alberto-Codes/vramfit/issues/597),
  a `chart:discuss` child of [chart #158](https://github.com/Alberto-Codes/vramfit/issues/158).
- **Authority:** The maintainer authorized reimplementation on 2026-09-05:
  “a but reimplement we do our own arch etc.” He funded about one pod-hour
  for a subsequent re-solve. His 2026-09-14 “go” commissioned this record.
  These rulings came through the task brief. They prohibit vendoring the
  reference patch or waiting for upstream.
- **Proposes amendments to:** [ADR-0016 decision 2](0016-imatrix-in-the-pack-path.md#decision)
  and [ADR-0018's assisted q0 method](0018-kquant-within-group-method.md#amendment-q0-imx-gets-built-2026-08-21-issue-350).
  The maintainer accepts this architecture through review. This record
  implements no encoder and starts no re-solve.

## Context

The 2026-09-05 rebuy changed the encoder for eleven Q2_0 stacks while
keeping the shipped 30B recipe's allocation. Its same-pod control measured
0.204318 mean KLD. The assisted arm measured 0.071428, with same-top
agreement rising from 83.127 % to 89.441 %. Both files contain
16,922,476,352 B. Their contents differ, as their SHA-256 hashes confirm.
The stock b10362 runtime evaluated the assisted arm.

The archived report records a 65.0 % KLD reduction and a 6.31-point
same-top gain. These measure the whole pack, not isolated stack damage.
The [evidence record](evidence/0032/README.md#rebuy-evidence) preserves
those results and distinguishes the same-pod control from the published file.

For scale, [ADR-0031](0031-refinement-pass.md) records the refinement
stage's 5.74 % improvement. The relative reductions differ by roughly
eleven times. This is scale context, not a matched experiment between
methods or a promise that their gains add.

vramfit already owns an assisted Q4_0 torch fit in
[`q0_assisted.py`](../../src/vramfit/adapters/outbound/scan/q0_assisted.py).
It uses candidate scales and the element weight
`qw[j] * sqrt(sigma2 + x[j]^2)`. Nominal 2 remains unassisted because
stock `quantize_q2_0` discards the matrix.
[`q0_ref.py`](../../src/vramfit/adapters/outbound/scan/q0_ref.py)
already reproduces the Q2_0 reference round trip.

The reference patch specifies a different Q2_0 scale search, described
in [issue #461](https://github.com/Alberto-Codes/vramfit/issues/461#issuecomment-5486773582).
It tries three signed scale seeds and four Lloyd iterations per seed.
It scores candidates after fp16 scale rounding against levels
`{-1, 0, 1, 2}`. The existing Q4_0 helper supplies precedent, not an
unchanged Q2_0 implementation. The patch is a specification to
reimplement and verify. vramfit must not carry or apply it.

## Experiment: can stock packing preserve pre-encoded tensors?

**Yes, on stock b10362 (`4801e3c56`).** The local probe supplied a
GGUF with a Q2_0 expert stack, an F16 attention tensor, and an F32 norm.
It invoked `llama-quantize --pure` with explicit Q2_0 and Q4_0
overrides, using Q2_K as the base type.

The quantizer preserved all 36,864 Q2_0 payload bytes and converted
the F16 tensor to Q4_0. It repeated that behavior with `--imatrix`.
Both calls exited 0. A Q4_0 override on the existing Q2_0 tensor
exited 1 and printed `requantizing from type q2_0 is disabled`.
No call supplied `--allow-requantize`.

The [commands, fixture source, hashes, and complete output](evidence/0032/README.md)
make the result reproducible. The first fixture lacked required model
metadata. Its archived failure preceded the corrected, successful probe.

The probe establishes tensor passthrough, including a three-dimensional
expert stack. It does not measure encoder accuracy, full-model loading,
production memory use, or passthrough on every llama.cpp version.

## Options and ADR-0016 decision 2

Decision 2 keeps matrix generation outside pack and calls pack a
“CPU subprocess driver.” It does not already authorize an in-process
vramfit tensor encoder. Merely running on the CPU does not settle
that boundary.

| Option | Relationship to decision 2 | Consequence |
| --- | --- | --- |
| Pre-encode selected Q2_0 tensors into a temporary base GGUF | A CPU subprocess preserves the driver's role. This ADR explicitly extends its responsibilities to that preprocessing step. | The probe proves stock passthrough. vramfit owns one encoder and selected-tensor GGUF rewriting. |
| Implement whole-file quantization inside vramfit | An in-process implementation contradicts the driver boundary. A subprocess preserves its literal form but replaces the delegated toolchain's responsibility. | Requires broader authority and ownership of every supported quantizer, override rule, metadata path, and file layout. |
| Add the encoder only to the scan meter | Leaves pack unchanged and therefore fits decision 2. | Measures a hypothetical assisted artifact. Stock packing still emits unassisted Q2_0, so those prices cannot support the funded re-solve. |

The first option needs no upstream change and carries no third-party patch.
The second adds ownership the measured gain does not require.
The third cannot deliver the measured benefit or matching provenance.

## Decision

1. **Select pre-encoding through a vramfit-owned CPU subprocess.**
   Amend ADR-0016 decision 2 to permit this preprocessing stage.
   The pack adapter remains a subprocess driver. Matrix generation
   remains outside pack. Stock `llama-quantize` retains final whole-file
   quantization and assembly.

   The preprocessor reads an immutable floating-point base and writes a
   temporary mixed GGUF. It replaces only tensors whose resolved final
   type is Q2_0 and whose fit uses the new encoder. It preserves other
   tensor payloads and metadata, with correct offsets and alignment.
   It does not overwrite the kept base or a published artifact.

   The final invocation retains `--pure`, the resolved overrides, and
   imatrix options. It never enables `--allow-requantize`. The implementation
   verifies selected payload bytes after packing and retains the existing
   final-type and size checks.

2. **Own the numerical fit once, behind the outbound adapters.**
   The scan meter and CPU preprocessor use the same Q2_0 encoding
   semantics. Scan measures the dequantization of the stored blocks,
   including fp16 scales. Domain code owns no torch or GGUF dependency.
   The plan step and base installation gain no heavy imports.

   Reimplement the specification within vramfit's architecture. Verify
   signed scales, rounding ties, zero blocks, and weighted candidate
   selection against reference fixtures. A copied patch or patched
   llama.cpp distribution does not satisfy this decision.

3. **Yes, the assisted q0 scan method changes with the shipping encoder.**
   Add assisted nominal-2 pricing alongside assisted nominal 4. Nominal 8
   keeps its reference arithmetic. Apply the same imatrix, expert-row
   mapping, exclusions, and zero-count fallback during scan and packing.
   Uncovered or excluded Q2_0 tensors retain stock reference behavior.

   Introduce a distinct serialized `within_group` token for this successor
   to `q0-imx`. The existing token keeps its stock-Q2_0 meaning.
   The implementation must carry the new identity through map fingerprints,
   checkpoints, recipes, and pack selection. It must record the encoder
   revision and matrix provenance with the packed result.

   Never relabel an old map or resume its nominal-2 cells under the new
   identity. Old recipes remain reproducible through their stock path.
   A recipe priced with the new method must use the new packing path.
   This record does not silently widen a currently accepted token.

4. **Gate the funded re-solve on matching scan and pack semantics.**
   First implement and verify the encoder and passthrough integration.
   Then produce a sensitivity map with the shipping encoder and its matrix.
   Only that map may price the new allocation. Preserve the existing
   measurement and runtime-frame requirements from ADR-0027.

   Record the base, matrix, map, recipe, encoder, toolchain, and final
   artifact identities for the re-solve. An old unassisted-Q2_0 map plus
   an assisted pack is a different experiment and cannot satisfy this gate.

## Consequences and implementation handoff

- vramfit owns one numerical encoder and a GGUF preprocessing stage.
  Stock decoders and block layouts remain sufficient.
- A temporary GGUF adds disk traffic and storage demand. The implementation
  must bound memory by chunks and measure full-file cost before the funded run.
- Matching payload hashes protect against lost pre-encoding. They do not
  prove the fit's numerical correctness. Reference fixtures address that separately.
- Toolchain changes require the passthrough probe again. The implementation
  must test overrides, matrix exclusions, zero-count experts, and the target
  row widths of 2688 and 1856.
- Same scan and pack semantics remove this encoder mismatch. They do not
  equate the torch scan frame with runtime damage or validate additive predictions.
- Acceptance requires a tiny end-to-end pack/load test with a stock runtime,
  plus scan reconstruction of the blocks that pack actually emits.
- The next task implements these requirements. The maintainer starts that task
  and the funded re-solve. [Issue #597](https://github.com/Alberto-Codes/vramfit/issues/597)
  tracks this handoff. This change contains only the record and its evidence.
