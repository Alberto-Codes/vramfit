# ADR-0032: Pre-encode assisted Q2_0 tensors before stock whole-file quantization

- **Status:** Accepted
- **Date:** 2026-09-14 (accepted 2026-09-15)
- **Note (2026-09-15, issue #597):** acceptance discharged the two
  forward-looking clauses below. Decision 3 defers the glossary's
  pack-side assisted amendment to acceptance. That amendment landed.
  [The glossary](../reference/glossary.md) now rules a pre-encoded
  tensor assisted and cites decision 3. Decision 5 reads ADR-0016's
  markers as contested and not superseded. Those markers now read
  **Amended by ADR-0032**, in
  [ADR-0016](0016-imatrix-in-the-pack-path.md)'s header and in the
  [index](index.md) row. The decision bodies below keep their
  original wording.
- **Origin:** [Issue #597](https://github.com/Alberto-Codes/vramfit/issues/597),
  a `chart:discuss` child of [chart #158](https://github.com/Alberto-Codes/vramfit/issues/158).
- **Authority:** Reimplementation was authorized on 2026-09-05, with
  approximately one pod-hour funded for a subsequent re-solve. This record
  was commissioned on 2026-09-14. These rulings prohibit vendoring the
  reference patch and prohibit waiting for upstream.
- **Amends:** two clauses. ADR-0016 carries two
  decision lists, so each reference below names its list.
  - [ADR-0016's original decision 2](0016-imatrix-in-the-pack-path.md#decision),
    the CPU-subprocess-driver clause.
  - [ADR-0016's 2026-08-21 amendment, decision 2](0016-imatrix-in-the-pack-path.md#amendment-the-assisted-shares-differ-2026-08-21-issue-278),
    the "cost of the width, not a toolchain handicap" clause. Decision 5
    below corrects it.
- **Adds to, and does not amend,**
  [ADR-0018's assisted q0 method](0018-kquant-within-group-method.md#amendment-q0-imx-gets-built-2026-08-21-issue-350).
  Its decision 1 reads that nominal 2 takes the reference path under the
  token `q0-imx`. That stays true, because decision 3 below keeps the
  token's stock-Q2_0 meaning and adds a distinct successor.
  [Issue #599](https://github.com/Alberto-Codes/vramfit/issues/599)
  names the successor and decides whether its record amends ADR-0018
  or this one.
- The maintainer accepts this architecture through review. This record
  implements no encoder and starts no re-solve.

## Context

The 2026-09-05 rebuy changed the encoder for eleven Q2_0 stacks while
keeping the shipped 30B recipe's allocation. Its `ctl-repro` block
measured 0.204318 mean KLD. Its `c2-assisted-q2_0` block measured
0.071428, and same-top agreement rose from 83.127 % to 89.441 %. Both
files contain 16,922,476,352 B. Their contents differ, as their SHA-256
hashes confirm. The stock b10362 runtime evaluated the assisted arm.

Every figure above names its archive block. The archive also holds a
`ctl-published` block, which records a different file and different
numbers. This record cites blocks and edits no published artifact.

The archived report records a 65.0 % KLD reduction and a 6.31-point
same-top gain. These measure the whole pack, not isolated stack damage.
The [evidence record](evidence/0032/README.md#rebuy-evidence) preserves
those results and names the source block of each figure.

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
`{-1, 0, 1, 2}`. The same comment records two further facts. The patch
changes the unassisted path too, so `quantize_row_q2_0_ref` runs the
same Lloyd search at weight 1.0. The published numbers do not separate
the search gain from the importance-matrix gain. The existing Q4_0
helper supplies precedent, not an unchanged Q2_0 implementation. The
patch is a specification to reimplement and verify. vramfit must not
carry or apply it.

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

## Options and ADR-0016's original decision 2

That clause keeps matrix generation outside pack and calls pack a
“CPU subprocess driver.” It does not already authorize an in-process
vramfit tensor encoder. Merely running on the CPU does not settle
that boundary. ADR-0016's 2026-08-21 amendment also numbers a decision
2, and this section never means that one.

| Option | Relationship to the driver clause | Consequence |
| --- | --- | --- |
| Pre-encode selected Q2_0 tensors into a temporary base GGUF | A CPU subprocess preserves the driver's role. This ADR explicitly extends its responsibilities to that preprocessing step. | The probe proves stock passthrough. vramfit owns one encoder and selected-tensor GGUF rewriting. |
| Implement whole-file quantization inside vramfit | An in-process implementation contradicts the driver boundary. A subprocess preserves its literal form but replaces the delegated toolchain's responsibility. | Requires broader authority and ownership of every supported quantizer, override rule, metadata path, and file layout. |
| Add the encoder only to the scan meter | Leaves pack unchanged and therefore fits the driver clause. | Measures a hypothetical assisted artifact. Stock packing still emits unassisted Q2_0, so those prices cannot support the funded re-solve. |

The first option needs no upstream change and carries no third-party patch.
The second adds ownership the measured gain does not require.
The third cannot deliver the measured benefit or matching provenance.

## Decision

1. **Select pre-encoding through a vramfit-owned CPU subprocess.**
   Amend ADR-0016's original decision 2 to permit this preprocessing stage.
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
   This record states no token, because the string is a vocabulary
   decision and ADR-0018 renamed one such token once already.
   [Issue #599](https://github.com/Alberto-Codes/vramfit/issues/599)
   owns the naming, and the implementation waits for it.
   The implementation must carry the new identity through map fingerprints,
   checkpoints, recipes, and pack selection. It must record the encoder
   revision and matrix provenance with the packed result.

   A pre-encoded tensor packs **assisted**. The label follows the fit,
   not the binary that performed it. The matrix weighted the encoder's
   candidate selection, so those bytes count toward the artifact's
   assisted share. A Q2_0 tensor the matrix does not cover, and one an
   exclusion drops, packs unassisted as before. The open questions below
   record that no measurement separates the matrix's contribution from
   the search's, which this accounting assumes.

   The glossary's pack-side rule reads assisted as “its type reads the
   matrix.” That wording describes stock `llama-quantize`, and it scores
   a pre-encoded tensor unassisted. This record does not edit the
   glossary's pack-side assisted rule, because it stays Proposed.
   Acceptance carries the matching glossary amendment, and the open
   questions below track it.

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
   The open questions below bound what this re-solve's prices may claim,
   because the measured gain does not separate the search from the matrix.

5. **Correct ADR-0016's 2026-08-21 amendment, decision 2.**
   That clause states: “The asymmetry is a cost of the width, not a
   toolchain handicap. No type reaches an assisted fit at 2.25 bits on
   rows of 2688 and 1856.” The rebuy refutes it. `Q2_0` is 2.25
   effective bits on exactly those rows
   ([ADR-0028](0028-expert-stack-type-table.md) decision 1). The
   `c2-assisted-q2_0` arm reached an assisted fit there and took mean
   KLD from 0.204318 to 0.071428.

   The conclusion inverts. It was a toolchain handicap, because stock
   `quantize_q2_0` discards the matrix. The width admits an assisted
   fit, and the stock encoder declines to compute one.

   The amendment's decisions 1 and 3 stand. Its assisted-share figures
   stand, because they measure what stock llama.cpp packed. ADR-0028's
   consequence “At nominal 2 and 8 the importance matrix does not shape
   the stack quantization” also stands, because it describes stock
   llama.cpp and this record changes no stock behavior.

   ADR-0016 carries a contested note beside the clause, and the ADR
   index marks its row. This record is Proposed, so those markers read
   contested and not superseded. Acceptance turns them into the
   correction this decision states.

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
- **A pre-encoded tensor loses ADR-0012 decision 3's record-and-continue
  floor.** Today a layer that no override reaches takes the `--pure`
  floor. The pack step records it in `PackResult.floored_layers`, prints
  one `warning:` line, and finishes the artifact (ADR-0012, 2026-08-16
  amendment, issue #307). Pre-encoding removes that outcome for the
  tensors it rewrites. A pre-encoded Q2_0 tensor that the quantizer's own
  matching then floors is an existing Q2_0 tensor at a different resolved
  type, so `llama-quantize` exits 1 with `requantizing from type q2_0 is
  disabled`. The negative control in
  [transcript.txt](evidence/0032/transcript.txt) records that exit.
  The operator sees a mid-pack abort and no artifact, after the
  preprocessor already wrote a full-size temporary GGUF. Issue #305 tracks
  the matcher divergence that reaches this case, where an override passes
  the #303 check and still changes no type. The implementation must
  refuse before the preprocessor writes, not after.
- Same scan and pack semantics remove this encoder mismatch. They do not
  equate the torch scan frame with runtime damage or validate additive predictions.
- Acceptance requires a tiny end-to-end pack/load test with a stock runtime,
  plus scan reconstruction of the blocks that pack actually emits.
- The next task implements these requirements. The maintainer starts that
  task and the funded re-solve.
  [Issue #597](https://github.com/Alberto-Codes/vramfit/issues/597) closed
  with this record. Two open tickets carry the handoff under
  [chart #158](https://github.com/Alberto-Codes/vramfit/issues/158).
  [Issue #599](https://github.com/Alberto-Codes/vramfit/issues/599) names
  the successor token.
  [Issue #601](https://github.com/Alberto-Codes/vramfit/issues/601) builds
  the encoder and the preprocessor, and it waits for acceptance of this
  record. This change contains only the record and its evidence.

## Open questions

- The successor `within_group` token has no name. Decision 3 requires a
  distinct token and states none, because the string is a vocabulary
  decision. [Issue #599](https://github.com/Alberto-Codes/vramfit/issues/599)
  owns it. The implementation cannot write a map before it closes.
- ~~The glossary's pack-side assisted rule needs its matching amendment.
  Decision 3 rules a pre-encoded tensor assisted.
  [The glossary](../reference/glossary.md) still defines the pack sense
  by which binary reads the matrix. The two records disagree on this
  tensor class until acceptance carries the amendment.~~ Resolved
  2026-09-15 by acceptance. The glossary's pack-side rule now reads a
  pre-encoded tensor assisted and cites decision 3.
- The preprocessor's full-file cost stays unmeasured. The probe wrote
  36,864 Q2_0 payload bytes on a synthetic fixture. Decision 1 reads the
  f16 base and writes the temporary mixed GGUF beside it, so the two
  coexist and the pack does not exist yet. The temporary file is that
  base with eleven stacks swapped to 2.25 bits, which makes it somewhat
  smaller than the base and several times the 16,922,476,352 B pack. The
  30B target's f16 reference measures 63,181,504,640 B
  ([card ledger](../../publication/nemotron-30b-a3b-fit16gib/card-ledger.md)).
  Scratch space sized against the pack falls short. The implementation
  measures peak memory and disk before the funded run.
- Whether the shared encoder reaches pack without torch stays unsettled.
  Decision 2 owns the fit once, and the pack path calls it. The shipped
  assisted fit imports torch.
  [ADR-0005](0005-heavy-deps-as-extras.md)'s 2026-09-04 amendment gives
  `vramfit[gguf]` "gguf-py and numpy and no torch" for the reads pack
  does itself. [ADR-0008](0008-hexagonal-architecture.md)'s import-linter
  contract "No heavy ML deps outside the scan adapter package" forbids
  torch across `vramfit`, and its one carve-out reads
  `vramfit.adapters.outbound.scan.* -> torch`. The pack adapter imports
  neither torch nor anything under `scan` today. This record neither
  widens that carve-out nor rules a second torch-free fit, which
  decision 2 forbids. Acceptance on 2026-09-15 settled neither, so the
  boundary stays open. The implementing task cannot choose it.
  [Issue #601](https://github.com/Alberto-Codes/vramfit/issues/601)
  needs the maintainer's ruling before an implementation picks a path.
- Passthrough holds on b10362 (`4801e3c56`) and on no other build. The
  probe tested one quantizer. Decision 1 forbids `--allow-requantize`,
  so a build that drops the passthrough refuses the pack rather than
  requantizing the payload silently. A toolchain change repeats the probe.
- What the re-priced map does to the allocation stays unknown. The rebuy
  changed the encoder and kept the shipped recipe's allocation, so no
  measurement on record says whether cheaper nominal-2 cells move the
  solver's choices. Decision 4 requires the new map before anyone answers.
- No measurement separates the search gain from the matrix gain. The
  `c2-assisted-q2_0` arm ran the patched encoder
  ([evidence](evidence/0032/README.md#rebuy-evidence)), and that patch
  also runs the Lloyd search unassisted at weight 1.0. The 0.204318 to
  0.071428 result therefore credits both changes together, and a
  weight-1.0 search would deliver part of it on tensors the matrix never
  covers. Running the two apart settles it, and the split that
  [issue #601](https://github.com/Alberto-Codes/vramfit/issues/601)
  builds is where that measurement belongs. Until the separation exists,
  decision 3's assisted-share accounting and any nominal-2 cell price
  drawn from it are not publishable numbers.
