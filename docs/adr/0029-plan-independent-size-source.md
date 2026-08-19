# ADR-0029: Plan reads tensor sizes from a source independent of the map

- **Status:** Proposed
- **Date:** 2026-08-19
- **Amends:** [ADR-0007](0007-recipe-solver-strategy.md). The solver
  prices every discovered group, not only the groups its input map
  carries.
- **Origin:** Maintainer ruling 2026-08-18 on #337, recorded in chart
  #158's Notes. #345 carries the source and the port's shape.

## Context

`vramfit plan` treats its input map's group list as the model.
`src/vramfit/domain/solver.py:557` sums the weight budget over
`sensitivity_map.groups` and reads no other size source. A group the map
omits contributes zero bytes.

#282 shipped `vramfit scan --groups`, so a map may now cover part of a
model by design. #328 used it. That run scanned 46 of the 210 stack
groups the 30B target discovers, at $4.23.

Those 46 groups hold 93.0 % of the parameters (#160). So `plan` against
that map prices 46 groups, counts 164 as zero bytes, and reports a fit
the packed model does not honor. The recipe omits 7.0 % of the weights.

The maintainer ruled the shape on 2026-08-18:

> `plan` gets a size source independent of the sensitivity map, and an
> uncovered group holds at reference precision.

The ruling refused two cheaper shapes. Recording a discovered-group
count on `ScanMeta` makes the failure visible rather than impossible.
Documentation alone accepts it. Only an independent size source stops
the map from defining the model.

Three facts measured on 2026-08-19 constrain the source.

The safetensors **index** carries no shapes. `model.safetensors.index.json`
holds `metadata` and `weight_map` only, and a `weight_map` entry maps a
tensor name to a shard filename. Its `metadata` does carry
`total_parameters` at 31,577,937,344 and `total_size` at 65,842,365,568.

The **shard headers** carry dtype and shape.
`scripts/backfill_tensor_sizes.py` already reads them for the same class
of fact, under ADR-0022 decision 4, at "a JSON parse, no torch". That
method satisfies ADR-0005's constraint that the plan step import without
torch.

A **base GGUF** exists only after a pack, and `plan` runs before packing.

## Decision

1. **The source is the checkpoint's safetensors shard headers.** Not the
   index, which carries no shapes. The adapter reads each shard's
   little-endian u64 header length, then parses that many bytes of JSON.
   It imports no torch.

   ADR-0022's Consequences anticipated this caller: the backfill script
   "earns a CLI command when a second consumer appears". `plan`'s size
   source is that second consumer. The adapter and the script share the
   header reader rather than duplicating it.

   Reading headers over HTTP range requests is **out of scope**. No
   record authorises a network read inside `plan`. A separate ticket
   carries it.

2. **The source excludes the MTP block.** The 30B checkpoint carries 270
   tensors under the `mtp.` root, against 6,242 under `backbone.` and one
   `lm_head`. Chart #158 records that GGUF numbers one layer stack, so
   backbone and MTP cannot pack together, and that a recipe states
   whether it packs the block at all. A source summing the whole
   checkpoint would overstate the weight budget by the block's 1.335B
   parameters.

3. **An uncovered group prices at bf16, and `plan` assigns every group it
   prices.** Two clauses, and the second carries the correctness.

   Reference precision means bf16, per the glossary: "the unquantized
   (bf16) model that perturbed models are compared against". Pricing an
   unmeasured group at the highest candidate would have the solver invent
   a fact. ADR-0022 decision 2 refuses that move for damage in the same
   words: "the solver does not invent one". Over-reservation is also the
   direction the records prefer, recorded as "the safe direction" in
   ADR-0006 and three explanation pages.

   A pricing convention alone cannot make the prediction true.
   `LlamaCppPacker.pack` runs `llama-quantize --pure` with
   `base_type(recipe)`, which is the recipe's precision **floor**. Any
   tensor no override covers reaches the artifact at that floor. So a
   recipe that prices a group at bf16 and then leaves it unnamed
   reproduces the #337 failure with the sign reversed. `plan` therefore
   emits an assignment for every group it prices.

4. **The recipe gains an F16 passthrough precision.**
   `EFFECTIVE_BITS[llama.cpp]` carries 8, 6, 5, 4, 3, and 2
   (`src/vramfit/domain/runtime.py:64`). No row expresses an unquantized
   group, so decision 3's assignment clause is unbuildable without one.
   The passthrough reaches the precision set, the effective-bits table at
   16.0 bits per weight, and the GGUF type map as `F16`.

   This closes a gap that reaches three tickets. #301 records the same
   missing passthrough disabling `vramfit validate`, because
   `Q0_REF_PRECISIONS` is `(8, 4, 2)` and no candidate holds a group at
   reference.

5. **The port returns a per-tensor record carrying bytes and dtype, not
   a bare integer.** `TensorSizeSource.tensor_sizes` returns a
   `Mapping[str, TensorSize]` keyed by checkpoint tensor name.
   `TensorSize` is a frozen domain dataclass holding `dtype` and
   `bytes`.

   A bare `Mapping[str, int]` would mirror
   `ImatrixCountSource.expert_stack_counts` and would hardcode the
   project's bf16 convention into the port.
   `scripts/backfill_tensor_sizes.py` states that convention as
   "element counts at 2 bytes per parameter". A checkpoint stored at
   fp8 or at fp32 prices differently, and reference precision is
   defined by what the checkpoint holds. Carrying the dtype lets the
   domain derive reference bytes instead of assuming two bytes per
   parameter. The adapter reports the header's dtype string verbatim
   and computes no convention of its own.

6. **Group aggregation lives in the domain. The adapter returns raw
   checkpoint tensor names.** The checkpoint stores routed experts
   individually. Layer 1's `up_proj` stack is 128 tensors,
   `backbone.layers.1.mixer.experts.0` through `127`. The map records
   the fused stack as one tensor at 1,277,165,568 bytes, which is
   638,582,784 parameters, which is 128 times 2688 times 1856. So
   something must sum 128 entries per stack group.

   That summation reads model structure, and structure is already a
   domain concept. `is_expert_stack` sits at
   `src/vramfit/domain/scan.py:126`. ADR-0008 keeps the domain pure, and
   an adapter that grouped tensors would become an authority on model
   structure. The adapter therefore stays a reader of bytes.

7. **A domain utility reconciles the naming roots, against an explicit
   root table and never a prefix wildcard.** The checkpoint roots at
   `backbone.`. The scan's discovered groups root at `model.`, measured
   on #328. ADR-0012 decision 2, as amended 2026-08-12, carries the
   naming families on the GGUF pack side only, so no record reconciles
   the roots for `plan`.

   The wildcard prohibition is not caution. Chart #158 records the
   failure it prevents, from #177: the imatrix name table supports
   `model.layers.N.` and `backbone.layers.N.` and no others, because "a
   prefix wildcard mapped a vision tower's `layers.5` onto the decoder's
   `blk.5` and would have priced it against the wrong columns". A size
   source carries the same hazard. It would price a vision tower's
   tensors against a decoder group.

   One shared utility serves `plan` and `validate` together. #301
   records the same mismatch disabling `vramfit validate`, and this ADR
   is its third appearance.

## Open questions

- **Whether an uncovered group enters the pinnable set.** Decision 3
  gives a caller no lever over the bf16 default, and `--pin` does not
  supply one today. `_expand_pins` matches a pattern against
  `map_.groups` (`src/vramfit/domain/solver.py:256`), so a pin reaches
  only groups the map carries and raises `PinError` otherwise. An
  uncovered group is therefore unpinnable, and the consequence below is
  unavoidable rather than chosen.

    Admitting the source's groups to the pinnable set would let a caller
    write `--pin "*=8"` and recover the 35-stack mix deliberately. It
    would also let a pin introduce a group name, which ADR-0007's "a
    pattern matching zero groups is a hard error" was written to
    prevent. The two readings conflict and a ruling settles them.

- **Whether a map and the source may disagree, and what `plan` does when
  they do.** The map records what the scan discovered. The source reports
  what the checkout holds now. A rename or a revision moves one and not
  the other. The map already carries `bytes_fp16` per group, so a
  disagreement is detectable. Whether it warns or refuses is unruled.

- **Whether the passthrough's 16.0 bits per weight is exact for every
  writer.** GGUF `F16` stores two bytes per weight with no block
  overhead. A checkpoint stored at bf16 converts to f16 without a size
  change. No measurement confirms this end to end.

## Consequences

- **The ruled mix moves from 35 of 46 expert stacks at `Q4_0` to 23 of
  46, when planning from #328's 46-group map.** The 164 uncovered dense
  groups price at 4.163 GiB against 2.212 GiB at nominal 8. That leaves
  11.613 GiB for the expert stacks against 13.564 GiB, which is 3.396
  bits per expert weight against 3.967.

    That cost is the price of an unmeasured dense block rather than a
    defect. It is recoverable. #328's cost table prices the whole
    210-group scan at $18.42, against the $4.23 already spent.

- **`plan` no longer silently omits weights.** A partial map produces a
  recipe covering every discovered group, so `predicted_total_bytes`
  states the whole model.

- **A whole-model recipe becomes emittable for the first time on this
  target.** Every arm #300 packed was a hand-authored 100-assignment
  recipe, because `plan` copies its input map's group names (#301). A
  solver reading #328's stack-keyed map now prices 210 groups. The map
  already ranks `up_proj` against `down_proj`, at 23 of 23 MoE layers, so
  the solver gains that distinction with no new term.

- **The plan step gains a checkpoint dependency it did not carry.**
  `plan` previously read one JSON map. It now also reads the
  checkpoint's shard headers, which requires the checkpoint on the
  machine running `plan`. Chart #158 records that big files stay on the
  rented pod and only small artifacts pull back. The reference box holds
  this target's 61.32 GiB checkpoint today, so the constraint binds a
  future target rather than this one. The deferred range-request ticket
  carries the general case.

- **A new port requires a verified-fake contract suite** under ADR-0009,
  and an outbound adapter under ADR-0008. The lazy-import gate applies:
  a module with an optional import needs a `[[tool.ty.overrides]]` entry
  or CI type-checking fails where local gates pass.

- **The root table becomes a maintained list, and a target it does not
  name refuses.** Decision 7 bars a prefix wildcard, so a checkpoint
  rooted at neither `model.` nor `backbone.` reaches no group. That is
  the designed outcome. A silent wildcard match would price a vision
  tower's tensors against a decoder group, which #177 measured and
  #186 fixed by adding names rather than a wildcard. Each new target
  costs one table entry.

- **The F16 passthrough reaches `vramfit validate`.** #301 records that
  the validation pass cannot hold a group at reference, so the clean
  experiment of perturbing the 46 stacks and holding the rest unchanged
  has no form today. The passthrough gives it one.

- **#350's frame skew stops being theoretical.** The `q0-ref` map prices
  nominal 4 unassisted, and `pack` applies `Q4_0` assisted through
  `quantize_row_q4_0_impl`. `quantize_q2_0` discards the matrix, so the
  skew lands on one width only. ADR-0018 predicts an unassisted meter
  sits at or above the packed damage for a covered tensor, so the map
  overstates nominal-4 damage against the artifact.

    No solver has read that skew yet, because every arm on this target
    was hand-authored (#301). This ADR is what first lets the greedy rule
    consume it, and the rule spends the cheap width against exactly that
    ratio. #350 carries whether the meter consumes the matrix. A recipe
    planned before that ruling reads a nominal-4 column measured in a
    frame the pack does not apply.
