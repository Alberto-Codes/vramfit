# ADR-0029: Plan reads tensor sizes from a source independent of the map

- **Status:** Accepted
- **Date:** 2026-08-19 (accepted 2026-08-19)
- **Amends:** [ADR-0007](0007-recipe-solver-strategy.md). The solver
  prices every discovered group, not only the groups its input map
  carries. Decision 4 also edits tables in
  [ADR-0012](0012-gguf-type-mapping.md) decision 1,
  [ADR-0013](0013-runtime-capability-in-recipes.md) decision 1, and
  [ADR-0028](0028-expert-stack-type-table.md) (listed 2026-09-04,
  #362).
- **Origin:** Maintainer ruling 2026-08-18 on #337, recorded in chart
  #158's Notes. #345 carries the source and the port's shape.
- **Note (2026-08-19, issue #345):** decision 7's closing sentence
  called the root mismatch "its third appearance". That reading is
  wrong and the decision is unaffected. Measured the same day: both
  maps this project holds emit `model.` group names, at
  `sensitivity-32k-kquant-imx.json` and
  `sensitivity-32k-q0-ref-stacks.json`, so the scan is consistent. The
  `backbone.` names live in one hand-authored campaign script, outside
  this repository, which hardcodes the string at line 28 of
  `campaign/build_recipes.py` in #300's run root.
  `src/vramfit/adapters/outbound/gguf/types.py` accepts either root, so
  those recipes packed and no gate reported it.
  So #301's `vramfit validate` refusal is a hand-authored artifact
  diverging from tool output, not a tool-level naming split. **Decision
  7 stands on the checkpoint-to-map direction alone**, where the split
  is real: the checkpoint roots at `backbone.` and every map roots at
  `model.`.
- **Amendment (2026-09-04, issue #362):** a fact-check pass and a
  peer-review pass ran after the merge, against the build in #358 and
  PR #360. Neither invalidates a decision. This amendment records what
  the record omitted.

    **Three gaps the build ruled.** The record lists three open
    questions and says none gates the build. Three it does not list
    did. #360 ruled each under a stated assumption.

    1. *Which naming root the emitted recipe carries.* Decision 7 says
       a utility reconciles the roots and never says in which
       direction. ADR-0012's 2026-08-12 (#180) amendment refuses a
       recipe naming two roots. #360 emits `model.` for every group
       (`MAP_ROOT` in `vramfit.domain.sizes`).
    2. *Which byte count prices a group both the map and the source
       carry.* Open question 2 asks whether `plan` warns or refuses
       on a disagreement. It never asks which number prices the
       group. #360 keeps the map's `bytes_fp16`.
    3. *Whether a foreign root refuses or reaches no group.* The
       Consequence at decision 7 says both. A source that reaches no
       group returns zero bytes per group, which is #337's failure
       with a new cause. #360 refuses, with `SizeSourceError` under
       the `VramfitError` root. The Consequence below is corrected.

    Decision 3 also does not say what `Assignment.damage` carries for
    an uncovered group. #360 writes 0.0, because a group held at
    reference is unperturbed. #301 records that an all-zero damage
    column broke the validation pass once.

    **Two silent table edits.** Decision 4 edits tables in two
    Accepted ADRs beyond ADR-0013, which #359 carries.
    ADR-0012 decision 1's type table gains `16 -> f16`, forced by
    `test_type_table_covers_the_llama_cpp_capability_set`, and
    `BASE_FTYPE_BY_BITS` gains `16 -> F16` beside it. ADR-0028's
    expert-stack tables gain a 16 row in `EXPERT_STACK_EFFECTIVE_BITS`
    and `EXPERT_STACK_TYPE_BY_BITS`. Decision 3 assigns every
    uncovered group, and `solver.py` routes an expert-stack group
    through the stack table. No uncovered stack group exists on this
    target's #328 map, where all 164 uncovered groups are dense. The
    decision is written as general. Both type names check out against
    the instrument: `parse_ggml_type` matches `ggml_type_name`
    case-insensitively, and `F16` is a valid positional ftype
    (`LLAMA_FTYPE_MOSTLY_F16`). The **Amends** header now lists all
    three records.

    **Two wrong statements.** Decision 5 says a bare
    `Mapping[str, int]` would mirror
    `ImatrixCountSource.expert_stack_counts`. That port returns
    `Mapping[str, tuple[int, ...]]`. The decision it supports is
    unaffected. ADR-0007 carried no reciprocal note while the index
    read "amended by 0029". The note landed 2026-09-04.

    **Smaller corrections**, applied in place: decision 4's "three
    tickets" names the one it reaches. Decision 7's "third
    appearance" sentence carries the 2026-08-19 retraction marker.
    The `validate` Consequence states which of #301's three blockers
    the passthrough clears. Glossary entries for uncovered group,
    passthrough precision, size source, and root table landed in
    PR #360.
- **Note (2026-09-04, issue #409):** decision 4's 16.0 bits per
  weight holds for a class the quantizer accepts, which packs through
  the `f16` override. It does not hold for an unquantizable class.
  `convert_hf_to_gguf.py` writes `ffn_gate_inp` and `ssm_conv1d` at
  F32 whatever `--outtype` asks, and the quantizer drops the override,
  so the packed file holds those classes at 32 bits. Publication #2's
  recipe priced its 46 passthrough groups 16,923,492 B under the
  packed bytes, against a 16,874,535 B margin, and fit on the residual
  overhead's over-reservation of the quantized classes. The passthrough
  now prices each group from the convert dtype table in
  `vramfit.domain.runtime`, 32.0 bits on both llama.cpp classes, and
  16.0 elsewhere. The recipe still records nominal 16. The scan skips
  those classes at discovery (#204), so a new map carries no cell for
  them. The size source keys such a tensor by its own name under every
  granularity, so it holds uncovered. Under `layer` granularity a
  layer group that absorbed it would hide its bytes behind a covered
  name.

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

   **Amendment (2026-09-04, issue #409):** a target whose family holds
   a class the quantizer refuses requires a size source. The scan
   skips that class at discovery (#204), so the map carries no bytes
   for it, and only the size source prices it. `plan` without
   `--checkpoint` refuses on such a map and names the flag. The
   refusal keys on the map naming the class's module (`mixer`) and
   none of the module's refused classes. A map that carries the class
   predates the skip and prices it itself. A runtime that serves no
   reference precision refuses the held class too. The class has no
   width on that runtime until a pack path for it exists, and no scan
   supplies one.

4. **The recipe gains an F16 passthrough precision.**
   `EFFECTIVE_BITS[llama.cpp]` carried 8, 6, 5, 4, 3, and 2 before
   this build (`src/vramfit/domain/runtime.py`). No row expressed an
   unquantized group, so decision 3's assignment clause was
   unbuildable without one. The passthrough reaches the precision set,
   the effective-bits table at 16.0 bits per weight, and the GGUF type
   map as `F16`. The table now carries 16 as well (#359).

   This closes a gap that ~~reaches three tickets~~ #301 records
   (corrected 2026-09-04, #362). #301 records the same missing
   passthrough disabling `vramfit validate`, because
   `Q0_REF_PRECISIONS` is `(8, 4, 2)` and no candidate holds a group at
   reference.

5. **The port returns a per-tensor record carrying bytes and dtype, not
   a bare integer.** `TensorSizeSource.tensor_sizes` returns a
   `Mapping[str, TensorSize]` keyed by checkpoint tensor name.
   `TensorSize` is a frozen domain dataclass holding `dtype` and
   `bytes`.

   A bare `Mapping[str, int]` ~~would mirror
   `ImatrixCountSource.expert_stack_counts` and~~ would hardcode the
   project's bf16 convention into the port. (That port returns
   `Mapping[str, tuple[int, ...]]`. Corrected 2026-09-04, #362.)
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
   failure it prevents, from #177. The imatrix name table supported
   `model.layers.N.` and `backbone.layers.N.` and no others at the
   time (the domain's `NAME_TABLE_ROOTS` owns the list today, per
   ADR-0012's 2026-09-04 amendment, #208):

   > A prefix wildcard mapped a vision tower's `layers.5` onto the
   > decoder's `blk.5` and would have priced it against the wrong
   > columns.

   A size source carries the same hazard. It would price a vision
   tower's tensors against a decoder group.

   One shared utility serves `plan` and `validate` together. #301
   records the same mismatch disabling `vramfit validate`, ~~and this
   ADR is its third appearance~~ (retracted by the 2026-08-19 note
   above, marked 2026-09-04, #362). The utility is shared because the
   checkpoint-to-map split is real for both commands.

## Open questions

- ~~**Whether an uncovered group enters the pinnable set.** Decision 3
  gives a caller no lever over the bf16 default, and `--pin` does not
  supply one today. An uncovered group is therefore unpinnable, and
  the consequence below is unavoidable rather than chosen.~~
  **Ruled 2026-08-22 by ADR-0007's #301 amendment: it does.** A pin
  reaches any checkpoint-discovered group, and a pinned uncovered
  group prices at the pinned width. The build lives in
  `vramfit.domain.pins`. ~~One caveat stands: a broad pattern such as
  `--pin "*=8"` also lands on unquantizable-class groups, which
  refuse every pin, so the ruled MoE mix pins its dense classes by
  name.~~ **Ruled 2026-09-04 by ADR-0007's #371 amendment:** a
  multi-group pattern skips each held group it sweeps and `plan`
  warns, so `--pin "*=8"` pins every group the quantizer accepts.

- **Whether a map and the source may disagree, and what `plan` does when
  they do.** The map records what the scan discovered. The source reports
  what the checkout holds now. A rename or a revision moves one and not
  the other. The map already carries `bytes_fp16` per group, so a
  disagreement is detectable. Whether it warns or refuses is unruled.

    **One case ruled 2026-09-04 (#409 review): `plan` warns on a
    folded refused class and keeps both prices.** A `layer` map
    scanned before the discovery skip (#204) lists `mixer.conv1d`
    and `mixer.gate` inside `model.layers.N`. The source holds each
    by its own name, so the plan prices the tensor twice: inside the
    group at its assigned width, and held at the convert dtype. The
    direction is conservative, about 9 MB plus the F32 bytes on the
    30B. `held_class_overlaps` in `vramfit.domain.sizes` finds each
    pair from the map's tensor list, and `plan` warns naming the
    map, the group, and the tensors. The prices stand. The group's
    `bytes_fp16` and damage curve were measured with the tensor
    inside, so no arithmetic separates them without a re-scan.
    A schema bump was the other remedy. It refuses every pre-skip
    map, including a map planned without `--checkpoint`, which
    prices the class correctly on its own. The tensor list already
    carries the evidence, so the warning costs no field. The
    general disagreement, a `bytes_fp16` mismatch on one group,
    stays unruled.

- ~~**Whether the passthrough's 16.0 bits per weight is exact for every
  writer.** GGUF `F16` stores two bytes per weight with no block
  overhead. A checkpoint stored at bf16 converts to f16 without a size
  change. No measurement confirms this end to end.~~
  **Answered 2026-09-04 on #409: it is not.** The converter writes an
  unquantizable class at F32, and the 2026-09-04 note above records
  the pricing. A per-type check against a synthetic packed layout now
  holds the prediction at the file's bytes
  (`tests/unit/adapters/test_predicted_vs_packed.py`).

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
  rented pod and only small artifacts return. The reference box holds
  this target's 61.32 GiB checkpoint today, so the constraint binds a
  future target rather than this one. The deferred range-request ticket
  carries the general case.

- **A new port requires a verified-fake contract suite** under ADR-0009,
  and an outbound adapter under ADR-0008. The lazy-import gate applies:
  a module with an optional import needs a `[[tool.ty.overrides]]` entry
  or CI type-checking fails where local gates pass.

- **The root table becomes a maintained list, and a target it does not
  name refuses.** Decision 7 bars a prefix wildcard, so a checkpoint
  rooted at neither `model.` nor `backbone.` ~~reaches no group~~
  **refuses with `SizeSourceError`** (corrected 2026-09-04, #362,
  which is what #360 built). That is the designed outcome. A silent wildcard match would price a vision
  tower's tensors against a decoder group, which #177 measured and
  #186 fixed by adding names rather than a wildcard. Each new target
  costs one table entry.

- **The F16 passthrough clears one of `vramfit validate`'s three
  blockers.** #301 records that the validation pass cannot hold a
  group at reference, so the clean experiment of perturbing the 46
  stacks and holding the rest unchanged has no form today. The
  passthrough gives it one. #301's other two blockers stand
  (corrected 2026-09-04, #362).

- **#350's frame skew stops being theoretical.** The `q0-ref` map prices
  nominal 4 unassisted, and `pack` applies `Q4_0` assisted through
  `quantize_row_q4_0_impl`. `quantize_q2_0` discards the matrix, so the
  skew lands on one width only. ADR-0018 predicts an unassisted meter
  sits at or above the packed damage for a covered tensor, so the map
  overstates nominal-4 damage against the artifact.

    No solver has read that skew yet, because every arm on this target
    was hand-authored (#301). This ADR is what first lets the greedy rule
    consume it, and the rule spends the cheap width against exactly that
    ratio. #350 resolved 2026-08-21: ADR-0018's amendment rules the
    `q0-imx` build, and #384 carries the assisted re-scan. A recipe
    planned before that map lands reads a nominal-4 column measured in a
    frame the pack does not apply, and the amendment's decision 5 makes
    it state the gap.
