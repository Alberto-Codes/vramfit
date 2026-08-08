# ADR-0022: Tensor-level precision arrives as within-layer protections

- **Status:** Proposed
- **Date:** 2026-08-07
- **Relation:** extends [ADR-0012](0012-gguf-type-mapping.md)
  decision 2. The amendment bullet lands there when this record is
  accepted.

## Context

The twelfth data point (in
[evaluating packed models](../explanation/evaluating-packed-models.md))
crossed a line no quantfit artifact had crossed. Probe G1 — the no-2
recipe's pack layout plus `attn_v` at `Q5_K` on 44 layers — beat the
baseline's mean KLD in budget: 0.1512 against 0.1584, 4.5 % lower,
outside the combined error bars. The probe was hand-driven through
`llama-quantize` overrides. The pipeline cannot reproduce it:
recipes assign one precision per layer group, and the GGUF backend
rejects anything finer (ADR-0012 decision 2, the declared v1
boundary).

The probe also fixed the shape the feature must take. G1 came from a
rule — hold one tensor class above the group floor — priced by size
alone, to the byte, against 102.1 MiB of headroom. No per-tensor
damage number entered the decision. Two candidate scopes existed for
lifting the boundary:

1. **True tensor granularity.** Scan per tensor, solve per tensor.
   The model has 438 weight tensors (the imatrix coverage census).
   At four precisions that is 1,752 cells. The group-level assisted
   scan measured ~7 minutes per cell — about 8.5 days of scanning,
   against 37 hours for the 328-cell group scan. The cost buys
   in-frame per-tensor prices, and
   [ADR-0021](0021-runtime-frame-measurement.md) closed the
   scan-frame refinement lane: the twelfth data point's two
   per-tensor phenomena — the quantizer fit collapse and the
   two-chunk instability — are both invisible in the scan frame.
2. **Protection rules.** Keep groups as the scan and assignment
   unit. Let the recipe hold named tensors above their group's
   precision. Price the hold by size only. This is exactly the
   mechanism G1 shipped with.

One discovery constrains any implementation. Under our importance
matrix, `Q5_K` reconstructs the front-stack `attn_v` tensors *worse*
than `Q3_K` — layer 1 by 5.1× (RMSE 0.0241 against 0.0048), layers
2 and 5 by 1.9× and 1.3×. The collapse reproduces bit-for-bit across
llama.cpp builds, and bartowski's imatrix avoids it 10×. A type
promotion under a fixed imatrix is not guaranteed to improve a
tensor. The 47-layer build that ignored this scored 9.594 PPL and
would have passed the smoke test. A per-tensor reconstruction check
— seconds of CPU through gguf-py — caught it.

## Decision

1. **Tensor-level precision enters the recipe as protections, not
   tensor-level groups.** A protection is an ordered glob pattern
   over tensor names plus a floor precision, for example
   `--protect "*.self_attn.v_proj=5"`. A protected tensor packs at
   the higher of its group's assignment and the floor. Layer groups
   stay the unit of scanning and assignment.
2. **The solver prices a protection by size only.** The byte cost
   of every hold enters the budget arithmetic before any downgrade
   runs. Predicted damage stays the group-level sum — the map has
   no per-tensor damage, and the solver does not invent one. The
   recipe records the protection patterns verbatim and the resolved
   (tensor, precision) pairs.
3. **The sensitivity map gains per-tensor sizes.** The scan records
   each group member's bytes at reference precision. For existing
   maps, a backfill reads the checkpoint's safetensors headers —
   a JSON parse, no torch — and writes an annotated map copy (the
   map-copy mechanism from ADR-0021 decision 4). A protection
   against a map without tensor sizes refuses with a named error.
4. **Pack drives protections as overrides ahead of group
   overrides.** `llama-quantize` applies the first matching
   pattern, so order encodes priority: `blk\.3\.attn_v` before
   `blk\.3\.`. A fixed class table maps HF tensor suffixes to GGUF
   suffixes (`v_proj` → `attn_v`, `o_proj` → `attn_output`, and
   peers). A protected tensor the table cannot map raises
   `PackError`.
5. **A reconstruction check guards every protected pack made with
   an imatrix.** The check dequantizes each protected tensor from
   the packed file and compares it against the f16 base. A
   protected tensor that reconstructs worse than its unprotected
   type is collapsed. The check names it and fails the pack
   acceptance. The smoke test cannot see this failure mode — the
   collapsed G1 build smoked clean.
6. **Tensor-granularity scanning stays out.** `--group-by tensor`
   remains unbuilt, and the GGUF backend keeps rejecting
   tensor-level *groups*. The 8.5-day scan buys prices in a frame
   ADR-0021 already closed for sub-4-bit decisions.

## Consequences

- G1 becomes reproducible from the CLI: `plan --protect` plus
  `pack`, no hand scripts. The KLD-crossing artifact gets a recipe
  file with provenance.
- The recipe schema gains an additive protections record. The map
  schema gains additive per-tensor sizes. Both changes are
  additive, so neither `quantfit_schema` bumps.
- Damage accounting is honestly incomplete. The plan's predicted
  damage ignores protection benefit. The evidence page carries the
  benefit claim, measured per artifact, not the recipe.
- The solver cannot trade a protection against a downgrade —
  protections carry no damage price, so the user supplies the
  judgment. The pin mechanism has the same character.
- Pack gains a mandatory post-pack step on protected packs, and a
  collapse means a second pack round with the named tensors
  excluded. G1 needed exactly one such round (layers 1, 2, 5).

## Open questions

- The reconstruction check's comparison reference. Candidates: a
  second pack at base types (4.6–17 minutes), an unweighted
  gguf-py re-quantize of the single tensor (cheap, but it cannot
  reproduce the assisted fit), or a sibling artifact when one
  exists. G1 used the sibling.
- Refuse-and-name against demote-and-repack. The check can fail
  the pack and list collapsed tensors, or pack can exclude them
  and repack once automatically.
- Whether evidence-backed default protections ship (`attn_v=5` on
  a 3-bit floor rediscovers the baseline's toolkit) or every
  protection stays user-directed.
- Whether the backfill is a CLI command or a repository script.
- Whether the reconstruction check also runs on unprotected packs.
  It is cheap, and it would have caught nothing so far — every
  collapse to date involved a promotion under an imatrix.
