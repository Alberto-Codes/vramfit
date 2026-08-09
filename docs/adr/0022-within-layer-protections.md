# ADR-0022: Tensor-level precision arrives as within-layer protections

- **Status:** Accepted
- **Date:** 2026-08-07 (accepted 2026-08-08)
- **Extends:** [ADR-0012](0012-gguf-type-mapping.md) decision 2. The
  amendment bullet lands there when this record is accepted.
- **Amendment (2026-08-09):** [ADR-0023](0023-imatrix-exclusions.md)
  adds a second remedy to decision 6's refusal — keep the promotion
  and drop the tensor's imatrix row (`--exclude-imatrix`).

## Context

The twelfth data point (in
[evaluating packed models](../explanation/evaluating-packed-models.md))
crossed a line no quantfit artifact had crossed. Probe G1 held
`attn_v` at `Q5_K` on 44 layers of the no-2 recipe's pack layout.
It beat the baseline's mean KLD in budget: 0.1512 against 0.1584,
4.5 % lower, outside the combined error bars. The maintainer drove
the probe by hand through `llama-quantize` overrides. The pipeline
cannot reproduce it. Recipes assign one precision per layer group,
and the GGUF backend rejects anything finer (ADR-0012 decision 2,
the declared v1 boundary).

The probe also fixed the shape the feature must take. G1 came from
a rule: hold one tensor class above the group's precision. Size
alone priced the rule, to the byte, against 102.1 MiB of headroom.
No per-tensor damage number entered the decision. Two candidate
scopes existed for lifting the boundary:

1. **True tensor granularity.** Scan per tensor, solve per tensor.
   The imatrix coverage report counts 438 weight tensors (437
   covered). At four precisions that is 1,752 cells. The
   group-level assisted scan measured ~7 minutes per cell — about
   8.5 days of scanning, against 37 hours for the 328-cell group
   scan. The cost buys per-tensor prices inside the scan frame,
   and the twelfth data point's two per-tensor phenomena are both
   invisible there: the quantizer fit collapse and the two-chunk
   instability.
2. **Protection rules.** Keep groups as the scan and assignment
   unit. Let the recipe hold named tensors above their group's
   precision. Price the protection by size only. This is exactly
   the mechanism G1 shipped with.

One discovery constrains any implementation. Under our importance
matrix, `Q5_K` reconstructs the front-stack `attn_v` tensors
*worse* than `Q3_K`. Layer 1 collapses 5.1× (RMSE 0.0241 against
0.0048), layers 2 and 5 by 1.9× and 1.3×. The fit collapse
reproduces bit-for-bit across llama.cpp builds, and bartowski's
imatrix avoids it 10×. A type promotion under a fixed imatrix is
not guaranteed to improve a tensor. The 47-layer build that
ignored this scored 9.594 PPL and would have passed the smoke
test. A per-tensor reconstruction check (seconds of CPU through
gguf-py) caught it.

## Decision

1. **Tensor-level precision enters the recipe as protections, not
   tensor-level groups.** A protection is an ordered glob pattern
   over tensor names plus a protection floor, for example
   `--protect "*.self_attn.v_proj=5"`. The pattern language is
   fnmatch, the same as `--pin` — never regex. A protected tensor
   packs at
   the higher of its group's assignment and the floor. Layer
   groups stay the unit of scanning and assignment.
2. **The solver prices a protection by size only, at effective
   bits.** In every candidate evaluation, a protected tensor
   prices at the higher of the candidate precision and its floor,
   through the runtime's effective-bits table (ADR-0014).
   Downgrading a protected group therefore frees fewer bytes, and
   the damage-per-byte ranking shifts accordingly. Predicted
   damage stays the group-level sum — the map has no per-tensor
   damage, and the solver does not invent one. The recipe records
   the protection patterns verbatim and the resolved
   (tensor, precision) pairs.
3. **Plan validates protections at solve time.** A floor the
   target runtime cannot serve is rejected through the ADR-0013
   capability table. A pattern that matches no tensor is rejected
   as a pin's no-match is. A protection on a flag-driven group
   (the embedding or the output head) is rejected with a pointer
   to `--pin` — those groups hold one tensor each. A floor at or
   below every matched group's assignment draws a no-op warning.
   Nothing about a protection is silent.
4. **The sensitivity map gains per-tensor sizes.** The scan
   records each group member's bytes at reference precision. For
   existing maps, a backfill reads the checkpoint's safetensors
   headers — a JSON parse, no torch — and writes an annotated map
   copy (the map-copy mechanism from ADR-0021 decision 4). Plan
   refuses a protection against a map without tensor sizes and
   names the missing field.
5. **Pack drives the resolved (tensor, type) pairs, never the raw
   floors.** Pack derives the quantizer's escaped regex overrides
   from the resolved pairs — user glob input never reaches
   `llama-quantize`. Protection overrides precede group overrides:
   the quantizer applies the first matching pattern, so order
   encodes priority (`blk\.3\.attn_v` before `blk\.3\.`). Emitting
   resolved types keeps a protection from demoting a tensor whose
   group assignment exceeds the floor. A fixed class table maps HF
   tensor suffixes to GGUF suffixes (`v_proj` → `attn_v`,
   `o_proj` → `attn_output`, and peers). A protected tensor the
   table cannot map raises `PackError`.
6. **A reconstruction check guards every protected pack made with
   an imatrix.** The check dequantizes each protected tensor from
   the packed file and compares it against the f16 base. The
   criterion: the tensor must reconstruct closer to f16 than it
   does at its unprotected type. The check marks a failing tensor
   as collapsed, names it, and fails the pack acceptance. How the
   check obtains the unprotected reference is an open question
   below. The smoke test cannot see this failure mode — the
   collapsed G1 build smoked clean.
7. **Tensor-granularity scanning stays out.** `--group-by tensor`
   remains unbuilt, and the GGUF backend keeps rejecting
   tensor-level *groups*. The 8.5-day scan prices tensors in a
   frame that sees neither phenomenon the twelfth data point
   surfaced.

## Consequences

- G1 becomes reproducible from the CLI: `plan --protect` plus
  `pack`, no hand scripts. The KLD-crossing artifact gets a recipe
  file with provenance.
- The recipe schema bumps. A schema-2 reader that dropped the
  protections record would silently pack a different artifact than
  the recipe intends. ADR-0013 already ruled that silent-drop case
  breaking. The map's per-tensor sizes are informational and stay
  within map schema 1.
- Damage accounting is honestly incomplete. The plan's predicted
  damage ignores protection benefit. The evidence page carries the
  benefit claim, measured per artifact, not the recipe.
- The validation pass quantizes protected tensors at their
  resolved precision, so its measured damage includes protection
  benefit while the predicted sum does not. The gap moves in the
  sub-additive direction — the safe one.
- The solver cannot trade a protection against a downgrade.
  Protections carry no damage price, so the user supplies the
  judgment, as with pins.
- Pack gains a mandatory post-pack step on protected packs. A fit
  collapse means one revision round: the user excludes the named
  tensors from the protection and re-plans. The packed file stays
  recipe-driven (ADR-0012 decision 3). G1 needed exactly one such
  round (layers 1, 2, 5).

## Resolutions

The questions this record left open closed with the implementation.

- **Reference: a second pack at unprotected types.** The pack
  command re-packs the same recipe with its protections stripped
  and dequantizes both files. gguf-py cannot quantize K-quants, so
  the cheap unweighted re-quantize was never available, and the
  second pack reproduces the assisted fit exactly — it is the
  sibling comparison G1 used, made general. The reference file is
  deleted after measurement.
- **Refuse and name.** The check halts, names the collapsed
  tensors, and keeps the file. The user excludes the tensors from
  the protection and re-plans. An automatic repack would emit an
  artifact no recipe describes (ADR-0012 decision 3).
- **No default protections.** Every protection is user-directed.
  The thirteenth data point weakened the case for a default:
  bartowski's imatrix reproduces the blk.1 collapse signature under
  our pack layout (RMSE 0.0245 against ours at 0.0241), so
  `attn_v=5` is not safe unconditionally.
- **The backfill is a repository script.**
  `scripts/backfill_tensor_sizes.py` reads safetensors headers and
  writes the annotated map copy. It earns a CLI command when a
  second consumer appears.
- **The check stays off unprotected packs.** A protected pack
  without an imatrix skips it with a printed note. The gate is
  mandatory exactly where the failure mode lives.

One thirteenth-data-point observation bounds the check's meaning: a
collapsed RMSE signature does not always destroy the model. The
same blk.1 signature cost +0.94 PPL under our imatrix (9.594
against 8.650) and cost nothing under bartowski's — his 47-layer
build scored 0.11 PPL *better* than its 44-layer sibling (8.646
against 8.752). The damage depends on *which* channels
the fit sacrifices, not the error magnitude alone. The check is
therefore a conservative gate: it refuses packs a cheaper fit would
serve better, and the one catastrophic case on record
(9.594 PPL) is exactly what it catches. The reconstruction output
of that catastrophic build was never saved — its RMSE figures rest
on the evidence page. Every check this record mandates writes its
raw measurements to the run log.
