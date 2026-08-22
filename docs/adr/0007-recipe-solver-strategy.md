# ADR-0007: Solver strategy for recipe selection

- **Status:** Accepted
- **Date:** 2026-07-27
- **Note (2026-07-28):**
  [ADR-0013](0013-runtime-capability-in-recipes.md) adds the deferred
  runtime-capability constraint as a filter on the candidate set.
- **Note (2026-07-29):**
  [ADR-0014](0014-per-type-effective-bits.md) revises the size model:
  when the target runtime has an effective-bits table, the solver
  prices candidates at per-type effective bits, not nominal bits.
- **Amendment (2026-08-21, issue #321):** the greedy solver gains a
  placement rule for the cheapest in-budget width on expert-stack
  groups. The maintainer ruled it 2026-08-21 in the #321 exchange.
  The rule carries two clauses.

    **The allocator refuses a second cheapest-width stack in a
    layer while a layer with none remains.** Two arms measure this
    clause on the 30B target. Both read one layer ranking and one
    6-up 5-down projection composition, and they differ in geometry
    alone. Arm 1 concentrates 11 cheap stacks on 6 layers and reads
    1.361520 mean PPL(Q)/PPL(base). The composition-matched arm
    spreads them over 11 layers and reads 1.276199, at 15.8 sigma.
    On mean KLD the gap reads 0.299049 against 0.360932, at 22.4
    sigma.

    **Within the layers the first clause admits, the allocator
    breaks ties toward the projection the stack-keyed map prices
    cheaper at the candidate width.** The first clause precedes
    this one. The `q0-ref` map prices `down_proj` cheaper at nominal
    2 in 23 of 23 MoE layers (#328's re-scan). At the same 11-layer
    geometry, the all-`down` probe reads 1.178594 against the
    composition-matched arm's 1.276199. Arm 4 spends every cheap
    stack on `up_proj` at near-matched geometry and reads 1.409476.
    Together the clauses move arm 1's result by 13.4 %. Inverting
    the layer ordering costs 16.0 % (#300, arm 5).

    The rule constrains which downgrades are candidates, and the
    decision's selection key is unchanged. The constraint is a
    deterministic function of the allocation state, so recipes stay
    deterministic and input-order invariant.

    Four bounds hold. ADR-0021's damage model prices no interaction
    term, and this rule exists because a measured interaction moved
    the result. ADR-0006's open question and
    [evaluating-packed-models](../explanation/evaluating-packed-models.md)
    carry the additivity evidence: six sub-additive measurements
    and one super-additive on a 2-bit-heavy recipe. #375 tracks the
    drift between those two carriers. ADR-0021 decision 4 barred
    the solver from the 2-bit width on this target, so the rule
    shipped unexercised there. **The bar lifted 2026-08-22 (#301)
    for expert-stack groups — the amendment below.** **Corrected
    2026-08-21 by the observed consequence below: the rule keys on the
    surviving floor and fires at nominal 4.** The measured
    recipes are hand-authored, so they test the map and the
    policy's shape, never this solver's own allocation (#301). #374
    carries the build, and the
    [#321 closing comment](https://github.com/Alberto-Codes/vramfit/issues/321#issuecomment-5372929524)
    carries the evidence.

    **Observed consequence (2026-08-21, issue #377): the rule keys on
    the surviving floor, so it fires at nominal 4.** The bound above
    reads that the rule ships unexercised on this target. ADR-0021
    decision 4 strips the 2-bit column by a map copy. #328's stack map
    carries precisions 4 and 2, so the strip leaves nominal 4 as the
    cheapest in-budget width and the rule fires there. A map carrying
    nominal 3 would leave 3 instead, because `servable_precisions` keeps
    it. PR #376 builds the rule at `candidates[-1]`, after the ADR-0013
    filter. **The bar itself is untouched.** No solver code reads it,
    and no 2-bit assignment becomes possible. Maintainer ruling
    2026-08-21: the rule keys on the surviving floor. Record:
    [#377](https://github.com/Alberto-Codes/vramfit/issues/377).
    **Corrected 2026-08-22 by the #301 amendment below: the bar
    lifts for expert-stack groups on this target, so a stack map
    keeping its 2-bit column makes nominal 2 the surviving floor.**

    **Observed consequence (2026-08-21, issue #350): the tie-break's
    nominal-4 input carries a measured unassisted skew.** The pack
    fits `Q4_0` with the imatrix and the `q0-ref` meter does not.
    #381 measured the assistance discounting `up_proj` at a median
    8.2 % and `down_proj` at 2.5 %, each over its own 23 stacks.
    So an unassisted map overprices `up_proj` against `down_proj`
    by about 6 % at nominal 4. Applied to #328's map, the two
    constants flip this rule's projection comparison in 4 of 23 MoE
    layers. The layers are 20, 27, 29, and 31, and three more sit
    inside the two-constant model's 4.5 % worst-case residual. The
    constants are reconstruction-error ratios, so the check bounds
    the exposure and measures no damage-frame flip. ADR-0018's
    2026-08-21 amendment rules the remedy: the `q0-imx` build. The
    tie-break reads the `q0-ref` column, with this exposure, until
    the assisted map lands.
    [#350's 2026-08-21 comment](https://github.com/Alberto-Codes/vramfit/issues/350#issuecomment-5375790074)
    carries the per-layer numbers.

- **Amendment (2026-08-22, issue #301):** the pin surface widens on
  two axes. A pin may name any width the target runtime serves. A
  pin may land on any checkpoint-discovered group
  ([ADR-0029](0029-plan-independent-size-source.md)), not only the
  map's groups, and a pinned uncovered group prices at the pinned
  width instead of holding at reference. The maintainer ruled both
  2026-08-22 in the #301 exchange. A pinned group never enters the
  ranking, so a pin orders no groups the map did not measure. At an
  unmeasured width the assignment records damage 0.0, the way an
  uncovered held group does, and `predicted_damage` sums measured
  marginals only — the #301 validation caveat stands. This gives
  the ruled MoE mix a form: a stack-keyed map beside dense pins at
  nominal 8. Before this amendment the campaign hand-authored every
  mixed recipe (#300, #387). One consequence: a pin at nominal 3 on
  an expert stack now plans, and pack refuses it (ADR-0028
  decision 2). [ADR-0021](0021-runtime-frame-measurement.md)'s
  2026-08-22 amendment lifts decision 4's bar for expert-stack
  groups on the 30B target. The placement rule above can then
  exercise at nominal 2 there, once a map keeping its 2-bit column
  solves. Record:
  [#301 ruling comment](https://github.com/Alberto-Codes/vramfit/issues/301#issuecomment-5377920461).

## Context

The plan step chooses one precision per group to minimize total predicted
damage subject to a byte budget — a multiple-choice knapsack problem (MCKP).
Instance sizes are small (a few hundred groups × ~4 precisions), so nearly
any strategy is computationally feasible. Candidates:

1. **Greedy by damage-per-byte-saved** — start everything at the highest
   precision, repeatedly downgrade the group with the best
   damage-cost/bytes-freed ratio until under budget. Simple, explainable,
   near-optimal in practice for convex damage curves.
2. **Exact MCKP via dynamic programming** — optimal for the modeled
   objective; instance is small enough to solve exactly.
3. **ILP solver** — also exact, but adds a dependency; overkill at this size.

The damage model is itself an approximation (ADR-0006's additivity
assumption), which bounds how much solver optimality is worth: an exact
optimum of an approximate objective is still approximate.

## Decision

Ship the **greedy damage-per-byte solver first** — its downgrade sequence
doubles as a human-readable explanation of *why* each group got its
precision. Add the exact DP solver later behind `--solver exact` as a check
on greedy's gap. Record the solver name in the recipe (`plan.solver`) for
reproducibility either way.

Specifics fixed at implementation (`vramfit.domain.solver` — implemented 2026-07-27, path per ADR-0008):

- Moves consider **all** lower candidate precisions, not just the next step
  down, so non-convex damage curves get direct multi-step jumps.
- Selection key is `(damage_delta / bytes_freed, group name, smallest
  downgrade)` — a total order, so recipes are deterministic and invariant
  to group input order.
- Pins are `fnmatchcase` globs. A pattern matching zero groups is a hard
  error, later pins override earlier ones, and pinned groups never move.
- Infeasibility is prechecked (minimum achievable total vs budget) and
  reported with the exact gap in bytes.
- The final downgrade is refined after the loop (2026-07-28): when a
  milder step of the same group also fits with less damage, it replaces
  the overshooting one. Greedy remains non-optimal globally — it does
  not apply improving moves when already under budget, and earlier
  groups may stay over-downgraded. Both are accepted gaps for
  `--solver exact` to quantify.
- The recipe records `format_overhead` and the full downgrade `trace`.

Resolved open questions: tie-breaking is specified above. Hard floors
stay deferred. The runtime-capability constraint landed as a candidate
filter (ADR-0013), and the size model moved to per-type effective bits
(ADR-0014) — both arrived as solver inputs without changing the recipe
format, as this section predicted.

## Consequences

- Recipes come with an explanation trace essentially for free.
- A future exact solver quantifies greedy's optimality gap on real maps —
  publishable data on whether solver sophistication matters at all here.
