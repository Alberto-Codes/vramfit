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
  groups. Maintainer ruling 2026-08-21, from the #321 exchange. Two
  clauses.

    **The allocator refuses a second cheapest-width stack in a layer
    while a layer with none remains.** Two arms measured the clause on
    the 30B target. They read one layer ranking and one 6-up 5-down
    projection composition, and they differ in geometry alone. Arm 1
    concentrates 11 cheap stacks on 6 layers and reads 1.361520 mean
    PPL(Q)/PPL(base). The composition-matched arm spreads them over
    11 layers and reads 1.276199, at 15.8 sigma — 22.4 sigma on mean
    KLD, at 0.299049 against 0.360932.

    **Within a layer the allocator takes the projection the
    stack-keyed map prices cheaper at the candidate width.** The
    q0-ref map prices `down_proj` cheaper at nominal 2 in 23 of 23
    MoE layers (#328). At matched spread geometry, the all-`down`
    arm reads 1.178594 and the composition-matched arm 1.276199.
    Arm 4 spent every cheap stack on `up_proj` and reads 1.409476,
    worse than concentrated arm 1. So each clause moves the result
    on its own, and together they move it 13.4 % against arm 1 —
    more than inverting the layer ordering costs, at 16.0 % (#300,
    arm 5).

    Three bounds. ADR-0021's damage model prices no interaction
    term, and this rule exists because a measured interaction moved
    the result — ADR-0006's open question carries the additivity
    evidence, at six sub-additive measurements and one
    super-additive on a 2-bit-heavy recipe. ADR-0021 decision 4
    still bars the solver from the 2-bit width on this target, so
    the rule ships unexercised there until that bar lifts. The
    measurements are hand-authored recipes, so they test the map
    and the policy's shape, never this solver's own allocation
    (#301). #321's closing comment carries the evidence trail.

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
