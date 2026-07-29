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

Specifics fixed at implementation (`quantfit.domain.solver` — implemented 2026-07-27, path per ADR-0008):

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
