# ADR-0007: Solver strategy for recipe selection

- **Status:** Proposed
- **Date:** 2026-07-27

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

## Decision (proposed)

Ship the **greedy damage-per-byte solver first** — its downgrade sequence
doubles as a human-readable explanation of *why* each group got its
precision. Add the exact DP solver later behind `--solver exact` as a check
on greedy's gap. Record the solver name in the recipe (`plan.solver`) for
reproducibility either way.

## Open questions

- Hard floors: is 2-bit ever allowed on attention groups regardless of what
  the scan claims, or do we encode structural priors as constraints?
- Should the solver see the runtime-capability table (ADR-0004) as a
  constraint set (only kernel-supported precisions per tensor type)?
- Tie-breaking under equal ratios — deterministic ordering matters for
  reproducible recipes.

## Consequences (if proposed decision is accepted)

- Recipes come with an explanation trace essentially for free.
- A future exact solver quantifies greedy's optimality gap on real maps —
  publishable data on whether solver sophistication matters at all here.
