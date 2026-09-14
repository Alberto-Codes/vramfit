# ADR-0031: The refinement pass measures in the runtime frame and records its search in a sidecar

- **Status:** Accepted
- **Date:** 2026-09-14 (UTC)
- **Origin:** Issue #590, resolving the question the 2026-09-11
  maintainer ruling left conditional on a neighbour winning. Nine
  neighbours won. The measured evidence is the closing comment on
  #486 and the C4 equal-byte neighbour sweep, an external measurement
  run of 2026-09-11 whose records live outside this repository. The
  figures that carry a decision are restated here, so no decision
  below rests on reaching that run.
- **Note (2026-09-14 UTC, validation pass):** the stage ran end to
  end on the published 30B recipe, on one rented H100 for 8.02 USD.
  The pod ran 2026-09-14T03:20:33Z to about 2026-09-14T05:37Z. That
  window falls on 2026-09-14 in UTC while the authorizing instruction,
  dated 2026-09-13, was written in a local zone still on the previous
  day. The
  control reproduced the published frame at 0.204220 mean KLD against
  0.204223 over 594 chunks. Five of fifteen arms measured below the
  control and four cleared the 7.8 sigma bar. The stage selected an
  arm at 0.192490, which is 5.74 percent better at 12.4 sigma paired.

    Those fifteen arms overlap the 2026-09-11 arms in zero positions,
    so decision 1 now rests on two independent samples. Spearman rho
    between predicted penalty and measured delta was **-0.171** here
    against +0.146 there, and Pearson r **-0.224** against +0.279. The
    correlation is near zero in both and changes sign between them.

    The consequence sharpened. A search ranking by the map would have
    picked the arm at +0.009580, which measured **worse** than the
    recipe it started from. The earlier run put map-ranking's cost at
    43 percent of the win. On this sample map-ranking does not lose
    part of the win, it goes backwards.
- **Amends:** [ADR-0007](0007-recipe-solver-strategy.md). The greedy
  solve no longer has the last word on a recipe's assignments. It
  keeps the whole plan step, and the refinement pass runs after it.

## Context

The solver ranks downgrades by damage measured one group at a time,
with every other group at reference precision (ADR-0006). Whether
that approximation picks the best assembled recipe was an open
question for the life of the project.

The C4 equal-byte neighbour sweep answered it on 2026-09-11. It ran
on the published 30B recipe on one rented H100, for 7.61 dollars
across sixteen arms. An arm swaps two assignments and spends the same
bytes, so it competes inside the same weight budget.

**Nine of fifteen neighbours measured lower full-window KLD than the
recipe they came from.** The best arm measured 0.191855 mean KLD
against the published 0.204223, which is 6.06 percent better at 14.4
sigma paired. Seven arms cleared 4 sigma. The control reproduced the
published figure at 0.08 sigma, and the same-file noise floor read
zero twice.

**The map does not order the neighbourhood it prices.** Spearman rho
between the map's predicted penalty and the measured delta is +0.146,
and Pearson r is +0.279. A search that ranked by the map would have
measured the top arm, stopped at 0.197942, and missed 43 percent of
the improvement.

That last number is the design constraint. Prediction earns the
greedy solve, where it orders thousands of candidates cheaply. It
does not earn the refinement step.

## Decision

1. **The refinement pass measures every candidate it considers.** It
   packs each candidate and evaluates it in the runtime frame. It
   never ranks candidates by the sensitivity map, and it never
   measures only a map-selected subset.

2. **A new port carries the runtime-frame measurement.** It returns
   per-chunk divergences against a reference base, which is what a
   paired test needs. The llama.cpp adapter drives
   `llama-perplexity --kl-divergence`.

   Two existing ports were considered and neither fits.
   `DamageMeter.measure_recipe` measures the torch scan frame, and
   [ADR-0021](0021-runtime-frame-measurement.md) binds this
   measurement to the runtime frame. `SmokeTester.smoke` returns one
   perplexity figure, and widening it would change every caller of an
   [ADR-0017](0017-post-pack-smoke-test.md) port for one new consumer.

   **This port is forced, not fitted.** The charting convention warns
   against adding a port so a ticket's wording fits, and #179 did
   exactly that. The distinction is the measurement frame. The
   authorized measurement is unreachable through any existing seam,
   so the port is the smallest way to deliver it rather than
   architecture added around it.

3. **The search record ships as a refinement sidecar, beside the
   recipe.** [ADR-0025](0025-evals-sidecar.md) already drew this
   line. A search record is evidence about an artifact, not plan
   arithmetic. The sidecar carries the `vramfit_schema` envelope, and
   breaking changes bump it.

4. **The recipe schema does not change.** A refined recipe names its
   pass in `plan.solver` and carries nothing else new. Every recipe
   the project has published was never refined, and none of them
   gains a field.

   `plan.trace` does not grow a refinement step. The trace holds
   downgrade steps that no longer explain a refined recipe's
   assignments, so adding to it would compound a field that is
   already wrong for this case.

5. **The frame carries every input whose substitution would change
   the number.** That is the rule. The instances are the runtime
   binary build, the hardware, the evaluation corpus content
   identity, the reference logits content identity, and the
   importance matrix identity. Each one, swapped, moves the measured
   divergence, so a record that omits it serializes two different
   passes identically. An input the pass did not use records as null
   rather than as an absent field, because "ran unassisted" is a
   claim and silence is not.

   The reference logits are the sharpest case. Every divergence in
   the record is computed against those bytes, so substituting them
   changes every figure the pass reports — more directly than the
   corpus or the matrix. Cost does not exempt them: the file reaches
   39.7 GB on the 49B target and is hashed once per pass, before the
   first arm runs, because a rule whose most load-bearing case is
   carved out reads as coverage it does not provide.

   This rule is written as a rule because four separate reviews of
   this change found the same shape: the measurement was sound and
   the record could not prove it — first the corpus identity, then
   the neighbourhood size, then the matrix, then the reference
   logits. A list of cases invites another. The rule generates them,
   and the fourth case was found by reading the rule rather than the
   code.

   **The sidecar records at minimum:** every arm evaluated, how many
   byte-neutral moves the neighbourhood held, the winner, the
   evidence bar the caller stated, the control result with its
   sigma, and the frame.

   The neighbourhood count is not the arm count. A pass measures the
   arms its budget affords, so 15 arms of 15 and 15 arms of 385 are
   different results and a record that states one number states
   neither. The count comes from the enumeration, before the stride
   samples it.

   It counts what was enumerated, never the outcome. A declined pass
   records the moves its enumeration found, which is zero only when
   the neighbourhood was genuinely empty. A pass that declines for
   another reason — a pin this map cannot resolve — records the moves
   it found, because a recipe with 385 moves the pass could not prove
   safe is not a recipe with no neighbourhood. Every decline the
   stage can reach enumerates first, so the count is always a real
   one and the record carries no separate not-enumerated value.

6. **The map's predicted delta records as provenance only.** The
   sidecar may carry it. Nothing may order, filter, or select on it.
   Spearman +0.146 is why this pass exists, and the record must not
   reintroduce map ranking through a stored field.

7. **The caller states the evidence bar, and no default exists.** The
   bar a result must clear belongs to the caller, not to the stage.
   The project's artifact precedent is 7.8 sigma.

8. **Declining is an outcome, not a failure.** A recipe with no legal
   swap reports that it has none.

   The published 49B recipe is that case, and the numbers are these.
   It allocates 82 groups. 81 sit at the 3-bit floor and one sits at
   8 bits. A swap needs two groups at different precisions, so every
   candidate pair on that recipe pairs the single 8-bit group against
   a 3-bit one — 81 ordered pairs, and no pair among the 81 groups at
   the floor, because a swap between two groups at one precision
   moves nothing. Each of those 81 pairs then fails the pricing test:
   the 8-bit group carries a different reference size from the
   stacks, so repricing the two at each other's precisions does not
   spend what they spent before. The neighbourhood is empty, and the
   pass reports that rather than forcing a move.

   The 49B refinement survey, an external measurement run, recorded
   the same result. It is restated here rather than cited, because a
   decision that cannot survive its citation being unreachable is not
   yet a record.

9. **Every packed arm is judged against the weight budget, and an arm
   that exceeds it is excluded rather than kept or dropped.** The
   pass calls `vramfit.domain.pack.weight_budget_margin`, the rule
   `vramfit pack` gates on, so one budget rule serves both stages.

   Byte-neutrality is a property of *predicted* bytes. The real GGUF
   size is a different number — `PREDICTED_BYTES_TOLERANCE` exists
   because nominal-bit predictions undershoot effective bits
   (ADR-0014) — so a swap whose predicted totals match can still pack
   over. On the 30B target a q_proj/o_proj swap moves two stacks that
   route through different effective-bits tables (ADR-0028), and
   their real block-and-padding costs drift in opposite directions.

   An over-budget arm stays in the record with its measurement and
   its negative margin, and leaves the selection. It cost card time,
   so dropping it from the record would make the arm count lie the
   way the declined-pass zero and "the recipe stands" once did. An
   arm that was never evaluated has no entry at all — that is the
   distinction a reader needs.

   A control that exceeds the budget stops the pass instead. Every
   other arm is read against it, so nothing downstream is readable.

## Consequences

- No outcome of the pass is a verdict on the recipe. The arms are a
  sample of the neighbourhood whenever the budget is smaller, so the
  command reports what it measured and the sidecar carries the
  fraction. A pass where no arm cleared the bar says exactly that.
- The pass costs packs and evaluations, never map arithmetic. The C4
  run measured 0.48 dollars per arm on this target, so a 15-candidate
  pass costs about 7 dollars and 1.5 hours on rented hardware.
- That cost does not scale to every target. Decision 8 is what keeps
  an unaffordable or degenerate target from being forced. A decline
  packs nothing and measures nothing, so it spends no card time — it
  still hashes the frame's inputs first, because the record names
  them by content whatever the outcome.
- **Byte-neutrality is priced, never assumed.** The pass prices every
  candidate group at its new precision through
  `vramfit.domain.solver.group_size_predictor`, the path the plan
  step used, and keeps a swap only when the two repriced groups spend
  what they spent before. Equal reference size is not the test. The
  predictor binds each group's effective-bits table from its measured
  row width, so two groups of one reference size price differently
  wherever the tables disagree — 2.25 against 2.625 bits per weight
  at nominal 2 (ADR-0028). The 30B target carries that pair: under
  `--group-by stack` its `q_proj` rows are 2688 wide and its `o_proj`
  rows 4096 wide, and both hold 4096 × 2688 elements. A swap priced
  by reference size would have understated such an arm by about
  516,096 bytes and spent more than the control inside the same
  weight budget.
- The pass also skips any pair naming a group the recipe already
  fixes — one a pin pattern covers, or one holding a tensor a
  protection pattern floors. The pack reads an arm's assignments and
  never its pins, so such a swap would pack against the constraint
  the plan records. This narrows which arms the pass measures, which
  is the subject of #591.
- **The stage does not re-derive a constraint the solver resolves.**
  Pins resolve through `vramfit.domain.pins.pinned_group_names` and
  protections through
  `vramfit.domain.protection.expand_protections`, the paths the plan
  step itself used. Each rule has exactly one resolution path, and
  the refinement pass calls it. A second implementation drifts from
  the first, and a drifted answer is how a refined arm violates a
  constraint its own plan records. Two rounds of review found this
  defect twice: a glob over assignment names that missed a folded
  merged-projection spelling, and a read of `protected_tensors` that
  missed every floor the assignment already met (issue #59).
- A pin the map alone cannot resolve declines. `pinned_group_names`
  resolves against the map's groups, so a pin spelled with a
  checkpoint-discovered (ADR-0029) or folded (#576) name is reported
  as missed, and the pass declines rather than measuring arms that
  may violate it. The function takes no parameters for widening that
  universe: the open question below is untaken, so the surface that
  would serve it returns with the caller that supplies it.
- The pass refuses a map that did not price the recipe. `refine` is
  the first command handed a recipe and a map as separate arguments,
  and a mismatched pair declines cleanly at exit 0 — a comparison
  that never happened, wearing the shape of a published no-winner
  result. Comparing `model_id` closes it.
- One definition states what clearing the evidence bar means.
  `vramfit.domain.paired.cleared_bar` decides which arm the pass
  selects and which winner a sidecar accepts, so no record can claim
  a winner selection would have refused.
- The solver keeps its approximation and its scope. ADR-0007 is
  amended in reach, not replaced.
- One more artifact rides a refined publication.
- Nothing here says a published pack should be replaced. Promoting an
  arm to an artifact needs a tier-3 slice and a serve test first.

## Open questions

- Which arms the pass measures when it cannot afford the whole
  neighbourhood, tracked in #591. The shipped stride is
  map-independent and untested. Measured 2026-09-14 UTC against the
  published 30B recipe: the neighbourhood holds 385 moves, the 15
  arms a stride selects overlap the 2026-09-11 arms in zero
  positions, and 12 of those 15 carry a predicted delta above +0.05
  where only 3 of the earlier arms sat. Decision 1 rules out ranking
  on the map. Whether it also rules out using the map to spread a
  sample is what #591 asks.
- Whether a pass should keep its packed arms for post-hoc
  inspection. It does not: `_measure` drops each arm's file once the
  meter has read it. The 30B target's arms are about 21 GiB each and
  a pass packs sixteen, so retaining them needs 336 GiB of pod disk
  to answer a question nothing has yet asked. A flag was considered
  and left out rather than shipped unused — a public option is a
  surface the project then owes.
- Whether to widen the pin match universe beyond the map's groups.
  The narrowing above is a choice, not a limit: `vramfit refine`
  opens the checkpoint one step before the pass runs, so the names
  that would widen the match are readable at that moment.

  Widening is not free, and the cost belongs in this record.
  `_resolve_row_widths` yields row widths alone — elements per row
  per group. The per-group byte sizes and folded-projection
  spellings a wider match needs come from
  `cli_plan_sizes.discovered_groups`, which `refine` never calls and
  neither the recipe nor the map stores. Widening therefore needs a
  second checkpoint read and a new call, not the reuse of a value the
  command already holds. `checkpoint_row_widths` also pins
  its granularity to `stack` where the plan-time read follows
  `map_.scan.group_by`, so a map grouped by layer or tensor would
  not even yield the same group names.

  Widening would replace a decline with a measured pass for a recipe
  planned under `--checkpoint` whose pin lands on an uncovered group.
  Nothing has measured that case, so the option stays untaken rather
  than refused.
- Whether the pass should search beyond one swap. Every arm measured
  so far moves exactly two assignments, and nothing prices a
  two-swap neighbourhood yet.
- Whether a winning arm's improvement survives a frame change. The
  C4 measurement ran on WikiText-2, and no arm has taken a tier-3
  slice.
- Whether an equal-byte move is the right neighbourhood at all. A
  move that spends fewer bytes than the budget allows was never
  measured.
