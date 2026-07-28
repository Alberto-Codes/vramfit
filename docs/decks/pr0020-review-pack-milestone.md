---
marp: true
theme: quantfit
paginate: true
footer: quantfit · PR #20
---

<!-- _class: lead -->

# quantfit — the pack step ships

The measuring is over: quantfit now builds real models, and the first one beats the standard approach.

**PR [#20](https://github.com/Alberto-Codes/quantfit/pull/20)** · 2026-07-28

---

## The problem

- Big AI models don't fit on affordable graphics cards. Shrinking
  them means storing each part with fewer bits.
- Most shrunk models use one-size-fits-all rules for which parts to
  squeeze. Nobody measures first.
- quantfit measures first. But until today, its output was just a
  plan on paper — a file saying "squeeze this part, protect that
  one". No runnable model existed.

---

## Before → After

**Before this work**

- quantfit could measure a model and write a shrink plan.
- No way to apply the plan. No proof the measuring helps.

**After this work**

- One command turns the plan into a working model file.
- The first model is built, tested, and it works.
- Head-to-head numbers against the standard method are in — and
  quantfit wins at the same size.

---

## What the numbers told us

We shrank a small pilot model (Qwen, 3 billion parts) to fit a
2 GB budget, and tested against the community's standard recipe.

- Our model makes **35 % less quality loss** on the standard text
  test than the standard method at the same budget.
- On a stricter test — how closely the shrunk model matches the
  original's behavior — ours is **23 % closer**.
- Bonus find: the plans were slightly too optimistic about file
  size. The new safety check caught it on the first try, and the
  fix is now part of the routine.

---

## What we built

- The **pack** command: plan in, working model out, in under a
  minute for the pilot model.
- A written rule book for how each precision level translates to
  the model file format, so results are repeatable.
- A safety check that weighs the finished file against the budget
  and refuses models that don't fit.
- A full activity log of every build, for the audit trail.
- Automated tests around all of it, plus an outside review pass.

---

## What's next

- Teach the planner two more precision levels the format offers —
  our tests show that's where more quality is waiting.
- Add a double-check that a plan's predictions hold before building.
- Then the main event: the 49-billion-part target model on a
  single consumer graphics card — the project's founding goal.
  Every step of that path is now built and rehearsed.

---

## Links

- The change: [PR #20](https://github.com/Alberto-Codes/quantfit/pull/20)
- The roadmap: [issue #8](https://github.com/Alberto-Codes/quantfit/issues/8)
- The evidence, with method details:
  [evaluating packed models](../explanation/evaluating-packed-models.md)
