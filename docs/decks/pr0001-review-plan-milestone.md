---
marp: true
theme: quantfit
paginate: true
footer: quantfit · PR #1
---

<!-- _class: lead -->

# quantfit

**Making a big AI model fit on the computer we already own.**

Milestone 1 review — July 27, 2026
[github.com/Alberto-Codes/quantfit/pull/1](https://github.com/Alberto-Codes/quantfit/pull/1)

---

# The problem

- Powerful AI models are big. The one we want is about **98 GB**.
- Our video card has **24 GB** of memory. The model does not fit.
- You can "shrink" a model to save space, but shrink it too much
  and it gets noticeably dumber.
- Shrinking everything the same amount does not work here:
  the size that fits ruins the model, and the size that keeps it
  smart doesn't fit.

---

# The idea

- Not all parts of a model are equally fragile.
- Some parts can be squeezed hard with no real harm.
  Others fall apart if you touch them.
- **quantfit measures which parts are which**, then squeezes
  each part just the right amount to fit our card.

---

# Before → After

**Before this milestone**
- The idea existed only as documents.
- "Will it even fit?" was a guess based on rough estimates.

**After this milestone**
- A working planning tool you can run today.
- It reads the real model's blueprint and tells us exactly
  how much room we have — no more guessing.
- Given measurements, it builds the full "shrink plan"
  automatically, and can explain every choice it made.

---

# What we built

- A **budget calculator**: how much memory is really free for the
  model after everything else takes its share.
- A **recipe maker**: decides how much to squeeze each part,
  staying inside the budget, and says why for every decision.
- If a plan is impossible, it says so honestly — and by how much.
- 88 automated tests check all of it on every change.

---

# What the numbers told us

- With the real model's blueprint: about **20 GB** is truly
  available for the model on our 24 GB card.
- That means squeezing to about **3.5 units of size** on average —
  the serving software we planned to use only supports 4.
- So there's a known gap to close. We wrote down four ways
  to close it and will pick one in the next phase.
- Finding this out **now**, on paper, cost nothing.
  Finding it after months of building would have hurt.

---

# What's next

- Build the **scanner**: the tool that measures which parts of
  the model are fragile. This is the science step.
- Then use those measurements with today's recipe maker
  to produce the first real shrunken model.
- Progress and decisions are tracked in the project repo:
  `github.com/Alberto-Codes/quantfit` (pull request #1).
