---
marp: true
theme: quantfit
paginate: true
footer: quantfit · PR #6
---

<!-- _class: lead -->

# quantfit — the measuring tool is built

The scanner that finds out which parts of an AI model can be shrunk safely.

**PR [#6](https://github.com/Alberto-Codes/quantfit/pull/6)** · merged July 28, 2026

---

## The problem

- We want to run a 100 GB AI model on a video card with 24 GB of memory
  (VRAM — the video card's memory).
- To fit, the model must be compressed. Compress the wrong parts and
  the model gets noticeably dumber.
- Nobody can tell you which parts are safe to compress. Today people
  guess, using one-size-fits-all rules.
- Our bet: **measure** each part instead of guessing. Until now we had
  no tool that could do that measuring.

---

## Before → After

**Before**
- The measuring step existed only as a design document.
- No way to test the project's core idea on a real model.
- A crash during a long measuring run would mean starting over.

**After**
- `quantfit scan` works end to end and is merged.
- It measures every part of a model at four compression levels and
  writes the results to a file the planner already understands.
- If a run crashes at hour six, restarting continues from hour six —
  every finished measurement is saved as it happens.

---

## What we built

- A scanner that gently compresses **one part at a time**, checks how
  much the model's answers drift, then puts the part back.
- **Save-as-you-go**: each measurement lands on disk the moment it
  finishes, with a safety tag so two different runs can never mix.
- Clear error messages for every way it can fail — no silent wrong
  numbers. Bad measurements stop the run instead of being recorded.
- The heavy machine-learning software is optional — everyday users of
  the planning tool install none of it.
- Two independent review passes (seven automated reviewers) found and
  fixed about 40 issues before merge.

---

## What the numbers told us

We ran the scanner on a small test model, on the real graphics card:

- Every part got **more damaged** as we compressed harder — exactly the
  curve the planner needs. The math behaves.
- Some parts were about **50 times more sensitive** than others.
- That difference is the whole point: spend memory on the sensitive
  parts, compress the tough ones hard. Guessing evenly wastes both.

---

## What's next

- The real 100 GB target model is downloading to our test machine now.
- Next step: the **first real scan** — several hours of measuring, made
  safe by save-as-you-go.
- That produces the first genuine sensitivity map, which feeds the
  planner, which produces the first measured compression recipe.
- The goal is unchanged: the big model, running well, on a card it has
  never fit before — with proof, not vibes.

---

## Links

- This work: [PR #6](https://github.com/Alberto-Codes/quantfit/pull/6)
- What comes next: [issue #8](https://github.com/Alberto-Codes/quantfit/issues/8)
- Project: [github.com/Alberto-Codes/quantfit](https://github.com/Alberto-Codes/quantfit)
