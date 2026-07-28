# ADR-0002: Three-stage scan → plan → pack CLI pipeline

- **Status:** Accepted
- **Date:** 2026-07-27

## Context

The workflow has three phases with very different costs: measuring
sensitivity (hours, GPU-bound), solving for a recipe (seconds, pure math),
and writing a quantized checkpoint (minutes, IO-bound). A monolithic
"quantize this model" command would re-run the expensive phase to iterate on
the cheap one — e.g. re-scanning for hours just to try a different VRAM
budget.

## Decision

Ship three separate CLI commands — `scan`, `plan`, `pack` — connected by
versioned JSON artifacts on disk (sensitivity map, recipe). Each stage is
independently re-runnable; each artifact is inspectable and shareable.

## Consequences

- Iterating on budgets is free: one scan supports unlimited `plan` runs
  (different VRAM targets, pins, context assumptions).
- Artifacts double as publishable outputs — a sensitivity map for a popular
  model is useful to people who never run `scan`.
- Two file-format contracts must be versioned and kept stable
  (`quantfit_schema` field); format churn breaks downstream stages.
- No single "just do it" command for now; a convenience wrapper can compose
  the three later without changing the architecture.
