# ADR-0001: Selective per-layer quantization as the core approach

- **Status:** Accepted
- **Date:** 2026-07-27

## Context

Models worth running (Nemotron Super 49B class) exceed a 24 GiB card at any
uniform precision that preserves quality: uniform 4-bit doesn't fit, uniform
3-bit fits marginally but degrades badly. Layer fragility is known to be
non-uniform, and existing tools encode that as fixed per-architecture
heuristics rather than measurement. antirez/ds4 demonstrated that a recipe
tuned to one specific model outperforms generic treatment of that model.

Alternatives considered: uniform quantization with better within-layer
methods (AWQ/GPTQ alone — doesn't change the fit arithmetic); pruning or
distillation (changes the model, orders of magnitude more compute); CPU/GPU
offload (changes the serving-speed contract entirely).

## Decision

quantfit's core mechanism is **measured, per-layer-group mixed-precision
quantization**: measure each group's damage curve empirically on the target
model, then assign precisions non-uniformly under a VRAM budget. Measurement
over heuristics; fitting one model well over fitting all models adequately.

## Consequences

- Requires a sensitivity scan per model — hours of compute — where heuristic
  tools are instant. This is the accepted price of measurement.
- Recipes are per-model artifacts and shareable; one person's scan of a
  popular model benefits everyone with the same hardware class.
- Within-group quantization method (RTN, AWQ, GPTQ) stays orthogonal and
  swappable; quantfit decides *how many bits where*, not *how to round*.
- The additivity assumption behind marginal scanning is a known risk,
  mitigated by a final whole-recipe evaluation (see ADR-0006).
