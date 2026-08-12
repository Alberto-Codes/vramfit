# ADR-0001: Selective per-layer quantization as the core approach

- **Status:** Accepted
- **Date:** 2026-07-27
- **Note (2026-07-28):** the serving path moved to GGUF/llama.cpp for
  the benchmark ([ADR-0010](0010-sub-4-bit-serving-path.md)). The
  "vLLM-served" clause below reads through that amendment.
- **Note (2026-07-29):** the orthogonality claim below met a
  measurement. The first 49B head-to-head lost to an imatrix quant,
  with ~81 % of the gap in the within-group method
  ([ADR-0012](0012-gguf-type-mapping.md) open questions). *How to
  round* is not separable from the outcome at 3-bit scale.

- **Amendment (2026-08-11, issue #161):** the decision below says
  "per-layer-group". It never said what a group is on a
  mixture-of-experts model, because the first targets were dense.
  Ruling: **the group is the unit a pack can address.** The
  sensitivity map gains a `stack` value for `--group-by`, which
  collapses one projection's routed experts into one group and keeps
  every other weight separate. On a dense model `stack` matches
  `tensor`, so this record's original claim is unchanged there.

    Two measurements forced it. llama.cpp fuses each layer's experts
    into one tensor carrying one quantization type, which gives 46
    addressable expert slots on Nemotron 3.5 Lightning 30B-A3B
    (#159). vLLM, TensorRT-LLM, and SGLang each resolve one algorithm
    per mixture-of-experts module, which gives 23 (#166). No runtime
    serves a per-expert precision. A key finer than the stack prices
    distinctions no pack can express, and a key coarser than it —
    the layer — cannot price `ffn_up_exps` against `ffn_down_exps`,
    which differ in shape and therefore in available precisions.

    The sensitivity-map schema bumped to 3 for the new value. Readers
    accept 2 and 3, because version 3 only widened the enum.

## Context

Models worth running (Nemotron Super 49B class) exceed a 24 GiB card at any
uniform precision that preserves quality: uniform 4-bit doesn't fit, uniform
3-bit fits marginally but degrades badly. Layer fragility is known to be
non-uniform. Existing tools mostly encode that as fixed heuristics
(llama.cpp k-quants); EXL2/exllamav2 *does* measure per-layer and mix
bitrates, but is tied to its own runtime and optimizes to an average
bits-per-weight rather than an explicit VRAM budget planned jointly with KV
headroom. antirez/ds4 demonstrated that a recipe tuned to one specific model
outperforms generic treatment of that model. vramfit's niche: EXL2-style
measurement, vLLM-served, with the budget math and the recipe as inspectable
first-class artifacts.

Alternatives considered: uniform quantization with better within-layer
methods (AWQ/GPTQ alone — doesn't change the fit arithmetic); pruning or
distillation (changes the model, orders of magnitude more compute); CPU/GPU
offload (changes the serving-speed contract entirely).

## Decision

vramfit's core mechanism is **measured, per-layer-group mixed-precision
quantization**: measure each group's damage curve empirically on the target
model, then assign precisions non-uniformly under a VRAM budget. Measurement
over heuristics; fitting one model well over fitting all models adequately.

## Consequences

- Requires a sensitivity scan per model — hours of compute — where heuristic
  tools are instant. This is the accepted price of measurement.
- Recipes are per-model artifacts and shareable; one person's scan of a
  popular model benefits everyone with the same hardware class.
- Within-group quantization method (RTN, AWQ, GPTQ) stays orthogonal and
  swappable; vramfit decides *how many bits where*, not *how to round*.
- The additivity assumption behind marginal scanning is a known risk,
  mitigated by the validation pass (see ADR-0006, measured twice
  since).
