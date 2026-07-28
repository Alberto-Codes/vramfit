# ADR-0003: North-star benchmark: Nemotron Super 49B on a 24 GiB RTX 4090

- **Status:** Accepted
- **Date:** 2026-07-27

## Context

"Selective quantization tool" is unfalsifiable without a concrete target. The
project needs one benchmark that (a) is impossible today, (b) becomes
possible only if the core idea works, and (c) runs on hardware we actually
own (RTX 4090, 24 GiB VRAM, 124 GB system RAM).

Nemotron Super 49B fits that profile: it does not fit a 4090 at uniform
4-bit, the required ~3.2 average bits/parameter is achievable *only* with
non-uniform assignment, and the model is open-weight with strong quality —
so "49B on a 4090" is a headline result people can reproduce.

Alternatives: a 70B-class dense model (needs average bits low enough that
even selective assignment likely wrecks it — a stretch goal, not a first
target); a 30B-class model (fits at uniform 4-bit — proves nothing).

## Decision

The project's acceptance test is **Nemotron Super 49B serving on a single
RTX 4090 via vLLM at 16k context, with measured quality loss** (vs the bf16
reference and vs Nemotron Nano as the "just run a smaller model" baseline).
Design decisions are evaluated by whether they move this benchmark.

## Consequences

- Every pipeline stage must handle a model larger than VRAM (scan can't hold
  the reference on-GPU; pack can't load the checkpoint whole).
- The tool is developed against one architecture first; generality across
  architectures is explicitly deferred, per the depth-over-breadth ethos.
- If the benchmark proves unreachable with acceptable quality, that result is
  published too — the measurement infrastructure is the durable artifact.
