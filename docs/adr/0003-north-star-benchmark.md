# ADR-0003: North-star benchmark: Nemotron Super 49B on a 24 GiB RTX 4090

- **Status:** Accepted, amended by
  [ADR-0010](0010-sub-4-bit-serving-path.md)
- **Date:** 2026-07-27
- **Amendment (2026-07-28):** the serving-runtime clause changes — the
  benchmark serves through llama.cpp, not vLLM. Everything else stands.
- **Note (2026-09-03, issue #482):** the three evaluation tiers of
  this benchmark ran on the reference RTX 4090, tiers 1 and 2 on
  2026-08-09 and tier 3 on 2026-08-10. Publication #1 shipped on
  2026-08-11. The README section "The result" carries the compact
  result and links the records.
- **Note (2026-09-04, issue #501):** the reference box measured the
  published pack at 16k context with q8_0 KV cache, all 81 layers on
  the card, llama.cpp b10326. llama.cpp has no fp8 cache type, and
  q8_0 is its nearest substitute for the card's fp8 target. It did
  not load: the KV cache failed to allocate over 20,409.48 MiB of
  weights with the desktop's 2,341 MiB resident. The result holds at
  fp8 by arithmetic, 22,201 MiB of buffers against 21,765 MiB free.
  The largest context that served is 10,240 tokens, at 21,674.82 MiB
  of device buffers. The 16k clause of this benchmark is unmet on the
  reference box as configured. The same day a headless RTX 4090
  served the pack at 16,384 context. That card was a rented RunPod
  4090 with 1 MiB resident, on the same llama.cpp release built with
  CUDA. Device buffers took 22,307.58 MiB, and nvidia-smi peaked at
  22,762 MiB. The 16k clause is met on a headless card. The evidence
  page's twentieth data point records the reference-box ladder and
  the headless serve as its addendum.

## Context

"Selective quantization tool" is unfalsifiable without a concrete target. The
project needs one benchmark that (a) is impossible today, (b) becomes
possible only if the core idea works, and (c) runs on hardware we actually
own (RTX 4090, 24 GiB VRAM, 124 GB system RAM).

Nemotron Super 49B — exact checkpoint
[nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)
— fits that profile: it does not fit a 4090 at uniform 4-bit (NVIDIA's own
NVFP4 quant is still over the card), the required ~3.2 average
bits/parameter is achievable *only* with non-uniform assignment, and the
model is open-weight with strong quality — so "49B on a 4090" is a headline
result people can reproduce. Bonus: the model is NAS-derived with
structurally heterogeneous layers, so uniform treatment is especially
unlikely to be optimal for it.

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
