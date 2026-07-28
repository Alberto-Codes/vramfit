# ADR-0004: vLLM as the first target runtime

- **Status:** Accepted
- **Date:** 2026-07-27

## Context

The pack step must emit a checkpoint some runtime can serve; supporting every
runtime from day one is breadth we've explicitly rejected. Candidates: vLLM
(maintainer has a working fork and quantization tooling built around it —
turboquant-vllm), llama.cpp/GGUF (largest hobbyist reach, k-quant precedent),
SGLang, TensorRT-LLM.

vLLM already supports mixed-precision checkpoints via compressed-tensors,
serves the target model class, and is where existing project expertise and
tooling live.

## Decision

`quantfit pack` targets **vLLM (compressed-tensors format) first**. GGUF
export is planned second — it reaches the llama.cpp audience that made
antirez-style selective GGUFs popular — and the recipe format stays
runtime-neutral so a pack backend is additive.

## Consequences

- Fastest path to the ADR-0003 benchmark on existing infrastructure.
- The recipe's precision choices must respect what vLLM kernels actually
  support per-tensor — the plan step needs a runtime-capability table, which
  is a real constraint a purely mathematical solver would miss.
- The 2/3-bit end of the recipe may be constrained by available vLLM kernel
  support before it's constrained by quality — a risk to validate early.
- llama.cpp users wait for the GGUF backend; acceptable, they're not the
  benchmark audience.
