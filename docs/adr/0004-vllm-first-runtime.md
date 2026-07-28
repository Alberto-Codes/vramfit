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
- **Confirmed constraint** (2026-07): compressed-tensors supports per-layer
  non-uniform schemes over W4A16, W8A16, W8A8, and the FP4 microscaling
  formats (NVFP4A16, MXFP4A16) — there are no 2/3-bit vLLM kernels. The
  candidate-precision set for vLLM recipes is effectively {8, 4-int, 4-fp}
  until that changes; sub-4-bit recipes need the future GGUF backend.
- `quantfit pack` should drive [llm-compressor](https://github.com/vllm-project/llm-compressor)
  rather than reimplement checkpoint writing.
- **Open tension with ADR-0003:** the north-star budget forces ~3.2 average
  bits/parameter, below vLLM's 4-bit kernel floor. Resolution paths, in
  rough order of preference: (a) the scan + budget math turns out friendlier
  than the estimate (e.g. FP8 KV, tighter overhead), (b) contribute or adopt
  a sub-4-bit kernel in vLLM (maintainer has a vLLM fork), (c) the GGUF
  backend becomes the benchmark path, (d) the benchmark model changes. Needs
  its own ADR before `plan` lands.
- llama.cpp users wait for the GGUF backend; acceptable, they're not the
  benchmark audience.
