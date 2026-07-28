# ADR-0010: The sub-4-bit serving path runs through GGUF

- **Status:** Proposed
- **Date:** 2026-07-28

## Context

ADR-0003 fixes the benchmark: Nemotron Super 49B on one 24 GiB RTX 4090
at 16k context, via vLLM. ADR-0004 records the conflict: the measured
weight budget is 18.94 GiB at fp16 KV and 20.47 GiB at fp8 KV, which
forces ~3.3–3.5 average bits/parameter. vLLM kernels floor at 4 bits.
ADR-0004 lists four resolution paths and defers the choice. The scan
step cannot fix its candidate precision set until this ADR decides.

Evidence, verified 2026-07-28:

- **(a) Friendlier math: closed.** ~49B parameters at uniform 4-bit
  weigh ~22.8 GiB before format overhead. That exceeds even the
  20.47 GiB fp8-KV budget by more than 2 GiB. No overhead tuning closes
  a gap that size.
- **(b) A vLLM sub-4-bit kernel: possible, but against the current.**
  Upstream closed the EXL3 integration request
  ([vllm#19896](https://github.com/vllm-project/vllm/issues/19896)) as
  not planned. Upstream moved in-tree GGUF support out to
  [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin),
  whose tested formats stop at 4 bits. The Marlin and Machete kernels
  cover 4-bit and 8-bit only. A kernel in the maintainer's vLLM fork
  stays possible — as a multi-month CUDA project that reproducers must
  then build from source.
- **(c) GGUF through llama.cpp: works today.** llama.cpp serves the
  exact target checkpoint (community GGUFs of v1_5 exist). Its quant
  types mix per-tensor down to 2 bits, and `llama-quantize
  --tensor-type` accepts per-tensor overrides — the exact mechanism a
  recipe needs. Size evidence for the target model: IQ3_XXS is
  18.2 GiB and fits the 18.94 GiB fp16-KV budget. IQ3_XS is 19.5 GiB
  and fits the 20.47 GiB fp8-KV budget.
- **(d) A different benchmark model: gives up ADR-0003.**

Path (c) changes what the benchmark claims. Heuristic 3-bit GGUFs of
the target already fit the card, so "49B on a 4090 at all" is no longer
the impossible thing. The impossible thing is doing it without wrecking
quality — no published quant of this model chooses its per-layer bits
from measurement.

## Decision (proposed)

1. **The scan measures candidate precisions {8, 4, 3, 2} per group.**
   Sensitivity is a property of the model, not of a runtime. The scan
   simulates quantization (quantize–dequantize, bf16 compute) and needs
   no runtime kernel, so this set is safe under every resolution path.
2. **The benchmark serves through llama.cpp.** `quantfit pack` grows a
   GGUF backend, promoted from "planned second" (ADR-0004) to first in
   line. This amends the serving-runtime clause of ADR-0003.
3. **The benchmark claim sharpens to quality at equal size.** The
   recipe must beat the size-matched heuristic baselines (IQ3_XXS and
   IQ3_XS of the target) and NVIDIA's NVFP4 quant on measured damage,
   inside the same budget.
4. **vLLM remains the first runtime for recipes at 4-bit and above.**
   ADR-0004 stands in that regime. A sub-4-bit vLLM kernel stays open
   as a stretch path that would return the benchmark to vLLM. Pursuing
   it gets its own ADR.

## Consequences

- The scan is unblocked. Its candidate set is {8, 4, 3, 2} and does not
  depend on any runtime's kernel table.
- The plan step gains a runtime-capability input: llama.cpp allows
  {8, 6, 5, 4, 3, 2}, vLLM allows {8, 4}. The solver is already
  precision-set agnostic (ADR-0007), so this lands as a filter on the
  candidate set.
- The pack milestone targets `llama-quantize` with per-tensor overrides
  before llm-compressor. llm-compressor returns when a ≥4-bit vLLM
  recipe is packed.
- Nominal bits and real bytes drift apart: GGUF 3-bit types spend
  3.4–3.9 effective bits/weight. `--format-overhead` absorbs the gap
  today. Pack must re-check real sizes against the budget.
- The headline of ADR-0003 stays: 49B, one 4090, 16k context, measured
  quality loss.

## Open questions

- Which GGUF types map to nominal 3 and 2 bits? K-quants need no extra
  input. I-quants need an importance matrix, which the scan's
  calibration pass can produce as a byproduct.
- Are byte predictions from nominal bits too coarse for the solver once
  GGUF types set the real sizes? If yes, the sensitivity map grows
  per-precision measured byte counts.
