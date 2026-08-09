# ADR-0010: The sub-4-bit serving path runs through GGUF

- **Status:** Accepted
- **Date:** 2026-07-28 (accepted 2026-07-28)
- **Amendment (2026-07-29):** the decision-3 claim was measured and
  lost. The first packed recipe fit the budget and scored PPL 9.917
  against the size-matched imatrix Q3_K_S at 8.532. The control
  experiment traced ~81 % of the gap to the importance matrix
  ([evaluating packed models](../explanation/evaluating-packed-models.md)).
  The size-matched baseline in practice is Q3_K_S, not the IQ3
  quants listed below. ADR-0012's i-quant question now gates the
  claim.
- **Note (2026-07-29, later):** the imatrix rematch ran under
  ADR-0016 and lost by 0.53 PPL at equal size and equal toolchain
  (9.061 vs 8.532 — the fifth data point). The toolchain gate is
  gone. What gates the claim now is the solver's additive damage
  model at 2-bit: the validation pass measured it super-additive by
  11.9× on the 32,768-token map's recipe, and that recipe packed
  worse than the pilot map's in every cell of the 2×2.
- **Note (2026-08-09):** the gate cleared. ADR-0021 banned unpriced
  2-bit, ADR-0022/0023 added within-layer protections with imatrix
  exclusions, and the fifteenth data point's end-to-end pack beat
  the size-matched Q3_K_S baseline on full-window KL divergence at
  7.8σ with the best nominal perplexity in the lane
  ([evaluating packed models](../explanation/evaluating-packed-models.md)).
  Decision 3's benchmark claim is achieved. Open question 3 (serving
  overhead at 16k context) stays open.

## Context

ADR-0003 fixes the benchmark: Nemotron Super 49B on one 24 GiB RTX 4090
at 16k context, via vLLM. ADR-0004 records the conflict: the measured
weight budget is 18.94 GiB at fp16 KV and 20.47 GiB at fp8 KV, which
forces ~3.3–3.5 average bits/weight. vLLM kernels floor at 4 bits.
ADR-0004 lists four resolution paths and defers the choice. The scan
step cannot fix its candidate precision set until this ADR decides.

Evidence, verified 2026-07-28:

- **(a) Friendlier math: closed.** The real checkpoint holds 49.87B
  parameters, which weigh ~23.2 GiB at uniform 4-bit before format
  overhead. That exceeds even the 20.47 GiB fp8-KV budget by almost
  3 GiB. No overhead tuning closes a gap that size.
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

## Decision

1. **The scan measures candidate precisions {8, 4, 3, 2} per group.**
   Sensitivity is a property of the model, not of a runtime. The scan
   simulates quantization (quantize–dequantize, bf16 compute) and needs
   no runtime kernel, so this set is safe under every resolution path.
2. **The benchmark serves through llama.cpp.** `quantfit pack` grows a
   GGUF backend, promoted from "planned second" (ADR-0004) to first in
   line. This amends the serving-runtime clause of ADR-0003.
3. **The benchmark claim sharpens to quality at equal size.** The
   recipe must beat the size-matched heuristic baselines (IQ3_XXS and
   IQ3_XS of the target) on measured damage inside the same budget.
   NVIDIA's NVFP4 quant does not fit the card (ADR-0003), so it serves
   as the over-budget quality reference, not a same-budget baseline.
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
  ~3.1–3.9 effective bits/weight (IQ3_XXS 3.14, Q3_K_M 3.91 on the
  target). `--format-overhead` absorbs the gap today. Pack must
  re-check real sizes against the budget.
- The headline of ADR-0003 stays: 49B, one 4090, 16k context, measured
  damage.

## Open questions

- ~~Which GGUF types map to nominal 3 and 2 bits? K-quants need no
  extra input. I-quants need an importance matrix, which the scan's
  calibration pass can produce as a byproduct.~~ Resolved by
  [ADR-0012](0012-gguf-type-mapping.md): K-quants for v1. The i-quant
  half escalated on 2026-07-29 to north-star-gating (see ADR-0012's
  open questions).
- ~~Are byte predictions from nominal bits too coarse for the solver
  once GGUF types set the real sizes? If yes, the sensitivity map
  grows per-precision measured byte counts.~~ Resolved by
  [ADR-0014](0014-per-type-effective-bits.md): yes — and exact
  per-type block constants replaced the guess, no map change needed.
- The 18.94 / 20.47 GiB budgets assume vLLM's 2 GiB runtime overhead
  and KV layout. Measure both under llama.cpp on the reference box —
  the budgets move if llama.cpp allocates differently. The 2026-07-29
  loop packed 20.30 GiB against the 20.47 GiB budget, but a measured
  llama.cpp overhead figure at 16k context still does not exist.
