---
status: draft
---

# Prior art and external references

> **Status: draft** — annotated pointers, not endorsements. Each entry says
> why it matters to quantfit; follow the link for the substance.

## Direct inspiration

**[antirez/ds4](https://github.com/antirez/ds4)**
:   Single-model inference engine for DeepSeek V4 Flash with hand-tuned
    selective quantization
    ([GGUFs](https://huggingface.co/antirez/deepseek-v4-gguf)). Source of the
    depth-over-breadth ethos ([ADR-0001](../adr/0001-selective-per-layer-quantization.md))
    and proof that per-model recipes beat generic ones.

## Quantization methods

**[GPTQ](https://arxiv.org/abs/2210.17323)**
:   Calibration-aware post-training quantization via second-order weight
    updates. A candidate within-group method — quantfit decides bits per
    group; GPTQ is one way to round within a group.

**[AWQ](https://arxiv.org/abs/2306.00978)**
:   Activation-aware weight quantization; protects the small fraction of
    outlier channels that carry outsized importance. Same role as GPTQ here,
    and evidence for non-uniform fragility.

**[llama.cpp k-quants + imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix)**
:   k-quants ship non-uniform per-layer recipes chosen by fixed heuristic;
    the imatrix tool adds measured activation statistics that bias
    *within-block* scale selection. Measurement below the bit-assignment
    level — complementary to, not competing with, quantfit's per-group
    decision. `llama-quantize --tensor-type` accepts per-tensor type
    overrides, which is the mechanism the GGUF pack backend drives
    ([ADR-0010](../adr/0010-sub-4-bit-serving-path.md)).

**[Unsloth dynamic GGUFs](https://huggingface.co/unsloth)**
:   Selective per-layer bit assignment shipped as ready-made GGUFs —
    the closest prior art by shipped output, where EXL2 below is
    closest by method. The recipes are expert heuristics, and Unsloth
    does not publish the selection evidence. quantfit bets that a
    measured, published sensitivity map beats this recipe class. See
    [the artifact ecosystem](../explanation/artifact-ecosystem.md).

**[EXL2 / exllamav2](https://github.com/turboderp-org/exllamav2)**
:   The closest prior art by method. Measured per-layer variable bitrate (2–8 bpw
    mixing) hitting a target *average* bits-per-weight, with a reusable
    measurement pass. Differences quantfit bets on: EXL2 is tied to its own
    runtime (quantfit targets vLLM), optimizes to an average bpw rather than
    an explicit VRAM budget planned jointly with KV headroom, and its
    measurement is not a standalone, inspectable artifact. Study its
    measurement pass before designing `scan`.

**[llm-compressor](https://github.com/vllm-project/llm-compressor)**
:   vLLM's official companion for producing compressed-tensors checkpoints
    (GPTQ, AWQ, FP8/INT4 schemes, non-uniform per-layer configs). The likely
    implementation backend for `quantfit pack` — quantfit's job is deciding
    the recipe, llm-compressor's is applying it.

**[Minitron](https://arxiv.org/abs/2408.11796)** (NVIDIA)
:   Pruning + distillation for shrinking LLMs. The alternative road not taken
    ([ADR-0001](../adr/0001-selective-per-layer-quantization.md)): changes the
    model, needs training compute.

## Target runtime

**[vLLM](https://github.com/vllm-project/vllm)** /
**[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)**
:   First pack target for ≥4-bit recipes and its mixed-precision checkpoint
    format ([ADR-0004](../adr/0004-vllm-first-runtime.md)). The
    runtime-capability table (which precisions have kernels) comes from
    here. Verified 2026-07: the Marlin and Machete kernels cover 4-bit and
    8-bit only. Upstream closed the EXL3 integration request
    ([vllm#19896](https://github.com/vllm-project/vllm/issues/19896)) as
    not planned. Upstream also moved in-tree GGUF support out to
    [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin),
    whose tested formats stop at 4 bits. On that evidence
    [ADR-0010](../adr/0010-sub-4-bit-serving-path.md) routes sub-4-bit
    recipes through the GGUF serving path.

## Target model

**[Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)** (NVIDIA)
:   The north-star target ([ADR-0003](../adr/0003-north-star-benchmark.md)):
    dense decoder Transformer derived from Llama 3.3 70B via Neural
    Architecture Search — layers are structurally heterogeneous, which
    strengthens the case for per-layer measurement. NVIDIA publishes an
    official [NVFP4 quant](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4)
    (~4-bit uniform, still over a 24 GiB card) — a quality baseline for
    quantfit recipes to beat. Nemotron Nano is the "just run a smaller
    model" baseline.

## Writing system

**[ASD-STE100](https://www.asd-ste100.org/)**
:   Simplified Technical English — the controlled-language standard distilled
    into [CLAUDE.md](../../CLAUDE.md)'s writing rules. Free download from the
    official site.
