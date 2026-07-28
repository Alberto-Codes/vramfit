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

**[llama.cpp k-quants](https://github.com/ggml-org/llama.cpp)**
:   Shipping non-uniform per-layer recipes (Q4_K_M etc.) chosen by fixed
    heuristic. The thing quantfit replaces with measurement.

**[Minitron](https://arxiv.org/abs/2408.11796)** (NVIDIA)
:   Pruning + distillation for shrinking LLMs. The alternative road not taken
    ([ADR-0001](../adr/0001-selective-per-layer-quantization.md)): changes the
    model, needs training compute.

## Target runtime

**[vLLM](https://github.com/vllm-project/vllm)** /
**[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)**
:   First pack target and its mixed-precision checkpoint format
    ([ADR-0004](../adr/0004-vllm-first-runtime.md)). The runtime-capability
    table (which precisions have kernels) comes from here.

## Target model

**[Nemotron](https://huggingface.co/nvidia)** (NVIDIA)
:   Open-weight model family. Super 49B is the north-star target
    ([ADR-0003](../adr/0003-north-star-benchmark.md)); Nano is the
    "just run a smaller model" baseline.

## Writing system

**[ASD-STE100](https://www.asd-ste100.org/)**
:   Simplified Technical English — the controlled-language standard distilled
    into [CLAUDE.md](../../CLAUDE.md)'s writing rules. Free download from the
    official site.
