---
status: draft
---

# How to pack a recipe

> **Status: draft** — `quantfit pack` is implemented for the GGUF
> backend and verified on Qwen2.5-3B against the reference box. The
> vLLM backend does not exist yet (ADR-0004, ADR-0010).

## Goal

Turn a [recipe](../reference/recipe.md) into a packed model llama.cpp
can serve, with the type mapping from
[ADR-0012](../adr/0012-gguf-type-mapping.md).

## Prerequisites

The pack step drives external tools — none ship with quantfit:

1. A llama.cpp checkout with built tools:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cmake -B llama.cpp/build llama.cpp -DCMAKE_BUILD_TYPE=Release
   cmake --build llama.cpp/build -j --target llama-quantize
   ```

2. A Python able to run `convert_hf_to_gguf.py`. It needs torch and
   sentencepiece. The quantfit `scan` extra covers torch — add
   sentencepiece beside it:

   ```bash
   uv pip install "quantfit[scan]" sentencepiece
   ```

   Without sentencepiece the Qwen converter dies with
   `ModuleNotFoundError` before it can fall back to the BPE vocab
   path.

## Basic invocation

```bash
uv run quantfit pack recipe-4GiB.json \
  --llama-cpp ~/llama.cpp \
  --out qwen2.5-3b-recipe-4GiB.gguf \
  --threads 14
```

The recipe's `model_id` names the checkpoint directory when it is a
local path — pass `--model` otherwise. The command converts the
checkpoint to an f16 base GGUF once (minutes at 3B scale), then
quantizes it with one type override per layer group. A second pack of
the same model reuses the base GGUF and skips the conversion.

## Reading the result

The human channel prints the re-check line:

```
packed 37 groups -> qwen2.5-3b-recipe-4GiB.gguf (2.11 GiB),
weight budget 2.00 GiB, margin ... under
```

The run log (`<stem>.runlog.jsonl`) carries the same run as machine
events — `size_checked` records `packed_bytes`,
`weight_budget_bytes`, `margin_bytes`, and `fits`. An over-budget
pack exits 1 and keeps the file, so you can inspect what overflowed.
Real bytes exceed the nominal-bit prediction because GGUF types
spend 6-31 % more effective bits than their nominal precision
(ADR-0012).

## Evaluating the packed model

The scoreboard lives in
[evaluating packed models](../explanation/evaluating-packed-models.md):
perplexity first, whole-model KL against the f16 reference second,
task benchmarks last. The base GGUF from the pack doubles as the
reference model for both tiers.
