---
status: draft
---

# How to pack a recipe

> **Status: draft** — `vramfit pack` is implemented for the GGUF
> backend, verified on Qwen2.5-3B, and packed the 49B target 169.7 MiB
> under budget on the first try. The vLLM backend does not exist yet
> (ADR-0004, ADR-0010).

## Goal

Turn a [recipe](../reference/recipe.md) into a packed model llama.cpp
can serve, with the type mapping from
[ADR-0012](../adr/0012-gguf-type-mapping.md).

## Prerequisites

The pack step drives external tools — none ship with vramfit:

1. A llama.cpp checkout with built tools:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cmake -B llama.cpp/build llama.cpp -DCMAKE_BUILD_TYPE=Release
   cmake --build llama.cpp/build -j --target llama-quantize llama-perplexity llama-imatrix
   ```

   `llama-perplexity` runs the post-pack smoke test (ADR-0017) —
   build it unless you plan to skip the smoke test. `llama-imatrix`
   generates the importance matrix used below — build it unless one
   already exists for your model.

2. A Python able to run `convert_hf_to_gguf.py`. The `pack` extra
   provisions it (torch, transformers, sentencepiece):

   ```bash
   uv pip install "vramfit[pack]"
   ```

   sentencepiece matters even for BPE models: the Qwen converter
   probes it first and dies with `ModuleNotFoundError` when it is
   absent, before it can fall back to the BPE vocab path.

## Basic invocation

```bash
uv run vramfit pack recipe-4GiB.json \
  --llama-cpp ~/llama.cpp \
  --out qwen2.5-3b-recipe-4GiB.gguf \
  --threads 14
```

The recipe's `model_id` names the checkpoint directory when it is a
local path — pass `--model` otherwise. The command converts the
checkpoint to an f16 base GGUF once (minutes at 3B scale), then
quantizes it with one type override per layer group. The embedding and
`lm_head` groups drive the quantizer's dedicated
`--token-embedding-type` and `--output-tensor-type` flags instead of
pattern overrides (ADR-0012). A second pack of
the same model reuses the base GGUF and skips the conversion.

## Packing with an importance matrix

The community baselines quantize imatrix-assisted, and the first 49B
head-to-head traced ~81 % of the quality gap to exactly that
([ADR-0016](../adr/0016-imatrix-in-the-pack-path.md)). Generate the
matrix once per (base GGUF, calibration text) pair with
`llama-imatrix`, then hand it to pack:

```bash
llama-imatrix -m model-f16.gguf -f calibration.txt -o model.imatrix.gguf
uv run vramfit pack recipe.json \
  --llama-cpp ~/llama.cpp \
  --imatrix model.imatrix.gguf \
  ...
```

Use the scan's calibration text — one text source feeds the whole
measured pipeline. The quantizer embeds the matrix's provenance in
the packed file, and the `model_packed` run-log event records the
path.

The packed file's `general.file_type` names the tensor type that
covers the most bytes in it, under `file_type` on the same event.
One field cannot name a mixed recipe, so it names the largest share
([ADR-0012](../adr/0012-gguf-type-mapping.md) decision 3, the
2026-09-04 amendment). Read the recipe for the mix.

The command also reads the matrix's `.counts` tensors against the
base GGUF before quantizing
([ADR-0026](../adr/0026-moe-expert-pricing.md) decision 5). An
expert the matrix counts zero times quantizes at the unassisted
fit with no warning from the quantizer, so pack warns instead and
records the `(stack, expert)` pairs under
`imatrix_zero_count_experts`. A matrix the reader cannot vouch for
halts the pack. The read needs gguf-py, which the pack extra
already provisions.

## The smoke test

A packed artifact can pass the solver, the validation pass, and the
size re-check and still be destroyed at inference — a 49B recipe did
exactly that on 2026-07-29
([ADR-0017](../adr/0017-post-pack-smoke-test.md)). Pass
`--smoke-text` so pack proves the artifact emits language before
anything downstream trusts it:

```bash
uv run vramfit pack recipe.json \
  --llama-cpp ~/llama.cpp \
  --smoke-text calibration.txt \
  ...
```

A few chunks through `llama-perplexity` (layer offload disabled —
the test never contends for the GPU) must land under the `--smoke-threshold`
ceiling (default 1000). Working artifacts measure 8–10, destroyed
ones ~10⁶. A failing smoke test exits 1 and keeps the file for
inspection. Without `--smoke-text` the command warns that the
artifact is unproven.

## The reconstruction check

The smoke test cannot see fit collapse — a 47-layer protected build
scored 9.594 PPL and would have smoked clean
([ADR-0022](../adr/0022-within-layer-protections.md)). On a
protected recipe packed with `--imatrix`, pack therefore runs the
reconstruction check automatically: it packs the same recipe once
more with the protections stripped, dequantizes every protected
tensor from both files, and compares each against the f16 base. The
reference file is deleted afterward. Budget one extra quantize run
(5–17 minutes at 49B scale, thread-count dependent) and transient
disk for a second full-size artifact beside `--out`. A collapsed tensor names itself in
the output, and the refusal prints the exact remedy flags: re-plan
with `--exclude-imatrix` for the named tensors — the tensor keeps
its promotion and quantizes without its imatrix row
([ADR-0023](../adr/0023-imatrix-exclusions.md)). Dropping the
tensor from `--protect` is the fallback when the exclusion has
already failed. G1 needed exactly one drop round (layers 1, 2,
and 5, before exclusions existed), and the fifteenth data point's
pack cleared the gate all-green with four exclusions.

## Reading the result

The human channel prints the re-check line:

```
packed 37 groups -> qwen2.5-3b-recipe-4GiB.gguf (1.99 GiB),
weight budget 2.00 GiB, margin 5.26 MiB under
```

The run log (`<stem>.runlog.jsonl`) carries the same run as machine
events — `size_checked` records `packed_bytes`,
`weight_budget_bytes`, `margin_bytes`, and `fits`. An over-budget
pack exits 1 and keeps the file, so you can inspect what overflowed.

## Sizes at plan time

The solver prices llama.cpp recipes at per-type effective bits
([ADR-0014](../adr/0014-per-type-effective-bits.md)): Q4_K costs 4.5
bits/weight in the prediction, exactly what `llama-quantize` writes.
Leave `--format-overhead` alone — the 0.005 default covers the file
metadata and unquantized tensors the per-type table cannot see. The
hand-tuned overheads this section used to recommend (0.10, then
0.105 for 5-bit mixes) are obsolete: they were one scalar chasing
the mix-weighted drift of whatever types the solver picked, and the
6/5/4 mix overflowed the value that fit its predecessor. The re-check
above stays as the backstop.

## Evaluating the packed model

The scoreboard lives in
[evaluating packed models](../explanation/evaluating-packed-models.md):
perplexity first, whole-model KL against the f16 reference second,
task benchmarks last. The base GGUF from the pack doubles as the
reference model for both tiers.

The published Gemma 4 fit24gib pack has its own serving guide:
[serve the pack on a rented H100](serve-gemma-4-fit24gib-on-a-rented-h100.md).
That H100 is the instrument behind the pack's divergence numbers.
