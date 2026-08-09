---
status: draft
---

# Getting started

> **Status: draft** — every step runs against real code. The same loop
> ran end to end on the 49B target on 2026-07-29
> ([evidence](../explanation/evaluating-packed-models.md)).

This tutorial walks you from a clean checkout to a measured
mixed-precision recipe, using a small model so the scan finishes in
minutes.

## Prerequisites

- Linux with an NVIDIA GPU (any modern card works for the tutorial
  model — CPU also works, just slower)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)

## 1. Install

The scan step needs the GPU stack, which lives behind an extra
(ADR-0005):

```bash
git clone https://github.com/Alberto-Codes/quantfit.git
cd quantfit
uv sync --extra scan
uv run quantfit version
```

## 2. Scan a small model

The scan measures damage on some text — give it a calibration file:

```bash
printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..200} > calibration.txt

uv run quantfit scan HuggingFaceTB/SmolLM2-135M \
  --calibration calibration.txt \
  --max-tokens 2048 \
  --out sensitivity.json
```

The scan quantizes one layer group at a time at each candidate
precision (8, 4, 3, 2 bits), measures the damage relative to the
full-precision model, and writes a
[sensitivity map](../reference/sensitivity-map.md). Progress prints per
cell, and every finished cell lands in
`sensitivity.checkpoint.json` — rerun the same command after an
interruption and it resumes.

## 3. Plan a recipe

Pick a deliberately tight VRAM budget so the solver has real work to do:

```bash
uv run quantfit plan sensitivity.json --vram 256MiB --kv-headroom 64MiB --out recipe.json
```

The output [recipe](../reference/recipe.md) lists every layer group and
its assigned precision, plus the downgrade trace explaining each
choice. Try loosening or tightening `--vram` and watch the assignments
move.

## 4. Validate the recipe

The validation pass replays the whole recipe in one pass and reports
the measured damage next to the solver's prediction (ADR-0006):

```bash
uv run quantfit validate recipe.json --calibration calibration.txt --max-tokens 2048
```

Compare the measured number against the prediction. Six of seven
real measurements came in sub-additive. The outlier measured 11.9×
above prediction on a 2-bit-heavy recipe — treat that direction as
a solve-again signal, not a pack input (ADR-0006).

## 5. Pack and serve

The pack step applies the recipe and emits a GGUF that llama.cpp can
serve ([ADR-0010](../adr/0010-sub-4-bit-serving-path.md)). It needs a
llama.cpp checkout with built tools and the `pack` extra
(`uv sync --extra scan --extra pack`):

```bash
uv run quantfit pack recipe.json --llama-cpp ~/llama.cpp --out packed.gguf
```

Pack converts the checkpoint to an f16 base GGUF once, drives
`llama-quantize` with the recipe's type mapping, and re-checks the
packed file's real bytes against the weight budget. The
[how-to guides](../how-to/scan-a-model.md) cover scanning models that
*don't* trivially fit.
