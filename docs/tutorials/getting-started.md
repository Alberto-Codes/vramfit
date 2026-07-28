---
status: draft
---

# Getting started

> **Status: draft** — steps 1–3 (install, scan, plan) run against real
> code today. Step 4 (pack) is design-stage, so the loop does not close
> yet.

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
precision (8, 4, 3, 2 bits), measures divergence from the
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

## 4. Pack and serve *(not yet implemented)*

The pack step will apply the recipe and emit a checkpoint a runtime can
serve — GGUF/llama.cpp first, per
[ADR-0010](../adr/0010-sub-4-bit-serving-path.md). Until it lands, the
recipe is the end of the loop. The
[how-to guides](../how-to/scan-a-model.md) cover scanning models that
*don't* trivially fit.
