---
status: sketch
---

# Getting started

> **Status: sketch** — the pipeline commands are not implemented; this
> tutorial describes the intended first-run experience.

This tutorial walks you from a clean checkout to a quantized model you can
serve, using a small model so the whole loop runs in minutes.

## Prerequisites

- Linux with an NVIDIA GPU (any modern card works for the tutorial model)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)

## 1. Install

```bash
git clone https://github.com/Alberto-Codes/quantfit.git
cd quantfit
uv sync
uv run quantfit version
```

## 2. Scan a small model

We use a small model first so a full sensitivity scan finishes quickly:

```bash
uv run quantfit scan <small-model-id> --out sensitivity.json
```

The scan quantizes one layer group at a time at each candidate precision,
measures divergence from the full-precision model, and writes a
[sensitivity map](../reference/sensitivity-map.md).

## 3. Plan a recipe

Pick a deliberately tight VRAM budget so the solver has real work to do:

```bash
uv run quantfit plan sensitivity.json --vram 4GiB --kv-headroom 1GiB --out recipe.json
```

The output [recipe](../reference/recipe.md) lists every layer group and its
assigned precision.

## 4. Pack and serve

```bash
uv run quantfit pack <small-model-id> --recipe recipe.json --out ./packed
```

Point vLLM at `./packed` and generate text. You have run the full
scan → plan → pack loop; the [how-to guides](../how-to/scan-a-model.md) cover
doing this on models that *don't* trivially fit.
