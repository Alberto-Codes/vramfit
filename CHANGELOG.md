# Changelog

## 0.1.0 (2026-08-11)

First published release.

vramfit fits a large open model onto one GPU. It measures how much damage
each layer group takes at each candidate precision, then solves for a
mixed-precision recipe that fits a hard VRAM ceiling. Most quantized
models pick precision by heuristic. vramfit measures first.

### The pipeline

* **`vramfit scan`** — quantize one layer group at a time. Measure output
  divergence against the full-precision reference. Write a sensitivity map.
* **`vramfit plan`** — solve for a recipe under a VRAM ceiling. The solver
  spends bits where the sensitivity map says they matter. It runs without
  torch.
* **`vramfit validate`** — replay a full recipe in one pass. Compare the
  measured recipe damage against the solver prediction.
* **`vramfit pack`** — apply a recipe and write a GGUF checkpoint. Pack
  quantizes with an importance matrix, guards protected tensors with a
  per-tensor reconstruction check, and smoke-tests every artifact.
* **`vramfit budget`** — report the VRAM arithmetic for a model shape,
  context length, and KV-cache dtype.

### The acceptance test

Nemotron Super 49B serves on a 24 GiB RTX 4090. On 2026-08-09 an
end-to-end pack beat the size-matched community imatrix quant on
full-window KL divergence: 0.2873 against 0.2959, 7.8σ paired. The same
artifact holds the best nominal perplexity in its lane, 8.517 against
8.532, at 112 MiB under budget. Tier 3 ran five task benchmarks on
2026-08-10 and returned five statistical ties, none past 0.8σ.

The packed model and the sensitivity-map dataset are published on
[Hugging Face](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF).

### Install

Python 3.12 or later.

```bash
pip install vramfit          # plan and budget
pip install "vramfit[scan]"  # adds the torch scan meter
pip install "vramfit[pack]"  # adds the GGUF pack toolchain
```

The base install carries typer and structlog only. torch, transformers,
and the GGUF toolchain stay behind extras, so the plan step installs
without a GPU stack ([ADR-0005](docs/adr/0005-heavy-deps-as-extras.md)).

### Limits of this release

* Pack targets GGUF and llama.cpp. A vLLM backend for 4-bit and wider
  recipes is not written yet
  ([ADR-0010](docs/adr/0010-sub-4-bit-serving-path.md)).
* The solver excludes 2-bit until runtime-frame prices exist. Current
  practice plans on a map copy without the 2-bit column
  ([ADR-0021](docs/adr/0021-runtime-frame-measurement.md)).
* Artifacts carry the `vramfit_schema` envelope key. Artifacts written
  before the rename carry the old key and do not load.
