# Changelog

## 0.1.0 (2026-08-11)

First published release.

vramfit fits a large open model on one GPU. It measures how much damage
each layer group takes at each candidate precision, then solves for a
mixed-precision recipe that fits a hard VRAM ceiling. Most quantized
models pick precision by heuristic. vramfit measures first.

### The pipeline

* **`vramfit scan`** — quantize one layer group at a time. Measure output
  divergence against the full-precision reference. Write a sensitivity map.
* **`vramfit plan`** — solve for a recipe under a VRAM ceiling. The solver
  spends bits where the sensitivity map says they matter. The plan step
  runs without torch.
* **`vramfit validate`** — replay a full recipe in one pass. Compare the
  measured recipe damage against the solver prediction.
* **`vramfit pack`** — apply a recipe and write a GGUF checkpoint. Pack
  takes an importance matrix through `--imatrix`. Given one, it guards
  protected tensors with a per-tensor reconstruction check. `--smoke-text`
  runs a smoke test on the packed model. Without those flags pack warns
  that the artifact is unproven.
* **`vramfit budget`** — report the VRAM arithmetic for a model shape,
  context length, and KV-cache dtype.

### The acceptance test

Nemotron Super 49B serves on a 24 GiB RTX 4090. On 2026-08-09 an
end-to-end pack beat the size-matched community imatrix quant on
full-window KL divergence: 0.2873 against 0.2959, 7.8σ paired. Perplexity
reads 8.517 against 8.532, which the interval calls a tie. The baseline
holds its one clear lead, top-token agreement at 83.4 % against 82.9 %.
The packed model sits 112 MiB under the weight budget. Tier 3 ran five
task benchmarks on 2026-08-10 and returned five statistical ties, none
past 0.8σ.

Hugging Face hosts both artifacts: the
[packed model](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF)
and the
[sensitivity-map dataset](https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps).
The
[evidence ledger](https://github.com/Alberto-Codes/vramfit/blob/main/docs/explanation/evaluating-packed-models.md)
records all sixteen data points behind those numbers.

### Install

Python 3.12 or later.

```bash
pip install vramfit          # plan and budget
pip install "vramfit[scan]"  # adds the torch stack for scan and validate
pip install "vramfit[pack]"  # the scan stack plus the GGUF converter deps
```

The base install carries typer and structlog only. torch and transformers
stay behind the extras, so the plan step installs without a GPU stack
([ADR-0005](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0005-heavy-deps-as-extras.md)).
Two consequences follow. `vramfit validate` builds the same torch-backed
meter as `vramfit scan`, so it needs the scan extra. `vramfit pack` needs
a llama.cpp checkout with built tools, which you supply with
`--llama-cpp`
([ADR-0012](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0012-gguf-type-mapping.md)).

### Limits of this release

* Pack targets GGUF and llama.cpp. vramfit has no vLLM backend for 4-bit
  and wider recipes yet
  ([ADR-0010](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0010-sub-4-bit-serving-path.md)).
* The solver does not buy 2-bit until a runtime-frame price exists. It
  solves on a copy of the sensitivity map with the 2-bit column removed
  ([ADR-0021](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0021-runtime-frame-measurement.md)).
* Artifacts carry the `vramfit_schema` envelope key. Readers reject the
  pre-rename key by name, so a quantfit-era artifact does not load.
