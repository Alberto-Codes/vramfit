# Changelog

## [0.2.0](https://github.com/Alberto-Codes/vramfit/compare/v0.1.0...v0.2.0) (2026-08-15)


### Features

* **adapters:** read an evals sidecar back through a port ([#259](https://github.com/Alberto-Codes/vramfit/issues/259)) ([a68390b](https://github.com/Alberto-Codes/vramfit/commit/a68390bbb716dd652f4fbf1951bc04006e774715))
* **cli:** state coverage counts in the pack echo, not every name ([#226](https://github.com/Alberto-Codes/vramfit/issues/226)) ([9d4401a](https://github.com/Alberto-Codes/vramfit/commit/9d4401af700208d9f9a334d1a773578644284479))
* **cli:** state the uncovered imatrix count instead of every name ([#224](https://github.com/Alberto-Codes/vramfit/issues/224)) ([b8ae709](https://github.com/Alberto-Codes/vramfit/commit/b8ae7098b96b0518ec50732fcecc78fdb231d8da))
* **config:** match guard rules on tokens rather than raw text ([04e544b](https://github.com/Alberto-Codes/vramfit/commit/04e544b25746a25dce755461b1e7d24f28918f55))
* **config:** route each guard rule by whether its check is mechanical ([#269](https://github.com/Alberto-Codes/vramfit/issues/269)) ([04e544b](https://github.com/Alberto-Codes/vramfit/commit/04e544b25746a25dce755461b1e7d24f28918f55)), closes [#246](https://github.com/Alberto-Codes/vramfit/issues/246)
* **pack:** map expert stacks and every layer naming family ([979bf40](https://github.com/Alberto-Codes/vramfit/commit/979bf40fdcdbc7b5e75dad342a15d01548216582)), closes [#180](https://github.com/Alberto-Codes/vramfit/issues/180)
* **pack:** map expert stacks through the ADR-0028 type table ([#231](https://github.com/Alberto-Codes/vramfit/issues/231)) ([a415cf9](https://github.com/Alberto-Codes/vramfit/commit/a415cf9400d444484f7aecc891870cfbc51e47fb)), closes [#228](https://github.com/Alberto-Codes/vramfit/issues/228)
* **pack:** report zero-count experts from the imatrix ([#218](https://github.com/Alberto-Codes/vramfit/issues/218)) ([9ed0c7b](https://github.com/Alberto-Codes/vramfit/commit/9ed0c7b9817081821e29d0d217507c24117acddf)), closes [#179](https://github.com/Alberto-Codes/vramfit/issues/179)
* **scan:** add the slice perturbation path to the meter ([#222](https://github.com/Alberto-Codes/vramfit/issues/222)) ([8788d4c](https://github.com/Alberto-Codes/vramfit/commit/8788d4cfb0559e9d9228cd40b39e037669800135))
* **scan:** key the sensitivity map on the pack-addressable stack ([#181](https://github.com/Alberto-Codes/vramfit/issues/181)) ([48ad52c](https://github.com/Alberto-Codes/vramfit/commit/48ad52cec39cce9d7219f5ced3f65322b6b4aa69))
* **scan:** map Nemotron-H dense tensor names to their imatrix entries ([8a6f374](https://github.com/Alberto-Codes/vramfit/commit/8a6f3741453ef690f11d7898380aef8bdd6b2a10)), closes [#186](https://github.com/Alberto-Codes/vramfit/issues/186)
* **scan:** read a fused expert stack's counts as one vector ([#214](https://github.com/Alberto-Codes/vramfit/issues/214)) ([fe8f478](https://github.com/Alberto-Codes/vramfit/commit/fe8f47841ef8b95848042b0ec43cee271b1b8806))
* **scan:** read an expert stack's imatrix rows ([#187](https://github.com/Alberto-Codes/vramfit/issues/187)) ([a3c7b92](https://github.com/Alberto-Codes/vramfit/commit/a3c7b92346bf57e3cfab694615e34c67a6263099)), closes [#177](https://github.com/Alberto-Codes/vramfit/issues/177)
* **scan:** record per-stack imatrix count summaries on the map ([#217](https://github.com/Alberto-Codes/vramfit/issues/217)) ([315d790](https://github.com/Alberto-Codes/vramfit/commit/315d7904dbdfb6fcbea65aaba5cf0798360321f8)), closes [#179](https://github.com/Alberto-Codes/vramfit/issues/179)


### Bug Fixes

* **adapters:** keep the map's derived note across a save ([#254](https://github.com/Alberto-Codes/vramfit/issues/254)) ([be37d61](https://github.com/Alberto-Codes/vramfit/commit/be37d615a4bab13a6e8855e96806a37912a2293a)), closes [#136](https://github.com/Alberto-Codes/vramfit/issues/136)
* **pack:** decode toolchain output with a replacing handler ([#250](https://github.com/Alberto-Codes/vramfit/issues/250)) ([de2590e](https://github.com/Alberto-Codes/vramfit/commit/de2590ea346e709e85413a7930f21e40bb504671)), closes [#247](https://github.com/Alberto-Codes/vramfit/issues/247)
* **pack:** report an unnamed signal by number, not a ValueError ([#258](https://github.com/Alberto-Codes/vramfit/issues/258)) ([d079c04](https://github.com/Alberto-Codes/vramfit/commit/d079c04fd9e1112fc4ce6dcb231f3a8e4243e81c))
* **scan:** close the imatrix reader's silent-mispricing paths ([a3c7b92](https://github.com/Alberto-Codes/vramfit/commit/a3c7b92346bf57e3cfab694615e34c67a6263099))

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
  pre-rename key by name, so a pre-rename artifact does not load.
