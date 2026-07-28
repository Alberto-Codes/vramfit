---
status: draft
---

# Why selective quantization

> **Status: draft** — the reasoning is well-grounded in published work; our
> own measurements don't exist yet.

## The arithmetic that forces the issue

Weights dominate a model's memory footprint: `parameters × bits ÷ 8`.
Nemotron Super 49B at bf16 is ~98 GB. On a 24 GiB RTX 4090:

| Uniform precision | Approx. weight size | Fits with KV headroom? |
|-------------------|--------------------|------------------------|
| 16-bit | ~98 GB | No |
| 8-bit | ~49 GB | No |
| 4-bit | ~26 GB | No — over the card's total, before KV |
| 3-bit | ~19.5 GB | Barely — and uniform 3-bit quality is poor |

Uniform quantization has no answer here: the bit-width that fits wrecks the
model, and the bit-width that preserves it doesn't fit.

## Why non-uniform works

Quantization error is not spent equally well everywhere. Consistent findings
across the quantization literature (GPTQ/AWQ lineage, llama.cpp k-quants,
NVIDIA's own Minitron work):

- **Embeddings, first blocks, last blocks** disproportionately affect output
  quality — cheap insurance to keep at high precision.
- **Attention projections** are typically more fragile than MLP weights.
- **Outlier channels** — a small fraction of weights with large magnitudes —
  carry outsized importance (the insight behind AWQ).
- **Most mid-stack MLP mass tolerates aggressive crushing** — and MLPs are
  where most of the parameters are.

Existing tools encode these as fixed heuristics. The fragility *profile*
differs per architecture — a hybrid or MoE model distributes importance
differently than a dense llama-style stack — so a recipe measured for the
actual target model should beat a generic one. That measured-per-model bet is
the project's core hypothesis, and the 49B-on-a-4090 benchmark is its test.

## The prior art we're standing on

- **antirez/ds4** — proved the depth-over-breadth ethos: selective
  quantization hand-tuned for one model (DeepSeek V4 Flash) beat generic
  runtimes' recipes for that model.
- **llama.cpp k-quants** — ship non-uniform layer recipes, but heuristic and
  generic across architectures.
- **AWQ / GPTQ** — calibration-aware weight quantization within a uniform
  target precision; quantfit's selectivity operates a level above (which
  precision per group), and can use these as the within-group method.
