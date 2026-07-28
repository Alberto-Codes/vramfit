[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![docs vetted](https://img.shields.io/badge/docs%20vetted-docvet-purple)](https://github.com/Alberto-Codes/docvet)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

# quantfit

Selective per-layer quantization to fit large open models on a single GPU.

## What is this?

**The problem.** A model's weights are billions of numbers, normally stored at
16 bits each — Nemotron Super 49B is ~98 GB at full precision, and an RTX 4090
has 24 GB. Quantization stores those numbers with fewer bits (8, 4, even 2),
trading a little accuracy for a lot of memory. But even uniform 4-bit puts 49B
parameters at ~26 GB — still doesn't fit — and uniform 3-bit wrecks quality,
because some parts of a transformer get badly stupid when you crush them.

**The insight.** Not all layers are equally fragile. Some tolerate 2–3 bits
with barely a ripple; others (attention projections, first/last blocks) fall
apart below 6–8 bits. Most published quantized models pick precision by crude
heuristic. `quantfit` *measures* which layers are which, then solves for the
best mixed-precision recipe that fits a *specific* model into a *specific*
VRAM budget:

1. **Scan** — quantize one layer group at a time at candidate precisions and
   measure output divergence against the full-precision reference. Output: a
   sensitivity map of which layers can survive being crushed.
2. **Plan** — a budget problem: given the map and a hard VRAM constraint
   (minus KV-cache headroom), spend bits where the scan says they matter and
   crush where it says they don't. Bits are cost, quality is value.
3. **Pack** — apply the recipe and emit a checkpoint the target runtime can
   actually serve (vLLM for ≥4-bit recipes, GGUF for the sub-4-bit benchmark
   path per [ADR-0010](docs/adr/0010-sub-4-bit-serving-path.md), Proposed).

**The goal:** NVIDIA **Nemotron Super 49B running on a 24 GB RTX 4090** — a
model that does not fit today, made to fit selectively, with measured (not
vibes-based) quality loss versus running a smaller model instead.

Philosophy borrowed from [antirez/ds4](https://github.com/antirez/ds4): depth
over breadth. One model profiled properly beats a generic recipe applied to a
hundred.

Docs live in [`docs/`](docs/index.md) (Diátaxis layout, every page carries a
maturity status). Design decisions are recorded as [ADRs](docs/adr/index.md).

## Status

Early scaffold — CLI skeleton only. The scan/plan/pack pipeline is being built
in the open. See [Issues](https://github.com/Alberto-Codes/quantfit/issues) for
the roadmap.

## Requirements

- Python 3.12+
- CUDA GPU (developed against an RTX 4090 / 24 GB)
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone https://github.com/Alberto-Codes/quantfit.git
cd quantfit
uv sync
```

## Quick Start

```bash
# Show the CLI
uv run quantfit --help

# Planned interface (not yet implemented):
quantfit scan  nvidia/Nemotron-Super-49B --out sensitivity.json
quantfit plan  sensitivity.json --vram 24GiB --kv-headroom 4GiB --out recipe.json
quantfit pack  nvidia/Nemotron-Super-49B --recipe recipe.json --out ./quantfit-49b
```

## Development

```bash
uv sync --dev
uv run ruff check .     # Lint
uv run ty check         # Types
uv run pytest           # Tests
uv run docvet check --all  # Docstring quality
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

[MIT](LICENSE)
