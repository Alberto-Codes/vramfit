[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![docs vetted](https://img.shields.io/badge/docs%20vetted-docvet-purple)](https://github.com/Alberto-Codes/docvet)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

# quantfit

Selective per-layer quantization to fit large open models on a single GPU.

## What is this?

Uniform quantization treats every layer the same — and leaves quality (or VRAM)
on the table. Some layers of a transformer tolerate 2-bit crushing with barely a
ripple; others (attention projections, first/last blocks) fall apart below 8-bit.

`quantfit` measures which layers are which, then solves for the best mixed-precision
recipe that fits a *specific* model into a *specific* VRAM budget:

1. **Scan** — perturb each layer group at candidate precisions and measure output
   divergence (sensitivity) against the full-precision reference.
2. **Plan** — treat recipe selection as a budget problem: spend bits where the
   scan says they matter, crush where it says they don't, land under the target
   VRAM with room for KV cache.
3. **Pack** — emit the quantized checkpoint for the target runtime (vLLM first;
   GGUF export planned).

The north-star benchmark: **NVIDIA Nemotron Super 49B on a 24 GB RTX 4090** —
a model that does *not* fit at uniform 4-bit, made to fit selectively with
measured (not vibes-based) quality loss.

Philosophy borrowed from [antirez/ds4](https://github.com/antirez/ds4): depth
over breadth. One model profiled properly beats a generic recipe applied to a
hundred.

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
