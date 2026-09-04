[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![docs vetted](https://img.shields.io/badge/docs%20vetted-docvet-purple)](https://github.com/Alberto-Codes/docvet)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Alberto-Codes/vramfit/blob/main/LICENSE)

# vramfit

Selective per-layer quantization to fit large open models on a single GPU.

## What is this?

**The problem.** A model's weights are billions of numbers, normally stored at
16 bits each — Nemotron Super 49B is ~98 GB at full precision, and an RTX 4090
has 24 GiB. Quantization stores those numbers with fewer bits (8, 4, even 2),
trading a little accuracy for a lot of memory. But even uniform 4-bit puts 49B
parameters at ~26 GB — still doesn't fit — and uniform 3-bit wrecks quality,
because some parts of a transformer get badly stupid when you crush them.

**The insight.** Not all layers are equally fragile. Some tolerate 2–3 bits
with barely a ripple; others (attention projections, first/last blocks) fall
apart below 6–8 bits. Most published quantized models pick precision by crude
heuristic. `vramfit` *measures* which layers are which, then solves for the
best mixed-precision recipe that fits a *specific* model into a *specific*
VRAM budget:

1. **Scan** — quantize one layer group at a time at candidate precisions and
   measure output divergence against the full-precision reference. Output: a
   sensitivity map of which layers can survive being crushed.
2. **Plan** — a budget problem: given the map and a hard VRAM constraint
   (minus KV-cache headroom), spend bits where the scan says they matter and
   crush where it says they don't. Bits are cost, quality is value.
3. **Validate** — replay the whole recipe in one pass and measure real
   recipe damage against the solver's prediction
   ([ADR-0006](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0006-sensitivity-metric.md)).
4. **Pack** — apply the recipe and emit a checkpoint the target runtime can
   actually serve. GGUF covers the sub-4-bit benchmark path, per
   [ADR-0010](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0010-sub-4-bit-serving-path.md). A vLLM backend for
   ≥4-bit recipes is planned.

**The goal:** NVIDIA **Nemotron Super 49B running on a 24 GiB RTX 4090** — a
model that does not fit at full precision, made to fit selectively, with
measured (not vibes-based) damage versus running a smaller model instead.
[The result](#the-result) records the measured outcome.

Philosophy borrowed from [antirez/ds4](https://github.com/antirez/ds4): depth
over breadth. One model profiled properly beats a generic recipe applied to a
hundred.

Docs live in [`docs/`](https://github.com/Alberto-Codes/vramfit/blob/main/docs/index.md) (Diátaxis layout, every page carries a
maturity status). Design decisions are recorded as [ADRs](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/index.md).

## The result

vramfit's acceptance test is Nemotron Super 49B on a 24 GiB RTX 4090
([ADR-0003](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0003-north-star-benchmark.md)).
The published pack passed its three evaluation tiers on 2026-08-10.

| | |
|---|---|
| Model | [nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) |
| Budget | Solver target: 24 GiB card, 16k context at fp8 KV. Weight budget 20.47 GiB |
| Pack | 20.36 GiB GGUF, 112 MiB under the weight budget |
| Comparator | bartowski Q3_K_S, 20.45 GiB, size-matched |
| KL divergence, 564 chunks | 0.2873 against 0.2959. Better on 369 of 564 chunks, 7.8σ paired |
| Perplexity | 8.517 ± 0.063 against 8.532 ± 0.064. A tie by the interval |
| Top-token agreement | 82.9 % against 83.4 %. The comparator leads |
| Task slice, five tasks | Five ties, largest delta 0.8σ ([ADR-0024](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0024-tier3-task-slice.md)) |
| Artifacts | [Packed model](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF), [sensitivity maps](https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps) |
| Evidence | [Evidence ledger](https://github.com/Alberto-Codes/vramfit/blob/main/docs/explanation/evaluating-packed-models.md), data points fifteen and sixteen. [Card ledger](https://github.com/Alberto-Codes/vramfit/blob/main/publication/model-card/card-ledger.md) |

Every measured number above comes from one instrument: llama.cpp
b10172 on the RTX 4090. Perplexity and KL divergence ran on held-out
WikiText-2, with KL against the f16 reference. Damage and KL values
from other models, maps, calibration sets, or instruments do not
compare with these
([ADR-0027](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0027-instrument-frame-matching.md)).

Two later publications carry their own cards, comparators, and ledgers.
They are a [30B MoE pack for 16 GiB](https://huggingface.co/Alberto-Codes/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib-GGUF)
and an
[image-capable Gemma 4 31B pack for 24 GiB](https://huggingface.co/Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF).

## Status

The full pipeline is implemented: `scan`, `plan`, `validate`, `pack`, plus
`budget` for the VRAM arithmetic. Pack quantizes with an importance matrix
(ADR-0016), guards protected packs with a per-tensor reconstruction
check (ADR-0022), and smoke-tests every artifact before trusting it
(ADR-0017).

The road to the result above ran through measured eliminations.
Importance-weighted rounding was worth 0.86 of the original
1.39-perplexity gap. 2-bit group membership decides whether damages
add: super-additive by 11.9× on one 2-bit set, sub-additive by 1.6×
on another. The solver buys nominal 2 on expert-stack groups where a
runtime-frame campaign priced the width. Dense groups keep nominal 2
out until a price shows it beats the alternatives at or below the
budget (ADR-0021, amended 2026-08-22). Within-layer protections plus
imatrix exclusions (ADR-0022, ADR-0023) closed the fit-collapse gap.

The [evidence page](https://github.com/Alberto-Codes/vramfit/blob/main/docs/explanation/evaluating-packed-models.md) records
all nineteen data points, through 2026-08-22. See
[Issues](https://github.com/Alberto-Codes/vramfit/issues) for the roadmap.

## Requirements

- Python 3.12+
- CUDA GPU for `scan` and `validate` (developed against an RTX 4090 /
  24 GiB). `plan`, `budget`, and `capacity` need no GPU. `pack` needs
  the pack extra and a llama.cpp checkout (ADR-0012).
- [uv](https://docs.astral.sh/uv/) for the development path

## Installation

Install vramfit from [PyPI](https://pypi.org/project/vramfit/). The
base install declares two dependencies, typer and structlog, and no
torch. `plan`, `budget`, and `capacity` run without a GPU
([ADR-0005](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0005-heavy-deps-as-extras.md)).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install vramfit
vramfit version
```

That is the whole install for the
[first-run tutorial](https://github.com/Alberto-Codes/vramfit/blob/main/docs/tutorials/first-run.md). Two extras add the
remaining commands:

```bash
pip install "vramfit[scan]"  # adds the torch stack for scan and validate
pip install "vramfit[pack]"  # the scan stack plus the GGUF converter deps
```

`vramfit validate` builds the same torch-backed meter as `vramfit scan`,
so both need the scan extra and a CUDA GPU. `vramfit pack` needs the
pack extra and a llama.cpp checkout with built tools, which you pass
with `--llama-cpp` ([ADR-0012](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0012-gguf-type-mapping.md)).
The pack extra does not install llama.cpp. The
[pack how-to](https://github.com/Alberto-Codes/vramfit/blob/main/docs/how-to/pack-a-recipe.md) shows the build.

To develop vramfit itself, clone the repository instead. See
[Development](#development).

## Quick Start

Run this in an empty directory, not in a clone. Skip the first two
lines if you installed above. It solves the published 49B sensitivity
map under 24 GiB. It then reads how much context the recipe leaves.
The run needs no GPU, no torch, and no model weights. Two downloads
total 108 KB, and both commands finish in under one second.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install vramfit

curl -LO https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps/resolve/main/sensitivity-64k-kquant-imx-no2-sized.json
curl -LO https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/resolve/main/config.json

vramfit plan sensitivity-64k-kquant-imx-no2-sized.json --vram 24GiB --kv-headroom 3616MiB --out recipe.json
vramfit capacity recipe.json --model-config config.json --kv-dtype fp8 --context 16384
```

The [first-run tutorial](https://github.com/Alberto-Codes/vramfit/blob/main/docs/tutorials/first-run.md) shows the
expected output of both commands and explains each line. The
[getting-started tutorial](https://github.com/Alberto-Codes/vramfit/blob/main/docs/tutorials/getting-started.md) covers
the scan, validate, and pack steps. Scan and validate need the scan
extra and a GPU. Pack needs the pack extra and a llama.cpp checkout.

## Development

```bash
git clone https://github.com/Alberto-Codes/vramfit.git
cd vramfit
uv sync --dev
uv run ruff check .     # Lint
uv run ty check         # Types
uv run pytest           # Tests
uv run docvet check --all  # Docstring quality
```

See [CONTRIBUTING.md](https://github.com/Alberto-Codes/vramfit/blob/main/CONTRIBUTING.md) for the full workflow.

## License

[MIT](https://github.com/Alberto-Codes/vramfit/blob/main/LICENSE)
