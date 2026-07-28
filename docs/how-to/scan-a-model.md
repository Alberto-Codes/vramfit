---
status: sketch
---

# How to run a sensitivity scan

> **Status: sketch** — `quantfit scan` is not implemented; this records the
> intended workflow and the operational questions we already know matter.

## Goal

Produce a [sensitivity map](../reference/sensitivity-map.md) for a model —
per layer group, per candidate precision, how much the model's output degrades.

## Basic invocation

```bash
uv run quantfit scan nvidia/Nemotron-Super-49B \
  --calibration wikitext \
  --precisions 8,4,3,2 \
  --out sensitivity.json
```

## Scanning a model that doesn't fit in VRAM

The interesting targets are exactly the models that don't fit — which means
the scan itself can't naively hold the full-precision reference on the GPU.
Planned approach (open question, see
[ADR-0006](../adr/0006-sensitivity-metric.md)):

- Stream layer groups to the GPU one at a time; keep the rest in system RAM
  (the reference box has 124 GB — enough for the 49B reference weights).
- Cache reference activations for the calibration set once, then replay them
  per candidate quantization instead of re-running the full model.

## Choosing calibration data

Sensitivity is measured *on some text*; the choice matters and is an open
question. Starting point: a small generic slice (wikitext) plus a slice that
matches your actual workload. Re-scan when your workload changes character.

## Cost expectations

A scan is `O(groups × precisions × calibration tokens)` forward passes over
perturbed models. For a 49B model expect hours, not minutes, on a single
4090. Scans should be resumable (`--resume`) — losing hour six to a crash in
hour seven is not acceptable.
