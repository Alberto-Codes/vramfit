---
status: draft
---

# How to run a sensitivity scan

> **Status: draft** — `quantfit scan` is implemented and verified on a
> tiny model (CPU). No full-size scan has run yet, so the cost numbers
> below are estimates.

## Goal

Produce a [sensitivity map](../reference/sensitivity-map.md) for a model —
per layer group, per candidate precision, how much the model's output
degrades.

## Prerequisites

The scan needs the GPU stack, which the base install does not carry
(ADR-0005):

```bash
uv pip install "quantfit[scan]"
```

## Basic invocation

```bash
uv run quantfit scan nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --calibration calibration.txt \
  --precisions 8,4,3,2 \
  --trust-remote-code \
  --out sensitivity.json
```

`--calibration` takes a plain UTF-8 text file. The north-star target
ships custom modeling code, hence `--trust-remote-code`.

## Resume

Every finished (group x precision) cell lands in
`sensitivity.checkpoint.json` immediately. Rerun the same command after
a crash and the scan continues at the first unmeasured cell. The
checkpoint carries the scan's fingerprint — change the model,
calibration, token count, grouping, or precisions and the scan refuses
the old checkpoint. Pass `--no-resume` to discard it.

## Scanning a model that doesn't fit in VRAM

The interesting targets are exactly the models that don't fit. The v1
meter loads the model with `--device auto`, which shards across GPU and
system RAM (the reference box holds the 49B reference in its 124 GB).
Reference distributions are computed once and cached on the CPU in
float16 — roughly 0.25 GiB per 1024 calibration tokens at a 128k
vocabulary, so budget `--max-tokens` against system RAM. Streaming
groups to the GPU one at a time is the planned optimization, not yet
built.

## Choosing calibration data

Sensitivity is measured *on some text*; the choice matters and is an open
question. Starting point: a small generic slice (wikitext) plus a slice
that matches your actual workload. Re-scan when your workload changes
character — the map records its calibration provenance, and damage
values are not comparable across calibration sets.

## Cost expectations

A scan is `O(groups x precisions)` calibration passes over perturbed
models. For the 49B target at 80 groups x 4 precisions, expect hours on
the reference box. That is why the checkpoint writes after every cell —
losing hour six to a crash in hour seven is not acceptable.
