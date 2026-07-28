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
  --max-tokens 32768 \
  --trust-remote-code \
  --out sensitivity.json
```

`--calibration` takes a plain UTF-8 text file. The north-star target
ships custom modeling code, hence `--trust-remote-code`. Lower
`--max-tokens` below the 131,072 default for large models — see the
memory math below. Pass `--gpu-memory` to cap GPU 0 model shards:
without a cap, `auto` sharding packs the card full and leaves no
workspace for activations.

## Resume

Every finished (group x precision) cell lands in
`sensitivity.checkpoint.json` immediately. Rerun the same command after
a crash and the scan continues at the first unmeasured cell. The
checkpoint carries the scan's fingerprint — change the model id,
calibration path, token count, grouping, precisions, or method and the
scan refuses the old checkpoint. Pass `--no-resume` to discard it.

The fingerprint records provenance, not content. It cannot detect new
weights or edited calibration text behind an unchanged path — do not
change either between a crash and its resume.

## Scanning a model that doesn't fit in VRAM — not yet

The v1 meter perturbs weights in place, which needs every quantizable
group on a real device. `auto` sharding offloads overflow modules and
exposes their weights as meta tensors — unperturbable, and a silent
perturbation no-op would record zero damage. The meter therefore
refuses to start when any group is offloaded, and names the groups.

Consequence: today a scan needs the model's quantizable groups to fit
on the card under `--gpu-memory`. The north-star 49B target does not
fit, so its first scan waits on offload-aware perturbation (through
accelerate's weights map) or group streaming — tracked in issue #16.
The reference distributions still cache on the CPU — roughly 0.25 GiB
per 1024 calibration tokens at a 128k vocabulary — so `--max-tokens`
budgets against system RAM either way.

## Reading the run log

Beside the map and checkpoint, the scan appends one JSON event per
line to `sensitivity.runlog.jsonl` — timings, damage, and memory
high-water marks. Any JSONL consumer works. One-liner analysis with
DuckDB:

```bash
duckdb -c "SELECT group_, bits, damage, seconds \
  FROM read_json_auto('sensitivity.runlog.jsonl') \
  WHERE event = 'cell_measured' ORDER BY seconds DESC LIMIT 10"
```

Splunk, Postgres `COPY`, and log collectors ingest the same file
unchanged.

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
