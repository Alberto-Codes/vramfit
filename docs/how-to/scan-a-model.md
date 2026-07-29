---
status: draft
---

# How to run a sensitivity scan

> **Status: draft** — `quantfit scan` is implemented and has produced
> full-size maps: Qwen2.5-3B (148 cells, ~1 h) and the 49B target
> (328 cells, 3 h 42 m, offload-aware per
> [ADR-0015](../adr/0015-offload-aware-scanning.md)).

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

## Scanning a model that doesn't fit in VRAM

Models larger than the card scan through accelerate's weights map
([ADR-0015](../adr/0015-offload-aware-scanning.md)). Under
`--gpu-memory`, `auto` sharding keeps overflow weights in host RAM,
and the meter perturbs those weights where they live. The forward
hooks stream the perturbed values to the GPU each pass, so damage
numbers for offloaded groups are exact. The run log's `meter_built`
event reports `offloaded_groups`.

Two limits remain. Weights offloaded beyond host RAM (disk spill)
are refused at construction — size `--gpu-memory` and system RAM so
the model fits both. And expect offloaded cells to run slower: every
forward pass streams the offloaded weights over PCIe. On the
north-star 49B at a 17 GiB cap, one 2048-token forward takes ~9 s,
so a 32,768-token cell costs ~145 s.

The reference distributions still cache on the CPU — roughly
0.25 GiB per 1024 calibration tokens at a 128k vocabulary — so
`--max-tokens` budgets against system RAM. At 49B scale the floor is
the offloaded weights (~76 GB) plus those distributions.

## Reading the run log

Beside the map and checkpoint, the scan appends one JSON event per
line to `sensitivity.runlog.jsonl` — timings, damage, and memory
high-water marks. Any JSONL consumer works. One-liner analysis with
DuckDB:

```bash
duckdb -c "SELECT \"group\", bits, damage, seconds \
  FROM read_json_auto('sensitivity.runlog.jsonl') \
  WHERE event = 'cell_measured' ORDER BY seconds DESC LIMIT 10"
```

Splunk, Postgres `COPY`, and log collectors ingest the same file
unchanged. Reruns and resumes append to the same file — filter on
``run_id`` to select one run. A run-log write failure warns once and
disables further events; the scan continues.

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
