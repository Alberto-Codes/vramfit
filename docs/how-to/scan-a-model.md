---
status: draft
---

# How to run a sensitivity scan

> **Status: draft** — `vramfit scan` is implemented and has produced
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
uv pip install "vramfit[scan]"
```

## Basic invocation

```bash
uv run vramfit scan nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
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
calibration path, token count, grouping, precisions, method, or
imatrix path, and the scan refuses the old checkpoint. Pass
`--no-resume` to discard it.

The fingerprint records provenance, not content. It cannot detect new
weights or edited calibration text behind an unchanged path — do not
change either between a crash and its resume.

## Pricing cells the way the pack quantizes

The default within-group method is round-to-nearest — fast, and
honest at 4 bits and above. Packed evidence settled the sub-4-bit
rule the hard way: kquant-priced and imatrix-assisted maps each
bought more 2-bit and packed worse
([ADR-0021](../adr/0021-runtime-frame-measurement.md) supersedes
[ADR-0019](../adr/0019-kquant-priced-maps.md) and
[ADR-0020](../adr/0020-imatrix-assisted-pricing.md)). Sub-4-bit
damage is measured in the runtime frame, and the solver does not
buy 2-bit without a runtime-frame price — current practice plans
on a map copy without the 2-bit column. The kquant method itself
remains available
([ADR-0018](../adr/0018-kquant-within-group-method.md)):

```bash
uv run vramfit scan ./model --calibration calibration.txt \
  --within-group kquant \
  --imatrix model.imatrix.gguf \
  --out sensitivity.json
```

`--imatrix` adds the pack's importance matrix to the fit (assisted
pricing, [ADR-0020](../adr/0020-imatrix-assisted-pricing.md),
Superseded — the mechanics remain, the sub-4-bit licensing claim
does not).
Use the same file the pack step will consume — the map records the
imatrix path, and the recipe carries it forward so the validation
pass and the pack can hold the frame. The command echoes the
coverage split ("imatrix covers N of M parameters"). `token_embd`
is the expected miss. Expect assisted cells to run longer than
unassisted kquant cells, and both to run longer than RTN — the
weighted fit searches more candidate scales.

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

Run capped scans with the default CUDA allocator. On the reference
stack (torch 2.13.0+cu130),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` silently
corrupted tensor memory under near-OOM pressure and surfaced as a
NaN damage at the first cell. Upstream tracks several
`expandable_segments` defects. Do not set it — if the cap OOMs,
lower `--gpu-memory` instead.

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
