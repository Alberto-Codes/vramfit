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
pip install "vramfit[scan]"
```

## Basic invocation

```bash
vramfit scan nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
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

Checkpoints written before the vramfit rename do not resume: the
checkpoint schema bumped with the envelope key (#118). The scan
rejects the old file — pass `--no-resume` to discard it and start
over.

## Pricing cells the way the pack quantizes

The default within-group method is round-to-nearest — fast, and
honest at 4 bits and above. Packed evidence settled the sub-4-bit
rule the hard way: kquant-priced and imatrix-assisted maps each
bought more 2-bit and packed worse
([ADR-0021](../adr/0021-runtime-frame-measurement.md) supersedes
[ADR-0019](../adr/0019-kquant-priced-maps.md) and
[ADR-0020](../adr/0020-imatrix-assisted-pricing.md)). Sub-4-bit
damage is measured in the runtime frame, and the solver buys a
width only against that frame's price. The 30B target's 2-bit price
arrived 2026-08-14 and failed, so current practice there plans on a
map copy without the 2-bit column. The kquant method itself
remains available
([ADR-0018](../adr/0018-kquant-within-group-method.md)):

```bash
vramfit scan ./model --calibration calibration.txt \
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
coverage split ("imatrix covers N of M parameters"). On a dense
llama-family model `token_embd` is the expected miss. A MoE model
misses far more. On Nemotron 3.5 Lightning 30B-A3B only 29 tensors
price assisted, because a k-quant fit needs rows that divide into
256-element super-blocks and most of this model's rows do not
([ADR-0020](../adr/0020-imatrix-assisted-pricing.md)). Read the
split before you trust the `kquant-imx` label.
Expect assisted cells to run longer than
unassisted kquant cells, and both to run longer than RTN — the
weighted fit searches more candidate scales.

## Scanning rows no K-quant reaches

`llama-quantize` applies no type whose block does not divide a
tensor's row length. `tensor_type_fallback` warns and substitutes a
compatible type instead — `Q2_K`, `Q3_K`, and `Q2_0` become `Q4_0`,
and `Q4_K` becomes `Q5_0`. `Q2_K`, `Q3_K`, and `Q4_K` block 256
elements, and Nemotron 3.5 Lightning 30B-A3B holds routed-expert
rows of 2688 and 1856. Neither divides. Those stacks carry 93.0 % of
the model's parameters, and
[ADR-0028](../adr/0028-expert-stack-type-table.md) packs them
at `Q8_0`, `Q4_0`, and `Q2_0` instead.

Scan them with the `q0` method, which ports those three block
quantizers
([ADR-0018](../adr/0018-kquant-within-group-method.md), 2026-08-17
amendment, token renamed by the 2026-08-18 amendment):

```bash
vramfit scan ./model --calibration calibration.txt \
  --group-by stack \
  --precisions 4,2 \
  --within-group q0 \
  --out sensitivity-q0.json
```

The method covers nominal 8, 4, and 2. It refuses nominal 3, which
[ADR-0028](../adr/0028-expert-stack-type-table.md) refuses at pack,
and 5 and 6 until ports exist.
`--imatrix` pairs with it (ADR-0018's 2026-08-21 amendment). The
map then records the `q0-imx` token. Nominal 4 fits with imatrix
weights through the ported `quantize_row_q4_0_impl`. Nominal 2 and
8 keep the reference arithmetic, because `quantize_q2_0` and
`quantize_q8_0` ignore the matrix. On a fused expert stack the
meter reads one imatrix row per expert, as `llama-quantize` applies
them. A parameter the imatrix does not cover prices unassisted, and
the coverage echo reports the split.

A `kquant` scan now refuses such a cell. The message names the
parameter, the type, the block size, and the row length. Nominal 8
never refuses there, because `Q8_0` blocks 32 elements and 32
divides both rows.

Maps priced under different methods do not compare. `scan.within_group`
is one token for the whole map, so a `q0-ref` map and a `kquant-imx`
map cannot merge.

## Pricing a subset of the groups

`--group-by stack` returns 210 groups on Nemotron 3.5 Lightning
30B-A3B. The 46 routed-expert stacks are the unit `vramfit pack`
addresses, so a campaign that needs those alone pays for 420 cells
without a filter. `--groups` names the subset:

```bash
vramfit scan ./model --calibration calibration.txt \
  --group-by stack \
  --groups backbone.layers.1.mixer.experts.up_proj,backbone.layers.1.mixer.experts.down_proj \
  --precisions 4,2 \
  --within-group q0 \
  --out sensitivity-stacks.json
```

The names must be keys `--group-by` produces, which are the checkpoint's
parameter names. They are not the GGUF tensor names a recipe packs
through — `group_key` collapses a routed-expert index and drops the
`.weight` suffix, so a stack reads
`backbone.layers.1.mixer.experts.up_proj` and never `blk.1.ffn_up_exps`.

A name that matches no discovered group halts the run, after the model
loads and before any cell measures. The halt names every unmatched name
at once, so one run reports every typo. Read the group names from a full
map's `groups[].name`, or from a `--group-by stack` run of a small model
in the same family.

!!! warning "Plan a narrowed map with `--checkpoint`"

    The map carries the selected groups alone. `vramfit plan` without
    `--checkpoint` refuses this map: it names the `mixer` module, and
    the scan skips that module's `conv1d` and `gate` classes (#204),
    so only a size source prices them
    ([ADR-0029](../adr/0029-plan-independent-size-source.md)
    decision 3, amended 2026-09-04). On a family with no such class
    the same command sums its budget over the groups the map holds
    and reads no other size source. A 46-of-210 map then prices 46
    groups and counts the other 164 as zero bytes, and the recipe
    reports a fit the packed model does not honor.

    `--checkpoint` prices the other 164 from the model's safetensors
    headers and holds each at reference precision (ADR-0029). Those
    groups are unmeasured, so the plan reserves reference bytes for
    them rather than spending the measurement it does not have. A
    skipped class prices at the 32 bits the converter wrote it at
    (#409).

The selection stays out of the fingerprint, because a group subset is
not provenance. So a narrow run and a wide run share one checkpoint on
purpose. Two consequences follow, and both save measurement time:

- A narrowed run reuses a wide run's cells for the groups it selects.
  It measures nothing that the checkpoint already holds.
- The cells in deselected groups stay in the checkpoint. A later, wider
  run reuses them instead of measuring them again. The narrowed run
  reports how many cells it ignored.

A selection narrows what a run measures and never what it checks. The
checkpoint validates against the whole model before any narrowing. A
cell outside the full grid, or a cell that repeats, halts the run
whatever the selection names.

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

### Channel-locked checkpoints need a model-turn frame

Some instruct checkpoints refuse raw prose. Gemma 4 31B IT-QAT
prices raw text at PPL ≈ 3,000 and the same text inside its own
answer channel at PPL 26–75 (#423). Scanning such a target on raw
text prices every cell against a distribution no user ever sees.

Wrap the calibration text in a
[model-turn frame](../reference/glossary.md) first:

```console
uv run python scripts/frame_calibration.py \
  --model ./model --text calibration.txt --out calibration-framed.txt
```

The script wraps ~512-token blocks in the checkpoint's own chat
template. It refuses a vocabulary where a frame marker is not one
special id, and refuses prose that itself encodes to special ids.
Then pass the framed file as `--calibration`. Four rules keep the
numbers comparable:

1. The frame holds constant across the reference, every arm, and
   any baseline. Record the frame text beside the map.
2. Instruments slice a raw token stream, so windows cross block
   boundaries. State that convention beside every published number.
3. `llama-imatrix` needs `--parse-special` to see the frame.
   `llama-perplexity` cannot parse special tokens — do not publish
   its numbers on framed text.
4. Verify the frame against the GGUF vocabulary too, with
   `llama-tokenize`. A conversion can mis-type a marker as a normal
   token, and `--parse-special` then spells it out as prose. The
   Gemma 4 bf16 conversion shipped `<|turn>` as a normal token
   (#423). The token counts must match the tokenizer-side count
   plus one prepended BOS.

## Cost expectations

A scan is `O(groups x precisions)` calibration passes over perturbed
models. For the 49B target at 80 groups x 4 precisions, expect hours on
the reference box. That is why the checkpoint writes after every cell —
losing hour six to a crash in hour seven is not acceptable.
