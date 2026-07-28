---
status: draft
---

# CLI reference

> **Status: draft** — `version`, `budget`, `plan`, and `scan` are
> implemented. `pack` is design-stage.

## `quantfit version`

Implemented. Prints the installed package version.

```console
$ quantfit version
quantfit 0.1.0
```

## `quantfit budget`

Implemented. Prints the VRAM budget breakdown. The `--kv-headroom` value
for `plan` is the sum of the KV-cache and runtime-overhead lines. The attention shape comes from
exactly one source: `--model-config` (a Hugging Face `config.json` —
DeciLM NAS configs with skipped-attention blocks are handled) or the
manual triple.

```
quantfit budget
  --vram SIZE            Total VRAM  [default: 24GiB]
  --context INT          Context length in tokens  [default: 16384]
  --kv-dtype TEXT        fp16 | bf16 | fp8  [default: fp16]
  --sequences INT        Concurrent sequences  [default: 1]
  --overhead SIZE        Runtime overhead reservation  [default: 2GiB]
  --model-config PATH    Model config.json to derive the shape from
  --attn-layers INT      Attention layer count (manual shape)
  --kv-heads INT         KV heads per layer (manual shape)
  --head-dim INT         Head dimension (manual shape)
```

Exits 1 when nothing is left for weights, and 2 on conflicting or missing
shape sources.

```console
$ quantfit budget --model-config config.json --vram 24GiB --kv-dtype fp8
attention layers      49  (KV 100352 bytes/token, fp8)
VRAM total            24.00 GiB
- KV cache            1.53 GiB  (16384 tokens x 1 seq)
- runtime overhead    2.00 GiB
= weight budget       20.47 GiB
```

## `quantfit plan`

Implemented. Solves a sensitivity map into a recipe under a VRAM budget.

```
quantfit plan SENSITIVITY_MAP
  --vram SIZE            Hard VRAM ceiling (e.g. 24GiB)  [required]
  --kv-headroom SIZE     Reserved for KV cache + runtime  [default: 4GiB]
  --pin TEXT             Pin groups to a precision, repeatable (glob=bits)
  --out PATH             Output recipe  [default: recipe.json]
  --format-overhead F    Quantization-format overhead fraction  [default: 0.05]
```

Pin semantics: patterns are case-sensitive `fnmatch` globs matched against
the full group name (`--pin "model.layers.0.*=8"`). A pattern that matches
no group is an error (typo detection). Later pins override earlier ones for
overlapping groups — repeating a pattern moves it to the last position.
Pins are recorded in the recipe in their effective order.

Exit codes: 1 when the map is invalid, the output is unwritable, or no
recipe fits the budget (the gap is reported). Exit 2 on malformed options
(`--pin` not of the form `pattern=bits` with positive bits, unparseable
sizes, negative `--format-overhead`).

## `quantfit scan`

Implemented. Measures per-group damage and writes a sensitivity map.
Requires the scan extra (`uv pip install "quantfit[scan]"`) — without
it the command exits 1 with the install hint.

```
quantfit scan MODEL
  --calibration PATH     Calibration text file (UTF-8)  [required]
  --out PATH             Output sensitivity map  [default: sensitivity.json]
  --precisions TEXT      Candidate bit-widths, strictly descending CSV,
                         2-bit floor, default per ADR-0010
                         [default: 8,4,3,2]
  --group-by TEXT        Grouping granularity (layer | tensor)  [default: layer]
  --max-tokens INT       Calibration token budget  [default: 131072]
  --device TEXT          Device map: auto | cpu | cuda  [default: auto]
  --trust-remote-code    Allow model repos with custom code (the
                         north-star target needs this)
  --resume / --no-resume Continue from the checkpoint file  [default: resume]
  --runlog PATH          Run-log path (JSONL)
                         [default: <stem>.runlog.jsonl]
  --gpu-memory SIZE      Byte cap on GPU 0 model shards (e.g. 17GiB),
                         parsed with the same grammar as --vram.
                         Requires --device auto. Keeps workspace free
                         for activations and quantization
                         [default: none]
```

Every run appends machine-readable events to a run log
(`<stem>.runlog.jsonl`, ADR-0011): scan_started, meter_built,
resume_loaded, one cell_measured per cell with damage, seconds, and
the RSS high-water mark, then scan_finished or scan_halted. Every
finished (group x precision) cell lands in a checkpoint file next
to `--out` (`<stem>.checkpoint.json`). A rerun of the same scan resumes
from it. The checkpoint carries the scan's fingerprint (model, metric,
calibration, token count, grouping, precisions, method) — a rerun with
any of those changed refuses the checkpoint instead of mixing numbers.
The fingerprint identifies provenance, not content: do not swap weights
or calibration text under an unchanged path between resumes.
`--no-resume` deletes the checkpoint first and says so.

The scan refuses a model whose quantizable groups get offloaded off
the card — offloaded weights cannot be perturbed, and measuring them
would record zero damage. Raise `--gpu-memory` or use a smaller model.

Exit codes: 1 when the scan extra is missing, the model or calibration
cannot load, sharding offloaded a quantizable group, the checkpoint
belongs to a different scan, a measurement fails (the checkpoint keeps
completed cells), a checkpoint write fails, or the map cannot be
written. Exit 2 on malformed `--precisions`, `--group-by`, or
`--gpu-memory`, a `--gpu-memory` without `--device auto`, or a missing
`--out` directory.

## `quantfit pack` *(planned)*

```
quantfit pack MODEL_ID
  --recipe PATH          Recipe produced by `quantfit plan`
  --runtime TEXT         Target runtime (GGUF first per ADR-0010)
  --out PATH             Output checkpoint directory
```
