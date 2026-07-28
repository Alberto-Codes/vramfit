---
status: draft
---

# CLI reference

> **Status: draft** — `version`, `budget`, `plan`, `scan`, and `pack`
> are implemented. `pack` covers the GGUF backend only (ADR-0010).

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

## `quantfit pack`

Implemented for the GGUF backend (ADR-0010, ADR-0012). Applies a
recipe through llama.cpp's quantizer: one f16 base GGUF conversion
(reused when present), then `llama-quantize` with one type override
per layer group and the embedding bound via
`--token-embedding-type`. The base type is the recipe's precision
floor, applied with `--pure`, so no heuristic mixing leaks in.

```
quantfit pack RECIPE
  --llama-cpp PATH       llama.cpp checkout with convert_hf_to_gguf.py
                         and build/bin/llama-quantize  [required]
  --model PATH           Model checkpoint directory
                         [default: the recipe's model_id]
  --out PATH             Packed model path  [default: packed.gguf]
  --base-gguf PATH       f16 base GGUF, reused when present
                         [default: <model name>-f16.gguf beside --out]
  --python-bin PATH      Interpreter for the convert script — needs
                         torch and sentencepiece  [default: current]
  --threads INT          Quantizer thread count  [default: 8]
  --runlog PATH          Run-log path (JSONL)
                         [default: <stem>.runlog.jsonl]
```

After quantizing, the command re-checks the packed file's real bytes
against `plan.weight_budget_bytes` — nominal-bit predictions
undershoot GGUF's effective bits (ADR-0012). Every run appends the
pack events to the run log: pack_started, gguf_converted (with
`reused`), model_packed (real bytes, base type, override count),
size_checked (margin and `fits`), then pack_finished or pack_halted
(stage: convert, quantize, or size_check).

Exit codes: 1 when the recipe is invalid, the model directory does
not exist, a toolchain stage fails, or the packed model exceeds the
weight budget (the file is kept). Exit 2 when the llama.cpp checkout
misses its tools or the `--out`/`--runlog` directory does not exist.
