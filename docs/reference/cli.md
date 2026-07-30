---
status: draft
---

# CLI reference

> **Status: draft** — `version`, `budget`, `plan`, `scan`, `pack`,
> and `validate` are implemented. `pack` covers the GGUF backend only
> (ADR-0010).

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
  --runtime TEXT         Target runtime the recipe is planned for
                         [default: llama.cpp]
  --format-overhead F    Overhead fraction on top of the size model
                         [default: 0.005 with an effective-bits
                         table, 0.05 without]
```

Size predictions follow
[ADR-0014](../adr/0014-per-type-effective-bits.md): a runtime with an
effective-bits table prices each precision at its real per-weight
cost (llama.cpp: Q4_K spends 4.5 bits, not 4), and the overhead
fraction covers only unquantized tensors and file metadata. A
runtime without a table (vLLM, or `--runtime` omitted via the API)
keeps the nominal-bits prediction and the 0.05 scalar.

Pin semantics: patterns are case-sensitive `fnmatch` globs matched against
the full group name (`--pin "model.layers.0.*=8"`). A pattern that matches
no group is an error (typo detection). Later pins override earlier ones for
overlapping groups — repeating a pattern moves it to the last position.
Pins are recorded in the recipe in their effective order.

Exit codes: 1 when the map is invalid, the output is unwritable, or no
recipe fits the budget (the gap is reported). Exit 2 on malformed options
(`--pin` not of the form `pattern=bits` with positive bits, unparseable
sizes, a negative, NaN, or infinite `--format-overhead`, or a
`--runtime` outside the capability table).

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

Groups that `auto` sharding offloads to host RAM measure through
accelerate's weights map (ADR-0015) — `meter_built` reports the count
as `offloaded_groups`. The scan refuses a model whose weights fall
beyond host RAM (disk spill) — an unperturbable weight would record
zero damage. Raise `--gpu-memory`, free host RAM, or use a smaller
model.

Exit codes: 1 when the scan extra is missing, the model or calibration
cannot load, sharding offloaded a quantizable group beyond host RAM,
the checkpoint
belongs to a different scan, a measurement fails (the checkpoint keeps
completed cells), a checkpoint write fails, or the map cannot be
written. Exit 2 on malformed `--precisions`, `--group-by`, or
`--gpu-memory`, a `--gpu-memory` without `--device auto`, or a missing
`--out` or `--runlog` directory.

## `quantfit validate`

Implemented. Runs the whole-recipe validation pass (ADR-0006). The
command quantizes every group to its assigned precision in one
calibration pass. The pass uses the scan's own quantization. The
command reports the measured damage next to the recipe's summed
marginal damages. The gap is the additivity assumption leaking.
Requires the scan extra — without it the command exits 1 with the
install hint.

```
quantfit validate RECIPE
  --calibration PATH     Calibration text file (UTF-8)  [required]
  --model TEXT           Model id or checkpoint path
                         [default: the recipe's model_id]
  --max-tokens INT       Calibration token budget  [default: 131072]
  --group-by TEXT        Grouping granularity (layer | tensor)  [default: layer]
  --device TEXT          Device map: auto | cpu | cuda  [default: auto]
  --trust-remote-code    Allow model repos with custom code
  --gpu-memory SIZE      Byte cap on GPU 0 model shards (e.g. 17GiB).
                         Requires --device auto  [default: none]
  --runlog PATH          Run-log path (JSONL)
                         [default: <recipe stem>.validation.runlog.jsonl]
```

With offloaded groups, the whole-recipe pass restores their originals
from the model's safetensors shards (ADR-0015), so `--model` must
point at a local safetensors directory — a bare hub id is refused
before any weight changes.

Use the scan's calibration file and token budget — damage values are
only comparable within one calibration set. The command refuses a
recipe whose groups do not match the model's discovered groups (wrong
model or wrong `--group-by`). A `--model` that differs from the
recipe's `model_id` prints a warning — the comparison assumes the
scanned model. The command reports the gap and does not gate on it:
the invalidation threshold is an open question in ADR-0006 until
measured gaps exist.

Every run appends events to a run log: validation_started,
meter_built, then validation_finished with predicted_damage,
measured_damage, gap, and ratio — or validation_halted (stage:
meter_build, group_match, or measure).

```console
$ quantfit validate recipe.json --calibration calib.txt --max-tokens 32768
validated 37 groups over 32768 tokens
summed marginal damage (predicted)  0.066107
whole-recipe damage (measured)      0.032240
gap -0.033867 (-51.2 % of predicted)
```

Exit codes: 1 when the recipe is invalid, the scan extra is missing,
the model or calibration cannot load, the recipe's groups do not match
the model's, or the measurement fails. Exit 2 on a malformed
`--group-by` or `--gpu-memory`, a `--gpu-memory` without `--device
auto`, or a missing `--runlog` directory.

## `quantfit pack`

Implemented for the GGUF backend (ADR-0010, ADR-0012). Applies a
recipe through llama.cpp's quantizer: one f16 base GGUF conversion
(reused when present), then `llama-quantize` with one type override
per layer group. The embedding assignment binds
`--token-embedding-type`. An `lm_head` group binds
`--output-tensor-type` with its own assignment — without one, the
embedding assignment pins an untied head (ADR-0012 as amended). The
base type is the recipe's precision floor, applied with `--pure`, so
no heuristic mixing leaks in.

```
quantfit pack RECIPE
  --llama-cpp PATH       llama.cpp checkout with convert_hf_to_gguf.py
                         and build/bin/llama-quantize  [required]
  --model PATH           Model checkpoint directory
                         [default: the recipe's model_id]
  --out PATH             Packed model path  [default: packed.gguf]
  --base-gguf PATH       f16 base GGUF, reused when present
                         [default: <model name>-f16.gguf beside --out]
  --python-bin PATH      Interpreter for the convert script — install
                         quantfit[pack] to provision it
                         [default: current]
  --threads INT          Thread count for the quantizer and the
                         smoke test  [default: 8]
  --imatrix PATH         Importance matrix for the quantizer
                         (ADR-0016)  [default: none]
  --smoke-text PATH      Text for the post-pack smoke test (ADR-0017)
                         [default: none — pack warns]
  --smoke-chunks INT     Smoke-test chunk count  [default: 2]
  --smoke-threshold F    Perplexity ceiling a passing smoke test
                         stays under  [default: 1000]
  --runlog PATH          Run-log path (JSONL)
                         [default: <stem>.runlog.jsonl]
```

After quantizing, the command re-checks the packed file's real bytes
against `plan.weight_budget_bytes` — nominal-bit predictions
undershoot GGUF's effective bits (ADR-0012). With `--smoke-text` it
then runs the smoke test: `--smoke-chunks` perplexity chunks through
`build/bin/llama-perplexity`, gated by the `--smoke-threshold`
ceiling (ADR-0017). Without the flag the command warns that the
packed model is unproven. Every run appends the pack events to the
run log: pack_started, gguf_converted (with `reused`), model_packed
(real bytes, base type, embedding and output tensor types, override
count, imatrix, uncovered tensors), size_checked (margin and
`fits`), smoke_tested when the smoke test ran (perplexity — null
when non-finite, with a text copy — threshold, chunks, `passed`),
then pack_finished (with `smoked`) or pack_halted (stage:
convert, quantize, size_check, or smoke).

Exit codes: 1 when the recipe is invalid, the model directory does
not exist, a toolchain stage fails, the packed model exceeds the
weight budget, or the smoke test fails (the file is kept). Exit 2
when the llama.cpp checkout misses a needed tool, `--imatrix` or
`--smoke-text` is not a file, `--smoke-threshold` is not positive,
or the `--out`/`--runlog` directory does not exist.
