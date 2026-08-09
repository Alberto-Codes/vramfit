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
  --protect TEXT         Hold tensors at a precision floor inside
                         their groups, repeatable (glob=bits)
  --exclude-imatrix TEXT Quantize matched protected tensors without
                         their imatrix rows, repeatable (glob)
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

Protection semantics
([ADR-0022](../adr/0022-within-layer-protections.md)): patterns are
the same `fnmatch` language, matched against full *tensor* names
(`--protect "*.self_attn.v_proj.weight=5"`). A protected tensor
packs at the higher of its group's assignment and the floor, priced
by size only — predicted damage stays the group-level sum. The
command rejects a floor the target runtime cannot serve, a pattern
that matches no tensor, a protection on a single-tensor group (use
`--pin` — the embedding and output head hold one tensor each), and a
map without per-tensor sizes. A rule whose floor every matched
group's assignment already meets draws a no-op warning. The recipe
records the rules verbatim and the resolved (tensor, precision)
pairs. A pair exists only where the floor exceeds the group's
assignment (issue #59). A matched tensor whose assignment already
meets the floor draws a per-tensor warning, and the recipe drops
that pair — it would quantize identically to the unprotected
reference and falsely fail the reconstruction check. A dropped
pair's `--exclude-imatrix` mark drops with it, and the warning says
so. An `--exclude-imatrix` pattern left with no surviving pair is
rejected — the pack would keep the imatrix rows the pattern names.
A glob that also matches unprotected tensors draws a warning
naming what it did not cover. The unprotected rows stay (ADR-0023).

Exit codes: 1 when the map is invalid, the output is unwritable, no
recipe fits the budget (the gap is reported), or a `--protect` or
`--exclude-imatrix` rule is rejected. Exit 2 on malformed options (`--pin` or `--protect` not of
the form `pattern=bits` with positive bits, unparseable sizes, a
negative, NaN, or infinite `--format-overhead`, or a `--runtime`
outside the capability table).

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
  --within-group TEXT    Within-group method: rtn | kquant (ADR-0018).
                         kquant prices cells with the ported K-quant
                         reference quantizers and pairs only with
                         precisions the port covers (8, 4, 3, 2)
                         [default: rtn]
  --imatrix PATH         GGUF imatrix for assisted K-quant pricing
                         (ADR-0020). Requires --within-group kquant.
                         Use the file the pack step will consume
                         [default: none]
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
calibration, token count, grouping, precisions, method, imatrix
path) — a rerun with
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

With `--imatrix` the kquant fit weighs every covered tensor with the
pack's importance matrix (assisted pricing, ADR-0020). The map records
the token `kquant-imx` and the imatrix path in `scan.imatrix`. The
command echoes the coverage split, and `meter_built` records it as
`imatrix_covered` and `imatrix_uncovered` — uncovered tensors price
unassisted, the `llama-quantize` fallback (`token_embd` is the
expected miss). A covered tensor whose rows do not divide into
256-element super-blocks joins the uncovered set instead of refusing
the scan. A scan is only comparable to a pack that consumes the
same imatrix file — the CLI resolves the path, and the map records
the resolved spelling.

Exit codes: 1 when the scan extra is missing, the model or calibration
cannot load, sharding offloaded a quantizable group beyond host RAM,
the checkpoint
belongs to a different scan, a measurement fails (the checkpoint keeps
completed cells), a checkpoint write fails, or the map cannot be
written. Exit 2 on malformed `--precisions`, `--group-by`,
`--within-group`, or `--gpu-memory`, a `--gpu-memory` without
`--device auto`, a `--within-group kquant` combined with precisions
the port does not cover, an `--imatrix` without `--within-group
kquant` or naming a missing file, or a missing `--out` or `--runlog`
directory.

## `quantfit validate`

Implemented. Runs the whole-recipe validation pass (ADR-0006). The
command quantizes every group to its assigned precision in one
calibration pass. The pass uses the scan's own quantization,
selected with `--within-group` and `--imatrix`. The pass only checks
additivity when its frame matches the map that priced the recipe
(ADR-0019) — a recipe that records its map's method resolves the
frame by itself, and the command refuses flags that contradict the
record. Recipes without the record leave the pairing to the caller,
with a warning. The
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
  --within-group TEXT    Within-group method: rtn | kquant (ADR-0018)
                         [default: the recipe's recorded method, or
                         rtn without a record]
  --imatrix PATH         GGUF imatrix for assisted K-quant measurement
                         (ADR-0020). Required when the recipe was
                         priced on an assisted map — use the map's
                         imatrix file  [default: none]
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
scanned model. An `--imatrix` that differs from the recipe's recorded
imatrix path also prints a warning — a different file contaminates
the additivity comparison. The command echoes the imatrix coverage
split like the scan does. The command reports the gap and does not
gate on it: the invalidation threshold is an open question in
ADR-0006 until measured gaps exist.

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
`--group-by`, `--within-group`, or `--gpu-memory`, a `--gpu-memory`
without `--device auto`, an `--imatrix` without the kquant method or
naming a missing file, a frame that contradicts the recipe's
recorded method, a `--within-group kquant` that meets recipe
assignments the ported quantizers do not cover, or a missing
`--runlog` directory.

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

A recipe priced on an assisted map records its imatrix — the command
warns when `--imatrix` is absent or names a different file, because
the pack would not match the map's frame (ADR-0020). A warning, not
a refusal: packing itself works either way.

A protected recipe drives its resolved (tensor, precision) pairs as
extra overrides, placed *before* the group overrides — the quantizer
applies the first matching pattern
([ADR-0022](../adr/0022-within-layer-protections.md)).

After quantizing, the command re-checks the packed file's real bytes
against `plan.weight_budget_bytes` — nominal-bit predictions
undershoot GGUF's effective bits (ADR-0012). On a protected pack made
with `--imatrix`, the reconstruction check then runs, mandatory
([ADR-0022](../adr/0022-within-layer-protections.md)): the command
packs the same recipe with its protections stripped as the
reference, dequantizes every protected tensor from both files with
gguf-py, and compares each against the f16 base. The reference file
is deleted after measurement. A tensor that does not reconstruct
strictly closer to f16 than its unprotected type is collapsed — the
command names it, keeps the file, and exits 1. The remedy is the
user's: re-plan with `--exclude-imatrix` for the named tensors
(ADR-0023) — the refusal prints the exact flags. A
protected pack without `--imatrix` skips the check with a note —
every known fit collapse involved a promotion under one. A recipe
that records protections but resolved no pairs also skips with a
note: every floor was a per-tensor no-op at plan time (issue #59).

A recipe with imatrix exclusions drives one `--exclude-weights`
flag per marked pair when the pack runs with `--imatrix`
([ADR-0023](../adr/0023-imatrix-exclusions.md)). The excluded
tensors keep their promoted types and quantize without their
imatrix rows. Packing such a recipe without `--imatrix` warns that
the exclusions change nothing.

With `--smoke-text` the command runs the smoke test: `--smoke-chunks` perplexity chunks through
`build/bin/llama-perplexity`, gated by the `--smoke-threshold`
ceiling (ADR-0017). Without the flag the command warns that the
packed model is unproven. Every run appends the pack events to the
run log: pack_started, gguf_converted (with `reused`), model_packed
(real bytes, base type, embedding and output tensor types, override
count, imatrix, uncovered tensors, excluded tensors), size_checked (margin and
`fits`), reconstruction_checked when the gate ran (per-tensor
protected and reference RMSE, `collapsed`, `passed`), smoke_tested
when the smoke test ran (perplexity — null
when non-finite, with a text copy — threshold, chunks, `passed`),
then pack_finished (with `smoked`) or pack_halted (stage:
convert, quantize, size_check, reconstruction, or smoke).

Exit codes: 1 when the recipe is invalid, the model directory does
not exist, a toolchain stage fails, the packed model exceeds the
weight budget, the reconstruction check finds a collapsed tensor
(the file is kept), or the smoke test fails (the file is kept).
Exit 2 when the llama.cpp checkout misses a needed tool,
`--imatrix` or `--smoke-text` is not a file, `--smoke-threshold` is
not positive, or the `--out`/`--runlog` directory does not exist.
