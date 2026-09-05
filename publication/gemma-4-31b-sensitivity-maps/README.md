---
license: cc-by-4.0
pretty_name: gemma-4-31B-it sensitivity maps
tags:
  - vramfit
viewer: false
---

<!--
Authored for issue #449 under the #401 identity grammar. This file
is the card of the dataset repo
Alberto-Codes/gemma-4-31B-it-sensitivity-maps. Upload this file
verbatim — the published card and this source must match.
-->

# gemma-4-31B-it-sensitivity-maps

This dataset carries the per-layer quantization sensitivity map and
the in-frame importance matrix of
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it),
measured on the
[QAT unquantized checkpoint](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
at revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`.
[vramfit](https://github.com/Alberto-Codes/vramfit) measured them.
A sensitivity map records one damage number per layer group and
candidate precision. Damage is the shift in the model's output
distribution when that group alone quantizes — mean final-logits KL
divergence against the bf16 reference. The map describes the base
model, not any quantized file. It contains no model weights.

The packed model solved from this map ships as
[gemma-4-31B-it-fit24gib-GGUF](https://huggingface.co/Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF).
Its card states that the map and the matrix stay in the run
archive. Issue #449 ruled on 2026-09-04 that both publish here.

## The map format

The map is one JSON file, `vramfit_schema` 3, in the form that
carries `group_by: layer`. The `scan` block records the
measurement frame: metric, calibration file, token count,
candidate precisions, grouping, within-group method, and imatrix
path. The `groups` list records 61 decoder groups: the token
embedding and the 60 language-model layers. Each group carries
its member tensors, its bytes at reference precision, its damage
per precision, and `tensor_bytes`, the per-tensor size split. The
[sensitivity map format](https://github.com/Alberto-Codes/vramfit/blob/main/docs/reference/sensitivity-map.md)
page specifies every field. The path fields record the reference
box's absolute paths. The `scan.calibration` and `scan.imatrix`
basenames match the files here.

The map holds the language model only. The vision tower and the
projector never enter it: the published pack ships the vendor's
projector as a separate sidecar, converted to Q4_K_M without a
measurement.

## The scan

| File | Calibration tokens | Within-group method | Imatrix | Started (UTC) | Cells |
|---|---|---|---|---|---|
| `sensitivity-32k-kquant-imx.json` | 32,768 | `kquant-imx` | yes | 2026-08-27 | 244 |

The scan covers the 61 groups at candidate precisions
{8, 4, 3, 2}, which is 244 cells. The `kquant-imx` method
round-trips each cell through llama.cpp's k-quant types with the
importance matrix below. The scan ran on the reference box under a
12 GiB GPU memory cap with the remaining groups offloaded to host
memory, and finished in one invocation (run id `d4e07748ae2a`,
2026-08-27 02:46 to 16:06 UTC).

The published recipe solved from this map. `recipe.json` in the
model repo records the full solve: the 24 GiB budget, the 9 GiB KV
headroom, the 0.005 format overhead, and the 81-step trace.

## Do not compare damage across files

Damage values are calibration-relative and frame-relative. They
compare only within one file. Do not rank damage across scans,
across calibration sets, or across models. Rank packed models by
measured quality at a fixed model and budget, never by raw damage.

## Solve a recipe

`vramfit plan` is pure Python and imports no torch. Solve your own
budget against the map:

```
uv run vramfit plan sensitivity-32k-kquant-imx.json \
  --vram 24GiB --kv-headroom 9GiB --format-overhead 0.005
```

Those three values reproduce the published solve. Pass your own
`--vram` and `--kv-headroom` for a different budget. `recipe.json`
in the model repo records every resolved value.

## The importance matrix

`gemma-4-31b-bf16-framed.imatrix.gguf` is the matrix the scan and
the pack consumed. llama-imatrix b10362 built it with
`--parse-special` over `calibration-framed.txt`: 356 chunks at
`n_ctx` 512 through the BF16 decoder GGUF converted from the same
checkpoint at the same revision. The pack step consumed this file
by name, so a pack that passes it reproduces the published bytes.

Coverage derives from the matrix's own entry names, never from a
label. Two absences matter:

- The matrix carries no `token_embd.weight` entry, so the token
  embedding quantized unassisted. This is expected at b10362.
- The matrix carries `attn_v` entries for 50 of 60 layers. The 10
  full_attention layers (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
  receive no `attn_v` activation from the b10362 graph, so those
  ten tensors quantized unassisted. This is a property of the
  instrument, not a defect in the matrix.

## Run log

The scan ships its run log, `sensitivity-32k-kquant-imx.runlog.jsonl`
— structured JSONL, one `cell_measured` event per cell between the
lifecycle events `scan_started`, `meter_built`, and
`scan_finished`. Every line carries `vramfit_runlog` 2. The scan
ran start to finish with no halt: 244 cell events. Every cell
event records the group, the bits, the measured damage, the
wall-clock seconds, and the process memory high-water mark. The
`meter_built` event lists the vision-tower tensors the matrix does
not cover. The scan never measures them.

## The calibration set

`calibration.txt` is the complete Project Gutenberg ebook of
*Pride and Prejudice*, unmodified, with the Project Gutenberg
header and license text intact — byte-identical to the calibration
file of the project's two earlier map datasets.

`calibration-framed.txt` is the same text inside the checkpoint's
own answer channel: 357 blocks of about 512 tokens, each wrapped in
the Gemma chat template by vramfit's `frame_calibration.py`. Gemma
4 31B IT-QAT prices raw prose at a perplexity near 3,000 and the
same prose inside its own channel at 26 to 75, so every
measurement here ran on the framed file. The scan names it in
`scan.calibration` and reads 32,768 tokens of it. The matrix ran
over all 182,404 tokens. The evaluation tiers ran the packed model
on held-out WikiText-2 test text, never on either file.

## Files and hashes

| File | SHA-256 |
|---|---|
| `sensitivity-32k-kquant-imx.json` | `553e13ab9e3f1b34291beed8acf53c14e04df20e2383921e4b4ae6cad4d931d7` |
| `sensitivity-32k-kquant-imx.runlog.jsonl` | `7461f0cdbaa72b003f8b4017ced61db85b3070ef1d6aa1dc611c43dd6ca23e22` |
| `gemma-4-31b-bf16-framed.imatrix.gguf` | `4a168bd7309f787d89f237ff84520980a4b9995975a31c758fd69e8e9d275c3d` |
| `calibration.txt` | `74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806` |
| `calibration-framed.txt` | `98ab7220cdda1c1c6cd57ccf072daa2a1d0a5890ad82f25e982d5bbc8234c55d` |

Hashes prove identity, not quality. The measurement evidence is
the run log beside the map.

## License

The map and the run log are CC-BY-4.0. The importance matrix is
activation statistics of the Gemma 4 checkpoint, which Google
DeepMind released under the Apache 2.0 license — see the
[Gemma 4 license note](https://ai.google.dev/gemma/docs/gemma_4_license).
The two calibration files are a Project Gutenberg ebook, public
domain in the United States, distributed with its Project
Gutenberg header intact. This dataset carries measurements of the
base model, not the base model's weights.

## Disagree with a number?

Re-run the scan. The
[vramfit repository](https://github.com/Alberto-Codes/vramfit)
documents the scan command, the meter, and the settings the run
log records. A map you measure yourself beats one you argue with.
