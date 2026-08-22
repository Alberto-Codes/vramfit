---
license: cc-by-4.0
pretty_name: NVIDIA-Nemotron-3.5-Lightning-30B-A3B sensitivity maps
tags:
  - vramfit
viewer: false
---

<!--
Authored for issue #404 under the #401 identity grammar. This file
is the card of the dataset repo
Alberto-Codes/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps.
Upload this file verbatim — the published card and this source must
match.

Open before upload: the dataset repo does not exist yet. Create it,
upload the five files in the hashes table, then this card.
-->

# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-sensitivity-maps

This dataset carries the expert-stack quantization sensitivity maps
of
[nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
(revision `ce38b6a`).
[vramfit](https://github.com/Alberto-Codes/vramfit) measured them.
A sensitivity map records one damage number per layer group and
candidate precision. Damage is the shift in the model's output
distribution when that group alone quantizes — mean final-logits KL
divergence against the bf16 reference. The maps describe the base
model, not any quantized file. They contain no model weights.

The packed model solved from the `q0-imx` map below ships as
[NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib-GGUF](https://huggingface.co/Alberto-Codes/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-fit16gib-GGUF).

## The map format

Each map is one JSON file, `vramfit_schema` 3, in the form that
carries `group_by: stack`. The `scan` block records the
measurement frame: metric, calibration file, token count,
candidate precisions, grouping, within-group method, and imatrix
path. The `groups` list records the model's 46 routed-expert
stacks. An expert stack is the 128 routed experts of one
projection in one layer, stored as a single tensor — the unit a
GGUF pack assigns a precision to. Each group carries its member
tensors, its bytes at reference precision, its damage per
precision, and `tensor_bytes`, the per-tensor size split. The
`q0-imx` map also carries `imatrix_counts`. The
[sensitivity map format](https://github.com/Alberto-Codes/vramfit/blob/main/docs/reference/sensitivity-map.md)
page specifies every field. The path fields record the measuring
machines' absolute paths. The `scan.calibration` basename matches
the file here, and the model repo publishes the `scan.imatrix`
matrix as `imatrix.gguf`.

The maps hold the expert stacks only. Dense groups never enter
them: the published recipe pins every quantizable dense class at
8-bit and prices it from the checkpoint's safetensors headers.

## The two scans

| File | Calibration tokens | Within-group method | Imatrix | Started (UTC) | Cells |
|---|---|---|---|---|---|
| `sensitivity-32k-q0-imx-stacks.json` | 32,768 | `q0-imx` | yes | 2026-08-21 | 92 |
| `sensitivity-32k-q0-ref-stacks.json` | 32,768 | `q0-ref` | no | 2026-08-18 | 92 |

Each scan covers the 46 expert stacks at candidate precisions
{4, 2}, which is 92 cells. The `q0` method round-trips each cell
through llama.cpp's `_0`-family quantizers (one fp16 scale per
block, no super-block), ported against the C reference and
bit-exact for the unassisted types. `q0-ref` fits unassisted.
`q0-imx` weights the nominal-4 fit with the pack's importance
matrix. The 2-bit quantizer ignores the matrix, so the two methods
differ at nominal 4 only. The model repo publishes the importance
matrix the `q0-imx` map and the pack consumed.

The two maps play different roles in the publication:

- **The published recipe solved from the `q0-imx` map.**
  `recipe.json` in the model repo records the full solve: the
  budget bytes, the nine pins, the format overhead, and the
  11-step trace.
- **The `q0-ref` map is the attribution bound.** Its damage
  ordering agrees through every rank the solve reads, and it
  derives the identical placement. So the published win credits
  the damage ranking under the placement rule, not the importance
  matrix.

## Do not compare damage across files

Damage values are calibration-relative and frame-relative. They
compare only within one file. Do not rank damage across scans,
across calibration sets, or across models. The two scans exist to
separate the ranking from the imatrix, not to be averaged. Rank
packed models by measured quality at a fixed model and budget,
never by raw damage.

## Solve a recipe

`vramfit plan` is pure Python and imports no torch. Solve your own
budget against the `q0-imx` map:

```
uv run vramfit plan sensitivity-32k-q0-imx-stacks.json \
  --checkpoint <checkpoint dir> --vram 16GiB
```

`--checkpoint` prices the dense groups the map does not hold. An
unstated `--kv-headroom` defaults to 4GiB, so this example solves
a smaller weight budget than the publication. The published solve
passed the ruled 15.776 GiB weight budget as `--vram` with
`--kv-headroom 0`, nine dense pins at 8-bit, and a 0.002 format
overhead. `recipe.json` in the model repo records every resolved
value.

## Run logs

Each scan ships its run log, `<scan name>.runlog.jsonl` —
structured JSONL, one `cell_measured` event per cell between the
lifecycle events `scan_started`, `meter_built`, and
`scan_finished`. Every line carries `vramfit_runlog` 2. Both scans
ran start to finish with no halt: 92 cell events each. Every cell
event records the group, the bits, the measured damage, the
wall-clock seconds, and the process memory high-water mark.

## The calibration set

`calibration.txt` is the complete Project Gutenberg ebook of
*Pride and Prejudice*, unmodified, with the Project Gutenberg
header and license text intact — byte-identical to the calibration
file of the project's first publication. Both scans name this file
in `scan.calibration` and read 32,768 tokens of it. The evaluation
tiers ran the packed model on held-out WikiText-2 test text, never
on this file.

## Files and hashes

| File | SHA-256 |
|---|---|
| `sensitivity-32k-q0-imx-stacks.json` | `68c79d24b87dd58012ac72e0a1c6d0137173b10b528057afee36bf2079bd00e5` |
| `sensitivity-32k-q0-imx-stacks.runlog.jsonl` | `8f74d51303c66c9b8d607ee53122a137fa7c9707495b2c9edc19c8d63b8dc60a` |
| `sensitivity-32k-q0-ref-stacks.json` | `0fbcf84fc5d471fbc7a943485256ac8521596cf605b3c836bc1622e6c6b4f55e` |
| `sensitivity-32k-q0-ref-stacks.runlog.jsonl` | `1ce8fd3d4287abb240a314f3972fbf756279f74feb9eadd934b1aa126992b073` |
| `calibration.txt` | `74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806` |

Hashes prove identity, not quality. The measurement evidence is
the run logs beside each map.

## License

The maps and the run logs are CC-BY-4.0. `calibration.txt` is a
Project Gutenberg ebook, public domain in the United States,
distributed with its Project Gutenberg header intact. This dataset
carries measurements of the base model, not the base model's
weights. OpenMDW 1.1 governs the base model repo and the
packed-model repo.

## Disagree with a number?

Re-run the scan. The
[vramfit repository](https://github.com/Alberto-Codes/vramfit)
documents the scan command, the meter, and the settings the run
logs record. A map you measure yourself beats one you argue with.
