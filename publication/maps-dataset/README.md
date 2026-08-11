---
license: cc-by-4.0
pretty_name: Llama-3_3-Nemotron-Super-49B-v1_5 sensitivity maps
tags:
  - vramfit
viewer: false
---

<!--
Authored for issue #85. This file is the card of the dataset repo
Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps.
The repo went public on 2026-08-11 (#83) under cc-by-4.0. Upload this
file verbatim — the published card and this source must match.
-->

# Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps

This dataset carries the per-layer quantization sensitivity maps of
[nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5).
[vramfit](https://github.com/Alberto-Codes/vramfit) measured them.
A sensitivity map records one damage number per layer group and
candidate precision. Damage is the shift in the model's output
distribution when that group alone quantizes — mean final-logits KL
divergence against the bf16 reference. The maps describe the base
model, not any quantized file. They contain no model weights.

The packed model built from the sized no-2 map below ships as
[Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF).

## The map format

Each map is one JSON file, `vramfit_schema` 2 (the envelope key
renamed with the tool, #118). The `scan` block records the
measurement frame: metric, calibration file, token count, candidate
precisions, within-group method, and imatrix path. The
three `rtn-block32` maps predate the last two fields — an absent
field reads as `rtn-block32` with no imatrix, per the format page.
The
`groups` list records 82 layer groups. Each group carries its member
tensors, its bytes at reference precision, and its damage per
precision. Annotated copies add `tensor_bytes`, the per-tensor size
split. The
[sensitivity map format](https://github.com/Alberto-Codes/vramfit/blob/main/docs/reference/sensitivity-map.md)
page specifies every field. The path fields record the reference
box's absolute paths. The basenames match the files here.

## The five scans

Five scans measured the model, each inside its own measurement frame:

| File | Calibration tokens | Within-group method | Imatrix | Started (UTC) | Cells |
|---|---|---|---|---|---|
| `sensitivity-8k.json` | 8,192 | `rtn-block32` | no | 2026-07-29 | 328 |
| `sensitivity-32k.json` | 32,768 | `rtn-block32` | no | 2026-07-29 | 328 |
| `sensitivity-64k.json` | 65,536 | `rtn-block32` | no | 2026-07-30 | 328 |
| `sensitivity-64k-kquant.json` | 65,536 | `kquant-ref` | no | 2026-08-02 | 328 |
| `sensitivity-64k-kquant-imx.json` | 65,536 | `kquant-imx` | yes | 2026-08-04 | 328 |

`started_at` records the last resume of a halted scan. The run logs
carry every earlier attempt. Each scan covers 82 layer groups at candidate precisions
{8, 4, 3, 2} — 328 cells. `rtn-block32` quantizes a perturbed group
with round-to-nearest in 32-element blocks. `kquant-ref` round-trips
each cell through ported llama.cpp reference quantizers
([ADR-0018](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0018-kquant-within-group-method.md)).
`kquant-imx` weights the within-group fit with the pack's importance
matrix
([ADR-0020](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0020-imatrix-assisted-pricing.md)).
The model repo publishes that imatrix as `imatrix.gguf`.

## Derived copies

Three files derive from `sensitivity-64k-kquant-imx.json`:

- `sensitivity-64k-kquant-imx-sized.json` adds `tensor_bytes` to
  every group, read from the checkpoint's safetensors headers.
  `vramfit plan --protect` requires the field.
- `sensitivity-64k-kquant-imx-no2.json` removes the 2-bit column and
  marks itself derived in a `derived` field. In-frame 2-bit prices do
  not predict the packed artifact
  ([ADR-0021](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0021-runtime-frame-measurement.md)).
- `sensitivity-64k-kquant-imx-no2-sized.json` applies both changes.
  **`vramfit plan` solved the published recipe from this file.**

## Do not compare damage across files

Damage values are calibration-relative and frame-relative. They
compare only within one file. Do not rank damage across scans, across
calibration sets, or across models. The five scans exist because the
measurement frame evolved — their numbers answer different questions,
not the same question five times. Rank packed models by measured
quality at a fixed model and budget, never by raw damage.

## Solve a recipe

`vramfit plan` is pure Python and imports no torch. Solve your own
budget against the sized no-2 map:

```
uv run vramfit plan sensitivity-64k-kquant-imx-no2-sized.json --vram 24GiB
```

The published recipe used this map with explicit protections and
imatrix exclusions. `recipe.json` in the model repo records that full
solve: the budget bytes, the 48 protections, the 4 exclusions, and
the 162-step trace.

## Run logs

Each scan ships its run log, `<scan name>.runlog.jsonl` — structured
JSONL, one `cell_measured` event per cell between the lifecycle
events `scan_started`, `meter_built`, and `scan_finished`. Every
line carries `vramfit_runlog` 2, the run-log envelope key that
renamed with the tool (#118). A halted
run also logs `scan_halted` and `resume_loaded`, and repeats the
start events. Every cell event records the
group, the bits, the measured damage, the wall-clock seconds, and the
process memory high-water mark.

## The calibration set

`calibration.txt` is the complete Project Gutenberg ebook of *Pride
and Prejudice*, unmodified, with the Project Gutenberg header and
license text intact. Every scan names this file in
`scan.calibration`. Each scan records the token count it read in
`scan.calibration_tokens`. The evaluation tiers ran the packed model
on held-out WikiText-2 test text, never on this file.

## Files and hashes

| File | SHA-256 |
|---|---|
| `sensitivity-8k.json` | `802d6a6189912b624e670e416c8acf57a479a0ecc5c2900a7dbd84dde29ea71e` |
| `sensitivity-8k.runlog.jsonl` | `9750e9f6b3561d62e6541bea7e5535d07335debae95693fc009e7bead866027a` |
| `sensitivity-32k.json` | `ddb39f1684c427927f2d693c3f043c518bcc7bf24e9636103ccc306c70197ec6` |
| `sensitivity-32k.runlog.jsonl` | `113259fc0fae646929bdf8ac7fb4dabb77563d0530b243b6d3f0dda140b0097b` |
| `sensitivity-64k.json` | `9c810fea55c84e80202b32321ab4df1fc4dbd4736aad2e081e6746e6622bf7dc` |
| `sensitivity-64k.runlog.jsonl` | `e20c1a935580ded5a9e486e1deb0f2d2c8eb6bf98b019122337f96ac55297140` |
| `sensitivity-64k-kquant.json` | `b69d13aa5bf5eccc27c387b4978b5d34de1721e06e0e1609b314e6d13e6a1a33` |
| `sensitivity-64k-kquant.runlog.jsonl` | `aed1b245d3b92c69dba6c6abf4d7eea829f38adda5e03a7b1103f91cc4109e10` |
| `sensitivity-64k-kquant-imx.json` | `35ca616e3f00d23f8f680450a476658edfeee1679a8e0e2593b5dbb3f22d507c` |
| `sensitivity-64k-kquant-imx.runlog.jsonl` | `475292803a7ff6828887f51eccc34e463a1f3ab28edbb527e1a0cafc2e60fc52` |
| `sensitivity-64k-kquant-imx-sized.json` | `bc7e8bec5d4662d62608699b665525d1ad14e7308182ac1700574dc8d376631c` |
| `sensitivity-64k-kquant-imx-no2.json` | `86b4e14f68edae3c82ab7fd49a2e806d3856b72d4b4feff53c0cfcaf0e10fdbe` |
| `sensitivity-64k-kquant-imx-no2-sized.json` | `3f0a914cc3b0889aa94fe2621f195fd398758c913d221eb4f5af19a7a08b6c36` |
| `calibration.txt` | `74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806` |

Task #121 re-uploaded every map and run log on 2026-08-11. The
hashes above are the post-rename values. The key rename changed no
measurement. Every damage number, token count, and timestamp is the
value the scan recorded. `calibration.txt` did not change. A copy you
downloaded before 2026-08-11 carries the old key, and vramfit rejects
it. Download the file again.

Hashes prove identity, not quality. The measurement evidence is the
run logs beside each map.

## License

The maps and the run logs are CC-BY-4.0.
`calibration.txt` is a Project Gutenberg ebook, public domain in the
United States, distributed with its Project Gutenberg header intact.
This dataset carries measurements of the base model, not the base
model's weights. The NVIDIA Open Model License and the Llama 3.3
Community License govern the base model repo and the packed-model
repo.

## Disagree with a number?

Re-run the scan. The
[vramfit repository](https://github.com/Alberto-Codes/vramfit)
documents the scan command, the meter, and the offload settings the
run logs record. A map you measure yourself beats one you argue with.
