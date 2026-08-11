---
license: cc-by-4.0
pretty_name: Llama-3_3-Nemotron-Super-49B-v1_5 sensitivity maps
tags:
  - quantfit
---

<!--
Authored for issue #85. This file is the card of the dataset repo
Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps.
The repo is private until ship (#83). The license choice
(cc-by-4.0 for the measurement data) awaits maintainer
confirmation before the flip — flagged on the #85 record.
-->

# Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps

This dataset carries the per-layer quantization sensitivity maps of
[nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5).
[quantfit](https://github.com/Alberto-Codes/quantfit) measured them.
A sensitivity map records one damage number per layer group and
candidate precision. Damage is the shift in the model's output
distribution when that group alone quantizes — mean final-logits KL
divergence against the bf16 reference. The maps describe the base
model, not any quantized file. They contain no model weights.

The pack solved from these maps ships as
[Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF).

## The map format

Each map is one JSON file, `quantfit_schema` 1. The `scan` block
records the measurement frame: metric, calibration file, token count,
candidate precisions, within-group method, and imatrix path. The
`groups` list records 82 layer groups. Each group carries its member
tensors, its bytes at reference precision, and its damage per
precision. Annotated copies add `tensor_bytes`, the per-tensor size
split. The
[sensitivity map format](https://github.com/Alberto-Codes/quantfit/blob/main/docs/reference/sensitivity-map.md)
page specifies every field.

## The five scans

Five scans measured the model, each inside its own measurement frame:

| File | Calibration tokens | Within-group method | Imatrix | Started (UTC) | Cells |
|---|---|---|---|---|---|
| `sensitivity-8k.json` | 8,192 | `rtn-block32` | no | 2026-07-29 | 328 |
| `sensitivity-32k.json` | 32,768 | `rtn-block32` | no | 2026-07-29 | 328 |
| `sensitivity-64k.json` | 65,536 | `rtn-block32` | no | 2026-07-30 | 328 |
| `sensitivity-64k-kquant.json` | 65,536 | `kquant-ref` | no | 2026-08-02 | 328 |
| `sensitivity-64k-kquant-imx.json` | 65,536 | `kquant-imx` | yes | 2026-08-04 | 328 |

Each scan covers 82 layer groups at candidate precisions
{8, 4, 3, 2} — 328 cells. `rtn-block32` quantizes a perturbed group
with round-to-nearest in 32-element blocks. `kquant-ref` round-trips
each cell through ported llama.cpp reference quantizers
([ADR-0018](https://github.com/Alberto-Codes/quantfit/blob/main/docs/adr/0018-kquant-within-group-method.md)).
`kquant-imx` weights the within-group fit with the pack's importance
matrix
([ADR-0020](https://github.com/Alberto-Codes/quantfit/blob/main/docs/adr/0020-imatrix-assisted-pricing.md)).
The imatrix itself publishes in the model repo as `imatrix.gguf`.

## Derived copies

Three files derive from `sensitivity-64k-kquant-imx.json`:

- `sensitivity-64k-kquant-imx-sized.json` adds `tensor_bytes` to
  every group, read from the checkpoint's safetensors headers.
  `quantfit plan --protect` requires the field.
- `sensitivity-64k-kquant-imx-no2.json` removes the 2-bit column and
  marks itself derived in a `derived` field. In-frame 2-bit prices do
  not predict the packed artifact
  ([ADR-0021](https://github.com/Alberto-Codes/quantfit/blob/main/docs/adr/0021-runtime-frame-measurement.md)).
- `sensitivity-64k-kquant-imx-no2-sized.json` applies both changes.
  **The published recipe was solved from this file.**

## Do not compare damage across files

Damage values are calibration-relative and frame-relative. They
compare only within one file. Do not rank damage across scans, across
calibration sets, or across models. The five scans exist because the
measurement frame evolved — their numbers answer different questions,
not the same question five times. Rank packed models by measured
quality at a fixed model and budget, never by raw damage.

## Solve a recipe

`quantfit plan` is pure Python and imports no torch. Solve your own
budget against the sized no-2 map:

```
uv run quantfit plan sensitivity-64k-kquant-imx-no2-sized.json --vram 24GiB
```

The published recipe used this map with explicit protections and
imatrix exclusions. `recipe.json` in the model repo records that full
solve: the budget bytes, the 48 protections, the 4 exclusions, and
the 162-step trace.

## Run logs

Each scan ships its run log, `<scan name>.runlog.jsonl` — structured
JSONL, one `cell_measured` event per cell between `scan_started`,
`meter_built`, and `scan_finished`. Every cell event records the
group, the bits, the measured damage, the wall-clock seconds, and the
process memory high-water mark.

## The calibration set

`calibration.txt` is the complete Project Gutenberg ebook of *Pride
and Prejudice*, unmodified, with the Project Gutenberg header and
license text intact. Every scan names this file in
`scan.calibration` and reads the token count its scan block records.
The evaluation of the packed model used held-out WikiText-2 test
text, never this file.

## Files and hashes

| File | SHA-256 |
|---|---|
| `sensitivity-8k.json` | `cfde5e56e746e85a541aa2419ba9eb646cb7f6a5851b25f4c64da0be401d9595` |
| `sensitivity-8k.runlog.jsonl` | `55e3fca0495c6c2eceab140b8978f35499e789d5fb88861f3ba85358ff64bed7` |
| `sensitivity-32k.json` | `56297986edf97ff270e719fa990f9d94076c16c79fd5f2c2b1abf3e399471ab8` |
| `sensitivity-32k.runlog.jsonl` | `a5798fe33e7c5008ff5557f974e5e9b731d57952e37e76aa5e1964eba4df8f81` |
| `sensitivity-64k.json` | `86b9c67c1f46343e3c65da859670f2e82b48bddb8b8623bc8f93790473661cad` |
| `sensitivity-64k.runlog.jsonl` | `e6ef9acb89ba5e941c303e5669a2df769a6ccb34d9f9d2f1b620ec3ed3d2d20f` |
| `sensitivity-64k-kquant.json` | `df0ecc92aae09c83e3f25ea611e9eae3c4e965c39e64eb5e1406adcbd5fff898` |
| `sensitivity-64k-kquant.runlog.jsonl` | `6f2560ac06952790422bbac5fe85da42d568082b04c67d7cba83f5fee895b845` |
| `sensitivity-64k-kquant-imx.json` | `b922131a67c8aee2e5ba0db4a0d35841b113fafad69634b59c306b3f2c7ef9ef` |
| `sensitivity-64k-kquant-imx.runlog.jsonl` | `764218554c4b13345fe4f343d0e4011c39cf53ed965804d724f617a64a589cd5` |
| `sensitivity-64k-kquant-imx-sized.json` | `9c8dccec604219d21279db2be62e39f157ab8a73311cd8ad103c06f9da98cde1` |
| `sensitivity-64k-kquant-imx-no2.json` | `f11b2ad01701837203f7a714bc0007caee42a728de6e7ddf76098243bd48a0fe` |
| `sensitivity-64k-kquant-imx-no2-sized.json` | `8f1abcf2c0a38bf50103858fb39f1d2e331f5e340699ea11438c79763adcef62` |
| `calibration.txt` | `74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806` |

Hashes prove identity, not quality. The measurement evidence is the
run logs beside each map.

## License

The measurement data — the maps and the run logs — is CC-BY-4.0.
`calibration.txt` is a Project Gutenberg ebook, public domain in the
United States, distributed with its Project Gutenberg header intact.
This dataset carries measurements of the base model, not the base
model's weights. The base model's licenses govern the model repos.

## Disagree with a number?

Re-run the scan. The
[quantfit repository](https://github.com/Alberto-Codes/quantfit)
documents the scan command, the meter, and the offload settings the
run logs record. A map you measure yourself beats one you argue with.
