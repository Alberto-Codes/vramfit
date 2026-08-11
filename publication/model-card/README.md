---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
base_model_relation: quantized
quantized_by: Alberto-Codes
pipeline_tag: text-generation
tags:
  - vramfit
  - gguf
  - imatrix
---

<!--
DRAFT — issue #81. All numbered dependencies are resolved. The card
ships at upload (#83), which removes the DRAFT line below.

Resolved 2026-08-10 (#85): the sensitivity-map dataset exists,
private until ship. The link below is real. Dataset card source:
publication/maps-dataset/README.md.

Resolved 2026-08-10 (#99): the derived tier-2 statistics now trace
to the analysis artifact `analysis/kld564-paired-q3ks.json`
(ADR-0025 dated note). The ledger carries its upload row and
SHA-256.

Resolved 2026-08-10: the #65 ruling (ADR-0025 amendment) covers every
baseline row (render-time join, baseline sidecars under `baselines/`).
The maintainer ruled the v1-vs-v1_5 flag: the publication carries v1_5,
corrected on the #79 record (#82). The #82 dry run fixed the file
names, with sha256s for the nine on-disk files — the ledger beside
this file (card-ledger.md) carries the upload file list.
-->

# Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF

**DRAFT — do not ship. Issue #81. Remove this line at upload.**

This repository carries one mixed-precision GGUF of
[nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5),
solved to fit a 24 GiB VRAM budget.
[vramfit](https://github.com/Alberto-Codes/vramfit) measured each layer
group's quantization damage — the shift in the model's output
distribution when that group quantizes — then solved the bit allocation
under the budget and packed through llama.cpp's quantizer. The
`fit24gib` marker names the budget the solver targeted. This pack has no
single quantization scheme, and the budget is the claim. Built with
Llama.

Every measured number below sits beside its baseline counterpart. The
card prints the losing numbers too.

## Quality beside size

Held-out WikiText-2 test set, full 564 chunks. Tier 2 measures KL
divergence (KLD) against the f16 base over two windows: the first 100
chunks and all 564. Lower is better for PPL and KLD. "Same top" is the
share of tokens where the quantized model and the f16 base agree on the
top token — higher is better.

| Model | File size | PPL ↓ | KLD 100 ↓ | KLD 564 ↓ | Same top (564) ↑ |
|---|---|---|---|---|---|
| **This pack** | 20.36 GiB | **8.517 ± 0.063** | **0.1538** | **0.2873** | 82.9 % |
| Q3_K_S (bartowski) | 20.45 GiB | 8.532 ± 0.064 | 0.1584 | 0.2959 | **83.4 %** |
| IQ3_XS (bartowski) | 19.47 GiB | 8.554 ± 0.063 | 0.1982 | 0.3309 | 81.7 % |
| IQ3_XXS (bartowski) | 18.18 GiB | 8.723 ± 0.065 | 0.2302 | 0.3665 | 80.1 % |
| UD-IQ3_XXS (unsloth) | 18.34 GiB | 8.697 ± 0.065 | 0.1805 | 0.3439 | 82.0 % |

This pack beats every in-budget baseline on KLD 100 and KLD 564. On PPL
it leads nominally everywhere — a tie by the interval against Q3_K_S and
IQ3_XS, a clear win over the two smallest. The size-matched Q3_K_S keeps
one lead: top-token agreement, 83.4 % against 82.9 %. The i-quants are
0.9–2.2 GiB smaller, and the size column is part of the comparison, not
a footnote.

Against Q3_K_S on the full window, this pack is better on 369 of 564
chunks (65 %). The mean gap is 0.0086. A paired per-chunk test puts the
difference at 7.8σ. The pack is also the first spike-free profile among
this model's recorded packs. Its worst per-chunk excess over the
baseline anywhere in 564 chunks is +0.05. The three chunks where earlier
packs of this model spiked (347, 502, 137) read 0.126, 0.124, and 0.106.
`analysis/kld564-paired-q3ks.json` in this repository records this
comparison and the per-chunk KLD pairs that produce it.

## What fit24gib means

The solver targeted a 24 GiB card serving 16k context at fp8 KV cache.
The recipe records the arithmetic:

| Quantity | Bytes | GiB |
|---|---|---|
| VRAM ceiling | 25,769,803,776 | 24.00 |
| KV cache + runtime headroom | 3,791,650,816 | 3.53 |
| Weight budget | 21,978,152,960 | 20.47 |
| Predicted pack size | 21,957,337,301 | 20.45 |
| Real packed file | 21,860,214,272 | 20.36 |

The real file lands 112.48 MiB under the weight budget. Size prediction
prices each GGUF type at its effective bits — Q4_K spends 4.5 bits per
weight, not 4. A 0.005 overhead fraction covers unquantized tensors and
file metadata.

## The recipe

The solver allocated 82 layer groups:

- `model.layers.0` at 8-bit (Q8_0).
- The other 81 groups at 3-bit (Q3_K), including the token embedding
  and the output head.
- 48 protected tensors hold `attn_v` above the group assignment:
  47 floors at 5-bit (Q5_K), one at 4-bit (Q4_K, `blk.3`). The card
  uses the GGUF name `attn_v` throughout — the recipe records the same
  tensors under their HF name, `self_attn.v_proj.weight`.
- 4 imatrix exclusions: `blk.{1,2,3,5}.attn_v.weight` quantize without
  their importance-matrix rows. Under this imatrix those rows collapse
  the weighted fit — excluded, each tensor reconstructs 2.0–4.3× closer
  to f16 than its unprotected reference (RMSE 0.001641 vs 0.004755,
  0.000574 vs 0.002229, 0.001136 vs 0.002303, 0.000501 vs 0.002178).

The pack ran base type Q3_K_S with `--pure` and 128 type overrides,
under the published importance matrix. A mandatory
reconstruction check compared all 48 protected tensors against an
unprotected reference pack: 48 of 48 reconstruct strictly closer to f16.

The recipe's predicted damage is 0.3905, summed from the sensitivity
map's per-group measurements. Damage values are one scan's measurements
on one measurement frame. Do not compare them across scans or across
models.

<details>
<summary>Per-group allocation and damage (82 groups, from the recipe — higher damage is worse)</summary>

| Group | Precision | GGUF type | Bytes | Damage | Protections |
|---|---|---|---|---|---|
| `model.embed_tokens` | 3 | Q3_K | 453,718,426 | 0.0149 | — |
| `model.layers.0` | 8 | Q8_0 | 537,447,629 | 0.0450 | — |
| `model.layers.1` | 3 | Q3_K | 371,668,747 | 0.0294 | attn_v floor 5-bit (Q5_K), imatrix excluded |
| `model.layers.2` | 3 | Q3_K | 371,668,747 | 0.0402 | attn_v floor 5-bit (Q5_K), imatrix excluded |
| `model.layers.3` | 3 | Q3_K | 370,614,928 | 0.1383 | attn_v floor 4-bit (Q4_K), imatrix excluded |
| `model.layers.4` | 3 | Q3_K | 371,668,747 | 0.0433 | attn_v floor 5-bit (Q5_K) |
| `model.layers.5` | 3 | Q3_K | 371,668,747 | 0.0039 | attn_v floor 5-bit (Q5_K), imatrix excluded |
| `model.layers.6` | 3 | Q3_K | 152,145,101 | 0.0009 | — |
| `model.layers.7` | 3 | Q3_K | 152,145,101 | 0.0008 | — |
| `model.layers.8` | 3 | Q3_K | 371,668,747 | 0.0022 | attn_v floor 5-bit (Q5_K) |
| `model.layers.9` | 3 | Q3_K | 371,668,747 | 0.0014 | attn_v floor 5-bit (Q5_K) |
| `model.layers.10` | 3 | Q3_K | 371,668,747 | 0.0008 | attn_v floor 5-bit (Q5_K) |
| `model.layers.11` | 3 | Q3_K | 190,181,376 | 0.0005 | — |
| `model.layers.12` | 3 | Q3_K | 371,668,747 | 0.0010 | attn_v floor 5-bit (Q5_K) |
| `model.layers.13` | 3 | Q3_K | 371,668,747 | 0.0008 | attn_v floor 5-bit (Q5_K) |
| `model.layers.14` | 3 | Q3_K | 371,668,747 | 0.0009 | attn_v floor 5-bit (Q5_K) |
| `model.layers.15` | 3 | Q3_K | 371,668,747 | 0.0008 | attn_v floor 5-bit (Q5_K) |
| `model.layers.16` | 3 | Q3_K | 371,668,747 | 0.0010 | attn_v floor 5-bit (Q5_K) |
| `model.layers.17` | 3 | Q3_K | 371,668,747 | 0.0012 | attn_v floor 5-bit (Q5_K) |
| `model.layers.18` | 3 | Q3_K | 371,668,747 | 0.0016 | attn_v floor 5-bit (Q5_K) |
| `model.layers.19` | 3 | Q3_K | 371,668,747 | 0.0013 | attn_v floor 5-bit (Q5_K) |
| `model.layers.20` | 3 | Q3_K | 371,668,747 | 0.0007 | attn_v floor 5-bit (Q5_K) |
| `model.layers.21` | 3 | Q3_K | 371,668,747 | 0.0007 | attn_v floor 5-bit (Q5_K) |
| `model.layers.22` | 3 | Q3_K | 371,668,747 | 0.0005 | attn_v floor 5-bit (Q5_K) |
| `model.layers.23` | 3 | Q3_K | 371,668,747 | 0.0005 | attn_v floor 5-bit (Q5_K) |
| `model.layers.24` | 3 | Q3_K | 371,668,747 | 0.0005 | attn_v floor 5-bit (Q5_K) |
| `model.layers.25` | 3 | Q3_K | 371,668,747 | 0.0004 | attn_v floor 5-bit (Q5_K) |
| `model.layers.26` | 3 | Q3_K | 371,668,747 | 0.0004 | attn_v floor 5-bit (Q5_K) |
| `model.layers.27` | 3 | Q3_K | 371,668,747 | 0.0005 | attn_v floor 5-bit (Q5_K) |
| `model.layers.28` | 3 | Q3_K | 371,668,747 | 0.0004 | attn_v floor 5-bit (Q5_K) |
| `model.layers.29` | 3 | Q3_K | 371,668,747 | 0.0004 | attn_v floor 5-bit (Q5_K) |
| `model.layers.30` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.31` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.32` | 3 | Q3_K | 371,668,747 | 0.0004 | attn_v floor 5-bit (Q5_K) |
| `model.layers.33` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.34` | 3 | Q3_K | 371,668,747 | 0.0007 | attn_v floor 5-bit (Q5_K) |
| `model.layers.35` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.36` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.37` | 3 | Q3_K | 371,668,747 | 0.0005 | attn_v floor 5-bit (Q5_K) |
| `model.layers.38` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.39` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.40` | 3 | Q3_K | 371,668,747 | 0.0004 | attn_v floor 5-bit (Q5_K) |
| `model.layers.41` | 3 | Q3_K | 371,668,747 | 0.0004 | attn_v floor 5-bit (Q5_K) |
| `model.layers.42` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.43` | 3 | Q3_K | 152,145,101 | 0.0003 | — |
| `model.layers.44` | 3 | Q3_K | 152,145,101 | 0.0003 | — |
| `model.layers.45` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.46` | 3 | Q3_K | 304,290,202 | 0.0003 | — |
| `model.layers.47` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.48` | 3 | Q3_K | 152,145,101 | 0.0003 | — |
| `model.layers.49` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.50` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.51` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.52` | 3 | Q3_K | 371,668,747 | 0.0005 | attn_v floor 5-bit (Q5_K) |
| `model.layers.53` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.54` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.55` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.56` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.57` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.58` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.59` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.60` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.61` | 3 | Q3_K | 76,072,551 | 0.0002 | — |
| `model.layers.62` | 3 | Q3_K | 29,885,645 | 0.0002 | — |
| `model.layers.63` | 3 | Q3_K | 29,885,645 | 0.0002 | — |
| `model.layers.64` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.65` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.66` | 3 | Q3_K | 29,885,645 | 0.0002 | — |
| `model.layers.67` | 3 | Q3_K | 29,885,645 | 0.0002 | — |
| `model.layers.68` | 3 | Q3_K | 59,771,290 | 0.0002 | — |
| `model.layers.69` | 3 | Q3_K | 29,885,645 | 0.0002 | — |
| `model.layers.70` | 3 | Q3_K | 29,885,645 | 0.0003 | — |
| `model.layers.71` | 3 | Q3_K | 371,668,747 | 0.0007 | attn_v floor 5-bit (Q5_K) |
| `model.layers.72` | 3 | Q3_K | 371,668,747 | 0.0006 | attn_v floor 5-bit (Q5_K) |
| `model.layers.73` | 3 | Q3_K | 371,668,747 | 0.0010 | attn_v floor 5-bit (Q5_K) |
| `model.layers.74` | 3 | Q3_K | 371,668,747 | 0.0010 | attn_v floor 5-bit (Q5_K) |
| `model.layers.75` | 3 | Q3_K | 371,668,747 | 0.0010 | attn_v floor 5-bit (Q5_K) |
| `model.layers.76` | 3 | Q3_K | 371,668,747 | 0.0026 | attn_v floor 5-bit (Q5_K) |
| `model.layers.77` | 3 | Q3_K | 371,668,747 | 0.0020 | attn_v floor 5-bit (Q5_K) |
| `model.layers.78` | 3 | Q3_K | 371,668,747 | 0.0027 | attn_v floor 5-bit (Q5_K) |
| `model.layers.79` | 3 | Q3_K | 371,668,747 | 0.0086 | attn_v floor 5-bit (Q5_K) |
| `lm_head` | 3 | Q3_K | 453,718,426 | 0.0214 | — |

</details>

## The solver's trace

The solver is greedy damage-per-byte: at each step it demotes the group
with the least predicted damage per byte freed. It recorded all 162
demotion steps in the recipe, so the solve replays from the artifact.
The last step is the honest one. With 47 protection floors priced in,
the solver demoted `model.layers.3` from 4-bit to 3-bit. That step freed
113,087,938 B at a predicted damage of 0.1129 — 29 % of the recipe's
total, spent on one step. The measurements below did not punish that
trade.

## Evaluation

Three tiers, one instrument: llama.cpp build b10172 (Vulkan), WikiText-2
test set held out from calibration, f16 KL references recorded once and
reused. Tier 3 ran lm-evaluation-harness 0.4.12 through an in-process
llama-cpp-python lane on the same build — full evaluation splits, no
`--limit`, zero context truncations.

**Tier 1 (perplexity) and tier 2 (whole-model KL)** are the table above.

**Tier 3 (task benchmarks)** ran a slice fixed before any run:
five tasks at leaderboard few-shot settings, on this pack and the
size-matched Q3_K_S baseline. Deltas inside the combined standard error
report as ties.

| Task (metric) | This pack | Q3_K_S baseline | Δ | Combined σ | Verdict |
|---|---|---|---|---|---|
| MMLU 5-shot (acc) | 0.7829 ± 0.0033 | 0.7827 ± 0.0033 | +0.0002 | 0.0047 | tie |
| GSM8K 5-shot (strict) | 0.9318 ± 0.0069 | 0.9242 ± 0.0073 | +0.0076 | 0.0101 | tie |
| HellaSwag 10-shot (acc_norm) | 0.8412 ± 0.0036 | 0.8379 ± 0.0037 | +0.0033 | 0.0052 | tie |
| Winogrande 5-shot (acc) | 0.7845 ± 0.0116 | 0.7861 ± 0.0115 | −0.0016 | 0.0163 | tie |
| ARC-Challenge 25-shot (acc_norm) | 0.6493 ± 0.0139 | 0.6604 ± 0.0138 | −0.0111 | 0.0196 | tie |

Five ties. The largest delta is 0.8σ (GSM8K, this pack nominally ahead).
This pack trails nominally on Winogrande and ARC-Challenge — both
deficits print here with their error bars. The project fixed the slice
before any run, so no result selected the tasks.

Read together: tier 2 ranks (the 7.8σ full-window KLD win over the
size-matched baseline), tier 3 certifies (five ties at equal size).

## Reproduce it

The repository ships the recipe, the importance matrix, the evals
sidecar, and the run log beside the weights. One command reproduces the
pack from the base checkpoint:

```
uv run vramfit pack recipe.json --llama-cpp <llama.cpp checkout> \
  --imatrix imatrix.gguf --out Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf
```

<!-- File names final per the #82 dry run. Local source artifacts:
     recipe-g1c-replication.json -> recipe.json,
     nemotron-49b-f16.imatrix.gguf -> imatrix.gguf,
     nemotron-49b-g1c-replication.gguf ->
     Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf. -->

The command converts the f16 base GGUF once, drives the recorded type
overrides and imatrix exclusions into `llama-quantize`, and runs the
reconstruction check itself. Expect all 48 protected tensors green.

The recipe is not a magic constant. It records the full solve: the
budget bytes, the 48 protections, the 4 exclusions, and the 162-step
trace. To solve for a different budget, run `vramfit plan` against the
published sensitivity map with your own `--vram`. The map is right
there:
[Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps](https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps),
file `sensitivity-64k-kquant-imx-no2-sized.json`.

## Guardrails and damage disclosure

<!-- Stance decided in #86: comply-and-disclose. Record: the
     guardrail-efficacy stance note in
     docs/explanation/evaluating-packed-models.md. -->

This pack modifies no guardrail. Quantization compresses every weight
tensor with one uniform lossy procedure. It does not bypass, disable,
or circumvent the base model's safety training, which ships in these
weights at lower precision.

Quantization can shift any model behavior, and guardrail behavior is no
exception. The tables above are the measured bound on that shift: tier
2 measures whole-model KL divergence against the f16 base over the full
held-out set, and tier 3 holds five statistical ties at equal size. The
NVIDIA Open Model License conditions distribution on keeping "a
substantially similar Guardrail appropriate for your use case" — this
card's damage disclosure is the evidence of that similarity.

One limit, stated plainly: damage measures the general output
distribution on held-out WikiText-2 text, not guardrail behavior
separately. Read this card as a damage disclosure, not a safety
certificate. Deploy this pack with the same system-prompt and
application-layer protections you would give the base model.

## License

The base model ships under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
and is built on Llama 3.3, so Meta's Llama 3.3 Community License also
applies. Both layers permit a quantized GGUF derivative on Hugging Face
with attribution. This repository carries both license texts and both
notice files. Built with Llama.

## Provenance

- Packed file SHA-256:
  `48271199ee97d5559caa6bb963162265a9fc35cb5c7ec2b181513f7c4c810122`
  (21,860,214,272 B).
- The run log beside the weights
  (`Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.runlog.jsonl`) records
  the pack events and the 48-tensor reconstruction check.
- The evals sidecar
  (`Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib.gguf.evals.json`,
  ADR-0025 schema 2) records all three tiers. Baseline sidecars sit
  under `baselines/`.
- Toolchain: llama.cpp b10172 quantizer, `convert_hf_to_gguf.py` for the
  f16 base, lm-evaluation-harness 0.4.12 with llama-cpp-python 0.3.34
  for tier 3.

Hashes prove identity, not quality. The evidence is the three tiers
above, and every number on this card traces to the evals sidecar and the
recipe published in this repository.
