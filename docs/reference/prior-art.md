---
status: draft
---

# Prior art and external references

> **Status: draft** — annotated pointers, not endorsements. Each entry says
> why it matters to quantfit; follow the link for the substance.

## Direct inspiration

**[antirez/ds4](https://github.com/antirez/ds4)**
:   Single-model inference engine for DeepSeek V4 Flash with hand-tuned
    selective quantization
    ([GGUFs](https://huggingface.co/antirez/deepseek-v4-gguf)). Source of the
    depth-over-breadth ethos ([ADR-0001](../adr/0001-selective-per-layer-quantization.md))
    and proof that per-model recipes beat generic ones.

## Quantization methods

**[GPTQ](https://arxiv.org/abs/2210.17323)**
:   Calibration-aware post-training quantization via second-order weight
    updates. A candidate within-group method — quantfit decides bits per
    group; GPTQ is one way to round within a group.

**[AWQ](https://arxiv.org/abs/2306.00978)**
:   Activation-aware weight quantization; protects the small fraction of
    outlier channels that carry outsized importance. Same role as GPTQ here,
    and evidence for non-uniform fragility.

**[Is (Selective) Round-To-Nearest Quantization All You Need?](https://arxiv.org/abs/2505.15909)** (2025)
:   Argues plain RTN with selective precision upgrades is a viable,
    cheaper alternative to calibration-based methods — external
    support for quantfit's selective-assignment premise (ADR-0001)
    from the opposite direction. Added 2026-08-04, with a caveat
    from the seventh data point: at sub-4-bit, RTN *over-prices*
    in-frame damage 2.0–3.9× against the K-quant fit the pack
    ships ([ADR-0018](../adr/0018-kquant-within-group-method.md)).
    RTN as a serving format and RTN as a pricing frame are
    different claims.

**[llama.cpp k-quants + imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix)**
:   k-quants ship non-uniform per-layer recipes chosen by fixed heuristic.
    The imatrix tool adds measured activation statistics that bias
    *within-block* scale selection. That measurement sits below the
    bit-assignment level — and the 2026-07-29 head-to-head showed it
    dominates there: the first packed 49B recipe lost to an imatrix
    Q3_K_S, with ~81 % of the perplexity gap traced to the imatrix
    ([evaluating packed models](../explanation/evaluating-packed-models.md)).
    `quantfit pack` consumes an importance matrix since
    [ADR-0016](../adr/0016-imatrix-in-the-pack-path.md). The imatrix
    rematch narrowed the gap to 0.53–0.62 PPL, and the gating item
    moved to the scan-to-runtime frame transfer
    ([the fifth and sixth data points](../explanation/evaluating-packed-models.md#the-sixth-data-point-the-converged-map-and-where-the-leak-moved)).
    `llama-quantize --tensor-type`
    accepts per-tensor type overrides, which is the mechanism the
    GGUF pack backend drives
    ([ADR-0010](../adr/0010-sub-4-bit-serving-path.md)).

**[Unsloth dynamic GGUFs](https://huggingface.co/unsloth)**
:   Selective per-layer bit assignment shipped as ready-made GGUFs —
    the closest prior art by shipped output. Dynamic 2.0
    (2025) claims per-layer sensitivity measurement. The method,
    code, and per-model evidence stay unpublished: a direct
    [transparency question](https://github.com/unslothai/unsloth/discussions/3523)
    closed with zero maintainer comments (re-checked 2026-07-31).
    quantfit bets that a measured, *published* sensitivity map beats
    this recipe class. The bet is now specifically about evidence,
    not about whether selective assignment works. See
    [the artifact ecosystem](../explanation/artifact-ecosystem.md).
    Found 2026-08-04: Unsloth ships
    [UD GGUFs of the north-star target](https://huggingface.co/unsloth/Llama-3_3-Nemotron-Super-49B-v1_5-GGUF)
    inside the 20.47 GiB weight budget — UD-Q2_K_XL (19.3 GB) and
    UD-IQ3_XXS (19.7 GB). They are candidate additional baselines
    for the evaluation tiers: a head-to-head against a shipped
    selective recipe for the exact model and budget, not only
    against the flat-with-protections Q3_K_S.

**[EXL3 / exllamav3](https://github.com/turboderp-org/exllamav3)**
:   The closest prior art by method, superseding EXL2 here
    (re-surveyed 2026-07-31). exllamav3 ships `util/measure.py`,
    which records KL-divergence contributions per candidate group
    between quant levels. Its `optimize.py` allocates bits from
    that measurement to a target *average* bits-per-weight. That is
    a working measure-then-solve loop. quantfit bets on
    four differences. EXL3 is tied to its own CUDA runtime — vLLM
    declined integration
    ([vllm#19896](https://github.com/vllm-project/vllm/issues/19896)).
    It optimizes to an average bpw, not an explicit VRAM budget
    planned jointly with KV headroom. Its measurement is not a
    standalone provenance-carrying artifact. It has no whole-recipe
    validation pass.

**[Adaptive-Quantization](https://github.com/bigattichouse/Adaptive-Quantization)**
:   A per-tensor GGUF recipe tool in quantfit's exact pack lane
    (writes `--tensor-type` files for `llama-quantize`), surveyed
    2026-07-31. It profiles *weight reconstruction SNR* per tensor —
    layer-local error, the metric class
    [ADR-0006](../adr/0006-sensitivity-metric.md) rejected because
    it ignores propagation. quantfit's validation measurements are
    evidence that propagation matters exactly where bit allocation
    matters most, and that the effect depends on which groups sit
    at 2-bit (ADR-0006, third and fourth measurements). Zero
    adoption signals at survey time. Its existence confirms the
    lane is visible. The differentiation is the metric and the
    validation, not the mechanism.

**[llm-compressor](https://github.com/vllm-project/llm-compressor)**
:   vLLM's official companion for producing compressed-tensors checkpoints
    (GPTQ, AWQ, FP8/INT4 schemes, non-uniform per-layer configs). The
    candidate backend for a future ≥4-bit vLLM pack path — the shipped
    `quantfit pack` drives llama.cpp instead (ADR-0012). quantfit's job
    is deciding the recipe, a backend's is applying it.

**[Minitron](https://arxiv.org/abs/2408.11796)** (NVIDIA)
:   Pruning + distillation for shrinking LLMs. The alternative road not taken
    ([ADR-0001](../adr/0001-selective-per-layer-quantization.md)): changes the
    model, needs training compute.

## Research on the additivity problem

Added 2026-07-31, after the third validation measurement found
super-additive joint damage (×11.9) on a 2-bit-heavy recipe.

**[CLADO](https://arxiv.org/abs/2307.05657)**
:   Names quantfit's measured problem: sensitivity-based
    mixed-precision methods assume per-layer errors are independent,
    and they are not. CLADO measures *pairwise* cross-layer error
    terms on a small calibration subset and solves the allocation as
    an Integer Quadratic Program. It is demonstrated on
    ImageNet-scale vision models. At survey time (2026-07-31) it
    was the only interaction-aware allocator found, and it ships
    in no quantization tool. CoopQ (below) has since reached LLM
    scale in research. CLADO remains
    the candidate algorithm shape for an interaction-aware `plan`,
    at O(groups²) extra measurement the existing meter can price.
    ADR-0006's fourth measurement lowers its urgency: converged
    marginals steered the solver sub-additive without interaction
    terms.

**[CoopQ](https://arxiv.org/abs/2509.15455)**
:   Interaction-aware mixed precision at LLM scale (added
    2026-08-04). Estimates layer sensitivities *and* inter-layer
    dependencies with Shapley values (SPQE), then assigns 2 or 4
    bits per layer as a binary quadratic optimization under a
    memory constraint. Evaluated on Llama-3, Gemma-2, and Qwen-3
    across three quantization backends. This is quantfit's 2-bit
    membership problem (ADR-0006, fourth measurement) treated as
    the objective, on serving-class models. No shipped tool or
    artifact pipeline found. If the additive solve stays the
    gating item after imatrix-assisted pricing, this and CLADO are
    the two published algorithm shapes to price.

**[GAMMA](https://arxiv.org/abs/2605.18475)** (2026)
:   Learns module-wise precision preferences post-training and
    converts them to exact budget-feasible assignments with an
    integer program — one training serves arbitrary budgets by
    re-solving only the IP (added 2026-08-04). Evaluated on Llama
    and Qwen at 8B–32B. The first surveyed method that solves to
    an explicit budget rather than an average bits-per-weight —
    that narrows one line of quantfit's differentiation. What no
    surveyed method ships remains the same: provenance-carrying
    artifacts for a specific card, a validation pass, and packed
    evidence.

**[Mixed-precision quantization for language models: techniques and prospects](https://arxiv.org/abs/2510.16805)** (survey, 2025)
:   The field map. It confirms two things about the landscape.
    Sensitivity-driven bit allocation is an active research area.
    The published work targets research benchmarks — none of the
    surveyed methods ship budget-solved, provenance-carrying
    artifacts for a specific card. The gap quantfit aims at is
    engineering-shaped, not algorithm-shaped.

## Target runtime

**[vLLM](https://github.com/vllm-project/vllm)** /
**[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)**
:   Planned ≥4-bit pack target and its mixed-precision checkpoint
    format ([ADR-0004](../adr/0004-vllm-first-runtime.md) as amended). The
    runtime-capability table (which precisions have kernels) comes from
    here. Verified 2026-07, re-verified 2026-08-04: the Marlin and
    Machete kernels cover 4-bit and
    8-bit only — 2026 upstream work widened *hardware* coverage
    (Marlin on Turing+), not bit widths, so the ADR-0013 table
    stands. Upstream closed the EXL3 integration request
    ([vllm#19896](https://github.com/vllm-project/vllm/issues/19896)) as
    not planned. Upstream also moved in-tree GGUF support out to
    [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin),
    whose tested formats stop at 4 bits. On that evidence
    [ADR-0010](../adr/0010-sub-4-bit-serving-path.md) routes sub-4-bit
    recipes through the GGUF serving path.

## Target model

**[Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)** (NVIDIA)
:   The north-star target ([ADR-0003](../adr/0003-north-star-benchmark.md)):
    dense decoder Transformer derived from Llama 3.3 70B via Neural
    Architecture Search — layers are structurally heterogeneous, which
    strengthens the case for per-layer measurement. NVIDIA publishes an
    official [NVFP4 quant](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4)
    (~4-bit uniform, still over a 24 GiB card) — a quality baseline for
    quantfit recipes to beat. Nemotron Nano is the "just run a smaller
    model" baseline.

## Writing system

**[ASD-STE100](https://www.asd-ste100.org/)**
:   Simplified Technical English — the controlled-language standard distilled
    into [CLAUDE.md](../../CLAUDE.md)'s writing rules. Free download from the
    official site.
