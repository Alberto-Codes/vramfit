---
status: draft
---

# Glossary

One term per concept, one concept per term. These are the project's canonical
nouns — docs, code, commit messages, and conversations use these words and no
synonyms for them. Coin a new term only with a new glossary entry in the same
change.

## Pipeline artifacts

**Sensitivity map**
:   The JSON output of `vramfit scan`: per layer group, per candidate
    precision, the measured damage. Schema in
    [sensitivity map format](sensitivity-map.md). Not "scan results",
    "profile", or "analysis".

**Recipe**
:   The JSON output of `vramfit plan`: one precision assignment per layer
    group plus budget accounting. Schema in [recipe format](recipe.md). Not
    "config", "plan file", or "quant scheme".

**Packed model**
:   The output of `vramfit pack`: a checkpoint a target runtime can serve,
    produced by applying a recipe. Not "quantized model" (ambiguous — every
    stage quantizes something).

**Evals sidecar**
:   The versioned JSON artifact that records a packed model's evaluation
    results (all three scoreboard tiers, their settings, and the toolchain
    that produced them), published beside the weights
    ([ADR-0025](../adr/0025-evals-sidecar.md)). Not "eval log" (the raw
    tool output) or "benchmark report".

**Analysis artifact**
:   The JSON record of a derivation across two or more evaluated
    artifacts, with the method, the input hashes, the results, and the
    derived per-chunk values (ADR-0025 dated note, 2026-08-10). One
    sidecar describes one artifact — a cross-artifact derivation lands
    here. Distinct from the bare word "analysis", a banned synonym for
    the sensitivity map.

## Measurement

**Layer group** (short: **group**)
:   The unit of scanning and precision assignment — a named set of tensors
    quantized together (e.g. one layer's attention projections). Granularity
    set by `--group-by`: `layer`, `stack`, or `tensor`.

**Expert stack**
:   One mixture-of-experts projection's experts, addressed as a single
    unit. llama.cpp fuses them into one tensor that carries one
    quantization type, so a pack assigns one precision to the whole
    stack (#159). The `stack` value of `--group-by` keys the
    sensitivity map on this unit (#161). Always write the term in
    full — the bare word "stack" already means the model's layer
    stack (see **Fit collapse**) and the serving stack (see **Target
    runtime**). Not "expert group" or "fused experts".

> **Ruled 2026-08-14 (#212).** "Stack cell" is the allowed cell
> form — see **Stack cell**. The full compound "expert-stack cell"
> stacks three nouns, so the cell form drops the first word. The
> carve-out covers this compound only, and the bare word "stack"
> stays reserved. ADR-0026's 2026-08-13 (#200) amendment writes
> the form and conforms. ADR-0026 decision 2 (demoted 2026-08-14)
> prices a stack as one cell and never writes the compound.

**Routing mass**
:   The share of a layer's imatrix counts held by the experts a recipe
    assigns one precision. It measures what a bit budget reaches, not
    how many experts it covers. On Nemotron 3.5 Lightning 30B-A3B, a
    budget that puts 18.5 % of expert parameters at the wider type
    reaches 18.5 % of routing mass without a split, and 31.9 % with
    one (#167). Not "routing share" or "expert mass".

**Stack cell**
:   A measurement cell whose group is one expert stack, as the
    `stack` value of `--group-by` produces (#161). The 2026-08-13
    (#200) amendment of ADR-0026 attributes damage inside one by
    slice perturbation. The #212 ruling under **Expert stack**
    allows the compound. A **slice cell** measures a dim-0 slice
    of the stack instead. Not "expert-stack cell" or "fused cell".

**Slice cell**
:   A measurement cell over a dim-0 slice of a fused expert stack.
    The meter quantizes the slice in place and keeps every other
    weight at reference precision (ADR-0026's 2026-08-13 (#200)
    amendment). A slice cell ranks or weights in the scan frame, and
    its damage number never sets a recipe's price (ADR-0021
    decision 2). Its two forms are the single-expert slice, which
    the probe measures, and the band slice. Not "sub-cell" or
    "partial group".

**Per-expert probe** (short: **probe**)
:   The stratified single-expert measurement that tests whether an
    expert's damage tracks its imatrix count on a model. One slice
    cell per sampled expert, at one precision. A refutation demotes
    ADR-0026 decision 2 and the recipe stays unweighted. Not
    "spot check" or "sensitivity sample".

**Probe arm**
:   A packed arm built to answer one allocation question, named by
    its ticket. The #321 probe arm places all 11 cheapest-width
    stacks on `down_proj`, one per layer. Distinct from the
    per-expert probe, which is a scan-frame measurement.

**Frequency band** (short: **band**)
:   A contiguous run of one expert stack's experts, ordered by
    imatrix count. The meter measures a band as one slice cell, and
    a split pack realizes a band as its own expert stack (ADR-0026's
    2026-08-12 (#167) amendment). Not "expert cluster" or "tier".

**Sensitivity**
:   A group's measured fragility: how much damage quantizing *it alone* to a
    given precision causes. The property the scan measures.

**Damage**
:   The divergence score itself — the number recorded per (group ×
    precision). Higher is worse. Metric choice tracked in
    [ADR-0006](../adr/0006-sensitivity-metric.md).

**Damage curve**
:   One group's damage as a function of precision — the cost function the
    solver consumes.

**Calibration set**
:   The text run through the model to measure damage. Damage values are only
    comparable within one calibration set.

**Marginal scanning**
:   Measuring one group at a time while the rest stays at reference
    precision. Implies the **additivity assumption**: total recipe damage ≈
    sum of marginal damages. Known blind spot; checked by the validation
    pass.

**Within-group method**
:   How the meter quantizes the weights inside a perturbed group.
    v1 is round-to-nearest with 32-element scale blocks
    ([ADR-0006](../adr/0006-sensitivity-metric.md), token
    `rtn-block32`). `kquant-ref` round-trips through the ported
    llama.cpp reference quantizers
    ([ADR-0018](../adr/0018-kquant-within-group-method.md)).
    `kquant-imx` is the same port with assisted pricing — the map
    then also records the imatrix path in `scan.imatrix`
    ([ADR-0020](../adr/0020-imatrix-assisted-pricing.md)).
    [ADR-0021](../adr/0021-runtime-frame-measurement.md) supersedes
    ADR-0020, and the method stays a valid scan option.
    `q0-ref` ports `Q2_0` and `Q4_0`, the block quantizers
    `llama-quantize` applies where no K-quant reaches a row. It covers
    nominal 8 as well, reusing ADR-0018's `Q8_0` port, so the method
    reads 8, 4, and 2
    ([ADR-0018](../adr/0018-kquant-within-group-method.md), 2026-08-17
    amendment, token renamed by the 2026-08-18 amendment). `q0-imx` is
    its assisted path: nominal 4 fits with imatrix weights through
    the ported `quantize_row_q4_0_impl`, and nominal 2 and 8 keep
    the reference arithmetic. The 2026-08-21 amendment rules the
    build.
    A method
    change is a new scan — the token lives in the fingerprint and in
    the map's `scan.within_group`, and the recipe carries its map's
    token for the validation pass. Not "quantization mode" or
    "simulation method".

**Assisted pricing**
:   Measuring a cell with the pack's imatrix weighting the
    within-group fit, through the ported `_impl` quantizers.
    `kquant-imx` carries it for the K-quants
    ([ADR-0020](../adr/0020-imatrix-assisted-pricing.md)).
    [ADR-0021](../adr/0021-runtime-frame-measurement.md) supersedes
    ADR-0020, and the method stays a valid scan option.
    **Unassisted** names the reference-path fit without weights.
    A tensor the imatrix does not cover always prices unassisted —
    the same fallback `llama-quantize` applies. `q0-imx` carries
    the same shape through `quantize_row_q4_0_impl`, and its reader
    accepts fused expert stacks — one imatrix row per expert
    ([ADR-0018](../adr/0018-kquant-within-group-method.md),
    2026-08-21 amendment). `Q2_0` has no assisted path, because
    `quantize_q2_0` ignores the matrix (ADR-0018, 2026-08-17
    amendment, token renamed by the 2026-08-18 amendment). Not
    "imatrix mode" or "weighted scanning".

    The same pair names the pack side. A tensor packs **assisted** when
    its type reads the matrix and the matrix covers its name, and
    **unassisted** otherwise. An artifact's **assisted share** is the
    fraction of its bytes that packed assisted
    ([ADR-0016](../adr/0016-imatrix-in-the-pack-path.md), 2026-08-21
    amendment). The scan sense and the pack sense can disagree on one
    tensor, because the two sides apply different fits.

**Reconstruction error**
:   The squared difference between a quantized tensor and its
    original, at `||q - w||² / ||w||²`, measured in weight space. It
    compares two perturbations against each other. It never enters a
    sensitivity map and it sets no price. **It is not damage** — #302
    measured a weight-space term and measured damage ordering apart on
    the 30B target. Not "quantization error" or "damage".

**Validation pass**
:   The whole-recipe check of the additivity assumption (`vramfit
    validate`): quantize every group to its recipe-assigned precision in
    one pass through the scan's own quantization, then compare the
    measured damage against the recipe's summed marginal damages.
    Committed in [ADR-0006](../adr/0006-sensitivity-metric.md). Not
    "verification" or "recipe eval" — evaluation of *packed* models is
    a different step.

**Reference**
:   The unquantized (bf16) model that perturbed models are compared against.

**bf16**
:   The source checkpoint's storage format, and the format the
    **Reference** holds. Two bytes per parameter. The scan measures
    against it, and `bytes_fp16` records a group's size in it.

**f16**
:   IEEE 754 binary16, and the **Base GGUF**'s storage format. Also two
    bytes per parameter. It is a different format from bf16, at a
    different exponent and mantissa split, and it lives in a different
    file. Say "f16" for the GGUF and "bf16" for the checkpoint.

**fp16**
:   Another spelling of **f16**. It survives here only in the artifact
    field `bytes_fp16`, which despite its name holds **bf16** bytes at
    two bytes per parameter. So the field is named for one format and
    records another, and the two happen to agree on width. Prefer
    "f16" for the format and "reference precision" for what the field
    holds. Renaming the field would bump `vramfit_schema` and break the
    published maps, so the name stays and this entry carries the
    meaning (#357).

**Instrument**
:   The execution half of a measurement frame, hardware and compute
    stack together: the accelerator, its streaming-multiprocessor
    count, the stack's **build identity**, and the offload split
    ([ADR-0027](../adr/0027-instrument-frame-matching.md)).
    The build identity is the torch build in the scan frame and the
    llama.cpp release in the runtime frame.
    The instrument fixes the frame's numerics. cuBLAS repeats
    bitwise only on one architecture and SM count under one
    toolkit, so an H100 and a 4090 are two instruments.
    An **instrument check** re-measures hold-out cells across
    instruments ([ADR-0021](../adr/0021-runtime-frame-measurement.md)
    decision 3, [ADR-0027](../adr/0027-instrument-frame-matching.md)
    decision 4). Not "device", "card", or "machine".

**Measurement frame** (short: **frame**)
:   The whole apparatus a damage number is measured inside: process,
    quantization path, calibration text, token count, and
    instrument.
    Damage values compare only within one frame — cross-process
    re-measurement of identical cells moved values 2.7–4.1x (the
    [ninth data point](../explanation/evaluating-packed-models.md)).
    The **scan frame** is the meter's apparatus:
    perturb weights inside the bf16 model, measure calibration KL.
    The **runtime frame** is the packed artifact under the runtime's
    own numerics ([ADR-0021](../adr/0021-runtime-frame-measurement.md)).
    **Frame-matched** describes a comparison run entirely inside one
    frame. Not "environment", "setup", or "context".

**Frame transfer**
:   The leap from an in-frame price to packed behavior. The
    [tenth data point](../explanation/evaluating-packed-models.md)
    isolated frame transfer as the leak behind the
    sub-4-bit losses: every scan-frame refinement improved in-frame
    prices and worsened the packed artifact
    ([ADR-0021](../adr/0021-runtime-frame-measurement.md)).

**Runtime-frame lane**
:   The measurement path that prices damage in the runtime frame.
    The lane quantizes the candidate group to its real packed type
    inside a real GGUF. It measures damage under the runtime's own
    numerics ([ADR-0021](../adr/0021-runtime-frame-measurement.md)
    decision 2, #40). The solver buys a width only against this
    lane's price (ADR-0021 decision 4). It names what a
    measurement runs through, not where it runs. Always write the
    term in full (the #213 ruling, under **Harness lane**). Not
    "packed lane" or "GGUF lane".

**Rented-GPU lane**
:   The measurement path on a rented card that holds the checkpoint
    resident, so scan, validate, imatrix, and evals run without
    offload (#40). #163 proved it on the 30B checkpoint. Its damage
    magnitudes never cross instruments, and an ordering crosses
    only after the map passes ADR-0027 decision 4's ordering bar.
    It names where a measurement runs, not what it runs through.
    Always write the term in full (the #213 ruling, under
    **Harness lane**). Not "rented measurement lane" or "cloud
    lane".

**Group spec**
:   A discovered layer group before measurement: name, member tensors,
    and size at reference precision. Code type
    `vramfit.domain.scan.GroupSpec`.

**Damage meter**
:   The port that measures one group's damage at one precision
    (`vramfit.ports.outbound.DamageMeter`). The torch adapter
    implements it behind the `scan` extra.

**Offloaded group**
:   A layer group whose weights `auto` sharding moved off the GPU to
    host RAM under the `--gpu-memory` cap. The meter perturbs it
    through accelerate's weights map
    ([ADR-0015](../adr/0015-offload-aware-scanning.md)). Not
    "swapped", "spilled", or "CPU group".

**Scan checkpoint**
:   The incremental record of finished (group x precision) cells that
    makes a scan resumable. Written next to the map as
    `<stem>.checkpoint.json`. Not "cache" or "state file".

**Run log**
:   The machine-readable event stream of one pipeline run: JSON Lines
    beside the run's artifacts (`<stem>.runlog.jsonl`), one versioned
    event per line, each stamped with the run's ``run_id``. The
    machine channel — human CLI output is the human channel and never
    mixes with it. Decided in
    [ADR-0011](../adr/0011-run-logs-and-error-root.md).

**Fingerprint**
:   The identity string that ties a scan checkpoint to one scan's
    recorded provenance: model, metric, calibration, token count,
    grouping, precisions, within-group method, and imatrix path. It
    identifies
    provenance, not content — swapping weights or calibration text
    under an unchanged path defeats it.

## Budgeting

**Precision**
:   Bits per weight for a group (e.g. 8, 4, 3, 2 — protection
    floors also use 5 and 6). Not "quant level" or "bit depth".

**Passthrough precision**
:   Nominal 16, the precision that holds a group at **reference**
    and packs it as GGUF `F16`
    ([ADR-0029](../adr/0029-plan-independent-size-source.md)
    decision 4). It spends exactly 16.0 effective bits, because
    `F16` carries no block scale. A recipe assigns it to every
    **uncovered group** without a pin
    ([ADR-0007 amendment 2026-08-22](../adr/0007-recipe-solver-strategy.md)).
    Not "no-quant" or "fp16 pin".

**Tensor size source**
:   The port that reads a checkpoint's per-tensor stored sizes for
    the plan step (`vramfit.ports.outbound.TensorSizeSource`). The
    safetensors adapter implements it by parsing each shard header,
    so `plan` needs no torch. It exists so the sensitivity map stops
    defining the model (ADR-0029).

**Discovered group**
:   A group the tensor size source finds in the checkpoint. The
    domain names it with the same rule the scan uses, so the two
    agree on what a group is. The solver prices every discovered
    group.

**Uncovered group**
:   A discovered group the sensitivity map does not measure. It
    prices at reference precision, and the recipe assigns it the
    **passthrough precision**. `pack` quantizes at the recipe's floor,
    so an unnamed group would reach the artifact below the bytes the
    plan reserved. It carries no damage curve, so the
    solver never downgrades it. Distinct from an **uncovered
    tensor**, which the importance matrix missed, and from a
    **floored layer**, which no override in the recipe reaches. The
    three name gaps between different pairs: the map and the model,
    the matrix and the model, the recipe and the model. Always the
    two-word compound. Not "missing group" or "unscanned group".

**Naming root**
:   The first segment of a checkpoint's parameter names —
    `backbone.` on the 30B target, `model.` on a llama-family
    checkpoint. Maps root at `model.`, so a domain table reconciles
    the two before a tensor reaches a group. The table is explicit
    and never a prefix wildcard, which once mapped a vision tower's
    `layers.5` onto the decoder's `blk.5` (#177). Not "prefix" or
    "namespace".

**Weight budget**
:   VRAM available for weights: card total minus KV headroom minus runtime
    overhead. What the solver packs against. Math in
    [VRAM budget math](../explanation/vram-budget.md).

**Ballast**
:   A CUDA allocation one process holds for a run's duration, so every
    other process on the card sees less free VRAM. It makes a 24 GiB
    card serve a test at a smaller size. `scripts/vram_ballast.py`
    holds it ([#164](https://github.com/Alberto-Codes/vramfit/issues/164)).

**Visible free VRAM**
:   Free VRAM a runtime reports under a ballast. It is what the ballast
    caps, and it is not the weight budget — the KV cache and runtime
    overhead still come out of it.

**KV headroom**
:   VRAM reserved for the KV cache (and growth) at the planned context
    length and concurrency. CLI flag `--kv-headroom`.

**KV layer**
:   One attention layer's KV-cache geometry: KV-head count, head
    width, window, storage factor, and whether the layer allocates
    KV at all (`vramfit.domain.budget.KVLayer`, #421). A shape is a
    tuple of KV layers. Not "layer config" or "cache entry".

**KV growth**
:   Bytes each context token adds across the global layers
    (`kv_growth_bytes_per_token`). Sliding layers stop at their
    window and contribute to the **window pool** instead, so growth
    alone never prices a mixed stack. Not "bytes per token" without
    the qualifier.

**Window pool**
:   The KV bytes the sliding layers hold once every window is
    saturated (`kv_window_pool_bytes`). A constant per sequence:
    800 MiB on Gemma 4 31B at fp16. Not "sliding cache" or "SWA
    buffer".

**Storage factor**
:   KV tensors a layer stores per cached token: 2 for an independent
    K and V pair, 1 when the model reuses one tensor for both
    (`attention_k_eq_v`, field `kv_tensors`). Not "KV factor".

**Pin**
:   A user-forced precision for a group, overriding the solver
    (`--pin "*.layers.0=8"`). Recorded verbatim in the recipe. A pin
    may name any width the target runtime serves, and it may land on
    any checkpoint-discovered group
    ([ADR-0007 amendment 2026-08-22](../adr/0007-recipe-solver-strategy.md)).
    At a width the map never measured, the assignment records damage
    0.0.

**Protection**
:   A precision floor for named tensors inside their layer groups
    (`--protect "*.self_attn.v_proj.weight=5"`). A protected tensor
    packs at its **protection floor** where the floor exceeds the
    group's assignment — distinct from the recipe's base-type floor
    (ADR-0012). Priced by size only, never by damage
    ([ADR-0022](../adr/0022-within-layer-protections.md)). Not
    "tensor pin" or "override" (that word belongs to pack).

**Per-tensor no-op**
:   A protected pair whose floor the group assignment already
    meets. The pair would quantize identically to the unprotected
    reference, and the reconstruction check would read the tie as
    a collapse. `plan` drops the pair and warns per tensor
    (issue #59). Its imatrix exclusion drops with it. Not a
    **dead rule** — that names a whole pattern.

**Dead rule**
:   A protection pattern that changed nothing: every tensor it
    governs already meets the floor, or a later rule overrides it.
    `plan` warns once per dead rule (ADR-0022). Not "no-op
    pattern" — the per-tensor case has its own term.

**Solver**
:   The algorithm that assigns precisions under the weight budget. Strategy
    tracked in [ADR-0007](../adr/0007-recipe-solver-strategy.md).

**Spread placement rule** (short: **placement rule**)
:   The solver's constraint on downgrades that take an expert stack
    to the cheapest in-budget width — the 2026-08-21 amendment in
    [ADR-0007](../adr/0007-recipe-solver-strategy.md). Clause 1: no
    layer takes a second cheapest-width stack while a layer that can
    still take one has none. Clause 2: within a layer, the projection
    the map prices cheaper at that width goes first. The rule narrows
    the candidates and the selection key is unchanged. Dense groups
    keep the plain damage-per-byte order. Lives in
    `vramfit.domain.placement`.

**Target runtime**
:   The serving stack a recipe is planned for, recorded in the recipe's
    `runtime` field (`--runtime`, default `llama.cpp`). Not "backend"
    (that word belongs to pack adapters) or "engine". Decided in
    [ADR-0013](../adr/0013-runtime-capability-in-recipes.md).

**Runtime capability**
:   The set of nominal precisions a target runtime can serve. The solver
    filters its candidate set through the capability table
    (`vramfit.domain.runtime.RUNTIME_CAPABILITIES`) so a recipe never
    assigns a precision its target runtime lacks kernels for.

**Trace**
:   The solver's ordered downgrade log, recorded in `plan.trace`. Replaying
    it from the starting state reproduces the assignments — it is the
    recipe's explanation. Not "log" or "history".

**Effective bits**
:   Bits per weight a quantization type really stores, block scales
    included — `Q4_K` spends 4.5 effective bits on a nominal 4-bit
    assignment. Recorded per runtime in
    `vramfit.domain.runtime.EFFECTIVE_BITS`
    ([ADR-0014](../adr/0014-per-type-effective-bits.md)). The solver
    prices sizes at effective bits when the target runtime has a
    table.

**Format overhead**
:   The fraction added to predicted sizes for what the size model does
    not price in. CLI flag `--format-overhead`, recorded in
    `plan.format_overhead`. With an effective-bits table it covers
    unquantized tensors and file metadata (default 0.005). Without
    one it also covers scales and zero-points (default 0.05).

## Packing

**Type mapping**
:   The pack backend's translation from nominal precision to a runtime
    quantization type. [ADR-0012](../adr/0012-gguf-type-mapping.md)
    fixes the GGUF tensor-type table (8→Q8_0, 4→Q4_K, 3→Q3_K,
    2→Q2_K) and a separate base-ftype table for the quantizer's
    positional argument. A routed-expert stack maps through its own
    table instead (8→Q8_0, 4→Q4_0, 2→Q2_0), and the backend refuses
    nominal 3 there
    ([ADR-0028](../adr/0028-expert-stack-type-table.md)). The
    solver prices these types at their effective bits (ADR-0014),
    and pack re-checks real sizes.

**Base GGUF**
:   The full-precision (f16) GGUF conversion of the source checkpoint
    that `llama-quantize` consumes. Created once per model, reused
    across packs. Not "intermediate file".

**Type override**
:   One (tensor pattern → quantization type) pair driven into the
    runtime's quantizer. One per layer group plus one per protected
    tensor, protections first, first match wins (ADR-0022). Code
    type `vramfit.domain.pack.TypeOverride`.

**Pack result**
:   The pack step's accounting record: real packed bytes plus the type
    mapping driven into the quantizer — base type, embedding and
    output-head flag types, the pattern overrides, the importance
    matrix path when one was used, the imatrix coverage record
    (uncovered tensors, and the exclusions the recipe instructed),
    the zero-count expert report — `(stack, expert)` pairs the
    matrix counts zero times
    ([ADR-0026](../adr/0026-moe-expert-pricing.md) decision 5) — and
    the **floored layers**. Code type
    `vramfit.domain.pack.PackResult`.

**Floored layer**
:   A layer the base GGUF numbers that no override in the recipe
    reaches. It packs at the recipe's base-type floor and the
    quantizer reports nothing, so the pack step names it
    ([ADR-0012](../adr/0012-gguf-type-mapping.md), the 2026-08-16
    #307 amendment). Distinct from an **uncovered tensor**, which the
    importance matrix missed. One names a gap between the recipe and
    the model, the other a gap between the matrix and the model. Not
    "unaddressed layer" or "missing layer".

**Importance matrix** (short: **imatrix**)
:   Per-weight activation statistics collected over a calibration run,
    consumed by the runtime's quantizer to weight its block fit
    (`llama-quantize --imatrix`). Generated in v1 by `llama-imatrix`
    against the base GGUF over the scan's calibration set
    ([ADR-0016](../adr/0016-imatrix-in-the-pack-path.md)). Not
    "calibration data" (that names the text) or "activation cache".

**Imatrix entry**
:   One tensor's statistics inside an importance matrix, holding one
    row per matrix. A dense tensor holds one row. An **expert stack**
    holds one row per expert. The HF checkpoint spells those experts
    as separate parameters (#177). At load, transformers fuses them
    into one 3D parameter (#202). An entry carries the
    column weights and the **imatrix count** together. Not "imatrix
    row", which names one row inside an entry.

**Imatrix count**
:   The chunk tally an importance matrix records for one matrix. A
    routed expert's count is how often the router fired it over the
    calibration corpus, so it measures routing frequency. A count
    above zero divides the sums, and a count of zero weighs every
    column 1, which is the unassisted fit
    ([ADR-0026](../adr/0026-moe-expert-pricing.md) decision 1). Not
    "routing count" or "sample count".

**Fit collapse**
:   A quantizer failure mode under a fixed imatrix: an imatrix row
    with extreme column dynamic range (the collapsed rows span
    10⁸–10¹³) destabilizes the weighted super-block scale fit at
    `Q4_K`/`Q5_K`, and the tensor reconstructs 5–15× *worse* than
    its unweighted fit (`Q3_K` only inflates ~1.4×). Range alone
    does not decide — the reconstruction check does. A tensor in
    this state is **collapsed**.
    Discovered on the front-stack `attn_v` tensors by the twelfth
    data point in
    [evaluating packed models](../explanation/evaluating-packed-models.md).
    The fourteenth data point on the same page isolates the
    mechanism and fixes it by per-tensor imatrix exclusion.
    Not "quantizer bug" — the fit does what the weighting tells it.

**Imatrix exclusion**
:   The fit-collapse remedy that keeps the promotion. One named
    protected tensor quantizes without its imatrix row and takes
    the clean unweighted fit (`llama-quantize --exclude-weights`,
    5.8–14.7× cleaner on the collapsed rows). Marked per protected
    tensor in the recipe by `vramfit plan --exclude-imatrix`
    ([ADR-0023](../adr/0023-imatrix-exclusions.md)). Not "imatrix
    miss" — that names an unintentional coverage gap.

**Reconstruction check**
:   The per-tensor proof that a promoted tensor reconstructs
    closer to the f16 base than its unprotected type does:
    dequantize the packed tensor and compare (gguf-py, seconds of
    CPU). Guards protected packs made with an imatrix — fit
    collapse is invisible to the smoke test
    ([ADR-0022](../adr/0022-within-layer-protections.md)). Not
    "fit check" or "dequant test".

**Smoke test**
:   The post-pack proof that a packed model emits language: a few
    perplexity chunks through the target runtime, gated by a
    perplexity ceiling
    ([ADR-0017](../adr/0017-post-pack-smoke-test.md)). Enabled by
    `vramfit pack --smoke-text`. Not "sanity check" or "quick eval" —
    the evaluation tiers are a different step.

## Evaluation

**Blind recipe**
:   A recipe that assigns precision without reading a sensitivity map.
    It holds a comparison's size constant, and the count of groups at
    each precision, so the comparison isolates allocation. The
    comparison tests whether the scan and the solver earn their cost.
    A blind recipe is a scoreboard row and never a published claim. It
    uses the recipe schema ([recipe format](recipe.md)). Not "uniform
    pack" (the outside field's name for the same role, refused below).
    Not the "size-matched baseline" (a published build the recipe may
    beat in public).

> **Ruled 2026-08-15 ([#265](https://github.com/Alberto-Codes/vramfit/issues/265)).**
> The term exists because no uniform-expert pack sits at chart #158's
> budget. The palette holds no type between 2.25 and 4.25 bits per
> weight. Below the band, uniform `Q2_0` leaves 4.594 GiB of the 14.5
> GiB unspent. Above it, the 46 expert stacks alone at `MXFP4` cost
> 14.534 GiB, over the whole budget with every other tensor at zero
> bytes. The MoE quantization literature calls a bit-blind split at a
> matched budget "Uniform"
> ([AlphaQ](https://arxiv.org/abs/2606.04980)). This project refuses
> that word, because on this target it names an artifact that cannot
> exist. The maintainer judged a self-authored opponent contrived as a
> headline. The recipe's published claim runs against published
> artifacts only.

> **Corrected 2026-08-16 ([#284](https://github.com/Alberto-Codes/vramfit/issues/284)).**
> The arithmetic above is keyed to the superseded 14.5 GiB budget. The
> budget is 15.776 GiB. At that budget uniform `Q2_0` leaves 5.870 GiB
> unspent, and the 46 expert stacks at `MXFP4` cost 14.534 GiB, which
> sits inside the budget rather than over it.
> [#288](https://github.com/Alberto-Codes/vramfit/issues/288) rules
> whether the term survives. The ruling above stands until it does.

**Tier-3 slice**
:   The fixed set of lm-evaluation-harness tasks and few-shot settings
    that certifies a publication candidate's capabilities: five tasks at
    leaderboard settings
    ([ADR-0024](../adr/0024-tier3-task-slice.md)). Fixed before any run,
    so cards compare across candidates and time. Not "benchmark suite"
    or "eval suite".

**Harness lane**
:   The recorded software path that drives a packed model through the
    tier-3 slice: the harness, the binding, and the llama.cpp build
    acting as one instrument
    ([ADR-0024](../adr/0024-tier3-task-slice.md)). The evals sidecar
    records it in its toolchain block, under the field name `lane`.
    Not "backend" (the harness's own term for one piece of the path).

> **Ruled 2026-08-14 (#213).** The bare short form "lane" is
> struck. It named this lane and the rented-GPU lane, and ADR-0028
> writes the runtime-frame lane beside them. Each compound writes
> in full from here on. Records written before this ruling keep
> the short form, and the sidecar's `lane` field name stays.

## Publication

**Identity grammar**
:   The naming rule for a published vramfit artifact:
    `<family-stem>-fit<N>gib-GGUF` for the repository, the repo id
    minus `-GGUF` plus `.gguf` for the weight file, the repo id as
    the card's H1. Ruled on #401, recorded in
    [evaluating packed models](../explanation/evaluating-packed-models.md#the-identity-grammar-from-publication-2).
    Not "naming convention" or "repo format".

**Family stem**
:   The upstream repository name after the org namespace, with its
    variant suffix removed. For publication #2:
    `NVIDIA-Nemotron-3.5-Lightning-30B-A3B`, from
    `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. Not "base
    name" or "model name" (ambiguous with the checkpoint id).

**Serve contract**
:   The runtime and context configuration a model card states to
    substantiate a `fit<N>gib` claim: toolchain, backend, offload,
    context, cap method, and concurrency bound. The claim holds
    inside this contract and nowhere else. Not "serving config".

## Architecture

**Domain**
:   The pure core (`vramfit.domain`): artifact types, budget math,
    solver. No IO, no frameworks — enforced by import-linter
    ([ADR-0008](../adr/0008-hexagonal-architecture.md)).

**Port**
:   A `typing.Protocol` in `vramfit.ports` naming a capability the
    application needs (e.g. `RecipeSink`). Outbound (driven) only today.

**Adapter**
:   An implementation touching the outside world. **Inbound** adapters
    drive the domain (the CLI); **outbound** adapters implement ports
    (JSON artifact files, HF configs).

**Envelope**
:   The serialization-level wrapper owned by the JSON adapters —
    notably the `vramfit_schema` version field, which domain objects
    never carry. Run logs carry `vramfit_runlog` instead, versioning
    one event line rather than a whole document.

> **Ruled 2026-08-11 (#118).** Both envelope keys renamed with the
> tool (chart #114). The rename was a breaking change, and every
> schema version bumped with it, including the run-log version. The
> rename PR (#120) reads only the two keys defined above: one term
> per concept. The re-upload task (#121) then edited the published HF
> files in place, re-hashed, and re-uploaded them. A reader that meets
> a pre-rename key names that key in the error (#154).

**Optional root**
:   A top-level import name a base install cannot resolve, because an
    extra or a dependency group provides it
    ([ADR-0005](../adr/0005-heavy-deps-as-extras.md)). Six today: gguf,
    numpy, safetensors, tokenizers, torch, transformers. Not "heavy
    dep" or "optional import" — the term names the import root, never
    the distribution.

**Ty override**
:   One `[[tool.ty.overrides]]` block in pyproject.toml that silences
    `unresolved-import` for the files allowed to import an optional
    root. Gated by `scripts/check_ty_overrides.py`. Always the
    two-word compound — the bare word "override" belongs to pack's
    **Type override**.

## Planning

**Chart**
:   The GitHub issue that indexes one multi-session effort: a
    destination, standing notes, the decisions so far, the fog, and
    the ruled-out scope. Decision tickets are its sub-issues. An
    index, never a decision store. Convention in
    [charting](../explanation/charting.md). Not "map" (that word
    belongs to the sensitivity map), "epic", or "roadmap".

**Decision ticket**
:   A chart's sub-issue that resolves one question, sized to one
    agent session. It closes with a pointer to the record that
    holds the decision: an ADR, an amendment, a data point, or a
    docs change. The ticket never stores the decision itself. Typed
    by label: `chart:research`, `chart:prototype`, `chart:discuss`,
    `chart:task`. A `chart:task` ticket records completed work and
    resulting facts instead of a decision.

**Claimable set**
:   The decision tickets a session may claim right now: open,
    unblocked by open dependencies, and unassigned. Not "frontier"
    (that word belongs to the quality-size frontier on the
    scoreboard).

**Fog**
:   An in-scope question not yet sharp enough to ticket, listed in
    the chart's `## Fog` section. Graduates to a decision ticket
    when the question can be stated precisely — answerability is
    not the test. Not "open question" (that names a section in ADRs
    and docs pages).

## Project

**Deck**
:   A Marp slide set that argues the project's present state to one
    audience, exported to PDF as a release asset. Two decks exist: review
    for stakeholders, deep dive for peer reviewers. A deck names a release
    and carries no `status` field. Convention in
    [decks](../decks/index.md). Not "milestone deck", "slide deck", or
    "presentation".

**North-star benchmark**
:   The acceptance test from
    [ADR-0003](../adr/0003-north-star-benchmark.md), as amended by
    [ADR-0010](../adr/0010-sub-4-bit-serving-path.md): Nemotron Super
    49B serving on a 24 GiB RTX 4090 via llama.cpp at 16k context, with
    measured damage lower than the size-matched heuristic GGUFs.

**Reference box**
:   The development machine the benchmark runs on: RTX 4090 (24 GiB),
    124 GB system RAM.
