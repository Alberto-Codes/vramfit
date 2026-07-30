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
:   The JSON output of `quantfit scan`: per layer group, per candidate
    precision, the measured damage. Schema in
    [sensitivity map format](sensitivity-map.md). Not "scan results",
    "profile", or "analysis".

**Recipe**
:   The JSON output of `quantfit plan`: one precision assignment per layer
    group plus budget accounting. Schema in [recipe format](recipe.md). Not
    "config", "plan file", or "quant scheme".

**Packed model**
:   The output of `quantfit pack`: a checkpoint a target runtime can serve,
    produced by applying a recipe. Not "quantized model" (ambiguous — every
    stage quantizes something).

## Measurement

**Layer group** (short: **group**)
:   The unit of scanning and precision assignment — a named set of tensors
    quantized together (e.g. one layer's attention projections). Granularity
    set by `--group-by`.

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

**Validation pass**
:   The whole-recipe check of the additivity assumption (`quantfit
    validate`): quantize every group to its recipe-assigned precision in
    one pass through the scan's own quantization, then compare the
    measured damage against the recipe's summed marginal damages.
    Committed in [ADR-0006](../adr/0006-sensitivity-metric.md). Not
    "verification" or "recipe eval" — evaluation of *packed* models is
    a different step.

**Reference**
:   The unquantized (bf16) model that perturbed models are compared against.

**Group spec**
:   A discovered layer group before measurement: name, member tensors,
    and size at reference precision. Code type
    `quantfit.domain.scan.GroupSpec`.

**Damage meter**
:   The port that measures one group's damage at one precision
    (`quantfit.ports.outbound.DamageMeter`). The torch adapter
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
    grouping, precisions, and within-group method. It identifies
    provenance, not content — swapping weights or calibration text
    under an unchanged path defeats it.

## Budgeting

**Precision**
:   Bits per weight for a group (8, 4, 3, 2). Not "quant level" or
    "bit depth".

**Weight budget**
:   VRAM available for weights: card total minus KV headroom minus runtime
    overhead. What the solver packs against. Math in
    [VRAM budget math](../explanation/vram-budget.md).

**KV headroom**
:   VRAM reserved for the KV cache (and growth) at the planned context
    length and concurrency. CLI flag `--kv-headroom`.

**Pin**
:   A user-forced precision for a group, overriding the solver
    (`--pin "layers.0.*=8"`). Recorded verbatim in the recipe.

**Solver**
:   The algorithm that assigns precisions under the weight budget. Strategy
    tracked in [ADR-0007](../adr/0007-recipe-solver-strategy.md).

**Target runtime**
:   The serving stack a recipe is planned for, recorded in the recipe's
    `runtime` field (`--runtime`, default `llama.cpp`). Not "backend"
    (that word belongs to pack adapters) or "engine". Decided in
    [ADR-0013](../adr/0013-runtime-capability-in-recipes.md).

**Runtime capability**
:   The set of nominal precisions a target runtime can serve. The solver
    filters its candidate set through the capability table
    (`quantfit.domain.runtime.RUNTIME_CAPABILITIES`) so a recipe never
    assigns a precision its target runtime lacks kernels for.

**Trace**
:   The solver's ordered downgrade log, recorded in `plan.trace`. Replaying
    it from the starting state reproduces the assignments — it is the
    recipe's explanation. Not "log" or "history".

**Effective bits**
:   Bits per weight a quantization type really stores, block scales
    included — `Q4_K` spends 4.5 effective bits on a nominal 4-bit
    assignment. Recorded per runtime in
    `quantfit.domain.runtime.EFFECTIVE_BITS`
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
    positional argument. The solver prices these types at their
    effective bits (ADR-0014), and pack re-checks real sizes.

**Base GGUF**
:   The full-precision (f16) GGUF conversion of the source checkpoint
    that `llama-quantize` consumes. Created once per model, reused
    across packs. Not "intermediate file".

**Type override**
:   One (tensor pattern → quantization type) pair driven into the
    runtime's quantizer. One per layer group, first match wins. Code
    type `quantfit.domain.pack.TypeOverride`.

**Pack result**
:   The pack step's accounting record: real packed bytes plus the type
    mapping driven into the quantizer — base type, embedding and
    output-head flag types, the pattern overrides, and the importance
    matrix path when one was used. Code type
    `quantfit.domain.pack.PackResult`.

**Importance matrix** (short: **imatrix**)
:   Per-weight activation statistics collected over a calibration run,
    consumed by the runtime's quantizer to weight its block fit
    (`llama-quantize --imatrix`). Generated in v1 by `llama-imatrix`
    against the base GGUF over the scan's calibration set
    ([ADR-0016](../adr/0016-imatrix-in-the-pack-path.md)). Not
    "calibration data" (that names the text) or "activation cache".

**Smoke test**
:   The post-pack proof that a packed model emits language: a few
    perplexity chunks through the target runtime, gated by a
    perplexity ceiling
    ([ADR-0017](../adr/0017-post-pack-smoke-test.md)). Enabled by
    `quantfit pack --smoke-text`. Not "sanity check" or "quick eval" —
    the evaluation tiers are a different step.

## Architecture

**Domain**
:   The pure core (`quantfit.domain`): artifact types, budget math,
    solver. No IO, no frameworks — enforced by import-linter
    ([ADR-0008](../adr/0008-hexagonal-architecture.md)).

**Port**
:   A `typing.Protocol` in `quantfit.ports` naming a capability the
    application needs (e.g. `RecipeSink`). Outbound (driven) only today.

**Adapter**
:   An implementation touching the outside world. **Inbound** adapters
    drive the domain (the CLI); **outbound** adapters implement ports
    (JSON artifact files, HF configs).

**Envelope**
:   The serialization-level wrapper owned by the JSON adapters —
    notably the `quantfit_schema` version field, which domain objects
    never carry. Run logs carry `quantfit_runlog` instead, versioning
    one event line rather than a whole document.

## Project

**North-star benchmark**
:   The acceptance test from
    [ADR-0003](../adr/0003-north-star-benchmark.md), as amended by
    [ADR-0010](../adr/0010-sub-4-bit-serving-path.md): Nemotron Super
    49B serving on a 24 GiB RTX 4090 via llama.cpp at 16k context, with
    measured damage lower than the size-matched heuristic GGUFs.

**Reference box**
:   The development machine the benchmark runs on: RTX 4090 (24 GiB),
    124 GB system RAM.
