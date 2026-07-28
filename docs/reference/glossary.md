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
    sum of marginal damages. Known blind spot; checked by a whole-recipe
    validation pass.

**Reference**
:   The unquantized (bf16) model that perturbed models are compared against.

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

**Trace**
:   The solver's ordered downgrade log, recorded in `plan.trace`. Replaying
    it from the starting state reproduces the assignments — it is the
    recipe's explanation. Not "log" or "history".

**Format overhead**
:   The fraction added to predicted sizes for quantization metadata
    (scales, zero-points). CLI flag `--format-overhead`, recorded in
    `plan.format_overhead`. Default 0.05 until measured per format.

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
    never carry.

## Project

**North-star benchmark**
:   The acceptance test from
    [ADR-0003](../adr/0003-north-star-benchmark.md): Nemotron Super 49B
    serving on a 24 GiB RTX 4090 via vLLM at 16k context with measured
    quality loss. [ADR-0010](../adr/0010-sub-4-bit-serving-path.md)
    (Proposed) would change the serving runtime to llama.cpp.

**Reference box**
:   The development machine the benchmark runs on: RTX 4090 (24 GiB),
    124 GB system RAM.
