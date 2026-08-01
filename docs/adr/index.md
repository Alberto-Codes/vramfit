# Architecture Decision Records

Significant design decisions are recorded here, one file per decision,
numbered in order of creation. An Accepted record's decision body is
immutable. Revisions arrive as dated **Amendment** or **Note** bullets
in the header, and open questions stay live — they accrete
measurements and strikethroughs. A full change of course gets a *new*
ADR that supersedes the old one.

## Lifecycle

| Status | Meaning |
|--------|---------|
| **Proposed** | Under consideration; the open questions are listed in the record. |
| **Accepted** | Decided; the codebase should conform to it. |
| **Deprecated** | No longer applies, without a direct replacement. |
| **Superseded by ADR-NNNN** | Replaced by a newer decision. |
| **Amended by ADR-NNNN** | Still in force; a later record revises one named clause. |

## Records

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-selective-per-layer-quantization.md) | Selective per-layer quantization as the core approach | Accepted |
| [0002](0002-scan-plan-pack-pipeline.md) | Three-stage scan → plan → pack CLI pipeline | Accepted |
| [0003](0003-north-star-benchmark.md) | North-star benchmark: Nemotron Super 49B on a 24 GiB RTX 4090 | Accepted, amended by 0010 |
| [0004](0004-vllm-first-runtime.md) | vLLM as the first target runtime | Accepted, amended by 0010 + 2026-07-29 |
| [0005](0005-heavy-deps-as-extras.md) | Heavy ML dependencies stay out of the base install | Accepted, amended by 0011 |
| [0006](0006-sensitivity-metric.md) | Sensitivity metric for the scan step | Accepted |
| [0007](0007-recipe-solver-strategy.md) | Solver strategy for recipe selection | Accepted |
| [0008](0008-hexagonal-architecture.md) | Hexagonal architecture, enforced by import-linter | Accepted |
| [0009](0009-testing-strategy.md) | Testing strategy — pyramid, verified fakes, properties | Accepted |
| [0010](0010-sub-4-bit-serving-path.md) | The sub-4-bit serving path runs through GGUF | Accepted |
| [0011](0011-run-logs-and-error-root.md) | Run logs as artifacts, and one error root | Accepted |
| [0012](0012-gguf-type-mapping.md) | The GGUF backend maps nominal bits to K-quant types | Accepted, amended by 0013 + 2026-07-29 |
| [0013](0013-runtime-capability-in-recipes.md) | Recipes record their target runtime | Accepted, amended by 0014 |
| [0014](0014-per-type-effective-bits.md) | The solver predicts sizes from per-type effective bits | Accepted |
| [0015](0015-offload-aware-scanning.md) | The meter perturbs offloaded groups through accelerate's weights map | Accepted |
| [0016](0016-imatrix-in-the-pack-path.md) | Pack consumes an importance matrix | Accepted |
| [0017](0017-post-pack-smoke-test.md) | A packed model proves it emits language before anything trusts it | Accepted |
| [0018](0018-kquant-within-group-method.md) | A K-quant-faithful within-group method behind a scan flag | Proposed |
| [0019](0019-kquant-priced-maps.md) | Sub-4-bit recipes solve on kquant-priced maps | Proposed |

## Template

```markdown
# ADR-NNNN: Title

- **Status:** Proposed | Accepted | Deprecated | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD

## Context

What forces are at play; why a decision is needed.

## Decision

What we chose, stated in the active voice.

## Consequences

What becomes easier, what becomes harder, what we gave up.
```
