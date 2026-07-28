# Architecture Decision Records

Significant design decisions are recorded here, one file per decision,
numbered in order of creation. Records are immutable once Accepted — a
change of course gets a *new* ADR that supersedes the old one.

## Lifecycle

| Status | Meaning |
|--------|---------|
| **Proposed** | Under consideration; the open questions are listed in the record. |
| **Accepted** | Decided; the codebase should conform to it. |
| **Deprecated** | No longer applies, without a direct replacement. |
| **Superseded by ADR-NNNN** | Replaced by a newer decision. |

## Records

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-selective-per-layer-quantization.md) | Selective per-layer quantization as the core approach | Accepted |
| [0002](0002-scan-plan-pack-pipeline.md) | Three-stage scan → plan → pack CLI pipeline | Accepted |
| [0003](0003-north-star-benchmark.md) | North-star benchmark: Nemotron Super 49B on a 24 GiB RTX 4090 | Accepted |
| [0004](0004-vllm-first-runtime.md) | vLLM as the first target runtime | Accepted |
| [0005](0005-heavy-deps-as-extras.md) | Heavy ML dependencies stay out of the base install | Accepted |
| [0006](0006-sensitivity-metric.md) | Sensitivity metric for the scan step | Proposed |
| [0007](0007-recipe-solver-strategy.md) | Solver strategy for recipe selection | Accepted |

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
