# ADR-0005: Heavy ML dependencies stay out of the base install

- **Status:** Accepted
- **Date:** 2026-07-27

## Context

The pipeline will eventually need torch, transformers, safetensors, and
runtime-specific packages — multi-GB installs with CUDA wheels. But the CLI
skeleton, artifact formats, and the plan-step solver are pure Python; forcing
a torch download to inspect a recipe JSON or run the test suite would make
development and CI needlessly heavy. Sibling projects (docvet) follow the
same principle: minimal runtime deps, everything else optional.

## Decision

The base install carries **typer only**. GPU-touching functionality lands
behind optional extras (planned: `quantfit[scan]`, `quantfit[pack]`), and the
default test suite must pass on a CPU-only machine — GPU-dependent tests
carry the `gpu` marker and skip cleanly without CUDA.

## Consequences

- CI stays fast and GPU-free; contributor setup is seconds, not tens of
  minutes.
- The plan step must remain importable without torch — which enforces a clean
  boundary between the solver (pure math on JSON) and the model-touching
  stages, an architecture win beyond install hygiene.
- Import-guarding at the extras boundary is ongoing discipline; a stray
  top-level torch import breaks the contract silently until CI catches it.
