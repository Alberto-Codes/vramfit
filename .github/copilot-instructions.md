# Copilot instructions for quantfit

quantfit measures per-layer quantization sensitivity of LLMs (`scan`),
solves mixed-precision recipes under a VRAM budget (`plan`), and packs
checkpoints for vLLM (`pack`). Only the plan step is implemented so far.

## Project stances (do not flag these as issues)

- **Artifacts are rejected, not normalized.** Invalid JSON artifacts raise
  `ArtifactError` with a JSON path; the loaders never silently fix input
  (e.g. `scan.precisions` must arrive strictly descending).
- **The glossary is law** (`docs/reference/glossary.md`): one term per
  concept. "Recipe" (never "config"), "damage" (never "loss" or "error"),
  "sensitivity map" (never "scan results"). Naming that follows the
  glossary is intentional.
- **No heavy ML deps in the base install** (ADR-0005): torch/transformers
  land behind future extras. Enforced by an import-linter forbidden
  contract. Torch lands later as outbound adapters behind ports.
- **Docs carry statuses** (`sketch/draft/stable`; ADRs
  `Proposed/Accepted`). A `sketch` page describing unimplemented behavior
  is a design artifact, not a doc bug.
- **Determinism is a contract** in the solver: integer byte math, total
  ordering of moves. Flag anything that threatens reproducibility.

- **Hexagonal layers are enforced** (ADR-0008): `adapters/inbound` →
  `adapters/outbound` → `ports` → `domain`; the domain imports no
  json/pathlib/os/io/typer. Import-linter fails the build on violations,
  so layering is a fact, not a suggestion.
- **File size cap**: 300/320 lines of actual code per module
  (`scripts/check_loc.py`); large modules are decomposed by policy.

## Review focus

- Correctness edge cases (rounding, empty inputs, boundary budgets) —
  prior reviews found real ones; keep looking there.
- Anything that could make the same map + pins + overhead produce two
  different recipes.
- Silent acceptance of invalid input (configs, artifacts, CLI options).
- Test gaps for new failure paths; CI has no GPU, so GPU-only tests must
  carry the `gpu` marker and skip cleanly.

## Conventions

- Python 3.12+, `from __future__ import annotations` everywhere, no
  relative imports, no `__main__.py`.
- Google-style docstrings with typed Attributes and Examples sections —
  enforced by docvet in CI; missing sections are build failures.
- Conventional commits `type(scope): description`; scopes:
  scan, plan, pack, cli, config, docs.
- Architecture decisions live in `docs/adr/`; cite ADR numbers when a
  review comment touches a recorded decision.
