# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**vramfit** measures per-layer quantization sensitivity, solves for a
mixed-precision recipe under a VRAM budget, validates the recipe against
the prediction, and packs the result for a target runtime. Acceptance test: Nemotron Super 49B serving on a 24 GiB RTX 4090
([ADR-0003](docs/adr/0003-north-star-benchmark.md)). Full pitch in
[README.md](README.md); design decisions in [docs/adr/](docs/adr/index.md).

## Trust levels — read this before building anything

Docs are written ahead of code. Every docs page carries a status
(`sketch | draft | stable`); ADRs carry `Proposed | Accepted`. The rules:

- **Do not write code against a `sketch` page or a `Proposed` ADR without
  flagging it to the user first.** Those record guesses, not commitments.
- Promote a page's status (`sketch → draft → stable`) in the same PR that
  lands the code proving it. Demote when code moves out from under a page.
- When you learn something that contradicts a `sketch`, fix the sketch —
  that's what it's for.

## Vocabulary

[docs/reference/glossary.md](docs/reference/glossary.md) is law: one term per
concept, one concept per term — in docs, code identifiers, commit messages,
and conversation. It's "recipe" (never "config"), "damage" (never "error" or
"loss"), "sensitivity map" (never "scan results"). Coining a new term
requires a glossary entry in the same change.

## Writing system

Distilled from ASD-STE100. Two modes.

**Strict mode** — ADRs, reference pages, README, error messages, CLI help,
commit messages, code comments. A misread here costs cycles:

1. One instruction per sentence, ≤ 20 words. Descriptive sentences ≤ 25.
2. Active voice; name the actor ("the solver rejects…", not "it is rejected").
3. Verbs for actions, not nouns: "analyze", never "perform an analysis".
4. No hedging stacks. Either state the fact or write it as an explicit
   **Open question**. "May potentially help improve" is banned.
5. No marketing adjectives: seamless, robust, powerful, blazing, cutting-edge.
   Show the number instead.
6. No semicolons. At most one em-dash per paragraph.
7. Single verbs over phrasal verbs: "remove", not "take off"; "examine", not
   "dig into".
8. State numbers and units: "24 GiB", not "large"; "~3.2 bits/param", not
   "very low".

**Flavored mode** — explanation pages, discussions, PR descriptions. Keep the
glossary discipline, active voice, and one-idea-per-sentence habit; range,
analogy, and voice are allowed. Explanation pages are where thinking happens —
do not strangle them.

Style rules govern *form*. The status system governs *trust*. Clean prose on
a `sketch` page is still a sketch.

## Architecture

Hexagonal, mechanically enforced by import-linter (ADR-0008):
`adapters/inbound` (CLI, composition root) → `adapters/outbound` (JSON
artifacts, HF configs, the torch scan meter, the GGUF pack toolchain) →
`ports` (Protocols) → `domain` (pure — no
json/pathlib/os/io/typer/logging/structlog, enforced). The `vramfit_schema` envelope
belongs to the JSON adapters, never to domain dataclasses. New external
integrations (torch, llm-compressor, runtimes) are outbound adapters
behind ports.

**File size rule:** modules cap at **300 lines of actual code (soft) /
320 (hard)** — code lines exclude comments and docstrings; the gate is
`scripts/check_loc.py` in pre-commit. Over the limit means decompose,
not excuse.

## Build & Development

```bash
uv sync --dev          # install with dev tools
uv run vramfit --help # CLI

# Quality gates (all must pass before PR)
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest                  # fast suite; push gate adds thorough+e2e+cov
uv run lint-imports            # hex layers + domain purity + no-torch (ADR-0008)
uv run python scripts/check_loc.py src   # 300/320 code-line cap
uv run docvet check --all
```

## Testing

Pyramid per ADR-0009, rules in `.claude/rules/pytest.md`:

- Default run = `unit` + `contract` (hermetic); `integration`/`e2e`/
  `gpu`/`slow` are deselected by addopts.
- **Every new port requires a verified-fake contract suite**
  (`tests/contract/`, fakes in `tests/fakes.py`).
- Solver invariants get hypothesis properties (`tests/strategies.py`);
  profiles `fast`/`thorough` via `HYPOTHESIS_PROFILE`.
- Pre-commit runs the fast suite; pre-push runs thorough + e2e +
  coverage. GPU tests must skip cleanly without CUDA.
- Test naming: `test_<what>_<condition>_<expected_result>`.

## Key Constraints

- Base install carries typer and structlog only (ADR-0011 amendment);
  torch/transformers land behind extras
  ([ADR-0005](docs/adr/0005-heavy-deps-as-extras.md)). The plan step must
  stay importable without torch.
- Artifact schemas carry `vramfit_schema`; breaking changes bump it.
- `from __future__ import annotations` at the top of every file.
- No relative imports; no `__main__.py`.

## Branching & Commits

- Single default branch `main`; feature branches `feat/<scope>-<description>`
  squash-merge via draft PRs.
- Conventional commits: `type(scope): description`.
  Scopes: scan, plan, pack, cli, config, docs, arch, domain, ports, adapters.
- No `Co-Authored-By` trailers.
