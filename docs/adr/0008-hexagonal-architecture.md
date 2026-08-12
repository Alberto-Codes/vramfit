# ADR-0008: Hexagonal architecture, enforced by import-linter

- **Status:** Accepted
- **Date:** 2026-07-28

## Context

The first milestone shipped as flat layering (`cli → solver/budget →
artifacts`). That was deliberate — four modules did not justify ports —
but the next milestones introduce real external seams: torch and model
loading (scan), llm-compressor and runtime backends (pack), Hugging Face
configs (already present). Retrofitting boundaries after those arrive is
the expensive order. The maintainer chose to adopt the full structure now,
with import-linter as the mechanical gate so the architecture is checked,
not aspirational.

## Decision

Ports-and-adapters with directional adapter packages, the common
convention in Python service codebases:

| Layer | Contents | May import |
|-------|----------|------------|
| `vramfit.adapters.inbound` | Typer CLI (driving adapter, composition root) | everything below |
| `vramfit.adapters.outbound` | JSON artifact files, HF config parsing | ports, domain |
| `vramfit.ports` | `Protocol` definitions (outbound only today) | domain |
| `vramfit.domain` | model dataclasses, solver, budget math | domain only |

Rules with teeth (the import-linter lookup map, in `pyproject.toml`):

1. **"Hexagonal layers"** — the table above, as a layers contract.
2. **"Domain is IO-free"** — the domain may not import `json`,
   `pathlib`, `os`, `io`, or `typer`. Serialization and file access are
   adapter concerns; consequently the `vramfit_schema` version field is
   an **envelope owned by the JSON adapters**, not a domain field.
3. **"No heavy ML deps"** — carried over from ADR-0005.

Ports are structural (`typing.Protocol`); adapters satisfy them by shape.
Only driven (outbound) ports are reified — the domain's inbound API is
its plain public functions, which is idiomatic Python hex.

The pipeline's *strongest* boundaries remain the artifacts themselves:
scan, plan, and pack are separate processes connected by versioned JSON
files — process-level ports that survive language and machine boundaries.
The in-process layers exist to keep each stage's internals swappable.

## Consequences

- scan and pack land as adapters (model loading, quantization backends,
  runtime writers) against ports the domain defines — the seams exist
  before the heavy code does.
- The `pack` runtime backend port has a designated home — filled by
  the GGUF backend (ADR-0010/0012, `RecipePacker`), with vLLM later
  per ADR-0004 as amended.
- More files and some indirection at today's size; accepted — the
  contracts make violations a failed commit rather than a review debate.
- Supporting gate: modules are capped at 300 (soft) / 320 (hard) lines
  of actual code (`scripts/check_loc.py`, pre-commit), which pushes
  toward decomposed adapters instead of grab-bag modules.
