# ADR-0011: Run logs as artifacts, and one error root

- **Status:** Proposed
- **Date:** 2026-07-28

## Context

The first real scan (issue #8) runs for hours and records results but
no telemetry: no per-cell timing, no memory high-water marks, no model
load duration. Debugging its CUDA out-of-memory failure (PR #13) used
raw process output. The project's differentiator is artifacts with
provenance — telemetry kept as loose text would be the one run output
without schema discipline.

Errors today have no shared root: `PinError(ValueError)`,
`InfeasibleBudgetError(Exception)`, `ArtifactError(ValueError)`,
`ScanExtraMissingError(RuntimeError)`. Each CLI call site enumerates
its own catch tuple. The review cycle found two mislabeled-error bugs
in exactly those tuples.

The human channel works: `typer.echo` progress lines and clean
`error:` messages. The gap is the machine channel.

## Decision (proposed)

1. **Two channels, never mixed.** `typer.echo` stays the human
   channel. Machine events go to a **run log**: JSON Lines, one event
   per line, written beside the run's artifacts as
   `<out stem>.runlog.jsonl`.
2. **The run log is an artifact.** Every line carries a
   `quantfit_runlog` version field. Events are named in the past tense
   (`scan_started`, `reference_pass_finished`, `cell_measured`,
   `scan_halted`). `cell_measured` carries group, bits, damage,
   seconds, and memory high-water marks. Append-only, crash-tolerant.
3. **structlog renders the events** and captures third-party library
   warnings into the same stream. structlog is pure Python and joins
   the base dependencies — this amends the base-dependency clause of
   [ADR-0005](0005-heavy-deps-as-extras.md) from "typer only" to
   "typer and structlog". Rejected alternative: a hand-rolled stdlib
   `json` emitter avoids the dependency but loses processor pipelines
   and stdlib-logging capture, and the maintainer standardizes on
   structlog.
4. **The domain stays log-free.** The import-linter forbidden list for
   `quantfit.domain` gains `logging` and `structlog`. Telemetry is an
   adapter concern.
5. **One error root.** The domain defines `QuantfitError`. Every
   quantfit exception inherits it while keeping its current base for
   compatibility (e.g. `ArtifactError(QuantfitError, ValueError)`).
   Adapters translate foreign exceptions (torch, transformers, OS)
   into `QuantfitError` subclasses at the port boundary. The CLI's
   outermost handler catches `QuantfitError`, prints the clean
   `error:` line, and emits `scan_halted` to the run log.

## Open questions

- The event set for `plan` and `pack` (start with `scan`, the only
  long-running stage).
- Whether run logs join the provenance story for third-party map
  submissions (issue #11).
- Whether `cell_measured` should carry GPU utilization, which costs an
  NVML dependency.

## Consequences

- Splunk, DuckDB, Postgres `COPY`, and any JSONL-speaking collector
  ingest run logs without adapters.
- Per-cell timing turns "which groups are slow" and future
  distributed-scan coordination into queries.
- The CLI's per-call-site catch tuples collapse toward one root — the
  class of mislabeled-error bug the review found becomes structurally
  harder to write.
- The base install grows by one pure-Python dependency and the
  ADR-0005 clause is amended on acceptance.
