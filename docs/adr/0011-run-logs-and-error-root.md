# ADR-0011: Run logs as artifacts, and one error root

- **Status:** Accepted
- **Date:** 2026-07-28 (accepted 2026-07-28)
- **Amendment (2026-08-11):** the tool renamed to vramfit (#118,
  chart #114). Decision 2's run-log envelope key renamed with it,
  and the run-log version bumped. The rename executed in #120.

## Context

The first real scan (issue #8) is a grid of ~330 calibration passes
and records no run history. It captures no per-cell timing, no memory
high-water marks, and no model load duration. Debugging its CUDA
out-of-memory failure (PR #13) used raw process output. The project's
differentiator is artifacts with provenance — run history kept as
loose text would be the one run output without schema discipline.

Errors today have no shared root: `PinError(ValueError)`,
`InfeasibleBudgetError(Exception)`, `ArtifactError(ValueError)`,
`ScanExtraMissingError(RuntimeError)`. CLI handlers enumerate ad-hoc
catch clauses per call site. The review cycle found two
error-labeling bugs at the adapter boundary, where the CLI translated
a foreign exception too eagerly.

The human channel works: `typer.echo` progress lines and clean
`error:` messages. The gap is the machine channel.

## Decision

1. **Two channels, never mixed.** `typer.echo` stays the human
   channel. Machine events go to a **run log**: JSON Lines, one event
   per line. The CLI writes it beside the run's artifacts as
   `<stem>.runlog.jsonl`.
2. **The run log is an artifact.** Every line carries a
   `vramfit_runlog` version field — deliberately distinct from
   `vramfit_schema`, which versions whole artifact documents where
   this versions one event line. The adapter names events in the past
   tense: `scan_started`, `meter_built`, `resume_loaded`,
   `cell_measured`, `scan_finished`, `scan_halted`. `cell_measured`
   carries group, bits, damage, seconds, and the host RSS high-water
   mark. Halt events carry `cells_kept`, null when the stage cannot
   know the count. Every line carries the run's `run_id`, so reruns
   and resumes stay separable in one file. The file appends and
   tolerates crashes — the reader drops a torn final line. A write
   failure warns once on the human channel and disables the run log:
   measurement work outlives its record.
3. **structlog renders the events.** structlog is pure Python. It
   joins the base dependencies, which amends the base-dependency
   clause of [ADR-0005](0005-heavy-deps-as-extras.md) from "typer
   only" to "typer and structlog". Rejected alternative: a hand-rolled
   stdlib `json` emitter avoids the dependency but loses processor
   pipelines, and the maintainer standardizes on structlog.
4. **The domain stays log-free.** The import-linter forbidden list for
   `vramfit.domain` gains `logging` and `structlog`. The run log is
   an adapter concern.
5. **One error root.** The domain defines `VramfitError`. Every
   vramfit exception inherits it while keeping its current base for
   compatibility (e.g. `ArtifactError(VramfitError, ValueError)`).
   Adapters translate foreign exceptions (torch, transformers, OS)
   into `VramfitError` subclasses at the port boundary. The CLI
   converges its handlers on the root over time — the full collapse
   of per-command catch clauses is an open question below.

## Open questions

- The event set for `plan` (start with `scan`, the only long-running
  stage). ~~The event set for `pack`.~~ Resolved by
  [ADR-0012](0012-gguf-type-mapping.md) decision 5 and implemented in
  `cli_pack`.
- A `reference_pass_finished` event: the reference pass runs inside
  the meter, invisible to the CLI. A dedicated event needs a port
  change, and the first `cell_measured` absorbs its seconds today.
- Full collapse of the CLI's per-command handlers onto the
  `VramfitError` root (the scan paths converge first).
- Translation of the torch meter's foreign RuntimeErrors, and its own
  stdlib raises, under the root.
- Whether run logs join the provenance story for third-party map
  submissions (issue #11).
- Whether `cell_measured` should carry the GPU memory high-water mark
  or utilization, which costs an NVML dependency.

## Consequences

- Splunk, DuckDB, Postgres `COPY`, and any JSONL-speaking collector
  ingest run logs without adapters.
- Per-cell timing turns "which groups are slow" and future
  distributed-scan coordination into queries.
- The CLI's ad-hoc catch clauses converge toward one root, and the
  translate-at-the-boundary rule targets the error-labeling bug class
  the review actually found.
- The base install grows by one pure-Python dependency, and
  acceptance amends the ADR-0005 clause.
