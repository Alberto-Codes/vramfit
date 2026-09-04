# ADR-0009: Testing strategy — pyramid, verified fakes, properties

- **Status:** Accepted
- **Date:** 2026-07-28
- **Note (2026-09-04, issue #207):** a test that pins a copied
  external contract stays under `unit`. Maintainer ruling 2026-09-04.
  The `contract` marker keeps its one meaning: a verified fake over a
  port. The decision text below records the placement.
- **Note (2026-09-04, issues #470 and #364):** pre-push gains two
  steps. One runs the `integration` tier and skips without the scan
  extra. One measures `scripts/` at its own coverage floor.
  Maintainer rulings 2026-09-04. The hook-stage text below records
  both.

## Context

The hex refactor (ADR-0008) gives "contract test" a precise meaning, and
the upcoming scan/pack milestones introduce expensive tiers (GPU, model
weights) that must not pollute the fast suite. Adapted from the
maintainer's agent-harness ADR-0012, minus the LLM eval axis. Pack has
since shipped and packed models are graded through the external
llama.cpp harness ([evaluating packed models](../explanation/evaluating-packed-models.md)).
A dedicated eval-axis ADR remains an open item. Two Copilot review rounds also showed our dominant
defect class is *validation edge cases* — exactly what property-based
testing hunts.

## Decision

**Pyramid with strict markers** (`--strict-markers` — every test declares
its tier):

- **`unit`** (most) — pure logic, no IO beyond tmp files. Includes
  **hypothesis property tests**: solver invariants (budget respected,
  determinism under permutation, trace replay, pins honored, exact
  infeasibility gaps) and codec round-trips. Profiles: `fast` (25
  examples) by default, `thorough` (200) via `HYPOTHESIS_PROFILE`.
  A test that pins a copied external contract also lives here, for
  example `_SUFFIX_TO_GGUF` held against `gguf.TensorNameMap`. No
  marker names that class (#207, ruled 2026-09-04).
- **`contract`** — **verified fakes**: one parametrized suite per port
  runs against the real adapter *and* its in-memory fake
  (`tests/fakes.py`), proving identical Protocol behavior including
  error types. When scan/pack add expensive ports (model loader,
  runtime writer), their orchestration gets tested against proven
  fakes — no GPU in CI.
- **`integration`** — real resources (model weights, GPU runtimes).
  Holds the real-torch scan and offload suites today. Tests must skip
  with an explicit reason when their resource is absent, never fail
  for absence.
- **`e2e`** — the installed console script via subprocess, full
  file-in/file-out flows.
- `gpu` / `slow` — orthogonal resource axes.

**Default run = unit + contract** (hermetic, sub-second), enforced by a
`-m` deselection in `addopts`; a CLI `-m` overrides it.

**Hook stages:** pre-commit runs the fast suite; **pre-push** runs the
thorough profile plus e2e plus the 90% coverage gate. Pre-push then
runs two more steps. A second coverage run measures `scripts/` at a
48% floor, set below the 50% measured on 2026-09-04 (#364). The last
step runs the `integration` tier without `gpu`. Its modules skip with
the reason "scan extra not installed" where torch is absent, so the
step costs nothing there and about 7 seconds where the extra is
installed (#470). CI mirrors the scripts run and not the integration
step, because CI installs no torch (ADR-0005). Install both hooks:
`pre-commit install --hook-type pre-commit --hook-type pre-push`.

**Fakes over mocks** at port boundaries — if a port is hard to fake, the
abstraction is wrong. Mocks only at external edges (clock, randomness).

## Consequences

- Commit friction stays sub-second while pushes carry the full proof.
- Mutation testing (mutmut) is **planned, not wired**: valuable as a
  suite audit once the scan milestone stabilizes; wrong for any hook.
  Run manually or nightly when adopted.
- The contract tier must grow with every new port — a port merged
  without a verified-fake suite is an unenforced protocol.
