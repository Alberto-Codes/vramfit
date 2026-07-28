---
marp: true
theme: quantfit
paginate: true
footer: quantfit · PR #1
---

<!-- _class: lead -->

# quantfit — plan step deep dive

Milestone 1: artifact schemas, VRAM budget math, greedy solver

Branch `feat/plan-solver-schemas-budget` · 2026-07-28
[github.com/Alberto-Codes/quantfit/pull/1](https://github.com/Alberto-Codes/quantfit/pull/1)

---

# Context

- Pipeline: **scan → plan → pack** (ADR-0002), connected by versioned
  JSON artifacts.
- Target: Nemotron Super 49B serving on a 24 GiB RTX 4090 via vLLM
  (ADR-0003, ADR-0004).
- Plan built first, deliberately:
  - ADR-0005: plan must be pure Python (stdlib + typer, **no torch**)
    → fully CPU-testable, cheap to iterate.
  - Its budget math turns the "does this even fit" question into
    arithmetic before the expensive GPU work starts.
- Scan (torch, 98 GB model) develops against these schemas later.

---

# Architecture (hexagonal, ADR-0008)

Layers enforced by import-linter — a violation is a failed commit:

- `domain/` — model dataclasses (self-validating), budget math,
  greedy solver. Pure: no json, pathlib, os, io, or typer.
- `ports/outbound.py` — Protocols (map source, recipe sink,
  shape source), each with a verified-fake contract suite.
- `adapters/outbound/` — JSON artifacts (schema envelope lives
  here), HF config parsing. `adapters/inbound/cli.py` — Typer,
  composition root.

Nothing imports torch (also a contract).

---

# Contract: sensitivity map (scan → plan)

```json
{
  "quantfit_schema": 1,
  "model_id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
  "scan": { "metric": "kl_divergence", "precisions": [8, 4, 3, 2], ... },
  "groups": [{
    "name": "model.layers.0.self_attn",
    "bytes_fp16": 100000000,
    "sensitivity": { "8": 0.0001, "4": 0.0042, "2": 0.417 }
  }]
}
```

- v1 rule: every group's sensitivity keys must equal `scan.precisions`.
- Validation errors carry JSON paths:
  `$.groups[3].sensitivity: key "4x" is not an integer precision`.
- Booleans explicitly rejected where numbers are expected
  (JSON `true` is a valid Python int).

---

# Contract: recipe (plan → pack)

```json
{
  "plan": {
    "weight_budget_bytes": 21474836480,
    "predicted_damage": 0.0871,
    "solver": "greedy-damage-per-byte",
    "pins": { "model.layers.0.*": 8 },
    "format_overhead": 0.05,
    "trace": [{ "step": 1, "group": "...", "from_bits": 8,
                "to_bits": 4, "ratio": 1.9e-12 }]
  },
  "assignments": [{ "group": "...", "bits": 8, "bytes": 50000000,
                    "damage": 0.0001 }]
}
```

- All plan fields are **required** — the loader rejects truncated or
  hand-edited artifacts rather than backfilling defaults.
- Replaying `trace` from the starting state (pinned groups at their
  pin) reproduces the assignments exactly (property-tested).

---

# Budget math — heterogeneity matters

Real target config (`config.json`, DeciLM/NAS):

- 80 blocks; **31 have attention deleted** (`no_op: true`).
- 49 attention blocks, GQA group size 8 → 8 KV heads × head_dim 128.
- FFN widths vary 10× (`ffn_mult` 0.5 → 5.25).

Consequences baked into the code:

- KV cost = **sum over attention layers**, never `layers × constant`
  (`domain/budget.py`).
- The HF config adapter parses DeciLM `block_configs` and llama
  configs, rejecting bad geometry: non-divisible GQA groups,
  non-boolean skip flags, invalid `head_dim` — never silent fallback.
- Hand-computed anchor test: 2 × 49 × 8 × 128 × 2 = **200,704
  bytes/token** at fp16.

---

# Measured results

`quantfit budget --model-config <nemotron> --vram 24GiB --context 16384`

| | fp16 KV | fp8 KV |
|---|---|---|
| VRAM total | 24.00 GiB | 24.00 GiB |
| − KV cache @16k | 3.06 GiB | 1.53 GiB |
| − runtime overhead | 2.00 GiB | 2.00 GiB |
| **weight budget** | **18.94 GiB** | **20.47 GiB** |

→ ~3.3–3.5 average bits/parameter over ~49B params — better than the
original ~3.2 estimate, still under vLLM's 4-bit kernel floor (see
risks slide). Re-run after the full refactor: **byte-identical**.

---

# Solver (ADR-0007, now Accepted)

Multiple-choice knapsack, greedy by damage-per-byte:

1. Start every group at highest scanned precision (or its pin).
2. Precheck: minimum achievable total vs budget →
   `InfeasibleBudgetError` with exact gap in bytes.
3. While over budget: apply the move minimizing
   `(damage_delta / bytes_freed, group_name, smallest_step)`.

---

<!-- _class: dense -->

# Solver — design points

- Moves consider **all** lower precisions → non-convex damage curves
  get direct multi-step jumps (8→2 without stopping at 3).
- Selection key is a total order → deterministic, input-order
  invariant (hypothesis-tested under permutation).
- Pins: `fnmatchcase` globs; zero-match = hard error; later overrides
  earlier; pinned groups never move (tested under budget pressure).
- The final downgrade is refined when a milder step also fits with
  less damage — kills the dominated-recipe case review found.
- Integer byte math (`ceil` once per size) — no float accumulation
  to threaten determinism.

---

# Why greedy (and what was rejected)

- Instance is small (hundreds of groups × ~4 precisions) — exact DP
  and ILP are both feasible.
- Greedy chosen first because its downgrade sequence **is** the
  explanation trace — "why is this group 4-bit?" has a literal answer.
- Damage model is itself approximate (marginal additivity, ADR-0006
  still Proposed) — an exact optimum of an approximate objective is
  still approximate.
- `--solver exact` planned later as a check on greedy's gap.

---

<!-- _class: dense -->

# Verification

- **156 tests, 99% coverage** across a marked pyramid (ADR-0009):
  unit + hypothesis properties, verified-fake contract suites per
  port, subprocess e2e. Fast suite < 1 s; pre-push runs the
  thorough profile.
- Properties: budget always respected, exact infeasibility gaps,
  determinism under permutation, trace replay, pins under pressure.
- **13 gates** on every commit: ruff (Sonar-adjacent set), ty,
  import-linter (3 contracts), 300/320 LOC cap, docs-reference
  check, docvet — all with `always_run`.
- Reviewed 8 ways: 3 Copilot rounds + 5 specialist agents — 50+
  findings triaged, every accept fixed and every decline explained
  on the PR threads.
- Not covered yet: GPU paths (`gpu` marker reserved) — scan is
  still a stub.

---

<!-- _class: dense -->

# Risks and open questions

- **ADR-0003 vs ADR-0004 tension (now measured):** budget forces
  ~3.3–3.5 avg bits; vLLM kernels floor at 4-bit (no 2/3-bit).
  - Paths: friendlier measured overheads · contribute sub-4-bit
    kernel · GGUF backend for the benchmark · change target model.
  - Deferred to its own ADR before `scan` fixes candidate precisions.
- Marginal-additivity assumption (ADR-0006, Proposed) — mitigation:
  whole-recipe validation pass after plan.
- `format_overhead` (5%) and runtime overhead (2 GiB) are planning
  constants until measured for real.

---

# Next milestone

- **Scan harness**: torch + offload behind `quantfit[scan]` extra
  (ADR-0005 keeps base install clean).
- Streams layer groups to GPU; 124 GB system RAM holds the bf16
  reference; KL divergence per (group × precision).
- Its output feeds today's `plan` unchanged — the artifact contract
  is already enforced from both sides.
- Before that: resolve the 4-bit-floor tension ADR.

---

# Links

- PR #1: `github.com/Alberto-Codes/quantfit/pull/1`
- ADRs: `docs/adr/` — 0007 Accepted; 0008 (hex) and 0009 (testing)
  added during this PR.
- Schemas: `docs/reference/sensitivity-map.md`, `docs/reference/recipe.md`
  (both promoted sketch → draft).
- Budget math: `docs/explanation/vram-budget.md` (promoted → stable,
  real numbers).
- Glossary: `docs/reference/glossary.md` — one term per concept.
- Roadmap for the next session: issue #2.
