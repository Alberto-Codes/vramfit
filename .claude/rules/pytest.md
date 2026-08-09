---
paths:
  - "**/test_*.py"
  - "**/tests/**/*.py"
  - "**/conftest.py"
---

# Pytest Rules

## Naming & layout

- Name tests `test_<what>_<condition>_<expected_result>`.
- Layout mirrors the hex layers: `tests/unit/domain/`,
  `tests/unit/adapters/`, `tests/contract/`, `tests/e2e/`,
  `tests/integration/`.
- Shared fixtures in `tests/unit/conftest.py`; hypothesis strategies in
  `tests/strategies.py`; port fakes in `tests/fakes.py`.

## Markers (registered in pyproject, `--strict-markers`)

Default run = `unit` + `contract` (hermetic, sub-second). See
[ADR-0009](../../docs/adr/0009-testing-strategy.md).

- `unit` — pure logic, tmp files at most. Hypothesis property tests live
  here (profiles: `fast` 25 examples default, `thorough` 200 on
  pre-push via `HYPOTHESIS_PROFILE`).
- `contract` — verified fakes: each port's suite is parametrized over
  the real adapter AND its `tests/fakes.py` fake, asserting identical
  Protocol behavior including error types. **Every new port needs one.**
- `integration` — real resources; must skip with an explicit reason
  when the resource is absent, never fail for absence.
- `e2e` — the installed `quantfit` console script via subprocess.
- `gpu` / `slow` — orthogonal axes; `gpu` tests skip cleanly without
  CUDA (CI has none).

## Principles

- Fakes over mocks at port boundaries; mocks (pytest-mock `mocker`,
  `autospec=True`) only at external edges. Patch where used, not where
  defined.
- New solver invariants become hypothesis properties, not just example
  tests — determinism is a contract (ADR-0007).
- One concept per test; plain `assert`; parametrize with `ids=[...]`.
- pytest-randomly is active: any order-dependence is a bug.

## Commands

```bash
uv run pytest                      # fast suite (unit + contract)
uv run pytest -m e2e               # console-script flows
HYPOTHESIS_PROFILE=thorough uv run pytest -m "not gpu and not integration"
```
