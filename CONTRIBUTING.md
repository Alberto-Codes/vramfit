# Contributing to quantfit

Thanks for your interest in contributing to quantfit! This guide covers everything you need to get started.

## Reporting Issues

Use the [Issues page](https://github.com/Alberto-Codes/quantfit/issues):

- **Bug Report** -- reproduce steps, environment info (GPU, driver, CUDA version), and logs
- **Feature Request** -- problem statement, proposed solution, acceptance criteria

## Prerequisites

- **Python** >= 3.12
- **uv** -- [install guide](https://docs.astral.sh/uv/getting-started/installation/)
- **git**
- **pre-commit** -- [install guide](https://pre-commit.com/#install)
- **CUDA GPU** -- only for `-m gpu` tests; everything else runs on CPU

## Fork and Clone

```bash
# 1. Fork the repo on GitHub (click "Fork" button)
# 2. Clone your fork
git clone https://github.com/<your-username>/quantfit.git
cd quantfit

# 3. Add the upstream remote
git remote add upstream https://github.com/Alberto-Codes/quantfit.git

# 4. Create a feature branch
git checkout -b feat/<scope>-<description>
```

## Development Setup

```bash
# Install in development mode with all dev dependencies
uv sync --dev

# Verify installation
uv run quantfit --help
```

## Pre-Commit Hooks

quantfit uses pre-commit hooks to catch issues before they reach CI. Pre-commit is not a project dependency -- if you don't have it yet, see the [install guide](https://pre-commit.com/#install). Then activate the hooks:

```bash
pre-commit install
```

Hooks that run on each commit:

| Hook | What it checks |
|------|---------------|
| yamllint | YAML syntax and formatting |
| actionlint | GitHub Actions workflow validity |
| ruff-check | Python linting |
| ruff-format | Python code formatting |
| uv-lock | Lockfile is in sync with pyproject.toml |
| uv-secure | Known vulnerabilities in the lockfile |
| ty | Type checking |
| pytest | Test suite (GPU tests excluded) |
| docvet | Docstring quality on staged files |

All hooks must pass before the commit succeeds.

## Quality Gates

All five gates must pass before opening a PR. Run them locally:

```bash
uv run ruff check .                  # Linting
uv run ruff format --check .         # Format check
uv run ty check                      # Type checking
uv run pytest -m "not gpu"           # Tests (CI enforces 90% coverage)
uv run docvet check --all            # Docstring quality
```

To auto-fix linting and formatting issues:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Coding Standards

- `from __future__ import annotations` at the top of every file
- Google-style docstrings on all public functions and classes
- Type hints on all function signatures (`list[str]`, not `List[str]`)
- 88-char soft limit (formatter), 100-char hard limit (linter)
- No relative imports (full package paths only)

## Testing

```bash
uv run pytest                  # All tests (GPU tests skip without CUDA)
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m gpu           # CUDA-required tests only
uv run pytest -m "not slow"    # Skip slow tests
uv run pytest -k test_name     # Single test by name
```

Test naming convention: `test_<what>_<condition>_<expected_result>`

Markers: `unit` (fast, no GPU), `integration` (real weights, slower), `gpu` (requires CUDA), `slow`.

GPU-dependent tests must carry the `gpu` marker and skip cleanly when CUDA is
absent -- CI runners have no GPU, so anything unmarked must pass on CPU.

## Commit Conventions

Commits follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): description
```

| Type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| docs | Documentation only |
| refactor | Code restructuring |
| test | Adding or updating tests |
| chore | Maintenance tasks |
| perf | Performance improvements |

**Scopes:** scan, plan, pack, cli, config, docs

**Examples:**

```
feat(scan): add per-layer KL-divergence sensitivity metric
fix(plan): respect kv-headroom when solving the bit budget
docs(pack): document vLLM checkpoint layout
```

Do not add `Co-Authored-By` trailers to commits.

## Pull Request Process

1. Push your branch to your fork: `git push -u origin feat/<scope>-<description>`
2. Open a **draft** PR against `upstream/main` (non-draft PRs trigger automated review prematurely)
3. PR title must follow conventional commits format: `type(scope): description`
4. All CI checks must pass before review
5. PRs are squash-merged to keep a linear history

## Dependency Vulnerabilities

CI runs `uv-secure` to scan for known vulnerabilities in the lockfile. If it flags something:

- **Fix exists?** Upgrade the package: `uv lock --upgrade-package <pkg>`. No suppression needed.
- **No fix?** Add the specific GHSA/CVE ID to the ignore list in `pyproject.toml` with an inline comment (package, version, description, "No fix available"). `allow_unused_ignores = false` stays set so stale suppressions fail CI and get cleaned up.

## Key Constraints

- Keep heavy ML dependencies (torch, transformers) out of the base install until the scan pipeline lands -- they will be optional extras
- Never use relative imports
- Never use `__main__.py` (entry point is `[project.scripts]`)
- Always target `main` for PRs (single-branch workflow)
