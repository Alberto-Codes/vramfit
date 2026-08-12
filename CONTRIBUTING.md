# Contributing to vramfit

Thanks for your interest in contributing to vramfit! This guide covers everything you need to get started.

## Reporting Issues

Use the [Issues page](https://github.com/Alberto-Codes/vramfit/issues):

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
git clone https://github.com/<your-username>/vramfit.git
cd vramfit

# 3. Add the upstream remote
git remote add upstream https://github.com/Alberto-Codes/vramfit.git

# 4. Create a feature branch
git checkout -b feat/<scope>-<description>
```

## Development Setup

```bash
# Install in development mode with all dev dependencies
uv sync --dev

# Verify installation
uv run vramfit --help
```

## Pre-Commit Hooks

vramfit uses pre-commit hooks to catch issues before they reach CI. Pre-commit is not a project dependency -- if you don't have it yet, see the [install guide](https://pre-commit.com/#install). Then activate the hooks:

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
| pytest | Fast suite: unit + contract (hermetic) |
| import-linter | Hex layers, domain purity, no heavy ML deps (ADR-0008) |
| loc-check | 300/320 code-line cap per module |
| doc-refs | Docs reference living module paths |
| docvet | Docstring quality on staged files |

All hooks must pass before the commit succeeds. A **pre-push** hook
additionally runs the full suite (thorough hypothesis profile, e2e via
subprocess, 90% coverage gate). Install both stages:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Quality Gates

All seven gates must pass before opening a PR. Run them locally:

```bash
uv run ruff check .                  # Linting
uv run ruff format --check .         # Format check
uv run ty check                      # Type checking
uv run pytest -m "not gpu and not integration"  # Tests (CI enforces 90% coverage)
uv run lint-imports                  # Hex layers + domain purity
uv run python scripts/check_loc.py src  # File size cap (300/320 code lines)
uv run python scripts/check_banned_terms.py  # Pre-rename tool name (#154)
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
- 300 code lines per module (soft), 320 hard -- decompose, don't excuse.
  Code lines exclude comments and docstrings (`scripts/check_loc.py`)
- No relative imports (full package paths only)

## Testing

```bash
uv run pytest                  # Default: fast suite (unit + contract)
uv run pytest -m e2e           # Console-script flows via subprocess
uv run pytest -m contract      # Verified-fake port suites only
uv run pytest -k test_name     # Single test by name
HYPOTHESIS_PROFILE=thorough uv run pytest -m "not gpu and not integration"
```

Test naming convention: `test_<what>_<condition>_<expected_result>`

Tiers per [ADR-0009](docs/adr/0009-testing-strategy.md): `unit` (pure
logic + hypothesis properties), `contract` (verified fakes — required
for every new port), `integration` (real resources, resource-gated),
`e2e` (console script), plus `gpu`/`slow` axes.

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
| ci | Workflows and release machinery |

**Scopes:** scan, plan, pack, cli, config, docs, arch, domain, ports, adapters

**Examples:**

```
feat(scan): add per-layer KL-divergence sensitivity metric
fix(plan): respect kv-headroom when solving the bit budget
docs(pack): document vLLM checkpoint layout
```

**Which types reach the changelog.** release-please lists `feat`, `fix`, and
`perf` in the release notes. Those three types cut a release. It hides `docs`,
`refactor`, `test`, `chore`, and `ci`, which cut no release on their own. Two
markers override the hidden list. A `!` breaking marker and a `Release-As:`
footer each force a release from any type. This repo writes docs ahead of
code, so a documentation commit must not mark a release on its own. Describe
user-facing documentation in the `feat` or `fix` commit that proves it.
[`release-please-config.json`](release-please-config.json) holds the
authoritative lists. A type absent from that file cuts no release and reaches
no changelog.

Do not add `Co-Authored-By` trailers to commits.

## Pull Request Process

1. Push your branch to your fork: `git push -u origin feat/<scope>-<description>`
2. Open a **draft** PR against `upstream/main` (non-draft PRs trigger automated review prematurely)
3. Fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md) -- remove HTML comments, keep visible content. The PR title becomes the squash-commit subject and (later) the release-please changelog entry, so it must follow conventional commits: `type(scope): description`
4. All CI checks must pass before review
5. PRs are squash-merged to keep a linear history

## CI Security Policy

Dated note, 2026-08-10 (ticket [#76](https://github.com/Alberto-Codes/vramfit/issues/76)).
This section records the fork-PR policy and the repository settings that
enforce it. Settings are not self-documenting, so this note is the record.
The author of a change to a setting, a secret, or a workflow trigger must
amend this note in the same PR.

**What it governs.** Two workflows exist: `ci.yml` (`pull_request` and
`push` on `main`) and `codeql.yml` (`pull_request`, `push`, and a weekly
`schedule`). Neither workflow references a secret. The repository defines
no Actions secrets, no Dependabot secrets, no environment secrets, and no
variables.

**Settings.** Verified 2026-08-10:

- The default `GITHUB_TOKEN` is read-only.
- Actions cannot create or approve pull requests.

Pending, blocked while the repository is private:

- Fork-PR workflow runs require maintainer approval for **all outside
  collaborators**. GitHub hides this setting on private repositories.
  The maintainer sets it the day the repository goes public
  ([#83](https://github.com/Alberto-Codes/vramfit/issues/83)).

**Rules.**

- Workflows must not use `pull_request_target`. That trigger grants the
  job the base repository's secrets and a write token. The bug lands
  when such a job checks out fork-PR code
  ([Securely using pull_request_target](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target)).
- Workflows must not check out fork-PR code in a job that holds a secret
  or a write token. One recorded exception: the `codeql.yml` analyze job
  holds `security-events: write` and checks out PR code. GitHub caps
  fork-PR tokens at read-only, so the write scope applies only to
  same-repository runs.
- The author of a workflow change that needs `pull_request_target` or an
  elevated checkout must amend this note in the same PR.

**Why.** The repository holds no secrets today, so a fork PR has nothing
to steal. The remaining risk is untrusted code running on shared runners
under the repository's identity. Public repositories get free standard
runners, so the exposure is runner abuse (cryptomining, egress), not a
bill. Approval-for-all closes that risk at the cost of one click per
fork PR. The `pull_request_target` ban prevents the bug class before it
can land.

**What fork contributors can expect.** GitHub gives fork PRs no secrets
and a read-only token. Every job that runs today passes under those
constraints. Once the repository is public, a maintainer approves each
fork-PR workflow run before it starts. **Open question:** the CodeQL
upload from a fork PR is untested here. GitHub documents the fork-PR
upload path as supported. The job stays skipped while the repository is
private. One known blemish: `uv-secure` crashes intermittently in CI
([#46](https://github.com/Alberto-Codes/vramfit/issues/46)). CI retries
the crash exit code once. Flag a suspected #46 crash on the PR. Do not
force-push to retrigger.

## Dependency Vulnerabilities

CI runs `uv-secure` to scan for known vulnerabilities in the lockfile.
For the intermittent crash in CI, see
[#46](https://github.com/Alberto-Codes/vramfit/issues/46) in the CI
Security Policy section above. If it flags something:

- **Fix exists?** Upgrade the package: `uv lock --upgrade-package <pkg>`. No suppression needed.
- **No fix?** Add the specific GHSA/CVE ID to the ignore list in `pyproject.toml` with an inline comment (package, version, description, "No fix available"). `allow_unused_ignores = false` stays set so stale suppressions fail CI and get cleaned up.

## Key Constraints

- Keep heavy ML dependencies (torch, transformers) out of the base install -- they live behind the `scan` and `pack` extras (ADR-0005)
- Never use relative imports
- Never use `__main__.py` (entry point is `[project.scripts]`)
- Always target `main` for PRs (single-branch workflow)
