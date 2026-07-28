<!--
AI: Remove all HTML comments when filling this template. Keep only visible content.

PR Title → squash commit subject (50 chars, imperative)
Format: type(scope): description
Types: feat | fix | docs | refactor | test | chore | perf

Scope: A noun describing a section of the codebase (per conventionalcommits.org).
  ✓ feat(scan): add KL-divergence sensitivity metric
  ✓ fix(plan): respect kv-headroom in the bit budget
  ✓ refactor(arch): split adapters by direction
  ✗ feat(034-feature): ...  ← NOT spec/issue numbers (breaks release-please)

Scopes: scan | plan | pack | cli | config | docs | arch | domain | ports | adapters

Breaking: add ! after scope → feat(cli)!: remove deprecated flag
-->
<!--
Why this change? Problem solved? Contrast with previous behavior.
e.g., "Uniform 4-bit does not fit the 49B target. This adds the solver that
spends bits per layer group under a hard VRAM budget."
-->

<!--
What changed? 2-4 bullets, imperative mood.
e.g., - Add greedy damage-per-byte solver with explanation trace
      - Record format_overhead in the recipe for reproducibility
-->
-

<!-- How to verify: command, manual steps, or "CI only" -->
Test: `uv run pytest -q`

<!--
Git trailers (one per line):
  Closes #123
  BREAKING CHANGE: remove deprecated solve() signature
-->
Closes #

<!--
═══════════════════════════════════════════════════════════════════════════
MULTI-COMMIT PRs (release-please)
═══════════════════════════════════════════════════════════════════════════
When a PR contains multiple logical changes that would normally be separate
commits, add additional conventional commit blocks as FOOTERS at the bottom
of the body (above the PR Review section). Release-please parses these to
generate proper changelog entries once releasing starts.

Format: blank line, then type(scope): description, then details

Example PR body structure:
───────────────────────────────────────────────────────────────────────────
Primary change description (associated with PR title).

- Bullet points for primary change

Test: `uv run pytest -q`

Closes #123

feat(domain)!: rename damage field for consistency

BREAKING CHANGE: `Assignment.loss` → `Assignment.damage`

docs(reference): add recipe trace field notes

- Document replayability guarantee
───────────────────────────────────────────────────────────────────────────

The PR title becomes the first changelog entry. Each footer block (starting
with a conventional commit type) becomes an additional entry.

Ref: https://github.com/googleapis/release-please#what-if-my-pr-contains-multiple-fixes-or-features
-->

---

## PR Review

### Checklist
- [ ] Self-reviewed my code
- [ ] All gates pass (`pre-commit run --all-files` and a push-stage run)
- [ ] Doc statuses promoted/demoted where code moved (CLAUDE.md trust rules)
- [ ] New ports have verified-fake contract suites (ADR-0009)
- [ ] Breaking changes use `!` in title and `BREAKING CHANGE:` in body

### Review Focus
<!-- Where should reviewers concentrate? Known limitations? -->

### Related
<!-- Other PRs, issues, ADRs for context -->
