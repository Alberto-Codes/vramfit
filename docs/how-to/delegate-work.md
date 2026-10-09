---
status: draft
---

# Delegate a bounded change

> **Status: draft** — ported from the `game-lab` sister repository on
> 2026-10-09. No vramfit dispatch has run through this procedure yet.
> The first accepted builder slice promotes it.

Use this procedure for supervised implementation, whatever the model or harness.
The supervisor selects work, decides boundaries, verifies behavior and handles publication.
The builder implements a scoped change and returns evidence.
Apply the [bounded execution limits](../../CLAUDE.md#bounded-execution) before dispatch.
[The shared roles](../reference/worker-runs.md#shared-roles) hold each agent's duties.

## 1. Make the task ready

An issue is the durable task record. The brief carries its accepted contract and the execution context.
Pass the exact accepted comment URL, or a versioned local contract when there is no remote.
Never substitute the latest comment for the accepted specification.

Call the `specifier` only when the issue has no decided, bounded contract.
For a behavior change, require a regression that fails before the fix.
Check that the failure shows missing behavior, not a broken fixture.
Keep the test and the implementation in the same dispatch when one builder can prove both.

A `chart:task` ticket carries extra limits.
It adds no port, no artifact schema field and no gate the authorizing record does not name.
Read [the charting rules](../../.claude/rules/charting.md) before you write its brief.

## 2. Size by behavior

Start with one behavior across about two to four production files, plus its tests and docs.
This range is a starting rule, not a measured limit.

| Task shape | Dispatch |
|---|---|
| Decided behavior with a failing acceptance test | Builder directly |
| Several independent behaviors | Split into named slices |
| Unresolved port, schema or ADR decision | Resolve the decision first (a `chart:discuss` ticket or the supervisor) |
| Small mechanical correction with an obvious proof | Supervisor, or a short repair brief |
| Broad failure after repeated repairs | Reassess the contract and split again |

## 3. Launch with a bounded brief

Replace every placeholder before dispatch. Link files instead of copying history.

```text
Role: bounded implementation worker. The supervisor owns acceptance and publication.
Supervisor / worker / harness: <actual assignments; requested model alias>
Issue and slice: <issue, one behavior>
Worktree / branch / base revision: <exact values>
Accepted specification: <exact reference or complete short contract>
Record: <the ADR, glossary entry or page the contract builds to>
Decision and reason: <settled shape and why>
Read first: <targeted files and evidence>
Allowed edits: <production, test and documentation paths>
Out of scope: <adjacent work and tempting incorrect fixes>

Acceptance:
- <observable invariant, exact command, expected result>
- <regression that must remain unchanged>
Run the behavioral regression before implementation. Preserve its red output.
Do not weaken the acceptance test to obtain green output.
Run focused checks during edits. Report required checks not run.

Do not commit, publish, or bypass guards or denied tools.
Preserve unrelated changes. Do not reset, clean, stash or restore them.
If required edits exceed the allowed scope, return the specific missing scope.

Return: changed paths, red/green commands and results, unrun checks,
remaining gaps. End with `Model: <model ID>`. Leave the diff for review and stop.
```

## 4. Accept behavior and finish the slice

Dispatch the `acceptance-reviewer` in a fresh session after the builder returns.
Read the diff against the agreed scope yourself as well. Self-review is not independent evidence.
Run the quality gates in `CLAUDE.md` on the finished change. Never report an unrun gate as green.
Promote or demote the docs pages the change proves or breaks, in the same PR.

When review finds a defect, return the failed assertion and a narrow correction brief to the builder.
After two unsuccessful repairs of the same defect, reassess the contract before another dispatch.
Commit only after the acceptance reviewer accepts the integrated result, including any repair.

## Build roles

| Role | Agent file | The supervisor calls it | Tools |
|---|---|---|---|
| Specifier | `specifier.md` | When the issue has no decided, bounded contract | Read, Grep, Glob, Bash, web. Edits no file. |
| Builder | `builder.md` | For each non-trivial behavior change | Every tool. Commits only on the user's explicit assignment. |
| Acceptance reviewer | `acceptance-reviewer.md` | After the builder returns, in a fresh session | Read, Grep, Glob, Bash. Edits no submitted file. |
| Researcher | `researcher.md` | For a cited survey a ruling or a `chart:research` ticket needs | Read, Grep, Glob, Bash, web, Context7. Edits no file. Decides nothing. |

The supervisor records each worker's `Model:` line in its session report.
A commit carries no model trailer. `CLAUDE.md` keeps AI attribution out of the git surface.
