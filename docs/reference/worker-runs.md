---
status: draft
---

# Worker run contract

> **Status: draft** — ported from the `game-lab` sister repository on
> 2026-10-09. The model table was checked that day. The roles are
> unproven on a vramfit dispatch.

This contract keeps the project's requirements separate from the model and harness that run them.
The [delegation procedure](../how-to/delegate-work.md) holds the task sequence and the acceptance rules.

## Roles

| Role | Responsibility |
|---|---|
| Supervisor | Select and groom work, settle decisions, own acceptance, keep continuity and publication boundaries |
| Specifier, when needed | Propose a bounded decision and an executable contract. Report uncertainty and evidence |
| Builder | Implement the assigned contract within its allowed paths and return evidence |
| Acceptance reviewer | Check the actual diff and exercise the defining behavior independently |
| Researcher | Answer one bounded question with sources. Decide nothing |

Roles describe responsibility, not model brands. A model name alone does not make a review independent.
The reviewer uses production evidence and its own probes. It does not adopt the builder's completion claim.
A change of model never widens permissions or authorizes a merge or a publication.

## Shared roles

Four agent files in `.claude/agents/` carry these roles: `specifier.md`, `builder.md`, `acceptance-reviewer.md` and `researcher.md`.
Each file is a stub. This section holds the duties once.
The supervisor calls each agent with the `model` its frontmatter names.
The specifier, the acceptance reviewer and the researcher load `.claude/hooks/agent_deny_writes.sh` as a Bash hook.
That hook blocks `git commit`, `push`, `tag`, `reset`, `rebase`, `clean`, `stash`, `restore` and `checkout`, with or without `-C <path>`.
It blocks `gh pr`, `gh issue` and `gh release` with `create`, `edit`, `close` or `merge`.
It allows `gh issue comment`, so an authorized specifier can post its contract.
It does not block a shell file write.

Each agent loads `CLAUDE.md`, its role below and the accepted contract before work starts.
Each agent reads the contract by its exact URL, or by the versioned local path the brief names.
Each agent reports missing context as a blocker, not as permission to guess.
The brief can narrow these duties. The brief never widens them past `CLAUDE.md`.

### Specifier

The specifier reads the issue, the cited files and the relevant records.
It returns one bounded contract under 150 words.
The contract names the decision, the allowed paths, the exclusions and the stopping condition.
The contract names the acceptance command and the red failure that command shows before the change.
A research, design or documentation contract names an observable acceptance instead of a code test.
The contract builds to the record. An Accepted ADR or a `stable` page outranks the ticket's wording.
The specifier reports each unresolved piece of evidence. It never selects a disputed source in silence.
It cites a source for each claim about a runtime, a library API or a model.
It edits no file. It posts a comment only when the brief authorizes the post.

### Builder

The builder records the baseline and preserves unrelated work. It edits only the assigned paths.
For a behavior change, it runs the acceptance test before the production edit.
It keeps the command and the red output.
It implements the change, then keeps the green output.
It never weakens a test or suppresses a gate.
It keeps the glossary vocabulary and the 300-code-line module cap.
It adds no port, no artifact schema field and no gate the contract does not name.
It returns the changed paths, the red and green evidence, the gate accounting and the gaps.
It commits only when the brief relays the user's explicit commit assignment. It never pushes, publishes or changes policy.

### Acceptance reviewer

The acceptance reviewer runs in a fresh session, apart from the builder and the supervisor.
It inspects the actual submitted revision and diff. It records their identity, including uncommitted changes.
It checks the allowed paths, the preserved behavior, each weakened test and each gate suppression.
It reads the parent outcome before the builder summary. It treats that summary as a claim.
It runs a positive probe and a counterexample probe through the real entry point.
It proves the acceptance check fails when the behavior is missing.
A faithful change can expose a weak contract. The reviewer reports that as a specification finding.
It reports each finding with its claim, evidence, impact and correction.
It edits no submitted file. It writes scratch files only where the brief names the path.
Its verdict is `accept`, `repair`, `reject` or `incomplete`. It never reports a partial review as clean.

### Researcher

The researcher answers one bounded question with sources. It runs on `sonnet`.
It reports every finding the brief's scope covers, with a URL and a read date on each number.
It marks a claim it cannot source as unverified. It never selects a winner. The supervisor decides.
It reads a published quant's GGUF header for tensor types. It never trusts the label.
It fetches runtime and library facts through Context7 before it cites a version.
It edits no file. It writes scratch only where the brief names the path.
It never writes a chart body. The charting session folds its gist into the chart.

### Model identity in the return

Each agent reports the exact model ID from its system context, or `unknown`.
It never substitutes the requested alias for a resolved identity.
It ends its return with the line `Model: <model ID>`.
The supervisor keeps that ID in its session report. No commit trailer carries it here.

## Model routing

Agent files name models by alias, never by full id. An alias moves to each new model of its tier on upgrade.
The SessionStart hook `~/.claude/hooks/model-aliases.sh` prints the resolved ids and reports `Aliases MOVED`. Update the table below when it does.
Claude Code resolves each alias to the current model of that family at dispatch.
The table below was checked on 2026-10-09 with Claude Code v2.1.296 on the Anthropic API provider.
Sources: [model-config](https://code.claude.com/docs/en/model-config), [sub-agents](https://code.claude.com/docs/en/sub-agents), [fast-mode](https://code.claude.com/docs/en/fast-mode).

| Alias | Resolves to | $/MTok in / out | Default effort | Role in this repo |
|---|---|---|---|---|
| `fable` | `claude-fable-5-1` | $10 / $50 | `high` | Supervisor when the user picks it. Escalation agent for a decision `opus` failed twice or `xhigh` did not settle |
| `opus` (= `default`) | `claude-opus-5-5` | $4 / $20 | `medium` | Supervisor default. `specifier`, `builder`, `acceptance-reviewer`, gate repairs |
| `sonnet` | `claude-sonnet-5-5` | $2 / $10 | `medium` | `researcher`, documentation reads, surveys (`general-purpose`, `claude-code-guide`) |
| `haiku` | `claude-haiku-5-5` | $0.10 / $0.50 at ≤100K prompt, $0.50 / $2.50 above | `medium` | File search, lookups, extraction, classification (`Explore`) |

- All four models have a 1M-token context, 128K output, always-on adaptive thinking and all five effort levels (`low` to `max`).
  The `[1m]` suffixes change nothing for them.
- `best` resolves to `fable` where it is available. Name `fable` explicitly instead, and only for escalation.
  `opusplan` plans on Opus and runs on Sonnet. It suits the main session only.
- Fast mode (`/fast`) works only on Opus, at $8 / $40 per MTok on Opus 5.5. It is the same model with faster output, not a cheaper one.
- After a CLI upgrade, compare the `Model:` line in a worker's return with this table.

**Resolution order for a sub agent's model.** The Agent tool's `model` parameter wins.
Then the agent's frontmatter `model`, then `CLAUDE_CODE_SUBAGENT_MODEL`, then the main session's model.
`CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1` puts the env var ahead of frontmatter.
So a sub agent with no `model` anywhere runs on the supervisor's model, which can be Fable.

**Effort.** `CLAUDE_CODE_EFFORT_LEVEL` wins, then the Agent tool's `effort`, then frontmatter `effort`.
The docs do not say what a sub agent gets when no `effort` is set. Assume it inherits the session's `effortLevel` (`high` in user settings).
Every agent file therefore sets `effort: medium`. Drop a lookup to `low` when quality holds.
Never run a multi-step task on `sonnet` at `low`.
Raise a builder to `xhigh` only after it fails at `medium`, and record why.

**Rules.** Pass `model` and `effort` on every Agent call, including calls to `Explore` and `general-purpose`.
`.claude/hooks/agent_model_gate.sh` enforces two of these rules as a `PreToolUse` hook on the Agent tool.
It denies a dispatch with no `model`. It asks before a `fable` dispatch whose brief has no `Escalation: <reason>` line.
It skips a `fork` and fails open on bad input. It does not check `effort`, because hooks cannot confirm the Agent tool exposes it.
Do not `fork` from a Fable session. A fork runs on the supervisor's own model.
`tests/unit/test_agent_hooks.py` pins both hooks' decisions.

## Preserve the baseline

Use one writer per checkout. Record HEAD and both staged and unstaged changes before dispatch.
Use an isolated worktree (`isolation: "worktree"`) for independent parallel work.
After the worker stops, compare the baseline and the return diff.
A clean final tree does not prove safety. A worker can discard an earlier uncommitted change and leave no diff.
Investigate a missing file, a restored file or an unexpected HEAD change before you accept the run.

## Observe completion and classify failure

A launch, a live process or a zero exit code alone does not prove completed work.
Confirm the final output, the changed files and the acceptance evidence.

| Failure class | Supervisor response |
|---|---|
| Specification | Correct the accepted contract and its revision. Do not blame a faithful implementation |
| Implementation | Return the failing input and expected behavior as a bounded repair |
| Acceptance test | Repair the fixture or oracle without weakening the intended behavior |
| Harness or configuration | Diagnose launch, model resolution, permissions and reasoning |
| Environment | Name the unavailable dependency and repair it within scope |
| External dependency | Record the blocking event and continue independent authorized work |
