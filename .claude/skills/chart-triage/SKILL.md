---
name: chart-triage
description: Triage a chart's claimable set and emit next-session prompts. Use when the maintainer asks what to work next on a chart, wants a hand-off prompt for a new session, or wants merged work checked against chart state.
---

# Chart triage

Status: draft — this skill shares the charting page's status and
promotes or demotes with it. The convention lives in
[docs/explanation/charting.md](../../../docs/explanation/charting.md).
Read that page before the first triage in a session.

Triage reads state and emits prompts. It resolves nothing: no
claims, no closures, no chart-body writes. Exception: it may fix
index rot it finds (a stale ticket range, a satisfied Notes
prerequisite) — maintenance, never resolution.

## Step 1 — compute the claimable set

Run the queries live. Never answer from memory or conversation
history — parallel sessions move the state.

```bash
# open sub-issues with assignee count and labels
gh api repos/{owner}/{repo}/issues/<chart>/sub_issues \
  --jq '.[] | select(.state == "open") |
        "#\(.number) \(.title) [\([.labels[].name] | join(","))] assignees:\(.assignees | length)"'

# a ticket is claimable only when every blocker is closed
gh api repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by \
  --jq '[.[] | select(.state == "open") | .number]'
```

## Step 2 — subtract reality

- Read the chart's `## Notes` for prerequisites that have no
  issue. Claimable does not mean ready.
- Flag stale claims: assigned, older than one day, no comment
  since assignment. Session discipline lets any session release
  them — report, do not release during triage.
- Diff recent merges against ticket assumptions: `git log
  --oneline -10` and `gh pr list --state merged --limit 5`. A
  merge can resolve a blocker, rot a ticket body, or satisfy a
  Notes prerequisite.

## Step 3 — rank by leverage

Judgment, not query. Weigh:

- Longest pole first: start multi-hour work before quick
  discussions, so it runs while discussions happen.
- Unblock count: prefer tickets that gate the most downstream
  tickets.
- Mode fit: `chart:discuss` needs the maintainer live,
  `chart:task` runs autonomously. Offer one prompt of each when
  both exist.
- External blockers (issues outside the chart) may need a plain
  dev session, not a chart session. Name them separately.

## Step 4 — emit prompts

Every prompt carries:

- The chart, the ticket to claim, and the one-decision boundary.
- The authorizing clause, quoted, with its ADR and decision
  number. A `chart:task` builds what that clause says and nothing
  more, so the session needs the words rather than a paraphrase.
- Pointers to research or evidence already on the ticket.
- Traps the ticket body cannot know: disk headroom, GPU
  serialization, instrument paths, parallel-session hazards.
- Open tickets touching the same surface, by number. A session
  that does not know a neighbor owns part of the scope builds
  into it.
- For discuss tickets: the expected record form (ADR, amendment,
  dated docs note), so the session budgets a PR.

Mark every mechanism as a hint, never a requirement. A prompt is
written before the work and can name an approach that turns out to
be impossible. Write "the counts read through `X` today", not
"read the counts through `X`". A session that finds the hint wrong
raises it on the ticket. It does not reshape the code until the
wording fits, and it does not widen an Accepted clause to make the
build land. #179 did both, and PR #195 did not merge.

A prompt authorizes nothing. Only a record does. Some work plainly
needs a decision first — a port, or an artifact schema field the
authorizing clause does not name. That ticket is not claimable. Say
so on the board and name the missing decision instead of emitting a
prompt.

Report the ranked board first, then the prompts. When the
conversation shows an earlier triage, state what changed since.
