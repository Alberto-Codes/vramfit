---
description: Rank the whole tracker by distance to a chart Destination and surface rot
allowed-tools: Bash(gh issue list:*), Bash(gh issue view:*), Bash(gh pr list:*), Bash(git log:*)
---

# Board

Status: draft — this command shares the charting page's status and
promotes or demotes with it.

`gh api` is deliberately absent from `allowed-tools`. The pattern
`Bash(gh api:*)` would also pre-approve `--method POST`, `PATCH`, and
`DELETE`, which is every write this command forbids below. The
dependency query in Step 1 prompts instead, once per issue it checks.

Rank every open issue, not one chart's sub-issues. Report what moves a
Destination, what blocks it, and what is rotting.

`chart-triage` answers a different question: what the next chart
session should claim, and it emits session prompts. This command
answers what matters across the whole tracker, including the issues no
chart owns. Type `/board` to decide where a session goes, then ask for
`chart-triage` once that decision lands on a chart. Nothing enforces
that order — a skill fires on its description, and this command fires
only when typed.

Report state and rank it. Resolve nothing. Claim nothing. Close
nothing.

## Step 1 — read the board live

Run the queries. Never answer from memory or conversation history,
because parallel sessions move the state.

```bash
# every open issue with labels, age, assignees, and body. Step 2 and
# Step 4 read prose for reopen triggers and blocker claims, so the
# body comes down in this one call rather than in 30 later ones.
gh issue list --state open --limit 200 \
  --json number,title,labels,createdAt,updatedAt,assignees,body \
  --jq '.[] | "#\(.number)\t\(.createdAt[0:10])\t\(.updatedAt[0:10])\t\([.labels[].name]|join(","))\tassignees:\(.assignees|length)\t\(.title)"'

# the live charts and their Destinations
gh issue list --label chart --state open --json number,title

# blockers, per issue that has any
gh api repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by \
  --jq '[.[] | select(.state == "open") | .number]'
```

Read each open chart's `## Destination` and its `## Notes`. The
Destination is the only definition of progress this project has.

## Step 2 — classify

Put every open issue in exactly one class:

- **Critical path** — closing it moves an open chart's Destination
  closer, directly or by unblocking something that does. Include
  issues outside the chart. A chart's Destination can hang on a plain
  issue, and the sub-issue list will not show it.
- **Blocked** — a real dependency edge is open. Prose saying "blocked
  on #X" is not an edge. Check the API, and report any issue whose
  prose claims a block the edge does not carry.
- **Correctness** — a defect in shipped behavior. Rank a silent
  failure above a loud one. This project's error philosophy exists to
  prevent silent failures, so one sitting open is worse than its age
  suggests.
- **Parked** — carries an explicit reopen trigger. Confirm the trigger
  has not already fired.
- **Meta** — process, docs surface, announcements, tooling.

## Step 3 — rank

Judgment, not query. In order:

1. Critical path, unblocked. The single item closest to a Destination
   leads.
2. Correctness defects, silent before loud, oldest before newest.
3. Critical path, blocked — name the blocker and whether it is moving.
4. Everything else, and say plainly that it is not on the path.

## Step 4 — report the rot

`chart-triage` Step 2 already defines the stale-claim and merge-rot
checks. Apply them across the whole tracker rather than restating
them, and add the two checks that only a whole-board view can make:

- A parked issue whose reopen trigger has fired.
- A correctness defect older than seven days.

Name each finding, or state that none exists.

## Step 5 — say what to do next

One recommendation, three items at most, each with its issue number
and one sentence of reasoning. Then state what the recommendation
deliberately leaves undone, so the maintainer can overrule it.

Print the ranked board before the recommendation. When the
conversation shows an earlier board, state what changed since.
