---
description: Rank the whole tracker by distance to a chart Destination and surface rot
allowed-tools: Bash(gh issue list:*), Bash(gh issue view:*), Bash(gh api:*), Bash(gh pr list:*), Bash(git log:*)
---

# Board

Rank every open issue, not one chart's sub-issues. Report what moves a
Destination, what blocks it, and what is rotting.

`chart-triage` answers a different question: what the next chart
session should claim, and it emits session prompts. This command
answers what matters across the whole tracker, including the issues no
chart owns. Run this to decide where a session goes. Run `chart-triage`
once that decision lands on a chart.

Report state and rank it. Resolve nothing. Claim nothing. Close
nothing.

## Step 1 — read the board live

Run the queries. Never answer from memory or conversation history,
because parallel sessions move the state.

```bash
# every open issue with labels, age, and assignees
gh issue list --state open --limit 200 \
  --json number,title,labels,createdAt,updatedAt,assignees \
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

Name each of these, or state that none exists:

- A parked issue whose reopen trigger has fired.
- An issue whose prose claims a block with no dependency edge.
- An assigned issue older than one day with no comment since
  assignment.
- A correctness defect older than seven days.
- A merged PR that rots an open issue's body. Check `gh pr list
  --state merged --limit 10` against the top-ranked issues.

## Step 5 — say what to do next

One recommendation, three items at most, each with its issue number
and one sentence of reasoning. Then state what the recommendation
deliberately leaves undone, so the maintainer can overrule it.

Print the ranked board before the recommendation. When the
conversation shows an earlier board, state what changed since.
