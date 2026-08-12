# Charting rules

Status: draft — this file shares the convention page's status and
promotes or demotes with it.

Apply these rules only when the session works a `chart`-labeled
issue or one of its `chart:*` decision tickets. Chart work has no
file path, so this file carries no path scope. Ignore it in every
other session.

The convention lives in
[docs/explanation/charting.md](../../docs/explanation/charting.md).
Read that page before charting or working a ticket. The page is the
source. This file only enforces the load-bearing rules.

Chart #70 was the proving run. It reached its Destination and closed
on 2026-08-12. The rules below carry its findings, several written
mid-run after the convention failed. When a rule fights
reality, flag the conflict to the maintainer. Do not silently obey
and do not silently deviate.

- A decision ticket never stores a decision. Close it with a
  pointer to the record: an ADR, an amendment, a data point, or a
  docs change. A `chart:task` ticket instead closes with what was
  done and the facts later tickets depend on. A `chart:research`
  ticket closes with its findings comment: gist, key facts, and
  primary-source links.
- Add a one-line gist plus both links to the chart's
  `## Decisions so far`.
- Claim before work: self-assign the ticket as the session's first
  write. Open and unassigned means unclaimed.
- Leave a progress comment on the ticket and unassign when the
  session ends without resolving. Any session may release a claim
  older than one day with no comment since assignment.
- The chart body has no lock. Re-read it immediately before
  writing and confirm the write survived.
- Enter a git worktree before editing repo files. Stage exact
  paths, never the whole tree.
- Background agents never write the chart body. The charting
  session folds research gists into Decisions so far.
- Resolve at most one decision ticket per session. Exception:
  `chart:research` tickets fire as parallel background agents at
  charting time.
- Fold overlapping issues at charting time. Absorb and close
  effort-scoped checklists. Each checkbox lands as a ticket, a
  Notes line, or fog. Leave issues that park strategy triggers
  open and cite them.
- Claimable does not mean ready. Check the chart's Notes for
  prerequisites that have no issue to block on.
- Discussion and prototype tickets resolve only through live
  exchange with the maintainer. Never answer the maintainer's side
  of a discussion.
- Strict mode (CLAUDE.md writing system) governs all chart and
  ticket text.
- After resolving, surface any new tickets. Graduate sharpened fog
  out of `## Fog`. Update or close tickets the decision
  invalidated.
- Land every deferral in the tracker before the session ends.
  Route it to a new ticket, a plain issue, or a comment on the
  owning ticket. An open question that lives only in prose is
  untracked.
