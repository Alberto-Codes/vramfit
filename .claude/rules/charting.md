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
on 2026-08-12. Most rules below carry its findings, several written
mid-run after the convention failed. Later charts add findings too, and
such a rule names the chart it came from. When a rule fights
reality, flag the conflict to the maintainer. Do not silently obey
and do not silently deviate.

- A decision ticket never stores a decision. Close it with a
  pointer to the record: an ADR, an amendment, a data point, or a
  docs change. A `chart:task` ticket instead closes with what was
  done and the facts later tickets depend on. A `chart:research`
  ticket closes with its findings comment: gist, key facts, and
  primary-source links.
- A `chart:task` never introduces a port. It never adds an
  artifact schema field the authorizing clause does not name. It
  never writes into an Accepted ADR beyond an observed
  consequence. Each is a decision. Open a `chart:discuss` instead.
- Refusing a malformed input no record addresses is a bug fix.
  Refusing a case a record already answers is a decision. #187
  refused a negative count, and that was right. #195 refused a
  file ADR-0026 says to report on, and that was wrong.
- Check precedent before adding a port or any comparable seam. Run
  `git log --follow -S'<symbol>' -- <path>`. Without `--follow` the
  history stops at the 2026-08 rename. Then read what kind of
  change introduced the last one. No `chart:task` has added a port.
- The ticket's wording is not authority. Build to the record when
  the two disagree. Raise the conflict on the ticket. Never
  reshape the code until the wording fits.
- Read the open tickets before building. Another may already own
  part of the scope. #179 pre-empted #191's echo policy and
  overlapped #194's coverage record.
- Add a one-line gist plus both links to the chart's
  `## Decisions so far`.
- Claim before work: self-assign the ticket as the session's first
  write. Open and unassigned means unclaimed.
- Leave a progress comment on the ticket and unassign when the
  session ends without resolving. Any session may release a claim
  older than one day with no comment since assignment.
- The chart body has no lock. Re-read it immediately before
  writing and confirm the write survived. Write the chart after
  the tracker edits land, or the entry describes a state the
  session then changes. #332's entry named one open blocker
  truthfully, and the same session wired a second 28 seconds
  later.
- A count or a measurement the session asserts comes from a
  command, run in that session. A figure a record or a ruling
  fixes cites the record instead. Use `git diff --stat` for a
  change's size, and the `blocked_by` call in the convention's
  Mechanics block for a blocker count. #332's entry reused a
  four-file count of one quantity as the cost of another, against
  22 changed files.
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
