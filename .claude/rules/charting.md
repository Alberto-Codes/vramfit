# Charting rules

Apply these rules only when the session works a `chart`-labeled
issue or one of its `chart:*` decision tickets. Chart work has no
file path, so this file carries no path scope. Ignore it in every
other session.

The convention lives in
[docs/explanation/charting.md](../../docs/explanation/charting.md).
Read that page before charting or working a ticket. The page is the
source. This file only enforces the load-bearing rules.

The page holds `sketch` status. No chart has proven the convention
yet. When a rule fights reality, flag the conflict to the
maintainer. Do not silently obey and do not silently deviate.

- A decision ticket never stores a decision. Close it with a
  pointer to the record: an ADR, an amendment, a data point, or a
  docs change. A `chart:task` ticket instead closes with what was
  done and the facts later tickets depend on.
- Add a one-line gist plus both links to the chart's
  `## Decisions so far`.
- Claim before work: self-assign the ticket as the session's first
  write. Open and unassigned means unclaimed.
- Resolve at most one decision ticket per session. Exception:
  `chart:research` tickets fire as parallel background agents at
  charting time.
- Discussion and prototype tickets resolve only through live
  exchange with the maintainer. Never answer the maintainer's side
  of a discussion.
- Strict mode (CLAUDE.md writing system) governs all chart and
  ticket text.
- After resolving, surface any new tickets. Graduate sharpened fog
  out of `## Fog`. Update or close tickets the decision
  invalidated.
