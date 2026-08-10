# Charting rules

Apply these rules when a session touches a `chart`-labeled issue or
its `chart:*` decision tickets. The full convention is
[docs/explanation/charting.md](../../docs/explanation/charting.md) —
read it before charting or working a ticket. The page is the source;
this file only enforces the load-bearing rules.

- A decision ticket never stores a decision. Close it with a pointer
  to the record: an ADR, an amendment, a data point, or a docs
  change. The chart's `## Decisions so far` gets a one-line gist
  plus both links.
- Claim before work: self-assign the ticket as the session's first
  write. Open and unassigned means unclaimed.
- Resolve at most one decision ticket per session. Exception:
  `chart:research` tickets fire as parallel background agents.
- HITL ticket types (`chart:discuss`, `chart:prototype`) resolve
  only through live exchange with the maintainer. Never answer the
  maintainer's side of a discussion.
- Strict mode (CLAUDE.md writing system) governs all chart and
  ticket text.
- After resolving: surface new tickets, graduate sharpened fog out
  of `## Fog`, and update or close tickets the decision invalidated.
