---
status: sketch
---

# Charting: planning work bigger than one session

> **Status: sketch** — process design recorded 2026-08-09. The first
> chart is #70 (go-public), created 2026-08-09 from the #66/#67/#11/#68
> cluster. The sections below fold in findings from that run. Promote
> this page when the chart reaches its destination.

## The gap this closes

The project already records decisions well. ADRs carry the
commitments, with amendments and supersessions dated in place. Docs
pages carry trust tiers (`sketch → draft → stable`). The scoreboard
page accretes numbered data points. Issues park triggers so phases
resurface instead of rotting.

What the project lacks is a **claimable set**: one queryable answer
to "what can I decide *right now*?" Open questions live smeared
across ADR "Open questions" sections, sketch pages, and the issues
that park triggers.
Ordering between them is implicit. When an effort spans many sessions
— the go-public push, the rented-GPU lane — each session re-derives
the state of the plan from scattered prose.

Charting fixes that by giving a multi-session effort one index issue
and a set of decision tickets with explicit blocking edges.

## Source and the one inversion

The mechanics adapt Matt Pocock's
[wayfinder skill](https://github.com/mattpocock/skills)
(`skills/engineering/wayfinder/`). Wayfinder plans large work as
decision tickets on the issue tracker, tracks fog of war, and walks a
frontier one session at a time. Most of that transfers directly.

One rule inverts. Wayfinder makes the ticket canonical: "a decision
lives in exactly one place — its ticket." Here the ticket is **never**
canonical. quantfit already has a decision store with a real
lifecycle, and a second store would fork the truth. So:

> **A decision ticket never stores a decision. It closes with a
> pointer to where the decision landed** — an ADR, an ADR amendment,
> a scoreboard data point, or a docs/glossary change in a named PR.

The chart indexes the route. The records carry the truth and the
trust. This also resolves the seam wayfinder leaves open (its
grilling flow can write both a ticket resolution *and* an ADR, with
no rule for which wins).

## The chart

A **chart** is one GitHub issue, label `chart`, that indexes a
multi-session effort. Its body holds five sections and nothing else:

| Section | Contents |
|---------|----------|
| `## Destination` | One or two lines: the state that ends the effort. |
| `## Notes` | Standing constraints for every session, and prerequisites that have no issue. Strict mode applies to all chart and ticket text. |
| `## Decisions so far` | One line per closed ticket: title link, one-line gist, link to the record. |
| `## Fog` | In-scope questions not yet sharp enough to ticket. |
| `## Out of scope` | Ruled-out work, one line each with the reason. Never resurrected in place. |

The chart is an index, not a store. It gists and links; it never
restates a decision. Open tickets do not appear in the body — they
are the open child issues, found by query.

## Decision tickets

Each child issue resolves one question and is sized to one agent
session. The body is a single `## Question`. The answer arrives only
at resolution: a closing comment with the gist and the pointer to the
record. Assets (research notes, prototypes) are linked, not pasted.
A research ticket gathers facts, not a decision, so the inversion
does not bind it. Its record is its own closing comment: one gist
line, the key facts, and links to primary sources.

Types, as labels:

| Label | Mode | Resolves by |
|-------|------|-------------|
| `chart:research` | Background | A background agent gathers facts against primary sources. |
| `chart:prototype` | Live | A throwaway artifact raises the fidelity of a discussion. |
| `chart:discuss` | Live | Conversation with the maintainer. The default type. |
| `chart:task` | Either | Real-world work that unblocks a decision (the only type that does rather than decides). |

A `chart:task` ticket has no decision to record. It closes with what
was done and the facts later tickets depend on, and its line in
Decisions so far records those facts.

Blocking uses GitHub's native issue dependencies, and children are
native sub-issues of the chart. The **claimable set** is the open,
unblocked, unassigned tickets.

Edges may point at issues outside the chart. Wire one whenever a
blocker has an issue. A real-world prerequisite without an issue
lives in the chart's Notes. Check Notes before claiming: claimable
does not mean ready.

## Fog discipline

The graduation test is precision, not answerability: **ticket when
you can state the question sharply** — even if it is still blocked —
**fog when you cannot**. Do not pre-slice fog into ticket-sized
pieces; one patch may graduate into several tickets, or none. Fog
only gathers toward the destination — anything else goes to Out of
scope.

Fog is effort-scoped and lives in the chart. Strategy-scoped
unknowns stay where they already live: sketch pages with a paired
issue that parks the triggers (the
[artifact ecosystem](artifact-ecosystem.md) / issue #11 pattern). A
chart may cite a sketch page; it does not replace one.

## Folding existing issues

A new chart often overlaps issues that already exist. Scope decides
their fate:

- **Effort-scoped issues** exist only to route this effort —
  checklists or umbrellas describing the chart's route. Absorb and
  close them. Each checkbox lands as a ticket, a Notes line, or
  fog. The closing comment maps every checkbox to where it landed
  and links the chart.
- **Strategy-scoped issues** park triggers for a sketch page beyond
  this effort. Leave them open. The chart cites them and rules
  their contents in or out of scope.

An issue that is neither — a plain feature or bug — stays where it
is. Wire a blocking edge when a ticket depends on it.

An open effort-scoped issue beside the chart forks the index, the
way a second decision store would fork the truth.

## Session discipline

- A session **claims** a ticket first, by self-assignment, before
  any work. An open, unassigned ticket is unclaimed.
- **One decision per session.** Research tickets are the exception:
  fire them as parallel background agents at charting time.
- Resolving a ticket ends with three writes: the closing comment
  with the pointer, the gist line in Decisions so far, and any
  newly surfaced tickets or graduated fog.
- A decision that invalidates existing tickets updates or closes
  them in the same session.
- Sessions may run in parallel on different tickets. The chart
  body has no lock. Re-read it immediately before writing. Re-read
  after writing to confirm the prior content and the new line both
  survived.
- Parallel sessions share one checkout unless isolated. A session
  that edits repo files enters its own git worktree first. Every
  session stages exact paths, never the whole tree.
- Background research agents never write the chart body. The
  charting session folds their gist lines into Decisions so far.
- A session that ends without resolving leaves a progress comment
  on its ticket and unassigns itself. Assignment then signals a
  live session or a crash. Every session claims as the same
  account, so a dead claim hides the ticket from the claimable
  set.
- Any session may release a claim older than one day that has no
  comment since assignment.

## Trust mapping

The chart carries no status of its own — it is an index, and its
trust is exactly the trust of the records it points at. The existing
rules keep governing: ADR lifecycle for commitments, page statuses
for docs, promote-in-the-proving-PR per CLAUDE.md. Fog is
pre-`sketch`: not even written from first principles yet, only named.

## When to chart

Chart only when the way to the destination is unclear **and** the
effort exceeds one session. If one session can plan it, plan it in
that session. If the destination discussion surfaces no fog, stop —
no chart.

## Mechanics (GitHub)

Verified against the GitHub REST docs and this repo on 2026-08-09.
Sub-issues and issue dependencies are enabled here.

```bash
# Once per repo: create the labels
gh label create chart
for t in research prototype discuss task; do gh label create "chart:$t"; done

# Create the chart, then add tickets as native sub-issues
gh issue create --label chart --title "<effort>" --body-file chart.md
gh api repos/{owner}/{repo}/issues/<chart>/sub_issues \
  -F sub_issue_id=<child-db-id>          # db id: gh api .../issues/<n> --jq .id

# Wire a blocking edge (second pass, after ids exist)
gh api --method POST \
  repos/{owner}/{repo}/issues/<child>/dependencies/blocked_by \
  -F issue_id=<blocker-db-id>

# Claimable set: open, unassigned sub-issues of the chart...
gh api repos/{owner}/{repo}/issues/<chart>/sub_issues \
  --jq '.[] | select(.state == "open" and (.assignees | length == 0))
        | "#\(.number) \(.title)"'
# ...minus any ticket with open blockers
gh api repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by --jq length

# Claim, resolve, close
gh issue edit <n> --add-assignee @me
gh issue comment <n> --body "<gist + pointer to record>"
gh issue close <n>
```

## Open questions

- **Label spelling.** `chart` / `chart:<type>` proposed here.
  Resolved 2026-08-09: chart #70 used the names unchanged.
  `chart:prototype` exists but no ticket has carried it yet.
- **Research-record size.** The closing comment is the record for a
  research ticket. Unknown whether one comment serves when the
  findings grow large.
- **First chart.** Resolved 2026-08-09: chart #70 indexes the
  go-public effort. The fold rules, the research-record rule, and
  the claimable-set caveat above came from that run.
- **Automation.** The convention runs by hand first. A repo skill
  (`.claude/skills/`) is worth writing only after two or three
  charts prove the shape. Partially resolved 2026-08-10: triage
  automated first (`.claude/skills/chart-triage/`), proven by
  repetition inside chart #70. Full charting automation still
  waits for the second or third chart.
- **CLAUDE.md pointer.** Add one when this page reaches `draft`, not
  before — the trust rules forbid building on a sketch.
