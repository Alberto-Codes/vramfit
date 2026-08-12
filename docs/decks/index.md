---
status: sketch
---

# Decks

> **Status: sketch** — recorded 2026-08-11 by
> [#124](https://github.com/Alberto-Codes/vramfit/issues/124). The ruling
> below is decided. The deck format and the filenames stay unproven until
> the first two decks land. That pair promotes this page to `draft`.

A deck argues the project's case to an audience. A docs page answers one
reader's question. That difference sets this convention.

## What a deck does

[Diátaxis](https://diataxis.fr/) governs the four documentation modes. A
deck is none of them. A deck is a delivery format, so it must not restate
an explanation page.

A deck does two things no docs page does:

1. **A deck argues in sequence.** A reader enters a docs page at any
   heading. A deck runs start to finish: thesis, evidence, conclusion.
2. **A deck orients a new reader.** That reader reaches the current state
   without reading every pull request.

## The two decks

| Deck | Audience | Reader's question |
|------|----------|-------------------|
| Review | Stakeholders and product owners | "What does this prove?" |
| Deep dive | Peer reviewers and architects | "How does the machinery work?" |

## Sources

Name each deck source by mode: `docs/decks/review.md` and
`docs/decks/deep-dive.md`. The front matter sets `theme: vramfit`.

## Currency

A deck describes the present state, so it names a release instead of a
date. The lead slide states the version, for example "as of v0.1.0".
Regenerate both decks at each release.

A deck carries no `status` field. The version on its lead slide replaces
it. A deck that names an older release declares its own staleness. This
page carries a status, because it records the convention instead of the
present state.

## PDFs

Marp builds a PDF from each deck source. The PDF is a release asset, not
a tracked file. `docs/decks/.gitignore` ignores `*.pdf`.

A release attaches the PDFs built from that release's deck sources. No
release backfills PDFs for an earlier version.

## Ruled 2026-08-11 (#124)

- Decks describe the present state. The per-pull-request deck format
  ends.
- PR #130 deletes the six pre-rename deck sources. Git never tracked
  their PDFs. Their measurements already live in the ADRs and the
  explanation pages.
- Releases attach deck PDFs from the first release forward.
- This ruling supersedes the deck clause of the #119 rename ruling.

**Not yet done.** The `deep-dive` and `review` skills still generate
per-pull-request decks.
[#131](https://github.com/Alberto-Codes/vramfit/issues/131) rewrites both
skills. [#132](https://github.com/Alberto-Codes/vramfit/issues/132) then
authors the first two decks.

## Build

```console
$ marp --theme-set docs/decks/theme.css --pdf docs/decks/<deck>.md
```
