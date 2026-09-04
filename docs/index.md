# vramfit documentation

Documentation follows the [Diátaxis](https://diataxis.fr/) framework: four
sections with four distinct jobs.

| Section | Job | Reader's question |
|---------|-----|-------------------|
| [Tutorials](tutorials/first-run.md) | Learning by doing | "Teach me" |
| [How-to guides](how-to/scan-a-model.md) | Task recipes | "How do I…?" |
| [Reference](reference/cli.md) | Facts about the machinery | "What exactly is…?" |
| [Explanation](explanation/why-selective-quantization.md) | Understanding | "Why is it like this?" |

Two tutorials open the docs. [First run](tutorials/first-run.md)
solves the published 49B map on a CPU in two commands, from the base
install. [Getting started](tutorials/getting-started.md) runs the
full loop on a small model with the GPU extras: scan, plan, validate,
pack.

Design decisions are recorded separately as [ADRs](adr/index.md). Three
pages anchor the rest: the [glossary](reference/glossary.md) (the
project's canonical vocabulary — one term per concept),
[prior art](reference/prior-art.md) (annotated external sources), and
the [architecture overview](explanation/architecture.md) (the pipeline
and the hexagonal layers, with diagrams).

## Page status

The project is being built in the open, so docs are written *ahead of* the
code they describe. Every page carries a `status` in its frontmatter, shown
under the title:

| Status | Meaning |
|--------|---------|
| `sketch` | Written from first principles; **no code verifies it yet**. Expect breaking changes. |
| `draft` | Partially backed by working code or real measurements; details may still shift. |
| `stable` | Verified against the current state of `main`. |

Every deck under `docs/decks/` describes the present state. A deck names
a release instead of a date and carries no status field. The
[deck conventions](decks/index.md) page carries a status like any other
page.

The rule of thumb: a `sketch` page records what we *think we know* — it is a
design artifact, not a promise. Pages are promoted (`sketch → draft → stable`)
in the PR that lands the code proving them, and demoted when code moves out
from under them.

ADRs use their own lifecycle (`Proposed → Accepted → Deprecated/Superseded`)
described in the [ADR index](adr/index.md).
