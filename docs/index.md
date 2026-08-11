# vramfit documentation

Documentation follows the [Diátaxis](https://diataxis.fr/) framework: four
sections with four distinct jobs.

| Section | Job | Reader's question |
|---------|-----|-------------------|
| [Tutorials](tutorials/getting-started.md) | Learning by doing | "Teach me" |
| [How-to guides](how-to/scan-a-model.md) | Task recipes | "How do I…?" |
| [Reference](reference/cli.md) | Facts about the machinery | "What exactly is…?" |
| [Explanation](explanation/why-selective-quantization.md) | Understanding | "Why is it like this?" |

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

Milestone decks under `docs/decks/` are dated point-in-time
snapshots. They carry no status field and do not track current
code.

The rule of thumb: a `sketch` page records what we *think we know* — it is a
design artifact, not a promise. Pages are promoted (`sketch → draft → stable`)
in the PR that lands the code proving them, and demoted when code moves out
from under them.

ADRs use their own lifecycle (`Proposed → Accepted → Deprecated/Superseded`)
described in the [ADR index](adr/index.md).
