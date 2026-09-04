---
status: draft
---

# How to publish a card or a maps dataset

> **Status: draft** — `scripts/publish_card.py` uploads through
> `huggingface_hub` and verifies the bytes. Its tests run the upload
> against a fake Hub client. No card has shipped through it yet.

## Goal

Push a publication's `README.md`, or a directory of sensitivity maps,
to its Hugging Face repo. Confirm the published bytes equal the source.
Record the SHA-256 in the publication's ledger.

Every card under `publication/` publishes byte-verbatim. The ledger
beside each card (`card-ledger.md`) records the hash of the published
copy. A dataset card carries its own hashes table instead.

## Prerequisites

- A Hugging Face login with write access to the target repo. Run
  `hf auth login` once, or export `HF_TOKEN`. The script reads the
  token through `huggingface_hub` and never prints it.
- `huggingface_hub`. The base install does not carry it, so run the
  script under `uv run --with huggingface_hub`.

## Publish one card

One command per card. The script resolves the model repo from the
card itself: the `quantized_by` front-matter field is the owner and
the H1 title is the repo name.

```bash
uv run --with huggingface_hub python scripts/publish_card.py \
  card publication/model-card
```

The script uploads `README.md`, downloads the committed revision back,
and compares every byte. On a match it prints the SHA-256 and a ledger
row. On a mismatch it exits with status 1 and prints both hashes.

Pass `--dry-run` first. It prints the target repo and the source hash
and uploads nothing:

```bash
uv run --with huggingface_hub python scripts/publish_card.py \
  card publication/model-card --dry-run
```

## Publish a maps dataset

Stage the files in one directory: the maps, their run logs, the
calibration file, the importance matrix when one publishes, and the
dataset card as `README.md`. Follow the layout of
`publication/nemotron-30b-a3b-sensitivity-maps/README.md`. The script
uploads every top-level file in the directory, files first and the
card last (the #85 pattern), and byte-matches each one. The script
never creates a repo: before the first run, create the dataset repo
on the Hub with `hf repo create <owner>/<name> --type dataset`.

```bash
uv run --with huggingface_hub python scripts/publish_card.py \
  dataset <staging dir> --repo Alberto-Codes/<model>-sensitivity-maps
```

`--repo` is required. A dataset card carries no field that names its
repo. Hidden files never upload.

## Record the ledger row

After a byte-match the script prints one row per file, in the shape
of the target table.

A model card prints the `| File | SHA-256 | Bytes |` row that the
30B and Gemma ledgers use:

```
| `README.md` | `<sha256>` | <bytes> |
```

Replace the `README.md` row in the upload table of that publication's
`card-ledger.md`. The 49B ledger keeps a `Local source` column in
place of `Bytes`: paste the hash into the existing row's SHA-256 cell
and keep the source note. Note the date and the commit that changed
the source.

A maps dataset prints the `| File | SHA-256 |` row that the dataset
cards' `## Files and hashes` table uses:

```
| `<file>` | `<sha256>` |
```

The card's own hashes table is the ledger. Paste each file's row into
the staged `README.md`, then rerun the same `dataset` command with the
same `--repo`. The rerun uploads the edited card last, and the Hub
records no new commit for a file whose bytes did not change. The
script byte-matches every file again, so the published card matches
the source.

## Related

- Issues #492 and #497 track the 2026-09-04 card wording changes that
  wait on a re-upload.
- Issue #449 decides whether the Gemma 4 31B map and importance matrix
  publish.
