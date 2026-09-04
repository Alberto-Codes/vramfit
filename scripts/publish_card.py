r"""Upload a publication's files to Hugging Face and verify the bytes.

The cards under ``publication/`` publish byte-verbatim. Each ledger
records the SHA-256 of the published copy. This script does the
upload, downloads the published copy back, and compares the bytes.
A mismatch exits nonzero. Two modes exist:

- ``card <publication dir>`` uploads ``README.md`` to the model repo
  the card names: ``<quantized_by>/<H1 title>``.
- ``dataset <dir> --repo <owner/name>`` uploads every top-level file
  in the directory to that dataset repo (the #85 pattern: files
  first, then the card).

Both modes print one ledger row per file, in the shape of the target
table. ``--dry-run`` prints the target repo and the hashes and
uploads nothing.

The script never prints the token. ``huggingface_hub`` reads it from
the local login (``hf auth login``) or ``HF_TOKEN``.

Examples:
    Dry run against the 49B card:

    ```console
    $ uv run --with huggingface_hub python scripts/publish_card.py \\
        card publication/model-card --dry-run
    ```
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

_TITLE = re.compile(r"^# (\S+)\s*$", re.MULTILINE)
_QUANTIZED_BY = re.compile(r"^quantized_by:\s*(\S+)\s*$", re.MULTILINE)


class UploadError(RuntimeError):
    """A file did not reach the Hub as the exact bytes on disk."""


class HubClient(Protocol):
    """The two ``huggingface_hub.HfApi`` calls this script needs."""

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> Any:
        """Commit one file. Returns the commit info (``oid`` names the revision)."""
        ...

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        revision: str | None,
        cache_dir: str,
        force_download: bool,
    ) -> str:
        """Fetch one file at ``revision``. Returns the local path."""
        ...


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo(card_text: str) -> str:
    """Return the model repo id from a card's own text.

    The H1 title is the repo name. The ``quantized_by`` front-matter
    field is the owner.
    """
    title_match = _TITLE.search(card_text)
    if title_match is None:
        raise UploadError("the card has no H1 title to name the repo")
    owner_match = _QUANTIZED_BY.search(card_text)
    if owner_match is None:
        raise UploadError("the card has no quantized_by field to name the owner")
    return f"{owner_match.group(1)}/{title_match.group(1)}"


def ledger_row(name: str, digest: str, size: int, repo_type: str) -> str:
    """Format the ledger row for one published file.

    A dataset card's hashes table is ``| File | SHA-256 |``. A model
    ledger's upload table adds a byte-count column.
    """
    if repo_type == "dataset":
        return f"| `{name}` | `{digest}` |"
    return f"| `{name}` | `{digest}` | {size:,} |"


def upload_and_verify(api: HubClient, path: Path, repo_id: str, repo_type: str) -> str:
    """Upload ``path`` under its own name, fetch it back, and compare bytes."""
    expected = path.read_bytes()
    info = api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Upload {path.name}",
    )
    revision = getattr(info, "oid", None)
    with tempfile.TemporaryDirectory(prefix="publish-card-") as cache:
        fetched = api.hf_hub_download(
            repo_id,
            path.name,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache,
            force_download=True,
        )
        published = Path(fetched).read_bytes()
    if published != expected:
        raise UploadError(
            f"{path.name}: published bytes differ from the source "
            f"({hashlib.sha256(published).hexdigest()} on the Hub, "
            f"{hashlib.sha256(expected).hexdigest()} on disk)"
        )
    return hashlib.sha256(published).hexdigest()


def _hub_api() -> HubClient:
    """Build the real client. Imported here so ``--dry-run`` needs no extra."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 — optional, lazy
    except ImportError as exc:
        raise UploadError(
            "huggingface_hub is not installed: run under "
            "`uv run --with huggingface_hub python scripts/publish_card.py …`"
        ) from exc
    return HfApi()


def _files_for(mode: str, directory: Path) -> list[Path]:
    if mode == "card":
        return [directory / "README.md"]
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if not files:
        raise UploadError(f"{directory} holds no files to upload")
    # The #85 pattern: files first, then the card.
    return [p for p in files if p.name != "README.md"] + [
        p for p in files if p.name == "README.md"
    ]


def _card_repo(directory: Path) -> str:
    card = directory / "README.md"
    if not card.is_file():
        raise UploadError(f"{directory} holds no README.md")
    return resolve_repo(card.read_text(encoding="utf-8"))


def run(argv: Sequence[str], api_factory: Callable[[], HubClient] = _hub_api) -> int:
    """Run the script. Returns the exit status."""
    args = _parse(argv)
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"error: {directory} is not a directory", file=sys.stderr)
        return 2
    try:
        if args.mode == "dataset":
            repo_id, repo_type = args.repo, "dataset"
        else:
            repo_id, repo_type = _card_repo(directory), "model"
        files = _files_for(args.mode, directory)
        print(f"target: {repo_type} repo {repo_id}")
        for path in files:
            print(
                f"source: {path} sha256 {sha256_of(path)} bytes {path.stat().st_size:,}"
            )
        if args.dry_run:
            print("dry run: nothing uploaded")
            return 0
        api = api_factory()
        for path in files:
            digest = upload_and_verify(api, path, repo_id, repo_type)
            print(f"verified: {path.name} matches the Hub byte for byte")
            size = path.stat().st_size
            print(f"ledger: {ledger_row(path.name, digest, size, repo_type)}")
    except UploadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="publish_card.py",
        description="Upload a publication's files to Hugging Face and verify the bytes.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    card = sub.add_parser(
        "card", help="upload one publication's README.md to its model repo"
    )
    dataset = sub.add_parser(
        "dataset", help="upload every top-level file in a directory to a dataset repo"
    )
    dataset.add_argument(
        "--repo", required=True, help="the target dataset repo id (<owner>/<name>)"
    )
    for p in (card, dataset):
        p.add_argument("directory", help="the publication directory")
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="print the target repo and hashes, upload nothing",
        )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
