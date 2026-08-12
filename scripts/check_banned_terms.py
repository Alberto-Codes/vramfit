"""Banned-term gate: the pre-rename tool name stays out of the tree.

The tool renamed to vramfit (#118, #120). Issue #154 ruled every tier
of the leftover mentions and swept the tree. A sweep without a gate
drifts back, because ADRs get amended and changelogs get appended.
This gate fails on any occurrence of the pre-rename name outside
`ALLOWLIST`.

The match ignores case, so it also catches the pre-rename exception
root and any title-case prose.

Each allowlist entry names a file whose subject *is* the pre-rename
name. An entry that matches nothing fails too: a stale entry widens
the gate without saying so, which is the drift this gate prevents.

Examples:
    Run against the tracked tree:

    ```console
    $ uv run python scripts/check_banned_terms.py
    checked 219 files, 13 allowed occurrences in 6 files
    ```
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

BANNED_TERM = "quantfit"

ALLOWLIST: dict[str, str] = {
    "publication/model-card/card-ledger.md": (
        "Ledger of the frozen run-root archive (#134). It records the "
        "envelope keys those files carry on disk, and the run root's path."
    ),
    "scripts/check_banned_terms.py": "This gate defines the term.",
    "src/vramfit/adapters/outbound/json_common.py": (
        "The envelope-key guard (#154 tier 2). The guard names the key it "
        "rejects, so the literal is the check."
    ),
    "tests/unit/adapters/test_recipe_json.py": "Proves the guard fires.",
    "tests/unit/adapters/test_scan_checkpoint_json.py": "Proves the guard fires.",
    "tests/unit/adapters/test_sensitivity_map_json.py": "Proves the guard fires.",
}

_PATTERN = re.compile(re.escape(BANNED_TERM), re.IGNORECASE)


def tracked_files(root: Path) -> list[Path]:
    """List the files git tracks or would add under a repository root.

    The listing includes untracked files that no ignore rule excludes.
    A new file carrying the term must fail on its first commit, not
    after it lands.

    Args:
        root: Repository root to list.

    Returns:
        Absolute paths, one per listed file.

    Raises:
        RuntimeError: If git is not on PATH. The gate must fail loudly
            rather than scan nothing.
    """
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is not on PATH — the gate cannot list files")
    # S603: every argument is a literal and `git` is a resolved absolute
    # path. No caller input reaches the command line.
    listing = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [root / name for name in listing.split("\0") if name]


def occurrences(path: Path) -> list[tuple[int, str, int]]:
    """Find every line of a file that carries the banned term.

    A file that does not decode as UTF-8 is binary and yields nothing.
    No tracked binary carries the term as text.

    Args:
        path: File to read.

    Returns:
        One ``(line number, stripped line, match count)`` triple per
        matching line, in file order.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [
        (number, line.strip(), len(_PATTERN.findall(line)))
        for number, line in enumerate(text.splitlines(), 1)
        if _PATTERN.search(line)
    ]


def report(
    root: Path, files: list[Path], allowlist: dict[str, str]
) -> tuple[list[str], int, int]:
    """Judge a file list against an allowlist.

    Args:
        root: Repository root the paths are reported relative to.
        files: Files to read.
        allowlist: Paths that may carry the term, each mapped to the
            reason it may.

    Returns:
        A ``(failures, allowed occurrences, allowed files)`` triple.
        Each failure is a formatted report line.
    """
    if not files:
        # A gate that scans nothing must fail loudly, not pass silently.
        return ["FAIL: the gate scanned no files"], 0, 0

    failures: list[str] = []
    allowed = 0
    matched_entries: set[str] = set()
    for path in files:
        name = path.relative_to(root).as_posix()
        hits = occurrences(path)
        if not hits:
            continue
        if name in allowlist:
            matched_entries.add(name)
            allowed += sum(count for _, _, count in hits)
            continue
        failures.extend(f"FAIL {name}:{number}: {line}" for number, line, _ in hits)

    failures.extend(
        f"FAIL {name}: allowlisted but carries no banned term — delete the entry"
        for name in sorted(set(allowlist) - matched_entries)
    )
    return failures, allowed, len(matched_entries)


def check(root: Path) -> tuple[list[str], int, int]:
    """Scan a repository against the module allowlist.

    Args:
        root: Repository root to scan.

    Returns:
        The `report` triple for every file git tracks or would add.
    """
    return report(root, tracked_files(root), ALLOWLIST)


def main() -> int:
    """Run the gate against the repository that holds this script.

    Returns:
        Process exit code: 1 when any file or allowlist entry fails,
        else 0.
    """
    root = Path(__file__).resolve().parent.parent
    failures, allowed, allowed_files = check(root)
    for line in failures:
        print(line)
    if failures:
        print(
            f"{len(failures)} banned-term failure(s). The tool renamed to "
            "vramfit (#118, #120). Issue #154 ruled the sweep. Extend "
            "ALLOWLIST only for a file whose subject is the old name.",
            file=sys.stderr,
        )
        return 1
    print(f"allowed {allowed} occurrences in {allowed_files} allowlisted files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
