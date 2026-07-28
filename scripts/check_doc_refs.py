"""Docs-reference gate: dotted ``quantfit.*`` paths in docs must exist.

Refactors move modules — markdown and docstrings do not notice. This
gate extracts every dotted ``quantfit.…`` reference from the living docs
AND the Python sources (docstring cross-references rot too) and verifies
each resolves to an importable module or an attribute of one. Decks
(``docs/decks/``) are excluded — they are dated point-in-time artifacts.

Examples:
    Run against the default doc set:

    ```console
    $ uv run python scripts/check_doc_refs.py
    checked 42 references
    ```
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

_REF = re.compile(r"\bquantfit(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")

DEFAULT_DOC_GLOBS = (
    "README.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "docs/**/*.md",
    ".github/copilot-instructions.md",
    "src/**/*.py",
    "tests/**/*.py",
)
EXCLUDED_PARTS = ("decks",)


def _resolves(ref: str) -> bool:
    """Report whether a dotted reference names a module or attribute.

    Args:
        ref: Dotted path, e.g. ``quantfit.domain.solver`` or
            ``quantfit.domain.solver.solve``.

    Returns:
        True when the path imports as a module, or its last component is
        an attribute of the importable parent.
    """
    try:
        if importlib.util.find_spec(ref) is not None:
            return True
    except ModuleNotFoundError:
        pass
    parent, _, attr = ref.rpartition(".")
    try:
        module = importlib.import_module(parent)
    except ModuleNotFoundError:
        return False
    return hasattr(module, attr)


def main(argv: list[str]) -> int:
    """Check every dotted quantfit reference in the doc set.

    Args:
        argv: Optional explicit file paths; defaults to the doc globs.

    Returns:
        Process exit code: 1 when any reference fails to resolve.
    """
    failures = 0
    files: list[Path] = []
    if argv:
        for a in argv:
            path = Path(a)
            if not path.is_file():
                print(f"FAIL {a}: not a file")
                failures += 1
                continue
            files.append(path)
    else:
        for pattern in DEFAULT_DOC_GLOBS:
            files.extend(Path().glob(pattern))
    checked = 0
    for path in sorted(set(files)):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        for ref in sorted(set(_REF.findall(path.read_text(encoding="utf-8")))):
            if ref.endswith(".git"):  # clone URLs, not module paths
                continue
            checked += 1
            if not _resolves(ref):
                print(f"FAIL {path}: unresolved reference {ref}")
                failures += 1
    if checked == 0:
        # Scanning nothing means the gate is misconfigured, not passing.
        print("FAIL: no references found to check")
        failures += 1
    print(f"checked {checked} references")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
