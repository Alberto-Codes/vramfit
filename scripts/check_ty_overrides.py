"""Ty-override gate: a lazy optional import needs its override entry.

CI installs no extras (`uv sync --locked --dev`). A dev venv that
carries `--extra scan` resolves torch, transformers, and gguf-py, so
`uv run ty check` passes locally on a module that fails in CI with
`unresolved-import`. The `[[tool.ty.overrides]]` list in pyproject.toml
suppresses that rule for the files ADR-0005 allows to import an extra.
The list is hand-maintained, so a new module lands without its entry
and the failure appears only after the push.

This gate fails when a file under `[tool.ty.src] include` imports an
optional root and no override covers it. It also fails an include
pattern that matches no file, because a stale pattern widens the gate
without saying so.

The gate reads imports with `ast`, so prose that names a module does
not count. `LlamaCppPacker`'s docstring says the interpreter "Must
import torch" and the gate ignores it.

Examples:
    Run against the repository:

    ```console
    $ uv run python scripts/check_ty_overrides.py
    covered 17 file(s) importing an optional root
    ```

See Also:
    - [ADR-0005](../docs/adr/0005-heavy-deps-as-extras.md): Why torch
      and gguf-py sit behind extras at all.
    - `scripts/check_banned_terms.py`: The other allowlist gate. It
      fails a stale entry for the same reason this one does.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final, NamedTuple

# Import roots the base install does not carry. Each maps to the
# reason it is optional. The extras that provide them are declared in
# `[project.optional-dependencies]` (ADR-0005).
OPTIONAL_ROOTS: Final[dict[str, str]] = {
    "accelerate": "scan extra — device mapping for the torch meter.",
    "gguf": "scan extra — gguf-py reads the imatrix and GGUF tensors.",
    "numpy": "rides in with gguf-py, never declared on its own.",
    "safetensors": "scan extra — shard restore for offload (ADR-0015).",
    "torch": "scan extra — the GPU stack the plan step must not need.",
    "transformers": "scan extra — checkpoint loading for the meter.",
}

# The rule an override must silence for a lazy import to type-check.
_SUPPRESSED_RULE: Final[str] = "unresolved-import"

# `[tool.ty.src] include`. The gate scans what ty checks, so a script
# under `scripts/` stays out — ty names that directory a search path,
# never a check target.
_SCANNED_ROOTS: Final[tuple[str, ...]] = ("src", "tests")


class Override(NamedTuple):
    """One `[[tool.ty.overrides]]` block.

    Attributes:
        patterns (tuple[str, ...]): The block's `include` globs, as
            written in pyproject.toml.
        suppresses (bool): Whether the block sets `unresolved-import`
            to `ignore`. A block that covers a file without silencing
            the rule leaves it failing in CI.
    """

    patterns: tuple[str, ...]
    suppresses: bool


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile one ty include glob to a full-match regular expression.

    `**` crosses directory separators and `*` does not. The patterns in
    use are literal paths and one recursive directory, so this covers
    them without a glob dependency.

    Args:
        pattern: Include glob, with `/` separators.

    Returns:
        A compiled pattern that matches a whole POSIX-style path.
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts) + r"\Z")


def read_overrides(pyproject: Path) -> list[Override]:
    """Read every `[[tool.ty.overrides]]` block from pyproject.toml.

    Args:
        pyproject: Path to the project's pyproject.toml.

    Returns:
        One `Override` per block, in file order. An empty list means
        the project declares no override.

    Raises:
        OSError: If the file cannot be read.
        tomllib.TOMLDecodeError: If the file does not parse.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    blocks = data.get("tool", {}).get("ty", {}).get("overrides", [])
    return [
        Override(
            patterns=tuple(block.get("include", ())),
            suppresses=block.get("rules", {}).get(_SUPPRESSED_RULE) == "ignore",
        )
        for block in blocks
    ]


def optional_imports(path: Path) -> list[tuple[int, str]]:
    """Find every import of an optional root in one Python file.

    The walk reaches imports at any nesting, so a function-level import
    and one under `if TYPE_CHECKING:` both count. Ty resolves both.

    Args:
        path: Python file to parse.

    Returns:
        One `(line number, root module)` pair per optional import, in
        line order. A file that does not parse yields nothing —
        `ruff` already fails a syntax error, and this gate must not
        duplicate that report.

    Raises:
        OSError: If the file cannot be read.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # `node.module` is None on a relative import. The project
            # bans those, and a relative import names no optional root.
            roots = [node.module.split(".")[0]] if node.module else []
        else:
            continue
        found.extend((node.lineno, root) for root in roots if root in OPTIONAL_ROOTS)
    return sorted(set(found))


def scanned_files(root: Path) -> list[Path]:
    """List the Python files git tracks under the scanned roots.

    The listing reads the index, so a newly staged module is covered on
    its first commit. An untracked scratch file stays out. It must not
    fail everyone's commit.

    Args:
        root: Repository root to list.

    Returns:
        Absolute paths, one per listed file.

    Raises:
        RuntimeError: If git is not on PATH. The gate must fail loudly
            rather than scan nothing.
        subprocess.CalledProcessError: If git rejects the listing.
    """
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is not on PATH — the gate cannot list files")
    # S603: every argument is a literal or a constant from this module,
    # and `git` is a resolved absolute path. No caller input reaches
    # the command line.
    listing = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z", "--cached", "--", *_SCANNED_ROOTS],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [root / name for name in listing.split("\0") if name.endswith(".py")]


def report(
    root: Path, files: list[Path], overrides: list[Override]
) -> tuple[list[str], int]:
    """Judge a file list against the override blocks.

    Args:
        root: Repository root the paths are reported relative to.
        files: Python files to parse.
        overrides: Blocks read from pyproject.toml.

    Returns:
        A `(failures, covered files)` pair. Each failure is a
        formatted report line.
    """
    if not files:
        # A gate that scans nothing must fail loudly, not pass silently.
        return ["FAIL: the gate scanned no files"], 0

    matchers = [
        (override, [glob_to_regex(pattern) for pattern in override.patterns])
        for override in overrides
    ]
    matched_patterns: set[str] = set()
    failures: list[str] = []
    covered = 0

    for path in files:
        name = path.relative_to(root).as_posix()
        for override, regexes in matchers:
            for pattern, regex in zip(override.patterns, regexes, strict=True):
                if regex.match(name):
                    matched_patterns.add(pattern)
        imports = optional_imports(path)
        if not imports:
            continue
        silencing = [
            override
            for override, regexes in matchers
            if override.suppresses and any(regex.match(name) for regex in regexes)
        ]
        if silencing:
            covered += 1
            continue
        failures.extend(
            f"FAIL {name}:{line}: imports {module!r} with no "
            f"[[tool.ty.overrides]] entry that ignores {_SUPPRESSED_RULE}"
            for line, module in imports
        )

    declared = {pattern for override in overrides for pattern in override.patterns}
    failures.extend(
        f"FAIL {pattern!r}: include pattern matches no file — delete the entry"
        for pattern in sorted(declared - matched_patterns)
    )
    return failures, covered


def check(root: Path) -> tuple[list[str], int]:
    """Scan a repository against its own pyproject.toml.

    Args:
        root: Repository root to scan.

    Returns:
        The `report` pair for every tracked Python file under the
        scanned roots.
    """
    return report(root, scanned_files(root), read_overrides(root / "pyproject.toml"))


def main() -> int:
    """Run the gate against the repository that holds this script.

    Returns:
        Process exit code: 1 when any file or include pattern fails,
        else 0.
    """
    root = Path(__file__).resolve().parent.parent
    failures, covered = check(root)
    for line in failures:
        print(line)
    if failures:
        print(
            f"{len(failures)} ty-override failure(s). CI installs no extras, "
            "so a lazy optional import needs an [[tool.ty.overrides]] include "
            "entry in pyproject.toml (ADR-0005). Add the source module and "
            "every test that imports the dep at module level.",
            file=sys.stderr,
        )
        return 1
    print(f"covered {covered} file(s) importing an optional root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
