"""Ty-override gate: a lazy optional import needs its override entry.

CI installs no extras (`uv sync --locked --dev`). A dev venv that
carries `--extra scan` resolves torch, transformers, and gguf-py.
`uv run ty check` then passes locally on a module that fails in CI
with `unresolved-import`. The `[[tool.ty.overrides]]` list in
pyproject.toml suppresses that rule for the files ADR-0005 allows to
import an extra. The list is hand-maintained, so a new module lands
without its entry and the failure appears only after the push.

The gate fails three ways:

- A file under `[tool.ty.src] include` imports an optional root and no
  override covers it.
- An include pattern covers no file that imports an optional root. A
  stale entry widens the gate without saying so.
- An include pattern uses a shape the gate cannot model. See
  `unsupported_reason`.

The gate reads imports with `ast`, so prose that names a module does
not count. `LlamaCppPacker`'s docstring says the interpreter "Must
import torch" and the gate ignores it.

`pytest.importorskip("torch")` names its module in a string, so the
gate does not see it. That call resolves at runtime and `ty` never
reports it, so the two agree.

Examples:
    Run against the repository:

    ```console
    $ uv run python scripts/check_ty_overrides.py
    covered 17 file(s) importing an optional root
    ```

See Also:
    - [ADR-0005](../docs/adr/0005-heavy-deps-as-extras.md): Why torch,
      transformers, and safetensors sit behind extras. The ADR names
      no other package, and `pyproject.toml` places gguf-py by the
      same rule.
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

# Import roots a base install cannot resolve. Each maps to what
# provides it. The list is hand-maintained, so it states a fact this
# gate cannot check: issue #244 carries the work to derive it.
# Measured 2026-08-15 against `uv sync --locked --dev` with no extras.
OPTIONAL_ROOTS: Final[dict[str, str]] = {
    "gguf": "gguf extra — gguf-py reads the imatrix and GGUF tensors.",
    "numpy": "gguf extra, and gguf-py depends on it too.",
    "safetensors": "scan extra — shard restore for offload (ADR-0015).",
    "tokenizers": "rides in with transformers, declared nowhere.",
    "torch": "scan extra — the GPU stack the plan step must not need.",
    "transformers": "scan extra — checkpoint loading for the meter.",
}

# The rule an override must silence for a lazy import to type-check.
_SUPPRESSED_RULE: Final[str] = "unresolved-import"

# Fallback when pyproject.toml names no `[tool.ty.src] include`. ty
# defaults to the project root, and the gate states a narrower guess
# rather than walking the whole tree.
_DEFAULT_SCANNED_ROOTS: Final[tuple[str, ...]] = ("src", "tests")


class TyOverride(NamedTuple):
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


def unsupported_reason(pattern: str, root: Path) -> str | None:
    """Say why the gate cannot model one include pattern.

    The gate models two shapes: a literal file path, and a prefix that
    ends in `/**`. Measured against ty 0.0.64, four other shapes read
    differently than a naive translation suggests. A bare directory and
    a trailing slash both recurse in ty and match nothing here, and a
    mid-pattern `**` also matches zero directories in ty.

    A gate that mis-models a pattern reports two failures at once, and
    the second tells the reader to delete a live suppression. So the
    gate refuses the shapes it cannot model.

    Args:
        pattern: Include glob, with `/` separators.
        root: Repository root, used to spot a directory path.

    Returns:
        The reason and the fix, or None when the gate models the
        pattern.
    """
    if pattern.endswith("/"):
        return f"a trailing '/'. Write '{pattern}**' instead"
    body = pattern.removesuffix("/**")
    if "*" in body or "?" in body:
        return "a wildcard outside a trailing '/**'. The gate models neither"
    if not pattern.endswith("/**") and (root / pattern).is_dir():
        return f"a directory. Write '{pattern}/**' instead"
    return None


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile one supported include glob to a full-match expression.

    The caller rejects every shape outside the two this models, so a
    trailing `**` crosses directory separators and nothing else globs.

    Args:
        pattern: Include glob, with `/` separators.

    Returns:
        A compiled pattern that matches a whole POSIX-style path.
    """
    if pattern.endswith("/**"):
        prefix = re.escape(pattern[: -len("**")])
        return re.compile(prefix + r".*\Z")
    return re.compile(re.escape(pattern) + r"\Z")


def read_overrides(pyproject: Path) -> list[TyOverride]:
    """Read every `[[tool.ty.overrides]]` block from pyproject.toml.

    Args:
        pyproject: Path to the project's pyproject.toml.

    Returns:
        One `TyOverride` per block, in file order. An empty list means
        the project declares no override.

    Raises:
        OSError: If the file cannot be read.
        tomllib.TOMLDecodeError: If the file does not parse.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    blocks = data.get("tool", {}).get("ty", {}).get("overrides", [])
    return [
        TyOverride(
            patterns=tuple(block.get("include", ())),
            suppresses=block.get("rules", {}).get(_SUPPRESSED_RULE) == "ignore",
        )
        for block in blocks
    ]


def read_scanned_roots(pyproject: Path) -> tuple[str, ...]:
    """Read `[tool.ty.src] include` from pyproject.toml.

    The gate scans what ty checks. Reading the value keeps the two from
    drifting apart, which a hardcoded copy could not do.

    Args:
        pyproject: Path to the project's pyproject.toml.

    Returns:
        The configured roots, or `_DEFAULT_SCANNED_ROOTS` when the file
        names none.

    Raises:
        OSError: If the file cannot be read.
        tomllib.TOMLDecodeError: If the file does not parse.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    include = data.get("tool", {}).get("ty", {}).get("src", {}).get("include")
    return tuple(include) if include else _DEFAULT_SCANNED_ROOTS


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
        duplicate that report. A file the index lists and the worktree
        lacks also yields nothing.

    Raises:
        OSError: If the file exists and cannot be read.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    except FileNotFoundError:
        # `git ls-files --cached` lists a file removed with plain `rm`.
        # The gate reports nothing rather than raising a traceback.
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


def scanned_files(root: Path, scanned_roots: tuple[str, ...]) -> list[Path]:
    """List the Python files git tracks under the scanned roots.

    The listing reads the index, so a newly staged module is covered on
    its first commit. An untracked file stays out. The gate must not
    fail a commit over someone's scratch file.

    Args:
        root: Repository root to list.
        scanned_roots: Directories to list, from `[tool.ty.src]
            include`.

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
    # S603: `git` is a resolved absolute path and every other argument
    # is a literal or a value read from the project's own pyproject.
    listing = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z", "--cached", "--", *scanned_roots],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    suffixes = (".py", ".pyi")
    return [root / n for n in listing.split("\0") if n.endswith(suffixes)]


def report(
    root: Path, files: list[Path], overrides: list[TyOverride]
) -> tuple[list[str], int]:
    """Judge a file list against the override blocks.

    A pattern is keyed by its block index as well as its text, so a
    dead entry still fails when another block spells it the same way.

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

    unsupported = [
        f"FAIL {pattern!r}: the include pattern uses {reason}"
        for block, override in enumerate(overrides)
        for pattern in override.patterns
        if (reason := unsupported_reason(pattern, root)) is not None
    ]
    if unsupported:
        # Every later judgement would rest on a pattern the gate reads
        # differently than ty does. Report the shapes and stop.
        return unsupported, 0

    matchers = [
        (block, override, [glob_to_regex(p) for p in override.patterns])
        for block, override in enumerate(overrides)
    ]
    earning: set[tuple[int, str]] = set()
    failures: list[str] = []
    covered = 0

    for path in files:
        name = path.relative_to(root).as_posix()
        imports = optional_imports(path)
        if not imports:
            continue
        silenced = False
        for block, override, regexes in matchers:
            for pattern, regex in zip(override.patterns, regexes, strict=True):
                if not regex.match(name):
                    continue
                # The entry earns its place only against a file that
                # imports an optional root, matching the allowlist rule
                # in `check_banned_terms.py`.
                earning.add((block, pattern))
                silenced = silenced or override.suppresses
        if silenced:
            covered += 1
            continue
        failures.extend(
            f"FAIL {name}:{line}: imports {module!r} with no "
            f"[[tool.ty.overrides]] entry that ignores {_SUPPRESSED_RULE}"
            for line, module in imports
        )

    declared = {
        (block, pattern)
        for block, override in enumerate(overrides)
        for pattern in override.patterns
    }
    failures.extend(
        f"FAIL {pattern!r}: the include pattern covers no file that imports "
        "an optional root — delete the entry"
        for _, pattern in sorted(declared - earning)
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
    pyproject = root / "pyproject.toml"
    return report(
        root,
        scanned_files(root, read_scanned_roots(pyproject)),
        read_overrides(pyproject),
    )


def guidance(failures: list[str]) -> str:
    """Choose the closing advice for a run's failures.

    A missing entry and a stale entry need opposite actions, so the
    gate names the one its failures call for.

    Args:
        failures: Report lines from `report`.

    Returns:
        One sentence pair naming what to do next.
    """
    stale = [line for line in failures if "delete the entry" in line]
    if len(stale) == len(failures):
        return (
            "Every failure names an include pattern that earns nothing. "
            "Delete the entry, then run `uv run ty check` against an "
            "extras-free venv to confirm the file still passes."
        )
    return (
        "CI installs no extras, so a lazy optional import needs an "
        "[[tool.ty.overrides]] include entry in pyproject.toml (ADR-0005). "
        "Add the source module and every test that imports the dep, at any "
        "nesting."
    )


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
            f"{len(failures)} ty-override failure(s). {guidance(failures)}",
            file=sys.stderr,
        )
        return 1
    print(f"covered {covered} file(s) importing an optional root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
