"""File-size gate: fail when a module exceeds the decomposition limit.

Counts *code* lines — lines carrying at least one real token, excluding
comments and docstrings — so docvet-mandated documentation never pushes a
file over the limit. Soft limit 300 (warn), hard limit 320 (fail):
anything larger gets decomposed, not excused.

Examples:
    Run against the source tree:

    ```console
    $ uv run python scripts/check_loc.py src
    checked 14 files
    ```
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

SOFT_LIMIT = 300
HARD_LIMIT = 320

_SKIP_TOKENS = frozenset(
    {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.ENDMARKER,
        tokenize.ENCODING,
    }
)


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Collect the line numbers occupied by docstrings.

    Args:
        tree: Parsed module AST.

    Returns:
        All 1-based line numbers inside module, class, or function
        docstrings.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
            and body[0].end_lineno is not None
        ):
            lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return lines


def count_code_lines(path: Path) -> int:
    """Count lines of actual code in a Python file.

    A line counts when it carries at least one token that is not a
    comment, and it is not part of a docstring.

    Args:
        path: The Python file to measure.

    Returns:
        The number of code lines.
    """
    text = path.read_text(encoding="utf-8")
    doc_lines = _docstring_lines(ast.parse(text))
    token_lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in _SKIP_TOKENS:
            continue
        token_lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(token_lines - doc_lines)


def main(roots: list[str]) -> int:
    """Check every Python file under the given roots.

    Args:
        roots: Directories to scan (defaults to ``src`` when empty).

    Returns:
        Process exit code: 1 if any file exceeds the hard limit, else 0.
    """
    failures = 0
    checked = 0
    for root in roots or ["src"]:
        paths = sorted(Path(root).rglob("*.py"))
        if not paths:
            # A gate that scans nothing must fail loudly, not pass
            # silently — a renamed root would otherwise disable it.
            print(f"FAIL {root}: no Python files found")
            failures += 1
            continue
        for path in paths:
            checked += 1
            n = count_code_lines(path)
            if n > HARD_LIMIT:
                print(f"FAIL {path}: {n} code lines (hard limit {HARD_LIMIT})")
                failures += 1
            elif n > SOFT_LIMIT:
                print(
                    f"WARN {path}: {n} code lines (soft limit {SOFT_LIMIT})",
                    file=sys.stderr,
                )
    print(f"checked {checked} files")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
