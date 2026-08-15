#!/usr/bin/env python3
"""PreToolUse guard for the tracker commands that skip a review step.

Four failures cost this project real work, and every one of them is a
shell command an agent runs without stopping:

- `gh pr ready` with no review cycle. PR #64 and PR #100 both reached
  ready without one, and both cycles then found defects.
- `gh pr merge` before the Copilot review is read. PR #156 merged 54
  seconds after the review posted, PR #176 five minutes after, and
  #176's flag was real.
- `gh pr create --body`, which bypasses the template silently. The
  title becomes the squash subject and release-please parses the body
  footers, so a freeform body breaks machinery rather than style.
- `gh issue create` with no prior search. Five sessions filed one
  runlog bug five times (#151, #155, #157, #185, #205).

The guard asks rather than blocks. A hook that cries wolf gets
disabled, and every guarded command has a legitimate use the guard
cannot see. Answering the question is the point.

The guard reminds. It does not enforce. `gh api --method PUT
repos/o/r/pulls/N/merge` reaches the same endpoint and matches no rule
here. Chasing every bypass is how a reminder becomes noise.

It fails open. A malformed payload allows the command, and an uncaught
error exits non-zero, which Claude Code reports without blocking the
tool.

Examples:
    The guard reads one PreToolUse payload on stdin and answers with a
    permission decision:

    ```console
    $ echo '{"tool_input": {"command": "gh pr merge 1"}}' \
        | python3 .claude/hooks/pr_guard.py
    {"hookSpecificOutput": {"hookEventName": "PreToolUse",
     "permissionDecision": "ask", "permissionDecisionReason": "..."}}
    ```

    An unguarded command prints nothing and exits 0:

    ```console
    $ echo '{"tool_input": {"command": "uv run pytest"}}' \
        | python3 .claude/hooks/pr_guard.py
    ```

See Also:
    - `.claude/settings.json`: Registers this guard on the Bash tool.
    - `.claude/commands/board.md`: The `/board` command, which answers
      where a session should go before any of these commands run.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from typing import Final, NamedTuple


class Rule(NamedTuple):
    """One guarded command shape.

    Attributes:
        pattern (re.Pattern[str]): Matches the command inside one
            shell segment.
        disarm (str | None): A flag that silences the rule when it
            appears as its own token in the same segment.
        text (str): The question to put to the user.
    """

    pattern: re.Pattern[str]
    disarm: str | None
    text: str


# Shell operators that separate one command from the next. The guard
# reads each segment alone, so a later command never disarms an
# earlier one.
_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"&&|\|\||;|\n|\|")

# A backslash before a newline continues one command onto the next
# line. It joins rather than separates, so it resolves first.
_CONTINUATION: Final[re.Pattern[str]] = re.compile(r"\\\s*\n")

_RULES: Final[tuple[Rule, ...]] = (
    Rule(
        re.compile(r"\bgh\s+pr\s+merge\b"),
        None,
        (
            "Has the Copilot review posted, and is every comment answered?\n"
            "PR #156 merged 54 seconds after the review posted. PR #176 "
            "merged five minutes after, and its flag was real."
        ),
    ),
    Rule(
        re.compile(r"\bgh\s+pr\s+ready\b"),
        None,
        (
            "Has the review cycle run on this diff?\n"
            "Fact-check every number, then peer-review for strict-mode "
            "style. Docs-only PRs are not exempt. PR #64 and PR #100 "
            "skipped it, and both times it found defects."
        ),
    ),
    Rule(
        re.compile(r"\bgh\s+pr\s+create\b"),
        "--body-file",
        (
            "Does this body follow .github/PULL_REQUEST_TEMPLATE.md?\n"
            "`gh pr create` bypasses the template silently. The title "
            "becomes the squash subject and release-please parses the "
            "body footers. Write the body to a file, then pass "
            "--body-file."
        ),
    ),
    Rule(
        re.compile(r"\bgh\s+issue\s+create\b"),
        None,
        (
            "Did you search the tracker for this symptom first?\n"
            "Search the filename or the error string, over all states, "
            "not the guessed cause. Five sessions filed one runlog bug "
            "five times. Comment on the match instead."
        ),
    ),
)


def tokens(segment: str) -> list[str]:
    """Split one shell segment into tokens.

    Quoting matters here. A `--body-file` inside a quoted message is
    one token's content, never a flag.

    A caller reads these tokens to find a flag that *silences* a rule,
    so an over-reported flag makes the guard quiet rather than loud. A
    segment this cannot tokenize therefore yields nothing, and every
    matching rule asks.

    Args:
        segment: One shell command, without separators.

    Returns:
        The tokens, or an empty list when the segment carries
        unbalanced quotes.
    """
    try:
        return shlex.split(segment)
    except ValueError:
        return []


def questions(command: str) -> list[str]:
    """Find every question one shell command should answer first.

    Args:
        command: The full command line the tool would run.

    Returns:
        One question per matched rule, in rule order, without
        duplicates. An empty list means no rule matched.
    """
    found: list[str] = []
    joined = _CONTINUATION.sub(" ", command)
    for segment in _SEPARATORS.split(joined):
        present = tokens(segment)
        for rule in _RULES:
            if not rule.pattern.search(segment):
                continue
            if rule.disarm is not None and rule.disarm in present:
                continue
            if rule.text not in found:
                found.append(rule.text)
    return found


def main() -> int:
    """Read one PreToolUse payload from stdin and decide on it.

    Returns:
        Always 0. The decision travels in the JSON on stdout, and a
        malformed payload allows the command rather than stopping the
        work.
    """
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "")
        found = questions(command) if isinstance(command, str) else []
    except (ValueError, AttributeError, OSError):
        # `json.JSONDecodeError` and `UnicodeDecodeError` subclass
        # `ValueError`. A payload that is not a mapping raises
        # `AttributeError` on `.get`. A broken pipe raises `OSError`.
        return 0

    if not found:
        return 0

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": "\n\n".join(found),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
