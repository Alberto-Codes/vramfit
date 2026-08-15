#!/usr/bin/env python3
"""PreToolUse guard for the tracker commands that skip a review step.

Four failures cost this project real work, and every one of them is a
shell command an agent runs without stopping:

- `gh pr ready` with no review cycle. The maintainer caught the cycle
  skipped twice, on PR #64 and PR #100. Both times it found defects.
- `gh pr merge` before the Copilot review is read. PR #156 merged 54
  seconds after the review posted, PR #176 five minutes after, and
  #176's flag was real (filed as #206).
- `gh pr create --body`, which bypasses the template silently. The
  title becomes the squash subject and release-please parses the body
  footers, so a freeform body breaks machinery rather than style.
- `gh issue create` with no prior search. One runlog bug was filed five
  times (#151, #155, #157, #185, #205) by five sessions.

The guard asks rather than blocks. A hook that cries wolf gets
disabled, and every one of these commands has a legitimate use the
guard cannot see. Answering the question is the point.

It fails open. Any error here allows the command, because a broken
guard must not stop the work.

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
import sys
from typing import Final

# Each rule pairs a command pattern with the question to ask. Order
# matters: the first match wins, so the narrower pattern comes first.
_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"\bgh\s+pr\s+merge\b"),
        (
            "Merging a PR. Has the Copilot review posted, and have you read "
            "and answered every comment? PR #156 merged 54 seconds after the "
            "review posted and PR #176 five minutes after, and #176's flag "
            "was real (#206). Confirm the review is triaged, or cancel and "
            "wait for it."
        ),
    ),
    (
        re.compile(r"\bgh\s+pr\s+ready\b"),
        (
            "Marking a PR ready. Has the review cycle run on this diff — a "
            "fact-check pass over every number and claim, and a peer-review "
            "pass for strict-mode style and overstatement? Docs-only PRs are "
            "not exempt, because their sha256s, dates, and issue refs are "
            "load-bearing. The maintainer caught this skipped on PR #64 and "
            "PR #100, and both times it found defects."
        ),
    ),
    (
        re.compile(r"\bgh\s+pr\s+create\b(?!.*--body-file)"),
        (
            "Creating a PR without --body-file. `gh pr create` bypasses "
            ".github/PULL_REQUEST_TEMPLATE.md silently. The title becomes the "
            "squash commit subject and release-please parses the body footers, "
            "so a freeform body breaks the changelog machinery. Write the body "
            "to a file against the template, then pass --body-file."
        ),
    ),
    (
        re.compile(r"\bgh\s+issue\s+create\b"),
        (
            "Filing an issue. Have you searched the tracker for the symptom "
            "first — the filename or the error string, over all states, not "
            "the guessed cause? One runlog bug was filed five times (#151, "
            "#155, #157, #185, #205) by five sessions that never looked. If a "
            "match exists, comment there with the new evidence instead."
        ),
    ),
)


def question(command: str) -> str | None:
    """Find the question one shell command should answer first.

    Args:
        command: The full command line the tool would run. A compound
            command is matched anywhere, so `a && gh pr merge` counts.

    Returns:
        The question to put to the user, or None when no rule matches.
    """
    for pattern, text in _RULES:
        if pattern.search(command):
            return text
    return None


def main() -> int:
    """Read one PreToolUse payload from stdin and decide on it.

    Returns:
        Always 0. The decision travels in the JSON on stdout, and an
        error path allows the command rather than stopping the work.
    """
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "")
        text = question(command) if isinstance(command, str) else None
    except Exception:
        # A broken guard must not block work. Any failure here allows
        # the command.
        return 0

    if text is None:
        return 0

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": text,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
