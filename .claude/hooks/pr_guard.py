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

**One rule per verifiable check, ruled 2026-08-15 on #246.** The guard
asked all four, and a measured session approved three prompts without
reading them. An unverifiable question fails at any frequency: the
guard cannot check whether a search ran, the maintainer cannot check
it at the prompt, and answering yes costs nothing. So each rule now
routes by whether its check is mechanical.

| rule | decision | who reads it |
|------|----------|--------------|
| `gh pr merge` | `ask` | the maintainer |
| `gh pr ready` | `ask` | the maintainer |
| `gh pr create` without `--body-file` | `deny` | the agent |
| `gh issue create` | context, no decision | the agent |

The two merge-shaped rules stay questions, because they are the
maintainer's decision and they fire about once per pull request.

No rule emits `allow`. `allow` skips the normal permission flow, so a
guard that used it to reach the agent would also grant permission the
settings never granted. `.claude/settings.json` ships with this
repository and a fresh clone carries no allowlist, so that would widen
what an agent runs unprompted on every other contributor's machine.
The `gh issue create` rule therefore emits `additionalContext` with no
decision at all.

The `--body-file` rule became a refusal. Its condition is a property
of the command line and needs no judgment.

The `gh issue create` rule stopped asking and started answering. It
runs the tracker search itself and hands the matches to the agent, so
nothing depends on a claim nobody can verify.

Field routing measured on Claude Code 2.1.233 (#246).
`additionalContext` reaches the agent with `allow` and with no
decision at all. `deny` plus `permissionDecisionReason` reaches the
agent. `ask` reaches only the maintainer, and `systemMessage` reaches
nobody. Re-measure before changing a decision value.

The guard still reminds rather than enforces. `gh api --method PUT
repos/o/r/pulls/N/merge` reaches the same endpoint and matches no rule
here. Chasing every bypass is how a reminder becomes noise.

It still fails open. A malformed payload allows the command, a failed
search reports that it failed, and an uncaught error exits non-zero,
which Claude Code reports without blocking the tool.

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
import shutil
import subprocess
import sys
from typing import Final, NamedTuple

# Most restrictive wins when one command matches several rules. The
# guard emits one decision, so a stricter match drops the search
# report (#246).
#
# `inform` is not a Claude Code decision value. It names a payload
# that carries `additionalContext` and **no** `permissionDecision`.
# Measured 2026-08-15 on 2.1.233: context arrives that way. Emitting
# `allow` instead would deliver the same text and also skip the normal
# permission flow, which would grant a permission the guard was never
# asked to grant.
_RANK: Final[dict[str, int]] = {"inform": 1, "ask": 2, "deny": 3}

# The tracker search runs inside a PreToolUse hook, and
# `.claude/settings.json` allows the hook 5 seconds. Leave room for
# process start and JSON encoding.
_SEARCH_TIMEOUT_SECONDS: Final[float] = 3.0
_SEARCH_LIMIT: Final[int] = 5

# Words too common to narrow a tracker search.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "does",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "no",
        "not",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "then",
        "to",
        "was",
        "what",
        "when",
        "which",
        "with",
        "without",
    }
)
_MIN_TERM_LEN: Final[int] = 3
_MAX_TERMS: Final[int] = 6


class Rule(NamedTuple):
    """One guarded command shape.

    Attributes:
        verb (tuple[str, ...]): The command words, matched as
            consecutive tokens.
        pattern (re.Pattern[str]): The same words anchored to the
            start of a segment, used only when a segment cannot be
            tokenized.
        disarm (str | None): A flag that silences the rule when it
            appears as its own token in the same segment.
        decision (str): ``ask``, ``deny``, or ``inform``. Only the
            first two name a Claude Code permission decision.
        text (str): The text the decision carries.
    """

    verb: tuple[str, ...]
    pattern: re.Pattern[str]
    disarm: str | None
    decision: str
    text: str


def rule(verb: str, disarm: str | None, decision: str, text: str) -> Rule:
    """Build one rule from its command words.

    Args:
        verb: The command words, space separated, e.g. ``gh pr merge``.
        disarm: A flag that silences the rule, or None.
        decision: ``ask``, ``deny``, or ``inform``.
        text: The text the decision carries.

    Returns:
        The rule, carrying both matchers.
    """
    words = tuple(verb.split())
    escaped = r"\s+".join(re.escape(w) for w in words)
    # Anchored on purpose. The fallback runs on raw text, where an
    # unanchored search matches a mention inside prose or backticks.
    # A heredoc with an apostrophe defeats `shlex`, and the old
    # unanchored fallback then fired on documentation that merely
    # named the command (#246). A real command opens its segment.
    return Rule(words, re.compile(rf"^\s*{escaped}\b"), disarm, decision, text)


def has_verb(present: list[str], verb: tuple[str, ...]) -> bool:
    """Report whether the tokens carry the verb as consecutive words.

    Token matching keeps a quoted mention from firing a rule.
    ``echo "gh pr create"`` is one token after `shlex`, so it names no
    command. A raw-text search matched it and refused the work, which
    the `deny` decision made expensive (#246).

    Args:
        present: Tokens of one shell segment.
        verb: The command words to find.

    Returns:
        True when the tokens contain the verb in order.
    """
    span = len(verb)
    return any(
        tuple(present[i : i + span]) == verb for i in range(len(present) - span + 1)
    )


# Shell operators that separate one command from the next, as tokens.
# The guard reads each segment alone, so a later command never disarms
# an earlier one.
#
# A newline is absent on purpose. `shlex` reads an unescaped newline
# as whitespace, so the only `\n` token it emits comes from a
# backslash continuation, which joins one command rather than ending
# it. Treating it as a separator put a continued command's flags in a
# different segment, where they no longer disarmed their rule.
_SEPARATOR_TOKENS: Final[frozenset[str]] = frozenset({"&&", "||", ";", "|"})

# The same operators as raw text, for the fallback path only.
_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"&&|\|\||;|\n|\|")

# A backslash before a newline continues one command onto the next
# line. It joins rather than separates, so it resolves first. `shlex`
# already does this, so only the fallback needs it.
_CONTINUATION: Final[re.Pattern[str]] = re.compile(r"\\\s*\n")

_ISSUE_CREATE_VERB: Final[tuple[str, ...]] = ("gh", "issue", "create")

_RULES: Final[tuple[Rule, ...]] = (
    rule(
        "gh pr merge",
        None,
        "ask",
        (
            "Has the Copilot review posted, and is every comment answered?\n"
            "PR #156 merged 54 seconds after the review posted. PR #176 "
            "merged five minutes after, and its flag was real."
        ),
    ),
    rule(
        "gh pr ready",
        None,
        "ask",
        (
            "Has the review cycle run on this diff?\n"
            "Fact-check every number, then peer-review for strict-mode "
            "style. Docs-only PRs are not exempt. PR #64 and PR #100 "
            "skipped it, and both times it found defects."
        ),
    ),
    rule(
        "gh pr create",
        "--body-file",
        "deny",
        (
            "Refused: `gh pr create` without `--body-file`.\n"
            "The flag bypasses .github/PULL_REQUEST_TEMPLATE.md "
            "silently. The title becomes the squash subject and "
            "release-please parses the body footers, so a freeform body "
            "breaks machinery rather than style. Write the body to a "
            "file, then pass --body-file."
        ),
    ),
    rule(
        "gh issue create",
        None,
        "inform",
        "",
    ),
)


def tokens(segment: str) -> list[str]:
    """Split one shell segment into tokens.

    Quoting matters here. A `--body-file` inside a quoted message is
    one token's content, never a flag.

    A caller reads these tokens to find a flag that *silences* a rule,
    so an over-reported flag makes the guard quiet rather than loud. A
    segment this cannot tokenize therefore yields nothing, and every
    matching rule fires.

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


def segments(command: str) -> list[tuple[str, list[str]]]:
    """Split one command line into its independent commands.

    Tokenizing first is what makes this safe. A quoted argument stays
    one token however many newlines or operators it carries, so a
    commit message or heredoc naming a guarded command never reads as
    that command. Splitting raw text first tore such an argument into
    fake segments, and a prose line opening with the verb then matched
    (#246).

    Args:
        command: The full command line the tool would run.

    Returns:
        One ``(segment, tokens)`` pair per command. A command the
        shell grammar cannot parse yields raw segments with no tokens,
        which sends the caller to the anchored regex.
    """
    present = tokens(command)
    if not present:
        joined = _CONTINUATION.sub(" ", command)
        return [(segment, []) for segment in _SEPARATORS.split(joined)]
    out: list[tuple[str, list[str]]] = []
    current: list[str] = []
    for token in present:
        if token in _SEPARATOR_TOKENS:
            out.append((" ".join(current), current))
            current = []
        else:
            current.append(token)
    out.append((" ".join(current), current))
    return out


def title_of(present: list[str]) -> str | None:
    """Read the issue title from one segment's tokens.

    Args:
        present: Tokens of one shell segment.

    Returns:
        The title, or None when the segment names none.
    """
    for flag in ("--title", "-t"):
        if flag in present:
            index = present.index(flag)
            if index + 1 < len(present):
                return present[index + 1]
    for token in present:
        if token.startswith("--title="):
            return token.split("=", 1)[1]
    return None


def search_terms(title: str) -> list[str]:
    """Reduce a title to the words worth searching.

    Args:
        title: The issue title as written.

    Returns:
        Up to `_MAX_TERMS` lowercase words, longest first, without
        stopwords. Empty when the title carries none.
    """
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", title.lower())
    keep = [w for w in words if len(w) >= _MIN_TERM_LEN and w not in _STOPWORDS]
    return sorted(set(keep), key=len, reverse=True)[:_MAX_TERMS]


def duplicate_report(title: str) -> str:
    """Search the tracker and describe what it found.

    The rule this serves used to ask whether a search had run. Nobody
    could verify that answer, so the guard runs the search instead
    (#246).

    Args:
        title: The issue title the command would file.

    Returns:
        Text for the agent. It names the matches, or states that the
        search could not run, which is itself worth knowing.
    """
    terms = search_terms(title)
    if not terms:
        return f'Tracker search skipped: no searchable words in "{title}".'
    gh = shutil.which("gh")
    if gh is None:
        return "Tracker search skipped: gh is not on PATH. Search before filing."
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                gh,
                "issue",
                "list",
                "--state",
                "all",
                "--search",
                " ".join(terms),
                "--limit",
                str(_SEARCH_LIMIT),
                "--json",
                "number,title,state",
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
        matches = json.loads(completed.stdout) if completed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        matches = None
    if matches is None:
        return "Tracker search failed to run. Search before filing."
    if not matches:
        return f"Tracker search for {terms} found nothing. Filing looks new."
    listed = "\n".join(
        f"  #{m['number']} [{m['state'].lower()}] {m['title']}" for m in matches
    )
    return (
        f"Tracker search for {terms} found {len(matches)} match(es):\n{listed}\n"
        "Read them before filing. Five sessions filed one runlog bug five "
        "times (#151, #155, #157, #185, #205). Comment on a match instead, "
        "or say in the body why this one is distinct."
    )


class Verdict(NamedTuple):
    """What the guard decided about one command.

    Attributes:
        decision (str): ``ask``, ``deny``, or ``inform``.
        reasons (list[str]): Text for an ``ask`` or ``deny``.
        context (list[str]): Text for the agent under ``inform``.
    """

    decision: str
    reasons: list[str]
    context: list[str]


def inspect(command: str) -> Verdict | None:
    """Decide what one shell command needs before it runs.

    Args:
        command: The full command line the tool would run.

    Returns:
        The verdict, or None when no rule matched.
    """
    reasons: list[str] = []
    context: list[str] = []
    decision: str | None = None
    for segment, present in segments(command):
        for candidate in _RULES:
            # Tokens decide when the command parses. An unparseable
            # command yields none, and the anchored regex then keeps
            # the rule loud rather than silent, without firing on a
            # mention inside the text.
            hit = (
                has_verb(present, candidate.verb)
                if present
                else candidate.pattern.search(segment) is not None
            )
            if not hit:
                continue
            if candidate.disarm is not None and candidate.disarm in present:
                continue
            if decision is None or _RANK[candidate.decision] > _RANK[decision]:
                decision = candidate.decision
            if candidate.verb == _ISSUE_CREATE_VERB:
                report = duplicate_report(title_of(present) or "")
                if report not in context:
                    context.append(report)
            elif candidate.text not in reasons:
                reasons.append(candidate.text)
    if decision is None:
        return None
    return Verdict(decision, reasons, context)


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
        verdict = inspect(command) if isinstance(command, str) else None
    except (ValueError, AttributeError, OSError):
        # `json.JSONDecodeError` and `UnicodeDecodeError` subclass
        # `ValueError`. A payload that is not a mapping raises
        # `AttributeError` on `.get`. A broken pipe raises `OSError`.
        return 0

    if verdict is None:
        return 0

    out: dict[str, str] = {"hookEventName": "PreToolUse"}
    if verdict.decision == "inform":
        # No `permissionDecision`, so the normal permission flow still
        # decides. With nothing to say the guard stays silent rather
        # than emit an empty payload.
        context = "\n\n".join(text for text in verdict.context if text)
        if not context:
            return 0
        out["additionalContext"] = context
    else:
        out["permissionDecision"] = verdict.decision
        if verdict.reasons:
            out["permissionDecisionReason"] = "\n\n".join(verdict.reasons)

    print(json.dumps({"hookSpecificOutput": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
