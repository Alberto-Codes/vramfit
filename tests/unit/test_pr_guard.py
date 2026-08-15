from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pr_guard as guard
import pytest

pytestmark = pytest.mark.unit

GUARD_PATH = Path(guard.__file__)

STUB_REPORT = "stub tracker report"

# Captured before the autouse stub replaces it, so the tests that
# exercise the real search can still reach it.
REAL_DUPLICATE_REPORT = guard.duplicate_report


@pytest.fixture(autouse=True)
def _no_real_tracker_search(monkeypatch) -> None:
    """Keep every test off the network.

    `gh issue create` now runs a tracker search. A unit suite must not
    reach GitHub, so the search is stubbed unless a test replaces it.
    """
    monkeypatch.setattr(guard, "duplicate_report", lambda title: STUB_REPORT)


def reasons(command: str) -> list[str]:
    verdict = guard.inspect(command)
    return [] if verdict is None else verdict.reasons


def decision(command: str) -> str | None:
    verdict = guard.inspect(command)
    return None if verdict is None else verdict.decision


@pytest.mark.parametrize(
    ("command", "expected_decision", "expected_phrase"),
    [
        ("gh pr merge 239 --squash", "ask", "Copilot review"),
        ("gh pr ready", "ask", "review cycle"),
        ('gh pr create --body "hi"', "deny", "--body-file"),
    ],
    ids=["merge", "ready", "create-freeform"],
)
def test_inspect_guarded_command_routes_to_its_decision(
    command, expected_decision, expected_phrase
) -> None:
    verdict = guard.inspect(command)

    assert verdict is not None
    assert verdict.decision == expected_decision
    assert len(verdict.reasons) == 1
    assert expected_phrase in verdict.reasons[0]


def test_inspect_issue_create_allows_and_carries_the_search(monkeypatch) -> None:
    # The rule stopped asking whether a search ran and started running
    # it (#246). Nothing here depends on an unverifiable answer.
    verdict = guard.inspect('gh issue create --title "a new symptom"')

    assert verdict is not None
    assert verdict.decision == "allow"
    assert verdict.reasons == []
    assert verdict.context == [STUB_REPORT]


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -q",
        "git push -u origin feat/x",
        "gh pr list --state merged",
        "gh issue view 158",
        "gh pr create --draft --body-file /tmp/body.md",
    ],
    ids=["unrelated", "push", "pr-list", "issue-view", "create-with-body-file"],
)
def test_inspect_unguarded_command_returns_no_verdict(command) -> None:
    assert guard.inspect(command) is None


@pytest.mark.parametrize(
    "command",
    [
        'echo "gh pr create --body x"',
        "printf 'gh pr merge 1'",
        'cat <<< "run gh pr ready when done"',
    ],
    ids=["echo-create", "printf-merge", "heredoc-ready"],
)
def test_inspect_quoted_mention_of_a_command_does_not_fire(command) -> None:
    # A quoted mention is one token after `shlex`, so it names no
    # command. A raw-text search matched it, and under `deny` that
    # refused legitimate work — hit while testing the guard itself.
    assert guard.inspect(command) is None


def test_inspect_deny_outranks_ask_when_both_match() -> None:
    # One decision travels to the caller, so the strictest wins.
    verdict = guard.inspect("gh pr create --body x && gh pr ready")

    assert verdict is not None
    assert verdict.decision == "deny"
    assert len(verdict.reasons) == 2


def test_inspect_ask_outranks_allow_when_both_match() -> None:
    verdict = guard.inspect("gh issue create --title x && gh pr merge 1")

    assert verdict is not None
    assert verdict.decision == "ask"


@pytest.mark.parametrize(
    "command",
    [
        "git push && gh pr merge 1",
        "cd /repo; gh pr ready",
        "gh pr merge 1 || echo failed",
        "gh pr merge 1 | tee log",
    ],
    ids=["and", "semicolon", "or", "pipe"],
)
def test_inspect_compound_command_matches_its_guarded_half(command) -> None:
    assert len(reasons(command)) == 1


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --draft --title x --base main --body-file b.md",
        "gh pr create \\\n  --title x \\\n  --body-file /tmp/b.md",
        "gh pr create --body-file b.md && gh pr view",
    ],
    ids=["flags-before", "multi-line", "compound-after"],
)
def test_inspect_create_with_body_file_returns_no_verdict(command) -> None:
    # The multi-line form is the shape an agent writes for a long
    # body. An earlier lookahead flagged it, because `.` in a regex
    # does not cross a newline.
    assert guard.inspect(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --body x && echo --body-file",
        "echo --body-file; gh pr create --body x",
    ],
    ids=["later-segment", "earlier-segment"],
)
def test_inspect_body_file_outside_the_create_segment_still_denies(command) -> None:
    # A mention of the flag elsewhere must not disarm the rule. An
    # earlier lookahead scanned the whole string and went silent.
    verdict = guard.inspect(command)

    assert verdict is not None
    assert verdict.decision == "deny"
    assert "--body-file" in verdict.reasons[0]


def test_inspect_repeated_command_reports_once() -> None:
    assert len(reasons("gh pr merge 1 && gh pr merge 2")) == 1


@pytest.mark.parametrize(
    "command",
    ["gh  pr   merge 1", "gh\tpr\tmerge 1"],
    ids=["extra-spaces", "tabs"],
)
def test_inspect_spaced_command_still_fires(command) -> None:
    assert len(reasons(command)) == 1


@pytest.mark.parametrize(
    "command",
    [
        'gh pr create --body "unterminated',
        'gh pr create --body "use --body-file',
        "gh pr create --body 'mentions --body-file and never closes",
    ],
    ids=["no-flag", "flag-after-open-quote", "flag-in-single-quote"],
)
def test_inspect_unbalanced_quotes_still_denies(command) -> None:
    # shlex raises on these, so the regex fallback runs. A whitespace
    # split would hand back `--body-file` as a token and silence the
    # rule, which is the wrong direction for a flag that disarms.
    assert decision(command) == "deny"


def test_tokens_unbalanced_quotes_reports_nothing() -> None:
    assert guard.tokens('--body "unterminated') == []


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["gh", "issue", "create", "--title", "x"], "x"),
        (["gh", "issue", "create", "-t", "y"], "y"),
        (["gh", "issue", "create", "--title=z"], "z"),
        (["gh", "issue", "create", "--body-file", "b.md"], None),
        (["gh", "issue", "create", "--title"], None),
    ],
    ids=["long", "short", "equals", "absent", "dangling"],
)
def test_title_of_reads_the_title_flag(present, expected) -> None:
    assert guard.title_of(present) == expected


def test_search_terms_drops_stopwords_and_short_words() -> None:
    terms = guard.search_terms("The scan prices 23 conv1d cells it never touches")

    assert "the" not in terms
    assert "it" not in terms
    assert "conv1d" in terms


def test_search_terms_empty_title_returns_nothing() -> None:
    assert guard.search_terms("") == []


def test_duplicate_report_without_gh_says_so(monkeypatch) -> None:
    monkeypatch.setattr(guard.shutil, "which", lambda name: None)

    report = REAL_DUPLICATE_REPORT("a real title here")

    assert "gh is not on PATH" in report


def test_duplicate_report_search_failure_says_so(monkeypatch) -> None:
    # A failed search must not pass as a clean bill of health.
    monkeypatch.setattr(guard.shutil, "which", lambda name: "/usr/bin/gh")

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="gh", timeout=3.0)

    monkeypatch.setattr(guard.subprocess, "run", boom)

    assert "failed to run" in REAL_DUPLICATE_REPORT("a real title here")


def test_duplicate_report_lists_every_match(monkeypatch) -> None:
    monkeypatch.setattr(guard.shutil, "which", lambda name: "/usr/bin/gh")
    found = [{"number": 262, "title": "Duplicate keys", "state": "OPEN"}]

    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(found), stderr=""
        ),
    )

    report = REAL_DUPLICATE_REPORT("duplicate keys load silently")

    assert "#262" in report
    assert "Duplicate keys" in report


def test_duplicate_report_no_match_says_filing_looks_new(monkeypatch) -> None:
    monkeypatch.setattr(guard.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[]", stderr=""
        ),
    )

    assert "looks new" in REAL_DUPLICATE_REPORT("wholly novel symptom")


def test_main_ask_command_emits_an_ask_decision(monkeypatch, capsys) -> None:
    payload = json.dumps({"tool_input": {"command": "gh pr merge 1"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    code = guard.main()

    assert code == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1, "a hook consumer parses one line of stdout"
    output = json.loads(out)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "ask"
    assert output["permissionDecisionReason"]
    assert "additionalContext" not in output


def test_main_deny_command_emits_a_deny_decision(monkeypatch, capsys) -> None:
    payload = json.dumps({"tool_input": {"command": 'gh pr create --body "x"'}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    guard.main()

    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "--body-file" in output["permissionDecisionReason"]


def test_main_issue_create_emits_allow_with_context(monkeypatch, capsys) -> None:
    # `additionalContext` reaches the agent only under `allow`,
    # measured on Claude Code 2.1.233 (#246).
    payload = json.dumps({"tool_input": {"command": "gh issue create --title x"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    guard.main()

    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["permissionDecision"] == "allow"
    assert output["additionalContext"] == STUB_REPORT
    assert "permissionDecisionReason" not in output


def test_main_allow_without_context_emits_nothing(monkeypatch, capsys) -> None:
    # An `allow` carrying no text would grant permission outright and
    # say nothing, which is worse than staying out of the way.
    monkeypatch.setattr(guard, "duplicate_report", lambda title: "")
    payload = json.dumps({"tool_input": {"command": "gh issue create --title x"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    guard.main()

    assert capsys.readouterr().out == ""


def test_main_unguarded_command_emits_nothing(monkeypatch, capsys) -> None:
    payload = json.dumps({"tool_input": {"command": "uv run pytest"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    code = guard.main()

    assert code == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        "",
        "{}",
        "[]",
        "null",
        '{"tool_input": {}}',
        '{"tool_input": 7}',
        '{"tool_input": {"command": 7}}',
    ],
    ids=[
        "unparseable",
        "empty-stdin",
        "empty-object",
        "list",
        "null",
        "no-command",
        "non-mapping-tool-input",
        "non-string-command",
    ],
)
def test_main_malformed_payload_fails_open(monkeypatch, capsys, payload) -> None:
    # A broken guard must allow the command. Blocking work on a hook
    # bug is worse than missing one reminder.
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    code = guard.main()

    assert code == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("payload", "expects_output"),
    [
        ('{"tool_input": {"command": "gh pr merge 1"}}', True),
        ('{"tool_input": {"command": "uv run pytest"}}', False),
        ("not json", False),
    ],
    ids=["guarded", "unguarded", "malformed"],
)
def test_guard_run_as_a_process_exits_zero(payload, expects_output) -> None:
    # The harness runs the file, not the function. This pins the
    # `sys.exit(main())` wiring and the real exit code.
    # S603: both arguments are this interpreter and this repository's
    # own guard path. No test input reaches the command line.
    done = subprocess.run(  # noqa: S603
        [sys.executable, str(GUARD_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0
    assert bool(done.stdout.strip()) is expects_output
