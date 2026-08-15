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


@pytest.mark.parametrize(
    ("command", "expected_phrase"),
    [
        ("gh pr merge 239 --squash", "Copilot review"),
        ("gh pr ready", "review cycle"),
        ('gh pr create --body "hi"', "--body-file"),
        ("gh issue create --title x", "search the tracker"),
    ],
    ids=["merge", "ready", "create-freeform", "issue-create"],
)
def test_questions_guarded_command_asks_its_question(command, expected_phrase) -> None:
    found = guard.questions(command)

    assert len(found) == 1
    assert expected_phrase in found[0]


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
def test_questions_unguarded_command_asks_nothing(command) -> None:
    assert guard.questions(command) == []


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
def test_questions_compound_command_matches_its_guarded_half(command) -> None:
    assert len(guard.questions(command)) == 1


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --draft --title x --base main --body-file b.md",
        "gh pr create \\\n  --title x \\\n  --body-file /tmp/b.md",
        "gh pr create --body-file b.md && gh pr view",
    ],
    ids=["flags-before", "multi-line", "compound-after"],
)
def test_questions_create_with_body_file_asks_nothing(command) -> None:
    # The multi-line form is the shape an agent writes for a long
    # body. An earlier lookahead flagged it, because `.` in a regex
    # does not cross a newline.
    assert guard.questions(command) == []


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --body x && echo --body-file",
        'gh pr create --body "use --body-file next time"',
        "echo --body-file; gh pr create --body x",
    ],
    ids=["later-segment", "quoted-mention", "earlier-segment"],
)
def test_questions_body_file_outside_the_create_segment_still_asks(command) -> None:
    # A mention of the flag elsewhere must not disarm the rule. An
    # earlier lookahead scanned the whole string and went silent.
    found = guard.questions(command)

    assert len(found) == 1
    assert "--body-file" in found[0]


def test_questions_two_guarded_verbs_asks_both_questions() -> None:
    # Reporting only the first would drop a warning the user needs.
    found = guard.questions("gh pr create --body x && gh pr ready")

    assert len(found) == 2
    assert any("--body-file" in text for text in found)
    assert any("review cycle" in text for text in found)


def test_questions_repeated_command_asks_once() -> None:
    found = guard.questions("gh pr merge 1 && gh pr merge 2")

    assert len(found) == 1


@pytest.mark.parametrize(
    "command",
    ["gh  pr   merge 1", "gh\tpr\tmerge 1"],
    ids=["extra-spaces", "tabs"],
)
def test_questions_spaced_command_still_asks(command) -> None:
    assert len(guard.questions(command)) == 1


def test_questions_unbalanced_quotes_still_asks() -> None:
    # shlex raises on this. The fallback must not go silent.
    found = guard.questions('gh pr create --body "unterminated')

    assert len(found) == 1


def test_main_guarded_command_emits_an_ask_decision(monkeypatch, capsys) -> None:
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


def test_main_two_matches_joins_both_reasons(monkeypatch, capsys) -> None:
    payload = json.dumps(
        {"tool_input": {"command": "gh pr create --body x && gh pr ready"}}
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    guard.main()

    reason = json.loads(capsys.readouterr().out)["hookSpecificOutput"][
        "permissionDecisionReason"
    ]
    assert "--body-file" in reason
    assert "review cycle" in reason


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
