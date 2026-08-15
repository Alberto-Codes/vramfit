from __future__ import annotations

import io
import json

import pr_guard as guard
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("command", "expected_phrase"),
    [
        ("gh pr merge 239 --squash", "Copilot review"),
        ("gh pr ready", "review cycle"),
        ('gh pr create --body "hi"', "--body-file"),
        ("gh issue create --title x", "searched the tracker"),
    ],
    ids=["merge", "ready", "create-freeform", "issue-create"],
)
def test_question_guarded_command_asks_its_question(command, expected_phrase) -> None:
    text = guard.question(command)

    assert text is not None
    assert expected_phrase in text


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -q",
        "git push -u origin feat/x",
        "gh pr list --state merged",
        "gh issue view 158",
        "gh pr create --draft --body-file /tmp/body.md",
    ],
    ids=[
        "unrelated",
        "push",
        "pr-list",
        "issue-view",
        "create-with-body-file",
    ],
)
def test_question_unguarded_command_asks_nothing(command) -> None:
    assert guard.question(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "git push && gh pr merge 1",
        "cd /repo; gh pr ready",
        "gh pr merge 1 || echo failed",
    ],
    ids=["and", "semicolon", "or"],
)
def test_question_compound_command_matches_its_guarded_half(command) -> None:
    assert guard.question(command) is not None


def test_question_body_file_after_other_flags_still_passes() -> None:
    # The negative lookahead must survive flags between the subcommand
    # and --body-file.
    command = "gh pr create --draft --title 'x' --base main --body-file b.md"

    assert guard.question(command) is None


def test_question_prefers_merge_over_a_later_rule() -> None:
    # Order matters: the first rule that matches wins, so a command
    # naming two guarded verbs reports the more consequential one.
    text = guard.question("gh pr ready && gh pr merge 1")

    assert text is not None
    assert "Copilot review" in text


def test_main_guarded_command_emits_an_ask_decision(monkeypatch, capsys) -> None:
    payload = json.dumps({"tool_input": {"command": "gh pr merge 1"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    code = guard.main()

    assert code == 0
    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "ask"
    assert output["permissionDecisionReason"]


def test_main_unguarded_command_emits_nothing(monkeypatch, capsys) -> None:
    payload = json.dumps({"tool_input": {"command": "uv run pytest"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    code = guard.main()

    assert code == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "payload",
    ["not json at all", "{}", '{"tool_input": {}}', '{"tool_input": {"command": 7}}'],
    ids=["unparseable", "empty", "no-command", "non-string-command"],
)
def test_main_malformed_payload_fails_open(monkeypatch, capsys, payload) -> None:
    # A broken guard must allow the command. Blocking work on a hook
    # bug is worse than missing one reminder.
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    code = guard.main()

    assert code == 0
    assert capsys.readouterr().out == ""
