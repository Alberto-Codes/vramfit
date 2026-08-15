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


def test_inspect_issue_create_informs_and_carries_the_search(monkeypatch) -> None:
    # The rule stopped asking whether a search ran and started running
    # it (#246). Nothing here depends on an unverifiable answer.
    verdict = guard.inspect('gh issue create --title "a new symptom"')

    assert verdict is not None
    assert verdict.decision == "inform"
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


def test_inspect_prose_naming_a_command_in_an_unparseable_segment_is_silent() -> None:
    # A heredoc carrying an apostrophe defeats `shlex`, so the regex
    # fallback runs on raw text. Unanchored, it fired on documentation
    # that merely named the command. This shape hit the guard live
    # while the guard was being written (#246).
    command = (
        "python3 - <<'EOF'\n"
        "s = s.replace('the maintainer's call', 'x')\n"
        "# `gh issue create` runs the tracker search itself\n"
        "EOF"
    )

    assert guard.inspect(command) is None


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "one\ngh issue create rule used it\nend"',
        'git commit -m "one\ngh pr create refuses now\nend"',
        'gh issue comment 246 --body "gh pr merge asks the maintainer"',
    ],
    ids=["issue-create-line", "pr-create-line", "body-mentions-merge"],
)
def test_inspect_multi_line_quoted_argument_is_one_token(command) -> None:
    # Splitting raw text on newlines tore a quoted argument into fake
    # segments, and a prose line opening with the verb then matched.
    # This fired live on a commit message, and under `deny` it would
    # have refused the commit (#246). Tokenizing first is the fix.
    assert guard.inspect(command) is None


def test_inspect_command_at_segment_start_survives_unbalanced_quotes() -> None:
    # The anchor must not cost a real command its rule.
    assert decision('gh pr create --body "oops') == "deny"
    assert decision("   gh pr merge 1") == "ask"


def test_inspect_deny_outranks_ask_when_both_match() -> None:
    # One decision travels to the caller, so the strictest wins.
    verdict = guard.inspect("gh pr create --body x && gh pr ready")

    assert verdict is not None
    assert verdict.decision == "deny"
    assert len(verdict.reasons) == 2


def test_inspect_ask_outranks_inform_when_both_match() -> None:
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

    assert "timed out" in REAL_DUPLICATE_REPORT("a real title here")


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


def test_duplicate_report_no_match_states_only_what_it_checked(monkeypatch) -> None:
    monkeypatch.setattr(guard.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[]", stderr=""
        ),
    )

    # "Filing looks new" asserted recall the search cannot support.
    # #205 reworded #151 and this title search would have missed it,
    # so the report states what it checked and what it misses (#246).
    report = REAL_DUPLICATE_REPORT("wholly novel symptom")

    assert "matched no open or closed issue" in report
    assert "#205" in report
    assert "looks new" not in report


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


def test_main_issue_create_emits_context_without_a_decision(
    monkeypatch, capsys
) -> None:
    # `additionalContext` reaches the agent with no decision at all,
    # measured on Claude Code 2.1.233 (#246). Emitting `allow` would
    # deliver the same text and also skip the permission flow, which
    # would grant a permission this guard was never asked to grant.
    payload = json.dumps({"tool_input": {"command": "gh issue create --title x"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    guard.main()

    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["additionalContext"] == STUB_REPORT
    assert "permissionDecision" not in output, (
        "a decision here would bypass the settings a fresh clone relies on"
    )
    assert "permissionDecisionReason" not in output


def test_main_inform_without_context_emits_nothing(monkeypatch, capsys) -> None:
    # An empty payload says nothing and risks being read as a
    # decision. Staying silent is the honest outcome.
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


@pytest.mark.parametrize(
    "command",
    ["gh pr create --help", "gh pr merge --help", "gh issue create -h"],
    ids=["create", "merge", "issue-create"],
)
def test_inspect_help_flag_is_exempt(command) -> None:
    # A refusal the agent cannot override must not stop it reading the
    # flags this guard is about.
    assert guard.inspect(command) is None


def test_inspect_heredoc_payload_is_not_read_as_commands() -> None:
    # shlex knows nothing about heredocs, so it tokenized the payload
    # as commands and refused a script that merely quoted one (#246).
    command = "python3 - <<'EOF'\ncases = ['cd /r && gh pr create --body x']\nEOF"

    assert guard.inspect(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --body-file b.md",
        "gh pr create --body-file=b.md",
        "gh pr create -F b.md",
    ],
    ids=["spaced", "equals", "short"],
)
def test_inspect_every_body_file_spelling_disarms(command) -> None:
    # gh spells it three ways. An exact-token test saw one and hard
    # refused a correctly formed command (#246).
    assert guard.inspect(command) is None


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/usr/bin/gh pr merge 5", "ask"),
        ("./gh pr merge 5", "ask"),
        ("/home/linuxbrew/.linuxbrew/bin/gh pr create --body x", "deny"),
    ],
    ids=["absolute", "relative", "brew"],
)
def test_inspect_path_qualified_gh_still_fires(command, expected) -> None:
    # gh is not always on PATH as a bare name. Comparing the first
    # token exactly lost every one of these.
    assert decision(command) == expected


def test_inspect_runs_one_search_per_command(monkeypatch) -> None:
    # One search per matching segment could spend three times the
    # hook's whole budget, and an overrun loses the decision too.
    calls: list[str] = []
    monkeypatch.setattr(
        guard, "duplicate_report", lambda title: calls.append(title) or "x"
    )

    guard.inspect(
        "gh issue create -t a && gh issue create -t b && gh issue create -t c"
    )

    assert len(calls) == 1


def test_inspect_report_travels_with_a_stricter_decision() -> None:
    # additionalContext is an independent key, so the search result no
    # longer dies when an ask or deny outranks it.
    verdict = guard.inspect("gh pr merge 1 && gh issue create --title x")

    assert verdict is not None
    assert verdict.decision == "ask"
    assert verdict.context == [STUB_REPORT]


def test_main_report_travels_with_a_stricter_decision(monkeypatch, capsys) -> None:
    payload = json.dumps(
        {"tool_input": {"command": "gh pr merge 1 && gh issue create --title x"}}
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    guard.main()

    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    assert output["additionalContext"] == STUB_REPORT


@pytest.mark.parametrize(
    ("stdout", "returncode", "stderr", "phrase"),
    [
        ("{}", 0, "", "unexpected shape"),
        ('{"message": "Not Found"}', 0, "", "unexpected shape"),
        ("null", 0, "", "unexpected shape"),
        ("not json", 0, "", "unparseable"),
        ("", 4, "gh: run gh auth login", "gh auth login"),
    ],
    ids=["empty-object", "message-object", "null", "garbage", "auth-failure"],
)
def test_duplicate_report_malformed_output_never_reads_as_clean(
    monkeypatch, stdout, returncode, stderr, phrase
) -> None:
    # A non-answer rendered as "found nothing" is the exact mislabel
    # this rule exists to prevent.
    monkeypatch.setattr(guard.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr
        ),
    )

    report = REAL_DUPLICATE_REPORT("duplicate json keys load silently")

    assert phrase in report
    assert "matched no open or closed issue" not in report


def test_duplicate_report_without_a_title_says_so() -> None:
    assert "names no title" in REAL_DUPLICATE_REPORT("")
