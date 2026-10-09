"""Pin the two agent hooks under ``.claude/hooks/``.

``agent_deny_writes.sh`` is a PreToolUse hook on Bash. The read-only
agents load it. Exit 2 blocks the call. ``agent_model_gate.sh`` is a
PreToolUse hook on the Agent tool. It denies a dispatch with no
``model`` and asks before a Fable dispatch with no ``Escalation:``
line. Both fail open without ``jq``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
DENY = HOOKS / "agent_deny_writes.sh"
GATE = HOOKS / "agent_model_gate.sh"

if shutil.which("jq") is None or shutil.which("bash") is None:
    pytest.skip("the hooks need bash and jq", allow_module_level=True)
BASH: str = shutil.which("bash") or "bash"


def _run(script: Path, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- fixed script path, JSON on stdin
        [BASH, str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def _deny(command: str) -> subprocess.CompletedProcess[str]:
    return _run(DENY, {"tool_name": "Bash", "tool_input": {"command": command}})


def _gate(tool_input: dict[str, object], tool: str = "Agent") -> dict[str, str] | None:
    result = _run(GATE, {"tool_name": tool, "tool_input": tool_input})
    assert result.returncode == 0, result.stderr
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)["hookSpecificOutput"]


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m x",
        "git -C /tmp/wt push origin main",
        "git checkout -- src",
        "git stash",
        "gh pr create --body-file b.md",
        "gh issue close 1",
        "gh release edit v1",
    ],
)
def test_deny_writes_history_or_record_write_blocks(command: str) -> None:
    result = _deny(command)
    assert result.returncode == 2
    assert "read-only agents" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "git status --short",
        "git diff HEAD",
        "gh issue comment 1 --body-file c.md",
        "gh issue view 1 --comments",
        "uv run pytest -q",
    ],
)
def test_deny_writes_read_or_comment_passes(command: str) -> None:
    assert _deny(command).returncode == 0


def test_model_gate_no_model_denies() -> None:
    out = _gate({"subagent_type": "Explore", "prompt": "find the solver"})
    assert out is not None
    assert out["permissionDecision"] == "deny"
    assert "explicit model" in out["permissionDecisionReason"]


def test_model_gate_fork_without_model_passes() -> None:
    assert _gate({"subagent_type": "fork", "prompt": "continue"}) is None


def test_model_gate_opus_passes() -> None:
    assert _gate({"model": "opus", "prompt": "build it"}) is None


def test_model_gate_fable_without_escalation_asks() -> None:
    out = _gate({"model": "fable", "prompt": "rule on the cap"})
    assert out is not None
    assert out["permissionDecision"] == "ask"


def test_model_gate_fable_with_escalation_passes() -> None:
    prompt = "Escalation: opus failed the sizing argument twice.\nRule on the cap."
    assert _gate({"model": "fable", "prompt": prompt}) is None


def test_model_gate_other_tool_passes() -> None:
    assert _gate({"command": "ls"}, tool="Bash") is None
