#!/bin/bash
# Block git-history and GitHub-record writes from the read-only agents.
# The specifier, the acceptance reviewer and the researcher load this hook.
# A PreToolUse hook on Bash. Exit 2 stops the call and returns stderr to the agent.
# Ported from ../game-lab/scripts/agent_deny_writes.sh on 2026-10-09.
COMMAND=$(jq -r '.tool_input.command // empty')
if printf '%s' "$COMMAND" | grep -qE -- 'git( -C [^ ]+)?\s+(commit|push|tag|reset|rebase|clean|stash|restore|checkout)|gh (pr|issue|release) (create|edit|close|merge)'; then
  echo "Blocked: read-only agents never write git history or edit GitHub records. Return a proposal to the supervisor." >&2
  exit 2
fi
exit 0
