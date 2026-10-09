#!/bin/bash
# Routing gate: a PreToolUse hook on the Agent tool.
# Deny a dispatch with no explicit `model`, so no sub agent silently inherits the supervisor's model.
# Ask before a `fable` dispatch whose prompt carries no `Escalation:` line naming the reason.
# Fails open: a parse error or missing jq lets the call through to normal permissions.
# Rules: docs/reference/worker-runs.md#model-routing.
# Ported from ../game-lab/scripts/hooks/agent_model_gate.sh on 2026-10-09.
INPUT=$(cat)
command -v jq >/dev/null || exit 0
TOOL=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null) || exit 0
case "$TOOL" in Agent|Task) ;; *) exit 0 ;; esac
MODEL=$(printf '%s' "$INPUT" | jq -r '.tool_input.model // empty')
TYPE=$(printf '%s' "$INPUT" | jq -r '.tool_input.subagent_type // empty')
PROMPT=$(printf '%s' "$INPUT" | jq -r '.tool_input.prompt // empty')

decide() {
  jq -n --arg d "$1" --arg r "$2" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:$d,permissionDecisionReason:$r}}'
  exit 0
}

# A fork always runs on the supervisor's model, so `model` does not apply to it.
[ "$TYPE" = "fork" ] && exit 0

if [ -z "$MODEL" ]; then
  decide deny "Routing gate: pass an explicit model (haiku: search/lookup, sonnet: research/docs, opus: build/test/review). See docs/reference/worker-runs.md#model-routing."
fi

case "$MODEL" in
  fable|best|claude-fable-*|claude-mythos-*)
    if ! printf '%s' "$PROMPT" | grep -qiE '^[[:space:]]*Escalation:'; then
      decide ask "Routing gate: a Fable dispatch is for a decision opus failed twice or xhigh did not settle. Add an 'Escalation: <reason>' line to the brief, or approve this one."
    fi ;;
esac
exit 0
