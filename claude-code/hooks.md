# Hooks

## What it is

Hooks are shell commands that execute **deterministically** at specific lifecycle events — not by model decision. Use them for auto-formatting, blocking protected files, auto-approving permissions, notifications, or enforcing invariants.

## Lifecycle events

| Event | When it fires |
|---|---|
| `SessionStart` | New session begins |
| `UserPromptSubmit` | User sends a message |
| `PreToolUse` | Before a tool call — can block it |
| `PostToolUse` | After a tool call completes |
| `Stop` | Session/turn ends |
| `Notification` | Claude requests attention |

(Exact event list may expand — check current docs.)

## Where it lives

`settings.json` under the `hooks` key:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ]
  }
}
```

## Matcher syntax

- Regex-like over tool names: `Edit|Write` matches both
- `mcp__.*` matches any MCP tool
- `Bash` matches shell calls
- Omit matcher to fire on all tools in that event

## Input/output contract

- **stdin**: JSON with event metadata (tool name, inputs, outputs, cwd, etc.)
- **stdout**: JSON (for structured responses) or plain text
- **exit code**: 0 = pass, 2 = block the action, other = error

## Plugin-bundled hooks

Plugins can ship hooks via `<plugin>/hooks/hooks.json` in the same schema.

## Gotchas

- Hooks run **sequentially** when multiple match (not parallel)
- Default timeout: 10 minutes
- `PostToolUse` can't undo an already-executed action — use `PreToolUse` to block
- Hook commands inherit the session's shell env
- Exit code 2 is the canonical "block" signal; other non-zero codes surface as errors

## Docs

- <https://code.claude.com/docs/en/hooks-guide.md>
