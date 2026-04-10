# Status Line

## What it is

The bar at the bottom of Claude Code's terminal UI — shows session info like model, context usage, permissions mode.

## Where it lives

`~/.claude/settings.json` under the `statusLine` key.

## Example

```json
{
  "statusLine": {
    "show": true,
    "format": "{model} • {usage}% • {mode}"
  }
}
```

## Configuration via agent

Claude Code ships a `statusline-setup` subagent specifically for configuring this — delegate to it if you want it done interactively.

## Gotchas

- Exact template variable list is **unverified** in public docs — the `statusline-setup` subagent is the authoritative source
- Some fields may only render in certain terminals

## Docs

- <https://code.claude.com/docs/en/settings.md> (statusLine section)
- Use the `statusline-setup` subagent
