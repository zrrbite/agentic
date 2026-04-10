# Settings

## What it is

`settings.json` configures Claude Code globally or per-project. The single most important config surface.

## Scopes (precedence: high → low)

| File | Scope |
|---|---|
| `.claude/settings.local.json` | Project local (gitignored, secrets) |
| `.claude/settings.json` | Project (committed) |
| `~/.claude/settings.json` | User |
| Managed / enterprise profile | Admin-enforced (can't override) |

## Common fields

```json
{
  "model": "claude-opus-4-6",
  "effortLevel": "high",

  "permissions": {
    "allow": ["Bash(git *)", "Read", "Edit", "Write"],
    "deny": ["Bash(rm -rf *)"]
  },

  "env": {
    "NODE_ENV": "development",
    "API_KEY": "..."
  },

  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "prettier --write ..." }]
      }
    ]
  },

  "statusLine": {
    "show": true,
    "format": "{model} • {usage}%"
  },

  "outputStyle": "default",

  "enabledPlugins": {
    "superpowers@claude-plugins-official": true
  },

  "extraKnownMarketplaces": {
    "my-team": "github.com/my-org/claude-marketplace"
  },

  "mcp": {
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem"]
      }
    }
  }
}
```

## Permissions cheat sheet

- `Bash(git *)` — allow any git command
- `Bash(npm test)` — allow exact command
- `Read`, `Edit`, `Write` — allow all file ops
- `mcp__github__*` — allow all tools from the `github` MCP server
- `deny` rules always win over `allow`

## Gotchas

- Changes are auto-detected but some apply **on next interaction**
- JSON syntax errors silently fall back to defaults — validate with `jq`
- Managed settings enforced by org cannot be overridden by user settings
- `.claude/settings.local.json` should be gitignored — put secrets there

## Docs

- <https://code.claude.com/docs/en/settings.md>
