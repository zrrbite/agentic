# Slash Commands

## What they are

Shortcuts invoked by typing `/name` in Claude Code. Built-in commands include `/help`, `/config`, `/mcp`, `/compact`, `/init`, `/plugin`, `/reload-plugins`, etc. Custom commands are now implemented as **skills** with `disable-model-invocation: true` (or via the legacy `.claude/commands/` files).

## Built-in highlights

| Command | Purpose |
|---|---|
| `/help` | Help menu |
| `/init` | Initialize CLAUDE.md in the current project |
| `/config` | Open settings UI |
| `/mcp` | Manage MCP servers |
| `/compact` | Manually compact context |
| `/plugin` | Plugin marketplace / install / list |
| `/reload-plugins` | Reload after install/edit |
| `/fast` | Toggle fast-output mode |
| `/clear` | Clear conversation |

## Custom command via skill (recommended)

`.claude/skills/deploy/SKILL.md`:
```markdown
---
name: deploy
description: Deploy the application to production
disable-model-invocation: true
---
Deploy to production:
1. Run tests
2. Build
3. Push to deployment target
4. Verify

Arguments: $ARGUMENTS
```

Invoke: `/deploy` or `/deploy staging` (arguments available via `$ARGUMENTS`).

## Legacy command file

`.claude/commands/deploy.md` still works but skills are the forward-compatible path.

## Gotchas

- Command names are case-sensitive
- Plugin commands are namespaced: `/<plugin>:<command>`
- For destructive commands, always set `disable-model-invocation: true` so Claude can't auto-fire them

## Docs

- <https://code.claude.com/docs/en/commands.md>
