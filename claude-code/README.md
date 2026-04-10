# Claude Code — extensibility surface

Claude Code is an agentic AI coding assistant from Anthropic, available as a CLI, VS Code extension, JetBrains plugin, desktop app, and web UI. It reads your codebase, edits files, runs commands, and integrates with external tools via the Model Context Protocol (MCP).

## The 5 main extension surfaces

| Surface | What it is | Where it lives |
|---|---|---|
| **MCP servers** | External tool integrations over an open protocol | `settings.json` `mcp` key or `claude mcp add` |
| **Plugins** | Shareable packages bundling skills/agents/hooks/MCP | `/plugin install <name>@<marketplace>` |
| **Skills** | Reusable task-specific prompts, auto- or manually invoked | `.claude/skills/<name>/SKILL.md` |
| **Subagents** | Specialized workers with isolated context windows | `.claude/agents/<name>.md` |
| **Hooks** | Deterministic shell commands on lifecycle events | `settings.json` `hooks` key |

## Plus these configuration surfaces

- **CLAUDE.md** — persistent instructions (project or user scope)
- **settings.json** — model, permissions, env, hooks, statusLine, enabledPlugins
- **Slash commands** — `/name` shortcuts (now implemented as skills)
- **Output styles** — customize tone/role/format
- **Status line** — terminal UI bottom bar customization
- **Keybindings** — `~/.claude/keybindings.json`

## Files in this folder

- [mcp-servers.md](mcp-servers.md) — MCP protocol, adding servers, notable ones
- [plugins.md](plugins.md) — plugin system, marketplaces, structure
- [skills.md](skills.md) — skills directory, auto-invocation, frontmatter
- [subagents.md](subagents.md) — built-in types, `.claude/agents/*.md` format
- [hooks.md](hooks.md) — lifecycle events, matchers, examples
- [slash-commands.md](slash-commands.md) — custom commands via skills
- [sdk.md](sdk.md) — Claude Code SDK / Agent SDK / headless mode
- [output-styles.md](output-styles.md) — customize response style
- [status-line.md](status-line.md) — terminal bottom bar
- [keybindings.md](keybindings.md) — keyboard shortcut config
- [settings.md](settings.md) — `settings.json` reference
- [ide-integrations.md](ide-integrations.md) — VS Code, JetBrains
