# Plugins

## What it is

Plugins package reusable skills, subagents, hooks, and MCP servers for sharing across projects or teams. Install with `/plugin install <name>@<marketplace>`. The official marketplace `claude-plugins-official` is built in; custom marketplaces can be GitHub repos, URLs, or local directories.

## Install flow

```bash
# add a custom marketplace
/plugin marketplace add obra/superpowers-marketplace

# install from it
/plugin install superpowers@superpowers-marketplace

# or install from the built-in official marketplace
/plugin install superpowers@claude-plugins-official

# reload after install
/reload-plugins
```

## Plugin structure

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json            # manifest: name, description, version
├── skills/
│   └── <skill>/SKILL.md
├── agents/
│   └── <agent>.md
├── hooks/
│   └── hooks.json
└── mcp/                        # optional bundled MCP servers
```

Minimal `plugin.json`:
```json
{"name": "my-plugin", "description": "My first plugin", "version": "1.0.0"}
```

## Local development

```bash
claude --plugin-dir ./my-plugin
# then invoke: /my-plugin:<skill-name>
```

## Gotchas

- Plugin skills are namespaced as `/<plugin-name>:<skill>`
- Plugin structure is strict — don't nest skills inside `.claude-plugin/`
- Plugins can't be uninstalled at runtime but can be disabled via settings

## Enabling in settings

```json
{
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true
  }
}
```

## Docs

- <https://code.claude.com/docs/en/plugins.md>
- Superpowers marketplace (reference impl): <https://github.com/obra/superpowers-marketplace>
