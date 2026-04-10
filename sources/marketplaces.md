# Plugin Marketplaces

Claude Code plugin marketplaces — places to `/plugin install` from.

## Official

- **`claude-plugins-official`** — built into Claude Code; no standalone GitHub repo. Surfaced automatically via `/plugin install <name>@claude-plugins-official`.

## Community

- [obra/superpowers-marketplace](https://github.com/obra/superpowers-marketplace) — Curated Claude Code plugins: skills, workflows, productivity (Superpowers, Elements of Style, Private Journal MCP, etc.)
- [wshobson/agents](https://github.com/wshobson/agents) — Production-ready marketplace of **77 plugins** bundling 182 agents, 16 orchestrators, 149 skills, 96 commands

## Adding a custom marketplace

```bash
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
/reload-plugins
```

## In `settings.json`

```json
{
  "extraKnownMarketplaces": {
    "my-team": "github.com/my-org/claude-marketplace"
  }
}
```
