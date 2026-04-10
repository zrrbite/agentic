# MCP Server Registries

Places to discover MCP servers.

## Verified GitHub-based

- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — **Official** reference implementations + community-built MCP servers. The canonical index.
- [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) — Community-curated MCP directory on GitHub (~84.5k stars)

## Web directories (unverified — blocked during research)

These are widely referenced but couldn't be confirmed from inside the research sandbox. Treat as probably-real-check-before-using:

- `mcp.so` — unverified
- `pulsemcp.com` (PulseMCP) — unverified
- `smithery.ai` (Smithery) — unverified
- `mcpservers.org` — unverified

## Adding an MCP server to Claude Code

```bash
# CLI
claude mcp add <name> <command> [args...]

# or directly in settings.json — see ../claude-code/mcp-servers.md
```
