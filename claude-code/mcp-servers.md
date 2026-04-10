# MCP Servers

## What it is

Model Context Protocol (MCP) is an open standard that connects Claude Code to external tools, data sources, APIs, and services like GitHub, Slack, databases, and custom tooling. MCP servers expose "tools" that Claude can call without manual setup.

## Where it lives

- `~/.claude/settings.json` (user scope)
- `.claude/settings.json` (project scope)
- `claude mcp add` CLI command

## Minimal example (stdio transport)

```json
{
  "mcp": {
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem"],
        "env": {}
      }
    }
  }
}
```

## Transports

- **stdio** — local process, command + args
- **HTTP / SSE** — remote servers, URL-based

## Gotchas

- MCP tools are namespaced as `mcp__<server>__<tool>` (important for hook matchers)
- stdio servers must be locally installed/available
- Remote HTTP/SSE servers need network access + permission rules
- Official server registry: `modelcontextprotocol/servers` (see `../sources/mcp-registries.md`)

## Docs

- <https://code.claude.com/docs/en/mcp.md>
- Official registry: <https://github.com/modelcontextprotocol/servers>
