# Agentic Coding — reference notes

A rough-notes knowledge base for exploring what **Claude Code** (and agentic coding tools in general) can do. The goal is to map out the full landscape: Claude Code's extensibility surface, external tools that interop with it, and the broader field of AI coding agents.

**Status**: living document, rough notes > polish.

## Layout

| Folder | What's inside |
|---|---|
| [`claude-code/`](claude-code/) | Claude Code's extensibility surface — MCP, plugins, skills, subagents, hooks, SDK, settings, IDE integrations |
| [`integrations/`](integrations/) | External tools that talk to Claude Code — OpenClaw, n8n, LangGraph, CrewAI, Letta |
| [`alternatives/`](alternatives/) | Competing agentic coding tools — OpenCode, Cursor, Aider, Cline, Codex, Continue, Windsurf, Zed, Goose |
| [`sources/`](sources/) | Where to find more — awesome-lists, plugin marketplaces, skill repos, MCP registries |
| [`experiments/`](experiments/) | Hands-on "I tried X" notes (dated files, added as you go) |

## Start here

- New to Claude Code extension points? → [`claude-code/README.md`](claude-code/README.md)
- Looking for prebuilt skills/plugins? → [`sources/skill-repos.md`](sources/skill-repos.md)
- Comparing tools? → [`alternatives/README.md`](alternatives/README.md)

## Conventions

- Rough notes, bullets over prose
- Every external claim links to a source — if no source, mark "unverified"
- Each file: *what it is → where it lives → minimal example → gotchas → docs link*
