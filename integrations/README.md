# Integrations — external tools that touch Claude Code's world

Tools outside Claude Code that either call it, share its LLM backend, or orchestrate agents alongside it.

## Relationship categories

- **Calls Claude Code** — spawns/manages CC sessions as a sub-tool
- **Shares LLM backend** — uses Anthropic Claude directly, parallel to CC
- **Orchestrates agents** — framework for building agent systems, Claude-backed or otherwise

## Files in this folder

| Tool | Type | Relationship |
|---|---|---|
| [OpenClaw](openclaw.md) | Personal AI assistant | Can spawn CC sessions remotely via chat; shares LLM backend |
| [n8n](n8n.md) | Workflow automation | Shares LLM backend (Anthropic Chat Model node) |
| [LangGraph](langgraph.md) | Agent orchestration | Shares LLM backend (via `langchain-anthropic`) |
| [CrewAI](crewai.md) | Multi-agent framework | Shares LLM backend (via LiteLLM) |
| [Letta](letta.md) | Memory-first agents | Shares LLM backend; ships `letta-code` CLI |

## Note

None of these embed Claude Code's runtime. The closest is **OpenClaw**, which markets "autonomous Claude Code loops" run remotely — effectively managing CC as a sub-process rather than linking against it.

If you want tools that actually link against CC's agent loop, see `../claude-code/sdk.md`.
