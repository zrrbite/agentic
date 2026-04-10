# Letta (formerly MemGPT)

## What it is

Open-source stateful agent platform focused on **long-term memory, self-editing context, and continual learning**. Model-agnostic; runs as a local/hosted server managing persistent agents via REST/SDK.

## Relationship to Claude Code

Ships a `letta-code` CLI (separate tool, not CC itself). Shares LLM backend — fully supports Anthropic Claude; docs recommend Opus-class models for best performance. Can act as a **memory layer behind other agent frameworks**.

## Install sketch

```bash
# CLI
npm install -g @letta-ai/letta-code
letta                                 # launch local agent server

# or SDK
pip install letta-client
# npm i @letta-ai/letta-client

export ANTHROPIC_API_KEY=sk-ant-...
```

## Use cases

- Agents that remember things across sessions
- Self-editing context windows — agent rewrites its own system prompt as it learns
- Persistent memory layer behind other frameworks (e.g. feed Letta memory into LangGraph/CrewAI)

## Links

- GitHub: <https://github.com/letta-ai/letta>
