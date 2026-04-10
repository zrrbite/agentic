# LangGraph

## What it is

Low-level orchestration framework from LangChain for building **stateful, long-running agents** with durable execution, graph-based control flow, human-in-the-loop interrupts, and memory primitives. Framework-agnostic — usable without the rest of LangChain.

## Relationship to Claude Code

Does **not** call Claude Code. Shares LLM backend — integrates Anthropic Claude via `langchain-anthropic` / `ChatAnthropic`. Orchestrates agents (including Claude-backed ones) as nodes in a graph.

## Install sketch

```bash
pip install -U langgraph langchain-anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from langgraph.graph import StateGraph
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-opus-4-6")

graph = StateGraph(...)
# build graph, add nodes, compile, invoke
```

## Use cases

- Agent workflows that need durability (checkpoints, resume from interrupt)
- Multi-agent graphs where Claude is one node
- Human-in-the-loop approval flows
- Long-running research tasks

## Links

- GitHub: <https://github.com/langchain-ai/langgraph>
