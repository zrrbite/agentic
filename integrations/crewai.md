# CrewAI

## What it is

Lean Python multi-agent framework (independent of LangChain) for orchestrating role-based "Crews" of cooperating agents, plus event-driven "Flows" for deterministic automation pipelines.

## Relationship to Claude Code

Does **not** call Claude Code. Orchestrates agents and shares LLM backend — supports Anthropic Claude via LiteLLM under the hood (set `model="anthropic/claude-..."` on the Agent).

## Install sketch

```bash
uv pip install 'crewai[tools]'
crewai create crew my_project
# .env: ANTHROPIC_API_KEY=sk-ant-...
```

```python
from crewai import Agent

agent = Agent(
    role="researcher",
    goal="...",
    llm="anthropic/claude-opus-4-6",
)
```

```bash
crewai run
```

## Use cases

- Multi-agent teams with distinct roles (researcher, writer, reviewer)
- Deterministic Flow pipelines with agent steps
- Role-play patterns where agents cooperate on a complex task

## Links

- GitHub: <https://github.com/crewAIInc/crewAI>
