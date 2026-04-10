# n8n

## What it is

Fair-code workflow automation tool (Zapier-style) with 400+ node integrations, self-hostable, supports code nodes and LangChain-style AI agent nodes.

## Relationship to Claude Code

Does **not** call Claude Code. Shares LLM backend — n8n ships an **Anthropic Chat Model** node (`n8n-nodes-langchain.lmChatAnthropic`) used inside AI Agent / LangChain nodes to orchestrate Claude as a step in workflows.

## Install sketch

```bash
npx n8n
# or: docker run -it --rm -p 5678:5678 n8nio/n8n

# In the UI:
# 1. Add "Anthropic Chat Model" node
# 2. Paste API key
# 3. Wire into "AI Agent" or "Basic LLM Chain" nodes
```

## Use cases

- Trigger Claude on webhook / schedule / file change
- Multi-step workflows where Claude is one LLM step among many
- Glue between SaaS tools with Claude reasoning in the middle

## Links

- Homepage: <https://n8n.io>
