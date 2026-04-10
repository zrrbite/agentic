# Subagents

## What it is

Subagents are specialized AI assistants defined in `.claude/agents/<name>.md`. Each runs in its **own context window**, isolated from the main conversation. Use them when a task would flood your main chat with logs/search results, or to parallelize work.

## Built-in types

| Agent | Use for |
|---|---|
| `general-purpose` | Broad multi-step research or execution |
| `Explore` | Read-only codebase inspection (fast, thorough tiers) |
| `Plan` | Design implementation plans without executing |
| `statusline-setup` | Configure status line via settings.json |
| `claude-code-guide` | Answer questions about Claude Code/SDK/API |

(Exact list varies by install — check via the `Agent` tool in session.)

## Custom subagent definition

`.claude/agents/code-reviewer.md`:
```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
model: claude-opus-4-6
tools: [Read, Grep, Bash]
---
You are an expert code reviewer. Analyze code for bugs, performance, and style.
Focus on security and testability.
```

## When to use subagents

**Good fit:**
- Broad codebase exploration (use `Explore`)
- Independent parallel tasks (dispatch 2-3 in parallel)
- Research that returns >20kB of tool output (keeps main context clean)
- Second-opinion code reviews (independent context = genuine independence)

**Bad fit:**
- Tasks where main agent already has all needed context
- Anything requiring tight back-and-forth with the user
- Trivial lookups (direct tool call is faster)

## Parallel dispatch

Spawn multiple subagents in a single message — they run concurrently. Max 3 parallel agents is a good rule of thumb.

## Gotchas

- Subagent context is **independent** — they don't see your conversation
- Brief them like a smart colleague walking in cold: goal, prior findings, expected output format
- `/compact` and context management work per-agent
- Agents can be run in background (`run_in_background: true`) for truly async work

## Docs

- <https://code.claude.com/docs/en/sub-agents.md>
