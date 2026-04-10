# Claude Code SDK / Agent SDK

## What it is

Programmatic access to Claude Code's agent loop. Two flavors:

1. **Headless CLI** — `claude -p "prompt"` for scripted non-interactive runs
2. **Agent SDK** — `@anthropic-ai/claude-agent-sdk` (TS) or `anthropic-agent` (Python) for embedding the agent loop in your own app

Also: **Managed Agents API** (`/v1/agents`, `/v1/sessions`) for server-hosted agents — typically enterprise tier.

## Headless CLI example

```bash
claude -p "Find and fix bugs in auth.py" \
  --allowedTools "Read,Edit,Bash" \
  --output-format json | jq '.result'
```

Common flags:
- `-p, --print` — non-interactive, print and exit
- `--allowedTools` — comma-separated allow list
- `--output-format` — `text` (default) or `json`
- `--bare` — skip CLAUDE.md, hooks, plugin discovery (good for CI)
- `--continue` — resume previous session
- `--model` — override default model

## TypeScript SDK

```ts
import { Agent } from "@anthropic-ai/claude-agent-sdk";

const agent = new Agent({
  model: "claude-opus-4-6",
  allowedTools: ["Read", "Edit", "Bash"],
});

const result = await agent.run("refactor src/auth.ts for testability");
console.log(result.output);
```

## Use cases

- **CI checks** — run agents on PRs in GitHub Actions
- **Cron jobs** — scheduled code audits, dependency updates
- **Custom UIs** — embed CC in Slack bots, web dashboards
- **Batch work** — run the same prompt across many repos

## Gotchas

- Bare mode skips your CLAUDE.md and hooks — intentional for reproducibility but don't forget
- Streaming requires Python 3.8+ or TypeScript targeting ES2017+
- Managed agents (`/v1/agents`) are a separate product — check pricing tier
- Rate limits apply at the API level, not per-session

## Docs

- Headless: <https://code.claude.com/docs/en/headless.md>
- SDK: <https://code.claude.com/docs/en/sdk>
- Managed Agents API: <https://docs.claude.com/en/api/agents>
