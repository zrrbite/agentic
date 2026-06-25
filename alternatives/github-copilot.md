# GitHub Copilot

## TL;DR

Microsoft/GitHub's AI coding assistant — the incumbent. Inline completions, Chat, agent mode, and code review, embedded in VS Code, Visual Studio, JetBrains, Neovim, the CLI, and github.com. Model-pluggable (GPT, Claude, Gemini, o-series). Billing is the defining trait: a **premium-request** quota model moving to **usage-based AI Credits on 2026-06-01**.

## Specs

- **License**: Proprietary (commercial). Plans: Free, Pro, Pro+, Business, Enterprise.
- **Architecture**: IDE extensions + CLI + web; no standalone editor (rides host IDE).
- **Models**: GPT-4o / GPT-4.1 (default, **0×**), plus premium models (Claude, Gemini Pro, o-series, GPT-4.5) drawn from a monthly quota.

## Cost model (the thing that actually bills)

The billable unit is the **premium request**, weighted by a per-model **multiplier**:

| Fact | Value |
|---|---|
| Default models (GPT-4o / GPT-4.1) | **0× — unlimited** on paid plans |
| Pro / Pro+ monthly premium quota | 300 / 1500 |
| Business / Enterprise (per user) | 300 / 1000 |
| Free | 50 |
| Overage | **$0.04 / request** |
| Multiplier range | **0.25×** (Gemini Flash) … **50×** (GPT-4.5) |
| Auto model selection | **0.9× discount** |
| Copilot code review | **13×** (from 2026-06-01) |

> 2026-06-01: all plans move to usage-based **AI Credits** — same levers, renamed unit.

### Saving premium requests
- **Model routing is the biggest lever** — do routine work on the 0× models; reserve premium models for the hard final moves; enable **Auto** for the 0.9× discount; avoid 50× models.
- **Inline comments cost 0** — a detailed comment before the cursor is treated as an instruction; the same ask in Chat costs a request.
- **Front-load context** (`copilot-instructions.md`) so the first answer is right — clarifying round-trips are where requests leak.
- **Watch quiet drains** — code review (13×) and agent-mode fan-out; prefer edit/ask mode for small changes.
- **Budgets** — set spending limits and monitor the per-user usage dashboard.

## Extensibility

The "skills" surface is a set of Markdown files (close cousins of CC's `.claude/` files):

- **`.github/copilot-instructions.md`** — repo-wide context auto-loaded by Chat.
- **`*.instructions.md`** (`applyTo:` glob) — path-scoped coding standards.
- **`*.prompt.md`** (`mode: agent|ask|edit`) — reusable slash-command tasks.
- **`*.chatmode.md`** — personas / tool presets you switch Chat into.
- **`skills/`, `agents/`, `plugins/`** — newer bundled formats; plugins install via `copilot plugin install <name>@awesome-copilot`.
- **MCP** — supported in agent mode.

Community hub: [`github/awesome-copilot`](https://github.com/github/awesome-copilot) (see [`sources/skill-repos.md`](../sources/skill-repos.md)).

## vs Claude Code

- **IDE/host-bound** vs CC's terminal-first design — better inline UX, weaker headless/CI story.
- **Request-quota billing** (premium requests / AI Credits) vs CC's token/usage API billing — Copilot's costs are easier to cap but its 0× default models make routine use effectively free.
- **Customization files mirror CC** — instructions ≈ CLAUDE.md, prompt files ≈ slash commands, chat modes ≈ subagents/output styles — so patterns port across both.

## Links

- Homepage: <https://github.com/features/copilot>
- Models & pricing: <https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing>
- Usage-based billing (2026): <https://github.blog/news-insights/company-news/github-copilot-is-moving-to-usage-based-billing/>
- awesome-copilot: <https://github.com/github/awesome-copilot>
