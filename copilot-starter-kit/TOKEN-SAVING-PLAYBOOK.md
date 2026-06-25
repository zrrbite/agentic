# GitHub Copilot — Token / Premium-Request Saving Playbook

> The thing that costs money on Copilot is the **premium request** (becoming
> **GitHub AI Credits** on 2026-06-01), *not* raw tokens. Every lever below is
> aimed at that unit.

## The billing model in one table

| Fact | Value |
|------|-------|
| Default models (GPT-4o / GPT-4.1) | **0× — unlimited** on paid plans |
| Pro / Pro+ monthly premium quota | 300 / 1500 |
| Business / Enterprise (per user) | 300 / 1000 |
| Free | 50 |
| Overage | **$0.04 / request** |
| Multiplier range | **0.25×** (Gemini Flash) … **50×** (GPT-4.5) |
| Auto model selection | **0.9× discount** |
| Copilot code review | **13×** (from 2026-06-01) |

## Lever 1 — Model routing (biggest win)
- **Default to 0× models** (GPT-4o / GPT-4.1) for routine completions, edits, Q&A.
- **Reserve premium models** (Claude, Gemini Pro, o-series, GPT-4.5) for the hard
  "final few moves after the strategy is locked."
- **Enable Auto model selection** in VS Code Copilot Chat → built-in 0.9× discount.
- **Avoid 50× models** unless truly warranted — one request ≈ 50 ordinary ones.

## Lever 2 — Cut the round-trips (where requests leak)
- **Inline comments instead of chat.** A detailed comment before the cursor is
  treated as an instruction and costs **0 premium** — the same ask in Chat costs
  a request.
- **Confirm before finishing.** Use the `confirm-first` chat mode so agent mode
  pauses for review instead of charging ahead and forcing corrective re-prompts.
- **Front-load context** via `copilot-instructions.md` so the first answer is
  right — you stop paying for clarifying request #2 and #3.

## Lever 3 — Watch the quiet drains
- **Copilot code review = 13×.** Use deliberately, not on every push.
- **Agent mode fans out** into many requests per task — prefer edit/ask mode for
  small, well-scoped changes.

## Lever 4 — Budget controls (org / enterprise)
- Set **spending limits / budgets** in GitHub billing to cap overage exposure.
- Monitor the per-user premium-request **usage dashboard** to find heavy patterns.

## Community "skills" to mine (don't author cold)
- **`github/awesome-copilot`** — official hub: `instructions/`, `prompts/`,
  `chatmodes/`, `skills/`, `agents/`, `plugins/`, `collections/`.
  Website: `awesome-copilot.github.io` (search + Learning Hub + `llms.txt`).
  Install plugins: `copilot plugin install <name>@awesome-copilot`.
- **`jaktestowac/awesome-copilot-for-testers`** — test-automation focused.
- Pull the **collection matching your stack** rather than the whole repo.

## Sources
- Models & pricing: https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
- Usage-based billing (2026): https://github.blog/news-insights/company-news/github-copilot-is-moving-to-usage-based-billing/
- Premium requests overview: https://docs.github.com/en/billing/concepts/product-billing/github-copilot-premium-requests
- Optimizing usage (community): https://github.com/orgs/community/discussions/163104
- Billing mechanics & anti-patterns: https://smartscope.blog/en/generative-ai/github-copilot/github-copilot-premium-request-optimization/
- awesome-copilot: https://github.com/github/awesome-copilot
