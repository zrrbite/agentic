# Premium-request spend self-audit

A short weekly ritual to keep Copilot spend low. Five minutes; catches the leaks
before they become an overage bill (or, after 2026-06-01, drained AI Credits).

> The billable unit is the **premium request** (→ **AI Credits**). Everything here
> is about finding where requests leak. Levers and economics live in
> [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md); routing in
> [`MODEL-ROUTING-CHEAT-CARD.md`](MODEL-ROUTING-CHEAT-CARD.md).

## Where to look

- **Personal:** GitHub → *Settings* → *Billing* → Copilot premium-request usage.
- **Org/Enterprise:** the per-user premium-request usage dashboard, plus billing
  *spending limits / budgets*.

## Weekly checklist

1. **Quota burn-down.** What fraction of the monthly quota is gone, and is it
   tracking ahead of the calendar? (e.g. 60% spent by day 10 = on pace to overrun.)
2. **Model mix.** What share of requests went to **premium** vs **Auto** vs **0×**?
   Routine work should sit on 0×/Auto. A premium-heavy mix is the #1 leak.
3. **50× sightings.** Any GPT-4.5 (50×) usage? Each is ~50 ordinary requests —
   confirm every one was warranted; re-scope the habit if not.
4. **Code review (13×).** How many runs? It is metered heavily — reserve it for PRs
   that matter, or substitute the kit's `code-reviewer` agent on a cheap model.
5. **Agent-mode fan-out.** Tasks that ballooned into many requests. Could they have
   been `ask`/`edit` instead? Were you using `confirm-first` to avoid re-prompts?
6. **Re-prompt rate.** Lots of "no, I meant…" corrections? That is missing context —
   strengthen `copilot-instructions.md` so the *first* answer lands.

## If a number is off → the fix

| Symptom | Fix |
|---|---|
| Quota burning too fast | Default the model picker to **Auto/0×**; reserve premium for hard steps |
| Premium-heavy model mix | Apply the [routing cheat card](MODEL-ROUTING-CHEAT-CARD.md); reach up only when Auto stalls |
| 50× usage | Stop selecting GPT-4.5 for routine work; re-scope the task |
| Frequent code-review runs | Run it on key PRs only; use the `code-reviewer` agent otherwise |
| Agent-mode blowups | Prefer `ask`/`edit`; use `confirm-first` and the `plan` prompt first |
| High re-prompt rate | Front-load context: fill/expand `copilot-instructions.md` and `*.instructions.md` |
| Overage risk (org) | Set a **spending limit / budget** in GitHub billing |

## The one-line takeaway

If only one number moves the bill, it is the **model mix** — keep routine work off
premium models and most of the spend takes care of itself.
