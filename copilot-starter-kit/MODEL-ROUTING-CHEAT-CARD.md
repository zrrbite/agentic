# Model-routing cheat card

Model routing is the **single biggest premium-request lever**. Glance here before
you pick a model. The rule in one line: **do the work on 0×, think on Auto, and
spend premium only on the hard final moves.**

> Multipliers move over time — confirm exact numbers in
> [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md) and the
> [models & pricing docs](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing).

## Task → model tier

| Task | Use | Why |
|---|---|---|
| Inline completions while typing | **0×** (default) | Unlimited; free |
| A detailed comment-as-instruction before the cursor | **0× / inline** | Costs **no** premium request at all |
| Boilerplate, renames, simple edits, doc comments | **0×** (GPT-4o / 4.1) | Trivial; no premium model earns its keep |
| Q&A about the codebase, "explain this" | **0×** or **Auto** | Auto's 0.9× only if it needs reasoning |
| Writing focused tests, small refactors | **Auto** | 0.9× discount; routes up only when needed |
| Multi-file refactor, tricky bug diagnosis | **Auto → premium** | Plan on Auto; escalate only for the hard step |
| Deep architecture / subtle concurrency / cross-file taint | **Premium** (Claude / Gemini Pro / o-series) | The "final few moves after the strategy is locked" |
| Anything routine | **never 50×** (GPT-4.5) | One request ≈ 50 ordinary ones |

## Decision flow

1. **Can a comment-as-instruction do it?** → write the comment (0 premium). Stop.
2. **Is it mechanical / boilerplate?** → 0× model. Stop.
3. **Does it need reasoning?** → **Auto** (0.9×). Let it route.
4. **Did Auto stall on genuinely hard reasoning?** → escalate to one premium model
   for *that step only*, then drop back down.
5. **Tempted by 50×?** → almost never justified. Re-scope the task instead.

## Watch the metered actions

| Action | Cost | Discipline |
|---|---|---|
| Copilot code review | **13×** | Use deliberately, not every push — or run the kit's `code-reviewer` agent on a cheap model |
| Agent mode | fans out into many requests | Prefer `ask`/`edit` for small, scoped changes |
| Premium model on routine work | full multiplier each turn | Default down to 0×/Auto |

## The habit

Lock **Auto** as your standing default in the model picker (built-in 0.9×), reach
*down* to 0× for mechanical work, and reach *up* to premium only for the rare hard
step. You will almost never need to touch a 50× model.
