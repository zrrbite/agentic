---
mode: agent
description: Diagnose a bug from a symptom and propose the minimal fix
---

# Debug

I will describe a symptom (error, wrong output, crash). Diagnose root cause first,
then fix — do not start editing on a guess.

1. **Restate the symptom** and the expected vs actual behavior in one line.
2. **Form the most likely hypothesis** from the code and the error/stack. If you
   need a fact you cannot infer (exact input, env, versions), ask **one** focused
   question before proceeding.
3. **Locate the root cause** — point to `file:line`. Do not treat a symptom site as
   the cause without tracing back to it.
4. **Propose the minimal fix**, apply it, and state how to verify (the command or
   the test that now passes). Note any other sites with the same latent bug.

Keep it scoped to this bug. Prefer the cheapest capable model; escalate only if the
root cause needs deep cross-file reasoning.
