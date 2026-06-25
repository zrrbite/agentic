---
mode: ask
description: Review the current diff for correctness and risks
---

# Review diff

Review the current changes (`${selection}` or the working diff). Report only what
matters — be concise to keep this a single request:

1. **Correctness bugs** — logic errors, off-by-one, null/UB, race conditions,
   memory issues (C/C++), unhandled errors.
2. **Security/risk** — unvalidated input, injection, unsafe casts/buffers.
3. **Reuse/simplification** — duplicated logic or existing utilities not used.

For each finding: file:line, what's wrong, suggested fix. Skip nits and style
that a formatter/linter already handles. If the diff is clean, say so plainly.
