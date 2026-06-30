---
mode: ask
description: Explain the selected code concisely (no edits)
---

# Explain code

Explain the selected code (or `${file}` if nothing is selected). Be concise — one
request should answer it fully:

1. **What it does** — purpose in 1-2 sentences.
2. **How** — the key steps / control flow, only the non-obvious parts.
3. **Watch-outs** — edge cases, side effects, assumptions, or footguns a caller
   must know.

Do not rewrite or "improve" the code — this is read-only. Skip line-by-line
narration of self-evident lines. Use the cheapest capable model.
