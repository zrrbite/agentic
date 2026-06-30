---
mode: ask
description: Draft a commit message for the staged/working changes
---

# Commit message

Draft a commit message for the current changes (`${selection}` or the working diff).

Format:
- **Subject:** imperative mood, <= 72 chars, no trailing period
  (e.g. `Fix off-by-one in ring buffer wrap`). Use a Conventional Commits prefix
  (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`) **only if the repo's
  history already uses them** — match existing style.
- **Body** (only if the change is non-trivial): what changed and *why*, wrapped at
  ~72 cols. Omit the body for small obvious changes.

Describe what the diff actually does — do not invent rationale you cannot see.
Output only the message, ready to paste. One request; do not iterate.
