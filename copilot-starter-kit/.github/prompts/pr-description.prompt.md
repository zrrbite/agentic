---
mode: ask
description: Draft a pull-request description from the current diff
---

# PR description

Draft a pull-request description for the current changes (`${selection}` or the
working/branch diff). One request — produce a complete, paste-ready body; do not
iterate.

Structure:
- **Title** — imperative, concise (e.g. `Add retry/backoff to upload client`).
- **Summary** — 1-3 sentences: what changed and *why*.
- **Changes** — short bullets of the meaningful changes (group by area if large).
  Skip mechanical noise (formatting, generated files).
- **Testing** — how it was verified (commands run / tests added). If you cannot
  tell from the diff, write `TODO: describe testing` rather than inventing it.
- **Risk / notes** — migrations, breaking changes, follow-ups — only if applicable.

Describe what the diff actually does — do not fabricate rationale, tickets, or test
results you cannot see. Match the repo's existing PR style if a template is present.
Output only the description. Use the cheapest capable model.
