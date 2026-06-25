---
mode: agent
description: Refactor the selection without changing behavior
---

# Refactor (behavior-preserving)

Refactor the selected code (or `${file}`):

1. State the smells you see and the target shape in 2-3 bullets first.
2. Apply the refactor **without changing external behavior**. Reuse existing
   utilities rather than adding dependencies.
3. Respect language rules from this repo's instruction files (RAII/smart pointers
   for C++, strict typing for TypeScript).
4. Preserve or update tests; do not delete coverage.
5. Summarize what changed and why in a few lines.

Do it in one pass. Do not expand scope beyond the selection.
