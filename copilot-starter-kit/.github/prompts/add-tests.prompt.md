---
mode: agent
description: Add focused tests for the selected code or file
---

# Add tests

For the selected code (or `${file}` if nothing is selected):

1. Identify the public behavior and the meaningful edge cases (boundaries, empty
   inputs, error paths). State them briefly before writing.
2. Write tests using **this project's existing test framework** — detect it from
   the repo (GoogleTest/Catch2 for C/C++, vitest/jest for TypeScript). Do not
   introduce a new framework.
3. Follow the structure and naming of existing tests in the nearest test dir.
4. Cover the happy path plus the edge cases — do not test trivial getters.
5. Show how to run the new tests.

Keep it to one pass; do not stop to ask what framework — infer it. Ask **one**
question only if behavior is genuinely ambiguous.
