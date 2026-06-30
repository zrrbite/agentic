---
mode: agent
description: Diagnose and fix a failing test in one pass
---

# Fix failing test

A test is failing. Fix it in one pass — do not stop to ask which test runner; infer
it from the repo (pytest, cargo test, GoogleTest/ctest, vitest/jest).

1. **Read the failure** — the assertion, the actual vs expected, the stack/line.
2. **Decide who is wrong** — the code under test or the test itself. State which in
   one line before changing anything. Do **not** "fix" a test by weakening its
   assertion to make it pass unless the test was genuinely incorrect.
3. **Make the minimal change** that addresses the root cause. Stay scoped to this
   failure; do not refactor unrelated code.
4. **Re-run** the specific test (show the command) and confirm it passes without
   breaking neighbors.

Ask **one** question only if the intended behavior is genuinely ambiguous.
