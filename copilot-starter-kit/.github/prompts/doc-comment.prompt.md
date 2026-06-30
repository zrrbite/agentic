---
mode: edit
description: Add doc comments to the selected API without changing code
---

# Doc comment

Add documentation comments to the selected code (or `${file}`). **Edit comments
only — do not change the code itself.**

- Use the language's idiomatic doc style: docstrings (Python), `///` rustdoc
  (Rust), Doxygen/`///` (C/C++), TSDoc/`/** */` (TypeScript).
- Document the contract: purpose, parameters, return value, errors/exceptions
  thrown, and any side effects or important preconditions.
- Be concise and accurate — describe what the code actually does, not aspirations.
  Do not restate obvious type information the signature already conveys.
- Skip trivial getters/setters and self-evident one-liners.

One pass. Do not reformat or refactor surrounding code.
