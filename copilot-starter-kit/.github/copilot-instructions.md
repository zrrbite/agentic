# Copilot Repository Instructions

> Auto-loaded by Copilot Chat for every request in this repo. Keep it short and
> factual — its job is to make the *first* answer correct so you avoid paying for
> clarifying round-trips. Values below are sensible defaults for a C/C++/TypeScript
> codebase; **edit the three "Project" lines to match this repo.**

## Project
- **What this is:** A C/C++/TypeScript project.  <!-- ← replace with one line -->
- **Primary language(s):** C++20 / C17 and TypeScript.  <!-- ← adjust -->
- **Package manager:** pnpm (TS); CMake fetches/vendoring for C/C++.  <!-- ← adjust -->

## Build / run / test (state once — never re-explain in chat)
- **Install:** `pnpm install` (TS); configure C/C++ with `cmake --preset default`.
- **Build:** `cmake --build build` (C/C++); `pnpm build` (TS).
- **Run:** see the project's run target / `pnpm dev`.
- **Test:** `ctest --test-dir build` (GoogleTest); `pnpm test` (Vitest).
- **Lint / format:** clang-format + clang-tidy (C/C++); `pnpm lint` + Prettier (TS).

## Conventions
- Match the style and structure of surrounding code.
- C/C++: RAII and smart pointers, `const`/`constexpr`-correctness, no leaks/UB.
- TypeScript: `strict` types, no implicit `any`, validate input at boundaries.
- Prefer reusing existing utilities over adding new dependencies.

## Working agreement (saves premium requests)
- Give complete, working answers in one pass; do not stop mid-task to ask trivia
  you can infer from the codebase.
- When a request is genuinely ambiguous, ask **one** focused question rather than
  guessing and producing a wrong answer that needs a corrective re-prompt.
- Keep responses scoped to what was asked — no unsolicited large rewrites.
