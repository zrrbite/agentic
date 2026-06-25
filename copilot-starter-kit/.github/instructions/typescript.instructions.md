---
description: Standards for TypeScript source files
applyTo: "**/*.{ts,tsx,mts,cts}"
---

# TypeScript instructions

Applied automatically to TypeScript files.

## Types
- **`strict` mode assumed.** No implicit `any`; type function boundaries explicitly.
- Prefer precise types and discriminated unions over `any`/`unknown` casts.
- Avoid non-null assertions (`!`) unless provably safe and commented.

## Style
- Match the existing module system (ESM vs CJS) and import ordering.
- Prefer `const`; immutable data where practical. Async/await over raw promises.
- No unused exports or dead code.

## Errors & validation
- Validate external input at the boundary ([zod / io-ts — set yours]); trust
  internal types thereafter.
- Throw `Error` subclasses, not strings.

## Build & test
- Tooling: **[tsc / esbuild / vite — set yours]**, package manager **[pnpm/npm]**.
- Add/extend tests in **[vitest / jest — set yours]** for new behavior.
- Run `[lint command]` and `[typecheck command]` before considering work done.
