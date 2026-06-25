---
description: Standards for TypeScript source files
applyTo: "**/*.{ts,tsx,mts,cts}"
---

# TypeScript instructions

Applied automatically to TypeScript files. Defaults below assume a modern
pnpm + Vite + Vitest toolchain; adjust per project.

## Types
- **`strict` mode on.** No implicit `any`; type function boundaries explicitly.
- Prefer precise types and discriminated unions over `any`/`unknown` casts.
- Avoid non-null assertions (`!`) unless provably safe and commented.

## Style
- Match the existing module system (ESM preferred) and import ordering.
- Prefer `const`; immutable data where practical. Async/await over raw promises.
- No unused exports or dead code.
- Formatting/lint owned by **Prettier + ESLint** — do not fight the configs.

## Errors & validation
- Validate external input at the boundary with **zod** (or the project's existing
  validator, e.g. Effect Schema); trust internal types thereafter.
- Throw `Error` subclasses, not strings.

## Build & test
- Package manager: **pnpm**. Build/bundle: **Vite** (libs: `tsc` for `.d.ts`).
- Type-check: `pnpm exec tsc --noEmit`. Lint: `pnpm lint`.
- Tests: **Vitest**, run via `pnpm test`. Add/extend tests for new behavior.
- Run type-check, lint, and tests before considering work done.
