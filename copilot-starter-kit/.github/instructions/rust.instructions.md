---
description: Standards for Rust source files
applyTo: "**/*.rs"
---

# Rust instructions

Applied automatically to Rust files. Keep guidance concrete so the first answer
compiles and passes clippy — re-prompting a borrow-checker fight costs premium
requests. Defaults assume a stable cargo toolchain; adjust per project.

## Language & idioms
- Target the project's edition (assume **2021** unless `Cargo.toml` says otherwise).
- Prefer ownership and borrowing over `clone()`; clone only when measured or clearly
  cheap. Avoid `unsafe` unless required, and document the invariants when used.
- Use iterators and combinators over manual index loops where it reads clearly.
- Prefer `&str`/`&[T]` parameters over owned `String`/`Vec<T>` for non-owning use.

## Errors & results
- Return `Result<T, E>` for fallible operations; do not `unwrap()`/`expect()` in
  library code or on external input. Reserve them for tests and provable invariants.
- Use `?` for propagation. Use **`thiserror`** for library error enums and
  **`anyhow`** for application-level error context — match what the crate already uses.
- Model absence with `Option<T>`, not sentinel values.

## Style
- Formatting is owned by **`rustfmt`** — do not hand-format against it.
- Keep code **clippy-clean**: `cargo clippy --all-targets` with no new warnings.
- Derive (`Debug`, `Clone`, `PartialEq`) rather than hand-implementing where it fits.

## Build & test
- Build: `cargo build`. Lint: `cargo clippy`. Format check: `cargo fmt --check`.
- Tests: `cargo test` — unit tests in-module under `#[cfg(test)]`, integration
  tests in `tests/`. Add/extend tests for new behavior.
- Run clippy, fmt, and tests before considering work done.
