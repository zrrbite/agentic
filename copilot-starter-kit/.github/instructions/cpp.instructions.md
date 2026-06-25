---
description: Standards for C and C++ source files
applyTo: "**/*.{c,h,cpp,cc,cxx,hpp,hh}"
---

# C / C++ instructions

Applied automatically to C and C++ files. Keep guidance concrete so the first
answer compiles — re-prompting a broken build costs premium requests.
Defaults below reflect a modern CMake + clang toolchain; adjust per project.

## Language & standard
- Target standard: **C++20** (C: **C17**). Do not use features beyond it.
- Use `#pragma once` for header guards.
- Prefer C++ idioms over C ones in `.cpp`; keep `.c` files C-only.

## Memory & safety
- Prefer RAII and smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw
  `new`/`delete`. Raw owning pointers only at clearly documented boundaries.
- No leaks, no use-after-free, no unchecked buffer arithmetic. Bounds-check.
- Initialize variables at declaration. Mark `const`/`constexpr` where possible.
- Prefer `std::string_view` / `std::span` for non-owning views.

## Style
- Formatting is owned by **`.clang-format`** — do not hand-format against it.
- Static analysis via **`.clang-tidy`**; keep changes clean of new warnings.
- Pass large objects by `const&`; return by value and rely on move/RVO.
- Errors: use **exceptions** at API boundaries where exceptions are enabled;
  in exception-disabled or embedded builds, use **`std::expected`** / error
  codes. Match what the target already does.

## Build & test
- Build system: **CMake** (Ninja generator). Typical: `cmake --build build`
  (or the project's `cmake --preset` if `CMakePresets.json` exists).
- Tests: **GoogleTest**, run via `ctest --test-dir build` (or the project's
  test target). Add/extend tests for new behavior.
