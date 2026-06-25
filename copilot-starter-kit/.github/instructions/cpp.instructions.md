---
description: Standards for C and C++ source files
applyTo: "**/*.{c,h,cpp,cc,cxx,hpp,hh}"
---

# C / C++ instructions

Applied automatically to C and C++ files. Keep guidance concrete so the first
answer compiles — re-prompting a broken build costs premium requests.

## Language & standard
- Target standard: **[C++20 / C17 — set yours]**. Do not use features beyond it.
- Header guards or `#pragma once` consistently with the existing file.
- Prefer C++ idioms over C ones in `.cpp`; keep `.c` files C-only.

## Memory & safety
- Prefer RAII and smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw
  `new`/`delete`. Raw owning pointers only at clearly documented boundaries.
- No leaks, no use-after-free, no unchecked buffer arithmetic. Bounds-check.
- Initialize variables at declaration. Mark `const`/`constexpr` where possible.

## Style
- Match surrounding formatting (assume `.clang-format` if present — do not fight it).
- Pass large objects by `const&`; return by value and rely on move/RVO.
- Errors: use [exceptions / `std::expected` / error codes — set yours] consistently.

## Build & test
- Build system: **[CMake / Make — set yours]**.
- Add/extend tests in **[GoogleTest / Catch2 — set yours]** for new behavior.
