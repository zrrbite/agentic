---
description: Standards for Python source files
applyTo: "**/*.py"
---

# Python instructions

Applied automatically to Python files. Keep guidance concrete so the first
answer runs — re-prompting a broken script costs premium requests. Defaults
below assume a modern uv/ruff/pytest toolchain; adjust per project.

## Language & types
- Target **Python 3.11+** unless the repo pins otherwise.
- **Type-hint public functions and class attributes.** Run clean under the
  project's type checker (mypy or pyright); no implicit `Any` across boundaries.
- Prefer `dataclasses` / `pydantic` models over loose dicts for structured data.
- Use `pathlib` over `os.path`; f-strings over `%`/`.format`.

## Style
- Formatting and lint are owned by **ruff** (and/or black) — do not hand-format
  against them. Keep changes free of new ruff warnings.
- Follow PEP 8 naming; keep functions small and single-purpose.
- Prefer comprehensions and the standard library over new dependencies.

## Errors & validation
- Raise specific exceptions, never bare `except:`. Do not swallow errors silently.
- Validate external input at the boundary (pydantic, or explicit checks); trust
  internal types thereafter.
- Use context managers (`with`) for files, locks, and connections.

## Build & test
- Environment / packaging: **uv** (`uv sync`, `uv run`) or the project's venv +
  `pip`/`poetry` — match what is present (`pyproject.toml` / `uv.lock`).
- Tests: **pytest**, run via `pytest` (or `uv run pytest`). Add/extend tests for
  new behavior; use fixtures and `parametrize` over copy-paste cases.
- Type-check and lint before considering work done.
