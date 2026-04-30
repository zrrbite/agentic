#!/usr/bin/env python3
"""Fast smoke test for the notebooks in this directory.

Catches the cheap-to-detect breakage *before* you push and trigger a
multi-minute CI run:

  - .ipynb files are valid JSON
  - Required structural fields are present
  - Each code cell's source parses as valid Python (`ast.parse`)

Does NOT execute notebooks — that's what render-snapshots.py and CI do.
Total runtime is sub-second across all 9 notebooks.

Usage:
    python check-notebooks.py            # all 0?-*.ipynb in this dir
    python check-notebooks.py 07-*.ipynb # one or more by name
    python check-notebooks.py --hook     # set this script up as a pre-push hook

Exit code is 0 on success, 1 on any failure — suitable for use as a
git hook or CI step.
"""
import argparse
import ast
import json
import os
import stat
import sys
from pathlib import Path


def _strip_jupyter_magics(source: str) -> str:
    """Replace IPython magic / shell lines with `pass` so ast.parse accepts them.

    Notebooks use `%pip install ...`, `!ls`, `%%time`, etc. These are valid
    Jupyter syntax but not valid Python. We don't try to parse them — just
    pass-through so the rest of the cell is checked normally.
    """
    out = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")):
            out.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def check_notebook(path: Path) -> list[str]:
    """Return a list of human-readable error messages; empty list = OK."""
    errs: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]
    except OSError as e:
        return [f"could not open: {e}"]

    if "cells" not in nb:
        return ["missing top-level 'cells' field"]
    if "nbformat" not in nb:
        errs.append("missing top-level 'nbformat' field")

    for i, cell in enumerate(nb["cells"]):
        if "cell_type" not in cell:
            errs.append(f"cell {i}: missing 'cell_type'")
            continue
        if cell["cell_type"] != "code":
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not source.strip():
            continue  # empty cells are fine

        try:
            ast.parse(_strip_jupyter_magics(source))
        except SyntaxError as e:
            line = e.lineno or "?"
            errs.append(f"cell {i}: SyntaxError at line {line}: {e.msg}")

    return errs


def install_hook() -> int:
    """Install this script as a pre-push hook for the current repo."""
    here = Path(__file__).resolve().parent
    repo_root = here
    while not (repo_root / ".git").is_dir():
        if repo_root.parent == repo_root:
            print("Could not find .git directory.", file=sys.stderr)
            return 1
        repo_root = repo_root.parent

    hooks_dir = repo_root / ".git" / "hooks"
    hook_path = hooks_dir / "pre-push"

    rel_to_root = here.relative_to(repo_root).as_posix()
    hook_body = f"""#!/usr/bin/env bash
# pre-push: notebook smoke test
# Installed by theory/code/check-notebooks.py --hook
set -e
cd "$(git rev-parse --show-toplevel)"
python {rel_to_root}/check-notebooks.py
"""
    hook_path.write_text(hook_body, encoding="utf-8")
    # chmod +x (best-effort on Windows; git for Windows respects it)
    try:
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass
    print(f"Installed pre-push hook at {hook_path.relative_to(repo_root)}.")
    print("To remove: rm .git/hooks/pre-push")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*",
                        help="notebook filenames or glob; default: all 0?-*.ipynb in this dir")
    parser.add_argument("--hook", action="store_true",
                        help="install this script as a git pre-push hook and exit")
    args = parser.parse_args()

    if args.hook:
        return install_hook()

    here = Path(__file__).parent.resolve()
    if args.notebooks:
        nbs: list[Path] = []
        for n in args.notebooks:
            p = Path(n) if Path(n).is_absolute() else (here / n)
            if "*" in str(p):
                nbs.extend(sorted(here.glob(p.name)))
            else:
                nbs.append(p)
    else:
        nbs = sorted(here.glob("0?-*.ipynb"))
    nbs = [n for n in nbs if n.exists()]

    if not nbs:
        print("No notebooks found.", file=sys.stderr)
        return 1

    failed = 0
    for nb in nbs:
        errs = check_notebook(nb)
        if errs:
            failed += 1
            print(f"FAIL  {nb.name}")
            for e in errs:
                print(f"        {e}")
        else:
            print(f"OK    {nb.name}")

    print()
    if failed:
        print(f"{failed} of {len(nbs)} notebook(s) failed.")
        return 1
    print(f"All {len(nbs)} notebook(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
