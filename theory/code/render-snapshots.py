#!/usr/bin/env python3
"""Render executed notebook snapshots to theory/code/snapshots/.

By default re-runs each notebook fresh and writes the result as markdown
(plus a folder of plot images per notebook). Run from the repo root or
from theory/code/, with the venv activated so `jupyter nbconvert` is
on PATH.

Notebooks listed in SKIP_EXECUTE are converted as-is (their existing
cell outputs are kept). This is for notebooks that need a GPU or other
resources we don't have in CI / local — e.g. 08 needs a Colab GPU.

Usage:
    python render-snapshots.py                    # all 0?-*.ipynb
    python render-snapshots.py 01-*.ipynb         # one or more by name
    python render-snapshots.py --no-execute ...   # convert without re-running
"""

# Notebooks that should be converted but NOT executed (e.g. need a GPU).
# These will be rendered with whatever cell outputs are already in the file.
SKIP_EXECUTE = {
    "08-sft-and-dpo.ipynb",   # needs Colab GPU; can't run on CPU or in CI
    "10-rag.ipynb",           # needs ANTHROPIC_API_KEY + sentence-transformers; CI has neither
}
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*",
                        help="notebook filenames or glob; default: all 0?-*.ipynb")
    parser.add_argument("--no-execute", action="store_true",
                        help="skip re-execution; convert with existing cell outputs")
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-cell execution timeout in seconds (default 600)")
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()
    out_dir = here / "snapshots"
    out_dir.mkdir(exist_ok=True)

    # Resolve which notebooks to render
    if args.notebooks:
        nbs: list[Path] = []
        for n in args.notebooks:
            p = Path(n)
            if not p.is_absolute():
                p = here / p
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

    print(f"Rendering {len(nbs)} notebook(s) -> {out_dir}\n")

    failed: list[tuple[str, str]] = []
    for nb in nbs:
        print(f"=== {nb.name} ===", flush=True)
        # Invoke jupyter via `python -m` so we don't depend on `jupyter.exe`
        # being on PATH (it isn't, on Windows, unless the venv is activated).
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "markdown",
            "--output-dir", str(out_dir),
            f"--ExecutePreprocessor.timeout={args.timeout}",
        ]
        if not args.no_execute and nb.name not in SKIP_EXECUTE:
            cmd.append("--execute")
        elif nb.name in SKIP_EXECUTE:
            print(f"  (skip-execute: {nb.name} listed in SKIP_EXECUTE)")
        cmd.append(str(nb))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            failed.append((nb.name, result.stderr.strip().splitlines()[-1] if result.stderr else "exit non-zero"))
            print(f"  FAILED: see stderr below")
            sys.stdout.write(result.stderr)
        else:
            print("  OK")

    print()
    if failed:
        print(f"{len(failed)} of {len(nbs)} notebook(s) failed:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        return 1

    print(f"All {len(nbs)} notebook(s) rendered successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
