# Jupyter notebooks — a quick guide

Notebooks are interactive Python documents made of **cells**. Each cell is either code or markdown. You run cells one at a time and the output (numbers, plots, errors) shows up directly below.

This guide assumes you've never used one. It's enough to follow the notebooks in this repo. For full Jupyter docs see [jupyter.org](https://jupyter.org).

## Why they exist

For exploration and teaching: write a paragraph of explanation, then a few lines of code that produce a plot, then more explanation. You can re-run any cell, change a parameter, see the new plot — without restarting the program.

The downside: cells can be run out of order, and state from one cell leaks into the next. This makes notebooks great for *learning*, less great for production code (which is why the later chapters in this repo are plain `.py` files).

## Installing

You need Python 3.10+ and `pip`. Then:

```bash
pip install jupyterlab numpy matplotlib
```

To launch:

```bash
jupyter lab
```

This opens a browser tab at `http://localhost:8888`. Navigate to a `.ipynb` file and double-click to open.

### Alternative: VS Code or Cursor

Both have first-class notebook support. Open the `.ipynb` file — VS Code/Cursor will prompt you to install the Jupyter extension if missing. You'll also need a Python interpreter selected (top-right of the notebook).

This is what most people use day-to-day; you don't really need `jupyter lab` if you have a good editor.

## Notebook anatomy

- A notebook is a vertical list of **cells**
- Two cell types: **code** (Python) and **markdown** (text/headings/links)
- Cells have an optional **output** area below them, populated when you run the cell

## Keyboard shortcuts (the only ones you need)

There are two modes: **edit mode** (cursor inside a cell, like a text editor) and **command mode** (cell selected, no cursor).

`Esc` enters command mode. `Enter` enters edit mode.

| In any mode | What it does |
|---|---|
| `Shift+Enter` | Run the current cell, move to the next |
| `Ctrl+Enter` | Run the current cell, stay |
| `Alt+Enter` | Run the current cell, insert a new cell below |

| Command mode (`Esc` first) | What it does |
|---|---|
| `B` | New cell **B**elow |
| `A` | New cell **A**bove |
| `D D` | **D**elete cell (D twice) |
| `M` | Switch cell to **M**arkdown |
| `Y` | Switch cell to code (p**Y**thon) |
| `Z` | Undo cell deletion |
| `↑` / `↓` | Move between cells |

## The kernel

A "kernel" is the Python process behind the notebook. It holds all your variables — once you `import numpy as np` in cell 1, every subsequent cell can use `np`.

Kernel commands you'll use (from the top menu or right-click):
- **Restart kernel** — clears all variables; fresh start. Use when something feels broken
- **Restart and run all** — the best sanity check. If your notebook only works when cells are run in some weird order, this catches it. *Run this before sharing a notebook*
- **Interrupt kernel** — stops a running cell (the equivalent of Ctrl+C)

## The classic gotchas

1. **Out-of-order execution.** You define `x = 5` in cell 3, then change cell 1 to print `x`. The print works because `x` is already in memory — but a colleague opening the notebook fresh and running top-to-bottom will hit a `NameError`. *Solution*: periodically Restart and Run All.

2. **Stale variables.** You delete a cell that defined `helper`, but `helper` is still in the kernel's memory until you restart. Code that uses it appears to work, but is broken on a fresh run.

3. **Long-running cells block the notebook.** A training loop will lock everything up until done. Print progress, or use `tqdm` for a progress bar.

4. **Plots don't show up.** With matplotlib in modern Jupyter this is rare, but if it happens, add `%matplotlib inline` at the top of the notebook.

5. **Restarting loses all your variables.** This is the *point* — but it can sting if you've spent an hour computing something. Save the data to a file (`np.save`, `pickle`) if it took real time to compute.

## Useful magic commands (start with `%`)

| Command | What it does |
|---|---|
| `%time <code>` | Time a single line |
| `%%time` | Time the whole cell (must be first line of cell) |
| `%matplotlib inline` | Show matplotlib plots inline (usually default now) |
| `!pip install <pkg>` | Run a shell command from inside a cell (note the `!`) |
| `?some_function` | Show docs for `some_function` |
| `??some_function` | Show source code |

## Workflow we recommend for this repo

1. Open the notebook in your editor (VS Code, Cursor, JupyterLab)
2. Read the markdown cell at the top — it links to the theory doc that explains the math
3. `Shift+Enter` your way down, reading and watching outputs
4. Pause to read longer markdown cells
5. When you finish, **Restart and Run All** to confirm everything works top-to-bottom
6. Try the **Exercises** at the bottom by editing cells in place

## When notebooks are the *wrong* tool

For training scripts, services, libraries, or anything that runs unattended → use a `.py` file. Notebooks lose:
- Reliable top-to-bottom reproducibility
- Clean git diffs (notebook JSON diffs are nasty without tools like [nbdime](https://nbdime.readthedocs.io/))
- Easy testing
- Production deployment

The repo switches to plain scripts at chapter 7 for exactly these reasons.

## Notebooks and git (briefly)

The `.ipynb` format is JSON containing both source *and* outputs. This means committing a notebook with executed cells creates a giant diff next time you re-run.

Two common practices:
- **Clear outputs before committing**: *Edit → Clear All Outputs* before saving. Smaller diffs, but you lose the rendered plots in GitHub
- **Keep outputs**: useful so the GitHub preview shows the full output. Diffs become noisy

For this repo: clear outputs before committing, since the notebooks are designed to be re-run and aren't long-running.
