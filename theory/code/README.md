# theory/code — code companions to the theory docs

Each chapter in `../` has a matching notebook or script here. Goal: make the math concrete by *building* it, not just reading it.

**Status**: living, added one chapter at a time.

## Why notebooks for early chapters, scripts for later

| Chapters 1–6 | Chapters 7+ |
|---|---|
| `.ipynb` notebooks | plain `.py` scripts |
| Math + plot + run cell, edit, re-run | Long-running training / inference loops |
| Best for *learning* | Best for *running* and version-controlling |

If you've never used a Jupyter notebook → **read [`NOTEBOOKS.md`](NOTEBOOKS.md) first**.

## Layout

| File | Companion to | What you'll build |
|---|---|---|
| [`01-mlp-from-scratch.ipynb`](01-mlp-from-scratch.ipynb) | [`../neural-networks.md`](../neural-networks.md) | A small MLP forward pass in NumPy, on a "two moons" toy dataset |
| `02-gradient-descent.ipynb` *(planned)* | [`../gradient-descent.md`](../gradient-descent.md) | Train the MLP with numerical gradients (slow but clear) |
| `03-micrograd-style.ipynb` *(planned)* | [`../backpropagation.md`](../backpropagation.md) | A tiny autograd engine (~150 lines), then the same MLP retrained orders of magnitude faster |
| `04-bpe-from-scratch.ipynb` *(planned)* | [`../tokenization.md`](../tokenization.md) | A working byte-pair encoder, tokenise some text |
| `05-attention-by-hand.ipynb` *(planned)* | [`../transformers.md`](../transformers.md) | One self-attention layer in NumPy; visualise attention weights |
| `06-mini-gpt.ipynb` *(planned)* | [`../transformers.md`](../transformers.md) | A tiny GPT in PyTorch (à la nanoGPT), trained on a small text |
| `07-training-loop.py` *(planned)* | [`../llm-training.md`](../llm-training.md) | Pretraining loop with logging, LR schedule, gradient clipping |
| `08-sft-and-dpo.py` *(planned)* | [`../llm-training.md`](../llm-training.md) | SFT then DPO on a toy preference set |
| `09-sampling-and-kvcache.py` *(planned)* | [`../inference.md`](../inference.md) | Generation with KV cache and sampling knobs |

## Setup

Python 3.10+ recommended. From the repo root:

```bash
# Create a virtualenv in this folder (one-time)
python -m venv .venv

# Activate it
.venv\Scripts\activate         # Windows (PowerShell or Git Bash)
# source .venv/bin/activate    # macOS / Linux

# Install dependencies for chapters 1-5
pip install numpy matplotlib jupyterlab

# Add for chapter 6+
pip install torch

# Add for chapter 7+
pip install transformers datasets
```

## Running a notebook

```bash
jupyter lab        # opens in browser at http://localhost:8888
```

Navigate to a `.ipynb` file and double-click to open. Or open the `.ipynb` file directly in **VS Code** or **Cursor** — both have built-in notebook support (install the Jupyter extension when prompted).

## Conventions for these notebooks

- Each notebook starts with a markdown cell linking back to its companion theory doc
- Every code cell should run top-to-bottom in order — *Restart and Run All* is the sanity check
- Plots over print statements when feasible — visual feedback beats numbers
- Each notebook ends with **Exercises** (small variations to try yourself)
- No external datasets — every notebook is self-contained, runnable offline
