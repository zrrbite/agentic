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

If the math feels rusty (derivatives, chain rule, matrix calculus, cross-entropy) → see [`MATH-PRIMER.md`](MATH-PRIMER.md) for curated free references.

## Layout

| File | Companion to | What you'll build |
|---|---|---|
| [`01-mlp-from-scratch.ipynb`](01-mlp-from-scratch.ipynb) | [`../neural-networks.md`](../neural-networks.md) | A small MLP forward pass in NumPy, on a "two moons" toy dataset |
| [`02-gradient-descent.ipynb`](02-gradient-descent.ipynb) | [`../gradient-descent.md`](../gradient-descent.md) | Train the MLP with numerical gradients (slow but clear) |
| [`03-backprop.ipynb`](03-backprop.ipynb) | [`../backpropagation.md`](../backpropagation.md) | Hand-derive backprop for the MLP, gradient-check against numerical, retrain ~100× faster |
| [`04-bpe-from-scratch.ipynb`](04-bpe-from-scratch.ipynb) | [`../tokenization.md`](../tokenization.md) | A working byte-pair encoder, train + encode/decode, compare to `tiktoken` |
| [`05-attention-by-hand.ipynb`](05-attention-by-hand.ipynb) | [`../transformers.md`](../transformers.md) | Self-attention + multi-head + causal mask in NumPy; visualise attention weights; show why we need positional encodings |
| [`06-mini-gpt.ipynb`](06-mini-gpt.ipynb) | [`../transformers.md`](../transformers.md) | Tiny GPT (~800K params, 4 layers) in PyTorch, trained on Tiny Shakespeare. Generate samples |
| [`07-training-loop.ipynb`](07-training-loop.ipynb) | [`../llm-training.md`](../llm-training.md) | Production-style training loop: LR schedule (warmup + cosine), gradient clipping, val tracking, checkpointing |
| [`08-sft-and-dpo.ipynb`](08-sft-and-dpo.ipynb) | [`../llm-training.md`](../llm-training.md) | **Colab GPU only.** Fine-tune SmolLM-360M with QLoRA + SFT to answer in haiku |
| [`09-sampling-and-kvcache.ipynb`](09-sampling-and-kvcache.ipynb) | [`../inference.md`](../inference.md) | Sampling strategies (greedy, temp, top-k, top-p) + KV cache from scratch + benchmark |
| [`10-rag.ipynb`](10-rag.ipynb) | (side quest) | RAG over this repo's own theory docs: embed → retrieve → generate. **Needs `ANTHROPIC_API_KEY`** |
| [`11-rag-hybrid-rerank.ipynb`](11-rag-hybrid-rerank.ipynb) | (side quest) | Upgrade 10's pipeline: BM25 + vector via reciprocal rank fusion + cross-encoder reranker |

## Setup

Python 3.10+ recommended. From this `theory/code/` directory:

```bash
# Create a virtualenv (one-time)
python -m venv .venv
```

Then **activate it** — the command depends on your shell:

| Shell | Activate command |
|---|---|
| **Git Bash** (Windows) | `source .venv/Scripts/activate` |
| **PowerShell** (Windows) | `.\.venv\Scripts\Activate.ps1` |
| **CMD** (Windows) | `.venv\Scripts\activate.bat` |
| **bash / zsh** (macOS / Linux) | `source .venv/bin/activate` |

You'll know it worked when your prompt gains a `(.venv)` prefix and `which python` (or `where python` on Windows) points inside `.venv`.

Then install dependencies **with the venv active**:

```bash
pip install numpy matplotlib jupyterlab    # chapters 1-5
pip install torch                          # chapter 6+
pip install transformers datasets          # chapter 7+
```

> **If you forget to activate** and run `pip install` against the system Python, packages land in user-site and `jupyter lab` won't be on PATH. You'll see PATH warnings during install. Fix: activate the venv and re-run `pip install`.

> **Microsoft Store Python on Windows** is sandboxed and causes friction with venvs and PATH. If you hit weird issues, install Python from [python.org](https://www.python.org/downloads/) instead — it's much smoother.

## Running a notebook

```bash
jupyter lab        # opens in browser at http://localhost:8888
```

Navigate to a `.ipynb` file and double-click to open. Or open the `.ipynb` file directly in **VS Code** or **Cursor** — both have built-in notebook support (install the Jupyter extension when prompted).

## Reading the notebooks without running them

Pre-rendered markdown copies (with plots inline) live in [`snapshots/`](snapshots/) — see the [snapshot index](snapshots/README.md) for the full reading order. They render natively on github.com; handy for reading on a phone or sharing without the recipient running anything.

Re-generate them locally with:

```bash
# from theory/code/, with the venv active
python render-snapshots.py                # all notebooks
python render-snapshots.py 02-*.ipynb     # one notebook
python render-snapshots.py --no-execute   # convert without re-running
```

GitHub Actions re-renders snapshots on every push to `main` that touches a notebook ([`.github/workflows/render-snapshots.yml`](../../.github/workflows/render-snapshots.yml)).

For a fast (sub-second) smoke test before pushing — JSON valid, Python syntax in code cells parses — without running the notebooks:

```bash
python check-notebooks.py            # all notebooks
python check-notebooks.py --hook     # install as a git pre-push hook (opt-in)
```

## Conventions for these notebooks

- Each notebook starts with a markdown cell linking back to its companion theory doc
- Every code cell should run top-to-bottom in order — *Restart and Run All* is the sanity check
- Plots over print statements when feasible — visual feedback beats numbers
- Each notebook ends with **Exercises** (small variations to try yourself)
- No external datasets — every notebook is self-contained, runnable offline
