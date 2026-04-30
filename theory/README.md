# LLM Theory — how the models work and how they're trained

Notes on the math and mechanisms behind LLMs: neural networks, gradient descent, transformers, training pipelines.

**Status**: living document, rough notes > polish.

## How each file is structured

Each topic file has two sections so you can stop where you want:

- **Intuition** — plain language, no math required. Read this first.
- **Mechanics** — equations, details, references to papers. Skip if you only need the gist.

## Code companions

Each topic has (or will have) a runnable notebook or script under [`code/`](code/) that builds the math from scratch. Start with [`code/NOTEBOOKS.md`](code/NOTEBOOKS.md) if you've never used Jupyter.

## Suggested reading order

1. [`neural-networks.md`](neural-networks.md) — what a network *is* before worrying about LLMs
2. [`gradient-descent.md`](gradient-descent.md) — how a network learns
3. [`backpropagation.md`](backpropagation.md) — how gradients flow through the network
4. [`tokenization.md`](tokenization.md) — turning text into numbers
5. [`transformers.md`](transformers.md) — the architecture behind GPT/Claude
6. [`llm-training.md`](llm-training.md) — pretraining, SFT, RLHF/DPO
7. [`inference.md`](inference.md) — what happens when you call a model

## Conventions

- Same as the parent repo: bullets > prose, sources required, mark "unverified" otherwise
- Equations in inline pseudo-LaTeX: `L = -Σ y log(ŷ)`
- Canonical references at the bottom of each file (arXiv IDs where applicable)

## If you want one-link starting points

- **Math behind an LLM, walked through end-to-end (video)** — [youtube.com/watch?v=xmkSf5IS-zw](https://www.youtube.com/watch?v=xmkSf5IS-zw) — recommended overview tying the pieces in this folder together
- **Visual intuition (free)** — 3Blue1Brown's *Neural Networks* playlist on YouTube
- **Hands-on code (free)** — Andrej Karpathy's *Neural Networks: Zero to Hero* playlist + nanoGPT
- **Reference textbook (free)** — Goodfellow, Bengio, Courville, *Deep Learning* (deeplearningbook.org)
