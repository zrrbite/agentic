# Transformers

The architecture every frontier LLM is built on. The 2017 paper title says it: *Attention Is All You Need*.

## Intuition

- Before transformers, language models used RNNs/LSTMs that read text left-to-right one token at a time, struggling with long-range dependencies and slow to train (no parallelism over the sequence)
- Transformer key idea: **attention** lets every token look at every other token directly, in parallel
- Per layer, each token (1) gathers information from other tokens via attention, (2) processes that information through a feedforward network
- Stack many layers (frontier LLMs: ~60–120) → deep contextual representations
- Training is fully parallel across the sequence (a huge wall-clock win over RNNs)

The mental model: imagine each word in a sentence sending out a "what am I looking for?" query and "what do I have to offer?" key. Words match up by query-key compatibility, and each word builds its new representation by pulling weighted information from the words it best matches with.

## Mechanics

### Self-attention

For each token, project its representation into three vectors:
- **Q** (query) — what this token is looking for
- **K** (key) — what this token offers
- **V** (value) — the actual content to share

Attention output for one head:
```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V
```
- `Q Kᵀ` → matrix of similarity scores between every pair of tokens
- divide by `√d_k` to keep the softmax well-scaled (otherwise variance grows with dim)
- softmax → weights summing to 1 per row
- multiply by `V` → each token gets a weighted sum of others' content

### Multi-head attention

- Run attention `h` times in parallel with different Q/K/V projections
- Each head can learn to focus on different patterns (syntax, coreference, position, ...)
- Concatenate heads, project back to `d_model`

### Causal (masked) attention

- For autoregressive LMs (GPT-style), token `t` must only attend to positions `≤ t`, never the future
- Achieved by setting future positions in `Q Kᵀ` to `-∞` before softmax
- Encoder transformers (BERT) use bidirectional attention; decoder-only LLMs use causal

### Feedforward / MLP block

- Per-position 2-layer MLP, typically expanding to `4 × d_model` then back
- This is where most of the parameters live
- Modern variants use **SwiGLU** or **GeGLU** gating instead of plain MLP

### A full transformer block

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```
- Pre-norm (LN before each sublayer) is the modern default; original paper used post-norm
- Residual connections are critical — they let gradients flow through dozens of layers

### Positional information

Self-attention is permutation-invariant by itself; positions must be injected:
- **Sinusoidal** (original transformer) — fixed, not learned
- **Learned absolute** (GPT-2) — one embedding per position, capped at training length
- **RoPE — Rotary Position Embeddings** (Su et al. 2021) — rotates Q/K vectors by position-dependent angles. Used in LLaMA, Mistral, most modern LLMs. Extends to longer contexts more gracefully
- **ALiBi** (Press et al. 2022) — additive bias to attention scores by relative distance

### Quadratic cost and what's done about it

- Attention is `O(n²)` in sequence length — the dominant cost at long contexts
- Mitigations:
  - **FlashAttention** (Dao et al. 2022) — same math, IO-aware kernel, much faster in practice
  - Sliding-window / local attention (Mistral)
  - Sparse / linear attention variants
  - State-space models like **Mamba** (Gu & Dao 2023) — alternative architecture, sub-quadratic

### Mixture of Experts (MoE)

- Replace the dense FFN with `N` experts, route each token to the top-k (usually 1–2)
- Only those experts run per token → much more capacity per FLOP
- Used in Mixtral, DeepSeek-V3, Qwen MoE; rumoured for GPT-4 and others (unverified)

### Decoder-only is the norm

- Original transformer was encoder-decoder (for translation)
- Modern LLMs are **decoder-only**: just a stack of causal-attention transformer blocks predicting the next token
- BERT-style encoder-only models still dominate retrieval and classification

## References

- Vaswani et al. 2017 — *Attention Is All You Need* ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
- Karpathy — *Let's build GPT* (YouTube) and **nanoGPT** ([github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT))
- Alammar — *The Illustrated Transformer* ([jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/))
- Su et al. 2021 — *RoFormer: Rotary Position Embedding* ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864))
- Press, Smith & Lewis 2022 — *ALiBi* ([arXiv:2108.12409](https://arxiv.org/abs/2108.12409))
- Dao et al. 2022 — *FlashAttention* ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
- Gu & Dao 2023 — *Mamba* ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))
- 3Blue1Brown — *Attention in transformers, visually explained* (YouTube)
