# Inference

What actually happens when you call a trained model. Where most user-facing knobs live (temperature, top-p, ...).

## Intuition

- A trained LLM is a function: token sequence → probability distribution over the next token
- "Inference" = repeatedly sample a next token and append it, until a stop condition (EOS token, length cap, stop sequence)
- Each generated token requires a full forward pass through every layer — but smart caching avoids re-doing work for tokens already processed

The user-facing knobs (temperature, top-p, etc.) all control the sampling step at the very end. The infrastructure (KV cache, batching, quantisation) controls how fast / cheap that loop runs.

## Mechanics

### Sampling strategies

Given the model's logits over the vocab, how do we pick the next token?

| Strategy | What it does | When to use |
|---|---|---|
| **Greedy** | Pick argmax | Deterministic; can be repetitive |
| **Temperature `T`** | Divide logits by T before softmax. T<1 sharpens, T>1 flattens, T=0 = greedy | The most common knob |
| **Top-k** | Keep top k tokens, renormalise, sample | Easy bound on randomness |
| **Top-p (nucleus)** | Smallest set of tokens whose cumulative prob ≥ p; sample from those (Holtzman et al. 2019) | Adapts to distribution shape; default in many APIs |
| **Min-p** | Threshold relative to top probability | Newer; arguably better than top-p |
| **Repetition / frequency / presence penalties** | Down-weight tokens that have already appeared | Prevents loops |

These are usually combined: e.g. `temperature=0.7, top_p=0.95, top_k=40`.

For deterministic behaviour: `temperature=0` (greedy). Note that with batching/floating-point non-determinism, even `temperature=0` is not bit-exact reproducible across hardware.

### KV cache

The single most important inference optimisation.

- During autoregressive generation, the K and V vectors for *past* tokens never change — only the new token's K/V get appended
- Without cache: every new token re-runs the whole sequence through attention → `O(n²)` per token, `O(n³)` total
- With cache: `O(n)` per token, `O(n²)` total

KV cache memory cost (per request):
```
2 · n_layers · n_heads · head_dim · seq_len · batch · bytes_per_value
```

This is *huge* for long contexts and is often the bottleneck:
- Mitigations: **Multi-Query Attention** (Shazeer 2019) and **Grouped-Query Attention** (Ainslie et al. 2023) share K/V across heads — used in LLaMA-2/3, modern open models
- **PagedAttention** (vLLM) manages KV cache memory like virtual memory pages, hugely reducing fragmentation

### Speculative decoding

(Leviathan et al. 2023; Chen et al. 2023)
- A small **draft model** proposes K tokens cheaply
- The big **target model** verifies all K in one parallel forward pass
- An accept-reject scheme guarantees the output distribution is *identical* to vanilla sampling from the target
- 2–3× speed-up typical, free quality-wise

### Quantisation

Trade tiny quality loss for big speed/memory wins. Weights stored at lower precision than they were trained at:
- **FP8 / int8** — common for serving; minimal quality loss
- **int4** (GPTQ, AWQ, GGUF/llama.cpp, bitsandbytes) — common for local inference
- KV cache can also be quantised (separately from weights)

### Batching for throughput

- **Static batching** — assemble a fixed-size batch, run together, wait for the slowest. Wasteful (sequences finish at different times)
- **Continuous / in-flight batching** (Yu et al. 2022, Orca; popularised by vLLM) — swap finished sequences out, slot new ones in mid-flight. Major throughput win for serving
- Almost universal in production LLM serving today

### Prefill vs decode

LLM inference has two distinct phases with very different performance characteristics:
- **Prefill** — process the entire prompt in one shot. Highly parallel, compute-bound. Fast even for long prompts
- **Decode** — generate tokens one at a time. Sequential, memory-bandwidth-bound. The slow part for long outputs

This is why "long input, short output" is much faster per token than "short input, long output" — and why prompt caching (Anthropic, OpenAI) gives such large speed-ups: it skips prefill entirely.

### Stop conditions

- EOS token (model emits an end-of-text token)
- Max tokens cap
- Stop sequences (caller provides strings; generation halts when matched)
- Tool-call boundaries in agent loops

## References

- Holtzman et al. 2019 — *The Curious Case of Neural Text Degeneration* (nucleus sampling) ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751))
- Shazeer 2019 — *Fast Transformer Decoding* (multi-query attention) ([arXiv:1911.02150](https://arxiv.org/abs/1911.02150))
- Ainslie et al. 2023 — *GQA: Grouped-Query Attention* ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245))
- Leviathan, Kalman & Matias 2023 — *Fast Inference from Transformers via Speculative Decoding* ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192))
- Chen et al. 2023 — *Accelerating Large Language Model Decoding with Speculative Sampling* ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318))
- Kwon et al. 2023 — *vLLM / PagedAttention* ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180))
- Yu et al. 2022 — *Orca: A Distributed Serving System* (continuous batching, OSDI'22)
- llama.cpp — [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — reference implementation for quantised local inference
