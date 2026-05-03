# 13 — RAG via OpenRouter free tier ($0, hosted)

Notebook 12 was *truly* free but slow (CPU inference, ~15-30s per query). This notebook is *also* free but fast — by using a hosted, free-tier model via [**OpenRouter**](https://openrouter.ai).

## What is OpenRouter?

OpenRouter is a unified API gateway in front of ~300 LLMs across providers (OpenAI, Anthropic, Google, Meta, Mistral, Cohere, ...). One API key, one base URL, one SDK shape — and you can swap models with one string change.

Crucially for our purposes: **some models are free**. Specifically, several open-weights models (Llama 3, Gemma 2, Mistral 7B, Phi-3, Qwen) are hosted by OpenRouter at no cost, with reasonable rate limits.

The trade-off vs notebook 12 (local Ollama):

| | Local Ollama (notebook 12) | OpenRouter free (this) |
|---|---|---|
| Cost | $0 | $0 |
| Latency | 15-30s on CPU | 2-5s |
| Setup | Install Ollama, pull a 2GB model | Sign up, get a free API key |
| Privacy | Fully local | Sent to OpenRouter & the model provider |
| Offline | Yes | No |
| Rate limits | None | Yes — free tier limits per model |

If privacy matters → notebook 12. If speed matters and you're OK sending queries to a third party → this one.

## What you'll see learning-wise

**This notebook uses the `openai` Python SDK, but talks to OpenRouter instead of OpenAI.** That's possible because OpenAI's API became a *de facto standard* — most providers (OpenRouter, Together, Groq, Anyscale, Fireworks, vLLM, llama.cpp's server, even Ollama) speak it. So the same code shape works against any of them. Pick your provider by the `base_url` and `api_key`.

## Setup

**1. Get a free OpenRouter API key.** [openrouter.ai/keys](https://openrouter.ai/keys) — sign up (Google/email), create a key. Set it as an environment variable before launching Jupyter:

```bash
# Git Bash / macOS / Linux
export OPENROUTER_API_KEY="sk-or-v1-..."

# PowerShell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

**2. Install the OpenAI SDK** (yes, even though we're not using OpenAI):

```bash
pip install openai
```

Plus `sentence-transformers` (already installed if you ran notebook 10 or 12).

**3. Free models to try.** OpenRouter tags free models with `:free` in the model ID. As of writing, popular free options include:

| Model ID | Size | Notes |
|---|---|---|
| `google/gemma-2-9b-it:free` | 9 B | Strong general-purpose, our default |
| `meta-llama/llama-3.1-8b-instruct:free` | 8 B | Comparable, sometimes better at code |
| `meta-llama/llama-3.2-3b-instruct:free` | 3 B | Smaller, faster |
| `mistralai/mistral-7b-instruct:free` | 7 B | Older but reliable |

Browse the live list at [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free). Free models can have rate limits or temporary unavailability — if one fails, swap in another.


```python
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

MODEL = "google/gemma-2-9b-it:free"  # swap if rate-limited or unavailable

if not os.environ.get("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY not set; generation cells will fail.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "missing"),
)
print(f"openai SDK pointed at OpenRouter. Default model: {MODEL!r}")
```

## Build the corpus and vector index (same as notebook 10)


```python
THEORY_DIR = Path("..")

def chunk_text(text, chunk_size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        para_break = text[start:end].rfind("\n\n")
        if para_break > chunk_size // 2:
            end = start + para_break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]

docs = []
for md_path in sorted(THEORY_DIR.rglob("*.md")):
    if "code" in md_path.parts:
        continue
    text = md_path.read_text(encoding="utf-8")
    for i, chunk in enumerate(chunk_text(text)):
        docs.append({
            "source": str(md_path.relative_to(THEORY_DIR)),
            "chunk_idx": i,
            "text": chunk,
        })

print(f"{len(docs)} chunks from {len(set(d['source'] for d in docs))} files")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(
    [d["text"] for d in docs],
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"vector index: {embeddings.shape}")

def retrieve(query: str, top_k: int = 4) -> list[dict]:
    q = embedder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q.T).squeeze()
    return [{**docs[i], "similarity": float(sims[i])} for i in np.argsort(-sims)[:top_k]]
```

## Generate via OpenRouter

This is the only thing that's different from notebook 10. Note the call shape: `client.chat.completions.create(...)` — pure OpenAI SDK, just pointed at a different host. Streaming works the same way the OpenAI SDK does (yields chunks with `.choices[0].delta.content`).


```python
SYSTEM_PROMPT = """You are a helpful assistant answering questions about LLMs and machine learning, grounded strictly in the CONTEXT provided in the user's message.

Rules:
1. Only answer based on the CONTEXT. Do NOT use parametric knowledge from your pretraining.
2. If the CONTEXT doesn't contain enough information, say "I don't have enough context to answer that fully" and explain what's missing.
3. Cite specific source files in your answer (e.g. "According to neural-networks.md, ...").
4. Be concise but complete. Use markdown formatting where helpful."""


def format_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"--- Source {i}: {c['source']} (similarity: {c['similarity']:.3f}) ---\n{c['text']}"
        )
    return "\n\n".join(parts)


def generate(query: str, chunks: list[dict], stream: bool = True) -> str:
    user_message = f"CONTEXT:\n\n{format_context(chunks)}\n\nQUESTION: {query}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    if not stream:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=1024,
        )
        return resp.choices[0].message.content

    parts = []
    for chunk in client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=1024, stream=True,
    ):
        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
        parts.append(delta)
        print(delta, end="", flush=True)
    print()
    return "".join(parts)


def ask(query: str, top_k: int = 4, show_chunks: bool = False) -> str:
    chunks = retrieve(query, top_k=top_k)
    if show_chunks:
        print("=== RETRIEVED CHUNKS ===")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] sim {c['similarity']:.3f}  |  {c['source']}")
            print(f"  {c['text'][:150]}...")
        print("\n=== ANSWER ===\n")
    t0 = time.perf_counter()
    answer = generate(query, chunks)
    print(f"\n(generation took {time.perf_counter() - t0:.1f}s)")
    return answer
```

## Try it


```python
_ = ask("What's the difference between SFT and RLHF?", show_chunks=True)
```


```python
_ = ask("How does the KV cache make inference faster?")
```


```python
_ = ask("Why do transformers need positional encodings?")
```

## Free-tier rate limits

OpenRouter's free models are rate-limited. Typical limits at the time of writing:

- 20 requests per minute
- 50-200 requests per day per model
- Limits vary per model and change over time — check the [pricing page](https://openrouter.ai/models?q=free) for current values

If you hit a limit, you'll get a `429`. Two recovery paths:

1. **Switch model** — change `MODEL` to a different `:free` ID. Free models have separate quotas, so switching gives you fresh budget
2. **Wait** — minute and daily limits roll over; `Retry-After` header tells you how long

For production-style use, you'd add retry-with-backoff around the call. We skip it here to keep the demo simple, but exercise 4 below has a sketch.

## The $0 trade space, summarised

| Path | Cost | Latency | Setup | Privacy |
|---|---|---|---|---|
| **Local Ollama** (notebook 12) | $0 | 15-30s on CPU | Install + pull model (~2 GB) | Fully local |
| **OpenRouter free** (this) | $0 | 2-5s | Sign up + API key | Sent to OR + provider |
| **HF Inference free tier** | $0 | 2-5s, more variance | HF account + token | Sent to HF + provider |
| **Together / Groq paid trials** | $1-25 free credit | 0.5-2s (Groq is fast) | Sign up | Sent to provider |

Pick by privacy needs first, then latency tolerance. Local-first if any of the corpus is sensitive; cloud-first if pure performance matters.

## Why this code shape is durable

Three notebooks now (10/12/13), three providers (Anthropic, Ollama, OpenRouter), and the **retrieval half of the code is identical**. Only the generator differs.

Real RAG systems exploit this — they wrap the generator behind an interface so they can:

- A/B test providers without rewriting the rest
- Fall back to a cheaper provider when the primary rate-limits
- Use different models for different query difficulties (a small free model for FAQs, a paid model for hard reasoning)

Frameworks like [LiteLLM](https://github.com/BerriAI/litellm) take this even further — they expose every provider behind the OpenAI SDK shape, so you write code once and route at runtime.

## Exercises

1. **Side-by-side compare.** Run the same query through notebooks 10 (Sonnet), 12 (local Ollama), and 13 (this). How do answers differ in citation accuracy, conciseness, hallucination rate?
2. **Provider router.** Wrap `generate()` so it tries OpenRouter first, falls back to Ollama on rate-limit. Best of both — fast when possible, never blocked.
3. **Cheap-paid hybrid.** Use OpenRouter free for the *retrieval-rerank* step (e.g., LLM-as-reranker) and Anthropic paid only for the *final generation*. Lowest total cost, almost no latency penalty
4. **Retry with backoff.** Wrap `generate()` with retry logic that handles `openai.RateLimitError`. On 429, sleep `Retry-After` seconds and try again. Bonus: rotate to a different free model after N retries
5. **Bigger free model.** Try `meta-llama/llama-3.1-70b-instruct:free` if available. Slower per query (it's 70 B params) but quality is closer to GPT-4. Worth it for harder questions?
6. **Through Groq for speed.** Groq's free tier is *fast* (200+ tok/sec on smaller models). Sign up, get a Groq key, point `client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=...)`. Same code, milliseconds-fast generation
