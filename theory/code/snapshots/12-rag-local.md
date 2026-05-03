# 12 — RAG, fully local (Ollama, $0)

Same pipeline as notebook 10 — chunk, embed, retrieve, generate — but the generator is a **local** LLM running on your CPU via Ollama. No API keys, no network calls (after the one-time model download), no per-query cost.

## Why local?

- **$0 marginal cost.** Send a million queries, pay nothing
- **Privacy.** Your retrieved docs never leave your machine. Useful when the corpus contains anything sensitive (internal wiki, code, customer data, private notes)
- **Offline.** Works on a plane, behind a corporate firewall, anywhere with no internet
- **No vendor lock-in.** Same code shape works against any provider with a similar SDK

## What you give up vs Anthropic / OpenAI / etc.

- **Speed.** ~15-30 seconds per RAG query on a modern CPU, vs ~2-3s for a hosted API. CPU inference is just slow. With a real GPU it'd be ~1s
- **Top-tier quality.** A 3B-param local model is closer to GPT-3.5 than to Claude Sonnet 4.6. For grounded RAG (where the model just composes from given chunks) this is *usually fine*; for hard reasoning or nuance it shows
- **Operational responsibility.** You manage the model, the disk space, the upgrades. Ollama makes this easy but it's still on you

For a teaching repo, learning project, or anything privacy-sensitive — **local wins**. For a production user-facing product where latency and quality are the bottleneck — pay for the API.

## The pipeline

Identical to notebook 10 except the last step:

```
[user query] ──embed──▶ [cosine similarity] ──top-k──▶ [retrieved chunks]
                                                              │
                                                              ▼
                                                  [Ollama (qwen2.5:3b)]   ← was Claude Sonnet 4.6
                                                              │
                                                              ▼
                                                          [answer]
```

## One-time setup

**1. Install Ollama** — desktop app + background server in one. Cross-platform installer:

- Windows / macOS / Linux: download from [ollama.com/download](https://ollama.com/download)
- Or via a package manager: `brew install ollama` (macOS), `winget install Ollama.Ollama` (Windows)

After install, Ollama runs as a background service on `http://localhost:11434`. It starts automatically on login.

**2. Pull a model** — pick one based on your RAM and patience:

| Model | Size | Speed (CPU) | Quality |
|---|---|---|---|
| `qwen2.5:3b` | ~2 GB | fastest | good for RAG |
| `llama3.2:3b` | ~2 GB | similar | similar |
| `qwen2.5:7b` | ~4.5 GB | ~2-3× slower | noticeably better |
| `llama3.1:8b` | ~5 GB | ~2-3× slower | noticeably better |

For this notebook, `qwen2.5:3b` is the recommended starting point — small enough that even modest hardware handles it. Open a terminal:

```bash
ollama pull qwen2.5:3b
```

First run downloads ~2 GB. Subsequent runs use the cached model.

**3. Install the Python client:**

```bash
pip install ollama
```

Plus the same `sentence-transformers` you used in notebook 10 (already installed if you ran that one).


```python
import re
import time
from pathlib import Path

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

MODEL = "qwen2.5:3b"

# Sanity check: is Ollama running and is the model pulled?
try:
    available = [m["model"] for m in ollama.list()["models"]]
    if MODEL not in available:
        print(f"WARNING: {MODEL!r} not found locally. Pull it first:")
        print(f"  ollama pull {MODEL}")
        print(f"\nAvailable models: {available}")
    else:
        print(f"OK — Ollama is running and {MODEL!r} is ready.")
except Exception as e:
    print(f"Could not reach Ollama on http://localhost:11434.")
    print(f"Make sure the Ollama app/service is running. Error: {e}")
```

## Build the corpus (same as notebook 10)

Self-contained — re-builds chunks and vector embeddings so this notebook runs without notebook 10.


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
```

## Retrieve (same as notebook 10)


```python
def retrieve(query: str, top_k: int = 4) -> list[dict]:
    q = embedder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q.T).squeeze()
    return [{**docs[i], "similarity": float(sims[i])} for i in np.argsort(-sims)[:top_k]]
```

## Generate with Ollama

The Python client is essentially identical to OpenAI/Anthropic — `chat()` takes a list of `{role, content}` messages, returns a response with the model's reply. The model is `qwen2.5:3b` running locally on your CPU.

**Streaming is on by default below** — for a 3B model on CPU you really want streaming. Without it, you'd stare at a frozen cell for 30 seconds before any output appears.


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


def generate_local(query: str, chunks: list[dict], stream: bool = True) -> str:
    user_message = f"CONTEXT:\n\n{format_context(chunks)}\n\nQUESTION: {query}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    if not stream:
        response = ollama.chat(model=MODEL, messages=messages)
        return response["message"]["content"]

    # Stream — print as we go, accumulate, return at the end
    parts = []
    for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
        delta = chunk["message"]["content"]
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
    answer = generate_local(query, chunks)
    print(f"\n(generation took {time.perf_counter() - t0:.1f}s)")
    return answer
```

## Try it

First query has the cold-start cost: Ollama loads the model into memory (~5-10s for a 3B model). Subsequent queries reuse the loaded model and are faster.


```python
_ = ask("What's the difference between SFT and RLHF?", show_chunks=True)
```


```python
_ = ask("How does the KV cache make inference faster?")
```


```python
_ = ask("Why do transformers need positional encodings?")
```

## Quality and speed: what to expect

Compared to Sonnet 4.6 (notebook 10):

| | Local (qwen2.5:3b on CPU) | Anthropic (Sonnet 4.6) |
|---|---|---|
| Cost per query | $0 | ~$0.01 |
| Latency | 15-30s typical (CPU) | 2-4s |
| Answer quality | Good for grounded RAG | Excellent |
| Citation accuracy | Mostly correct | Very reliable |
| Hallucination on missing context | Sometimes invents | Usually says "I don't know" |
| Privacy | Fully local | Sent to Anthropic |
| Offline | Yes | No |

**Where local underperforms most:** subtle multi-step reasoning over chunks, or queries where context is incomplete and the model needs to gracefully decline. A bigger local model (`qwen2.5:7b` or `llama3.1:8b`) closes most of this gap, at the cost of 2-3× slower inference and 2-3× the RAM.

**Where local is genuinely fine:** grounded factual Q&A from given chunks. The model just needs to compose, not invent. Which is most of RAG.

## Where this fits in the trade space

Three tiers, all running the same RAG code on top:

| Tier | Generator | Cost | Use when |
|---|---|---|---|
| **Local** (this notebook) | Ollama / llama.cpp | $0 | Privacy-critical; learning; offline; high-volume internal use |
| **Cheap hosted** | Haiku 4.5, Gemini Flash, GPT-4-mini, Together.ai, Groq | ~$0.001-0.01 / query | Most user-facing apps; quality matters more than $0 |
| **Top-tier hosted** | Sonnet 4.6, Opus 4.7, GPT-4 | ~$0.01-0.10 / query | Hard reasoning; nuanced output; willingness to pay for the gap |

Pick by the actual constraint, not by default.

## Exercises

1. **Bigger local model.** `ollama pull qwen2.5:7b`, change `MODEL` above, re-run. Slower (~2-3×) but noticeably better. Worth it?
2. **Side-by-side Anthropic vs local.** If you have an `ANTHROPIC_API_KEY`, paste in the `generate()` from notebook 10 and add a cell that runs both on the same query. How visible is the quality difference?
3. **OpenRouter free tier.** Some models are free via [OpenRouter](https://openrouter.ai/models?q=free) (e.g., `google/gemma-2-9b-it:free`). Pip-install `openai`, point its `base_url` at OpenRouter, swap in `generate()`. Lower latency than local CPU; still $0
4. **Hugging Face Inference API free tier.** Sign up for an HF account, get a free token. Free tier is rate-limited but works for low-volume RAG. Same SDK story as OpenRouter
5. **Latency optimisation.** Time each stage of the RAG pipeline (embed, retrieve, generate). On local, generation dominates. What % of time is each stage?
6. **Hybrid pipeline: cheap local rerank + paid generator.** Use a local cross-encoder to rerank (notebook 11), then call Sonnet only for the final generation. Best of both — almost no cost, almost no latency penalty
