# 10 — RAG (Retrieval-Augmented Generation)

Side-quest: build a small RAG system over this repo's own `theory/` docs, so you can ask questions like *"what's the difference between SFT and RLHF?"* and get answers grounded in our own writing.

## What is RAG, and why does it matter?

LLMs have **parametric knowledge** — what they learned during pretraining, frozen in the model weights. That works for general-purpose questions, but breaks down for:

- **Recent information** — anything after the training cutoff
- **Private documents** — your repo, your company's wiki, internal databases
- **Niche domains** — content the model didn't see enough of to memorise
- **Specific facts** — where the model can hallucinate plausibly-wrong details

**Retrieval-Augmented Generation** sidesteps these limits. At query time:

1. **Retrieve** the most relevant documents from a corpus you control
2. **Augment** the LLM's prompt by stuffing those documents in as context
3. **Generate** an answer grounded in the retrieved content

Same model, but now answering with up-to-date, private, domain-specific context.

## Why it's great

- **No retraining.** The LLM stays the same; you update the corpus instead. Fine-tuning a model on new docs costs hours of GPU time. Adding to a vector index takes seconds.
- **Citations.** You can show *which* documents the answer came from, dramatically reducing the trust gap.
- **Cost.** You pay per query, not per training run. RAG over 100k docs costs less than fine-tuning on 1k examples.
- **Privacy.** Your docs never enter the model's weights. They live in your retrieval system, on your infrastructure.
- **Freshness.** A new document is searchable the moment it's indexed.

## Trade-offs

RAG isn't free. The honest list:

- **Latency.** Every query is embed + search + LLM call. ~2–5 seconds, vs. 0.5–2s for a direct LLM call
- **Retrieval quality matters more than the model.** A great LLM with bad chunks gives bad answers
- **Prompt size grows.** Each query carries the retrieved chunks → more input tokens, more cost
- **You only know what you indexed.** Anything not in the corpus, RAG can't answer

## What we'll build

A RAG pipeline over **this very repo's `theory/` docs**. By the end you'll be able to ask questions and get answers grounded in our own writing — with the source chunks shown for inspection.

## The pipeline

End-to-end, on CPU:

```
[Markdown docs]
      │ chunk + embed (one-time)
      ▼
[NumPy vector store]   ← stable index, built once
      │
      │   at query time:
      │
[user query] ──embed──▶ [cosine similarity] ──top-k──▶ [retrieved chunks]
                                                              │
                                                              ▼
                                                  [Claude Sonnet 4.6]
                                                              │
                                                              ▼
                                                          [answer]
```

Four components, each ~10 lines of code:

1. **Embedder** — `sentence-transformers/all-MiniLM-L6-v2` turns text into 384-dim vectors. CPU-friendly, ~80 MB download, runs in seconds
2. **Vector store** — a plain NumPy array. No external DB needed for our small corpus (~150 chunks)
3. **Retriever** — cosine similarity ranking. With normalized embeddings, this is just `embeddings @ query_emb.T`
4. **Generator** — Claude Sonnet 4.6 composes the answer from the retrieved chunks

Real systems swap each piece for something heavier (better embedders, ChromaDB/Qdrant, hybrid search, rerankers) — but the conceptual loop is identical.

## Setup

Two new packages:

```bash
pip install sentence-transformers anthropic
```

And an **Anthropic API key** in the `ANTHROPIC_API_KEY` environment variable. Get one from [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys), then set it *before* launching Jupyter:

```bash
# Git Bash / macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

> **Cost note.** This notebook uses Sonnet 4.6 (`$3 / $15` per million input/output tokens). A typical RAG query here costs ~$0.01. Running every demo cell end-to-end is well under $0.50. Your Claude Code Max subscription does **not** cover API calls — these bill against your separate API credit balance.


```python
import os
from pathlib import Path

import numpy as np
import anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    print(f"API key present (starts with {api_key[:10]}...)")
else:
    print("WARNING: ANTHROPIC_API_KEY not set.")
    print("  Retrieval cells will run; generation cells will fail until you set it.")

print(f"anthropic SDK version: {anthropic.__version__}")
```

## Step 1: build the corpus

We'll use this repo's `theory/` directory — every `.md` file becomes part of the corpus.

**Chunking matters.** Each chunk is what we'll retrieve and pass to the LLM:
- Too small → not enough context per chunk for the LLM to compose a good answer
- Too large → retrieval gets noisy (one chunk covers many topics; relevant signal gets diluted) and you waste tokens

~500–1000 characters works for prose. Real systems use smarter chunking (split on markdown headers, keep paragraphs together, never break inside a code block). We'll start simple: fixed-size chunks with a small overlap so context doesn't get cut at chunk boundaries.


```python
THEORY_DIR = Path("..")  # we're in theory/code/, theory docs are in ../

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into ~chunk_size-char chunks with overlap.

    Tries to break at paragraph boundaries (double newlines) when possible,
    so chunks usually end on a complete idea.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Prefer breaking at the last \n\n in the second half of the candidate
        candidate = text[start:end]
        para_break = candidate.rfind("\n\n")
        if para_break > chunk_size // 2:
            end = start + para_break
        chunks.append(text[start:end].strip())
        start = end - overlap  # back up by overlap chars for context continuity
    return [c for c in chunks if c]

# Read every .md in theory/, but skip code/ to avoid re-indexing notebooks' README
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

print(f"Loaded {len(docs)} chunks from {len(set(d['source'] for d in docs))} files")
print()
print("Sample chunk:")
print(f"  source: {docs[0]['source']}")
print(f"  text:   {docs[0]['text'][:200]}...")
```

## Step 2: embed the chunks

We need to turn each chunk into a vector — a 384-dim point in space — such that *semantically similar chunks land near each other*. "What is the KV cache?" should be near a chunk about KV caching, even if the wording is different.

**`all-MiniLM-L6-v2`** is the workhorse for this:

- 23 M params (tiny by LLM standards)
- ~80 MB on disk
- 384-dim output vectors
- Trained on 1 B+ sentence pairs to put related sentences near each other in vector space
- Runs in seconds on CPU

Not as good as larger models like `all-mpnet-base-v2` (~440 MB) or modern OpenAI/Cohere/Voyage embedding APIs, but for a teaching demo it's perfect — fully local, no API cost.

We use `normalize_embeddings=True` so each vector has unit length, which means cosine similarity reduces to a plain dot product.


```python
from sentence_transformers import SentenceTransformer

print("Loading embedding model (first run downloads ~80MB)...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(f"Embedding {len(docs)} chunks...")
texts = [d["text"] for d in docs]
embeddings = embedder.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Memory:           {embeddings.nbytes / 1024:.1f} KB total")
```

## Step 3: retrieve

Given a query, embed it the same way, find the top-k closest chunks by cosine similarity, return them.

Because we normalised the embeddings, cosine similarity equals the dot product — no extra normalisation needed at query time.


```python
def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """Return the top-k most similar chunks to the query."""
    q_emb = embedder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).squeeze()  # cosine similarity
    top_idx = np.argsort(-sims)[:top_k]
    return [
        {**docs[i], "similarity": float(sims[i])}
        for i in top_idx
    ]

# Sanity check — retrieve for a known-in-corpus query
results = retrieve("What is the KV cache?", top_k=3)
for r in results:
    print(f"\nsim {r['similarity']:.3f}  |  {r['source']}")
    print(f"  {r['text'][:200]}...")
```

## Step 4: generate with Claude

Now we hand the retrieved chunks to **Claude Sonnet 4.6** and ask it to compose an answer from them.

The system prompt is doing real work here. We need to tell the model:

- **Use only the retrieved context** — not its parametric knowledge. This prevents the model from confidently answering from its general training when our docs disagree.
- **Cite sources** — name the file each claim came from, so the user can verify.
- **Say "I don't know"** when the context is insufficient. Hallucination defense #1.

We also enable **prompt caching** on the system prompt with `cache_control: {"type": "ephemeral"}`. The system prompt is the same on every query, so we cache it and pay ~10% of the cost on repeat reads. (Note: Sonnet 4.6 has a 1024-token minimum cacheable prefix — our system prompt is too small to actually hit the cache, but the pattern is the same as for production RAG with longer instructions, and you'll see `cache_creation_input_tokens` / `cache_read_input_tokens` in the response either way.)


```python
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

SYSTEM_PROMPT = """You are a helpful assistant answering questions about LLMs and machine learning, grounded strictly in the CONTEXT provided in the user's message.

Rules:
1. Only answer based on the CONTEXT. Do NOT use parametric knowledge from your pretraining.
2. If the CONTEXT doesn't contain enough information to answer, say "I don't have enough context to answer that fully" and explain what's missing.
3. Cite specific source files in your answer. Format: "According to neural-networks.md, ..." or "(See backpropagation.md, source 2)".
4. Be concise but complete. Use markdown formatting (lists, code blocks) where helpful.
5. If multiple sources contradict each other, note it explicitly."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for the user message."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"--- Source {i}: {c['source']} (similarity: {c['similarity']:.3f}) ---\n"
            f"{c['text']}"
        )
    return "\n\n".join(parts)


def generate(query: str, chunks: list[dict], model: str = "claude-sonnet-4-6"):
    """Send query + retrieved chunks to Claude. Returns (text, usage)."""
    user_message = (
        f"CONTEXT:\n\n{format_context(chunks)}\n\n"
        f"QUESTION: {query}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_message}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return text, response.usage


# First end-to-end call
chunks = retrieve("What is the KV cache?", top_k=4)
answer, usage = generate("What is the KV cache?", chunks)

print(answer)
print()
print(f"Tokens: {usage.input_tokens} input, {usage.output_tokens} output")
if getattr(usage, "cache_read_input_tokens", None):
    print(f"Cache read:     {usage.cache_read_input_tokens} tokens")
if getattr(usage, "cache_creation_input_tokens", None):
    print(f"Cache creation: {usage.cache_creation_input_tokens} tokens")
```

## Step 5: end-to-end

Wrap retrieval + generation into one function that's nice to use interactively.


```python
def ask(query: str, top_k: int = 4, show_chunks: bool = False) -> str:
    """End-to-end RAG: retrieve, generate, print, return the answer."""
    chunks = retrieve(query, top_k=top_k)

    if show_chunks:
        print("=== RETRIEVED CHUNKS ===")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] sim {c['similarity']:.3f}  |  {c['source']}")
            print(f"  {c['text'][:150]}...")
        print("\n=== ANSWER ===\n")

    answer, usage = generate(query, chunks)
    print(answer)
    print(f"\n({usage.input_tokens} in, {usage.output_tokens} out tokens)")
    return answer
```

## Try it out

Each query: see the chunks the retriever picked, then the model's grounded answer.


```python
_ = ask("What's the difference between SFT and RLHF?", show_chunks=True)
```


```python
_ = ask("Why do transformers need positional encodings?")
```


```python
_ = ask("How does the KV cache make inference faster, and what's the cost?")
```


```python
_ = ask("What's the relationship between LoRA and QLoRA?")
```

## RAG vs no RAG

Same question, two paths:

- **With RAG**: retrieve from our corpus → Sonnet → grounded, cited answer
- **Without RAG**: ask Sonnet directly

For broad questions covered by Sonnet's pretraining ("what is gradient descent?"), the no-RAG answer might be just as good. But for **repo-specific** questions, RAG wins clearly because Sonnet has never seen our docs.

Watch this:


```python
def ask_no_rag(query: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    print(text)
    print(f"\n({response.usage.input_tokens} in, {response.usage.output_tokens} out tokens)")
    return text


QUESTION = "What's in the MATH-PRIMER.md file in the agentic.git repo's theory/code/ folder?"

print("=" * 70)
print("WITHOUT RAG (Sonnet has never seen our repo):")
print("=" * 70)
ask_no_rag(QUESTION)

print("\n" + "=" * 70)
print("WITH RAG:")
print("=" * 70)
ask(QUESTION)
```

## What's next

This is RAG at its simplest. Real production systems add:

- **Better chunking**: split on markdown headers, keep code blocks intact, respect semantic boundaries
- **Hybrid search**: combine vector similarity (semantic) with keyword search (BM25). Each catches what the other misses — vector for meaning, keyword for exact technical terms
- **Reranking**: retrieve top-50 cheaply, then rerank to top-5 with a slower-but-better model (e.g. Cohere Rerank, BGE Reranker)
- **Multi-step / agentic RAG**: the LLM decides what to query for next based on what it found. Useful for multi-hop questions ("compare A and B" needs separate retrievals)
- **Real vector DBs**: ChromaDB, Qdrant, FAISS, pgvector, Pinecone, Weaviate. Scale to millions of docs and millisecond search
- **Anthropic's `citations` feature**: structured per-claim citations with character offsets. More precise than the inline source quoting we ask for

For production RAG, [`langchain`](https://github.com/langchain-ai/langchain) and [`llama-index`](https://github.com/run-llama/llama_index) bundle most of this together so you don't have to wire it yourself.

## Exercises

1. **Bigger embedder.** Swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` (~440 MB, more accurate). Re-run the demos. Does retrieval improve on hard or ambiguous queries?
2. **Header-aware chunking.** Replace `chunk_text` with one that splits on `## Heading` boundaries. Does it improve retrieval for queries that map to a specific section?
3. **BM25 hybrid search.** `pip install rank-bm25`, build a keyword index alongside the vector index. For each query, retrieve from both and merge. Does it catch things vector search misses (typos, exact technical terms like `claude-sonnet-4-6`)?
4. **Re-rank top-20 → top-4.** Retrieve more chunks, then have a small model (or even Sonnet itself in a separate call) score them for relevance and pick the best 4 before generating. Does answer quality improve?
5. **Anthropic citations.** Enable the `citations` feature on `messages.create` (use `document` content blocks with `citations: {enabled: True}`). Compare to our manual "Source N" prompting — citations are more reliable.
6. **Cheaper generator.** Try Haiku 4.5 (`claude-haiku-4-5`) instead of Sonnet 4.6. Does answer quality drop noticeably? By how much do costs drop? (~3× for input, ~3× for output.)
7. **Streaming UX.** Switch `messages.create()` for `client.messages.stream()` so the answer appears word-by-word — much better for interactive use, same total cost.
