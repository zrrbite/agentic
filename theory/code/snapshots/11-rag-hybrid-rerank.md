# 11 — RAG with hybrid search and reranking

Builds on notebook 10. Same corpus, same generator (Claude Sonnet 4.6) — but a noticeably better retrieval pipeline.

## What was wrong with notebook 10?

Pure vector search is *good at meaning, bad at exact terms*. It happily finds a chunk about "adapting models cheaply" when you ask about "fine-tuning" — but it can struggle with:

- **Exact technical terms** (`claude-sonnet-4-6`, `KV cache`, `nucleus sampling`) — the embedding may not preserve uncommon strings well
- **Code identifiers and acronyms** (`SFT`, `DPO`, `LoRA`, `BM25`) — short tokens with little semantic context
- **Typos and spelling variants** — `transfomer` (typo) gets embedded somewhere weird
- **Rare proper nouns** — anything not in the embedding model's training distribution

Two well-established fixes layer on top of vector search:

1. **Hybrid search**: combine vector similarity with **BM25** keyword search. BM25 is decades old (1994) — it doesn't understand meaning at all, but it's *unbeatable* at exact term matching. Each retriever catches what the other misses.
2. **Reranking**: retrieve a larger candidate set cheaply (e.g. top-20), then use a slower but more accurate model to score each `(query, chunk)` pair and pick the best 4. Catches false positives from both retrievers.

## The new pipeline

```
[user query]
      │
      ├──── vector embed ────▶ cosine sim ──top-20──┐
      │                                              │
      └──── tokenize ────────▶ BM25 score  ──top-20──┤
                                                     │
                                Reciprocal Rank Fusion (top-10)
                                                     │
                                                     ▼
                                  Cross-encoder rerank (top-4)
                                                     │
                                                     ▼
                                                Claude Sonnet 4.6
                                                     │
                                                     ▼
                                                  [answer]
```

Three components new in this notebook:

- **BM25** via `rank_bm25` — sparse keyword retrieval, ~5 lines of code
- **Reciprocal Rank Fusion** — combines vector and BM25 rankings without needing to calibrate scores
- **Cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — ~80 MB, runs on CPU; jointly encodes (query, chunk) for a sharper relevance score than separate-encoder cosine similarity

## Setup

Same as notebook 10 plus one new package:

```bash
pip install rank-bm25
```

If you ran notebook 10, you already have `sentence-transformers`, `anthropic`, and the embedder cached locally.


```python
import os
import re
from pathlib import Path

import numpy as np
import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("WARNING: ANTHROPIC_API_KEY not set; generation cells will fail.")
print(f"anthropic SDK: {anthropic.__version__}")
```

## Build the corpus (same as notebook 10)

Self-contained — re-builds chunks and vector embeddings from scratch so this notebook runs without notebook 10.


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
```


```python
# Vector index — same model as notebook 10
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(
    [d["text"] for d in docs],
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"vector index: {embeddings.shape}")
```

## BM25 keyword index

BM25 (Best Matching 25, Robertson & Walker 1994) is the standard sparse retrieval algorithm. It scores a (query, document) pair using:

- **Term frequency** in the document — more matches = higher score, but with diminishing returns (a doc with 50 matches isn't 50× as relevant as one with 1)
- **Inverse document frequency** — common words like "the" count for nothing; rare words like `claude-sonnet-4-6` count for a lot
- **Document length** — penalises very long docs (which would otherwise win by accident)

It needs **tokenized text** to work — a list of words per document. We'll use a simple lowercase + word-character tokenizer; production systems would use a proper tokenizer with stemming (`running` → `run`) and stopword removal.


```python
def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric word tokenization. Good enough for English prose + code identifiers."""
    return re.findall(r"\w+", text.lower())

tokenized_docs = [tokenize(d["text"]) for d in docs]
bm25 = BM25Okapi(tokenized_docs)
print(f"BM25 index built over {len(tokenized_docs)} documents")
print(f"sample tokens: {tokenized_docs[0][:15]}")
```

## Vector vs BM25: where they disagree

Same query, two retrievers. Look at the top result from each — they often differ, and *that's the point* (we want both signals).


```python
def retrieve_vector(query: str, top_k: int = 20):
    q = embedder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q.T).squeeze()
    return [int(i) for i in np.argsort(-sims)[:top_k]]

def retrieve_bm25(query: str, top_k: int = 20):
    scores = bm25.get_scores(tokenize(query))
    return [int(i) for i in np.argsort(-scores)[:top_k]]

# Where they agree, where they disagree
for query in [
    "What is RLHF?",                               # acronym — BM25 should shine
    "How do models learn from human feedback?",   # paraphrase — vector should shine
    "claude-sonnet-4-6",                           # exact identifier — BM25 huge advantage
]:
    v_top = retrieve_vector(query, top_k=3)
    b_top = retrieve_bm25(query, top_k=3)
    print(f"\n>>> {query!r}")
    print(f"  vector top-3: {[docs[i]['source'] for i in v_top]}")
    print(f"  BM25   top-3: {[docs[i]['source'] for i in b_top]}")
    overlap = len(set(v_top) & set(b_top))
    print(f"  overlap: {overlap}/3")
```

## Reciprocal Rank Fusion

Now we have two ranked lists. How to merge them?

**Bad idea**: linearly combine the scores (`α · vector_score + (1-α) · bm25_score`). Vector cosine similarity is in [-1, 1]; BM25 scores can go to 30+. They're not comparable, so you'd need to calibrate per-corpus, and it's brittle.

**Good idea**: **Reciprocal Rank Fusion** (Cormack et al. 2009). Each retriever votes for documents based on *rank*, not score:

```
rrf_score(d) = sum over retrievers r of  1 / (k + rank_r(d))
```

where `rank_r(d)` is the document's rank in retriever `r`'s output (1 = top), and `k=60` is a smoothing constant (the standard).

Why it works:

- **Score-free** — only ranks matter, so no calibration needed
- **Top-heavy** — rank 1 contributes `1/61 ≈ 0.0164`; rank 100 contributes `1/160 ≈ 0.0062`. The top of each list dominates
- **Documents in *both* lists win** — if a doc is rank 3 from vector and rank 5 from BM25, its score combines

This is the standard fusion for hybrid search; you'll see it in Elasticsearch, Vespa, Weaviate, and most production retrievers.


```python
def rrf_fuse(*ranked_lists, k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Each input is a list of doc indices in rank order."""
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

def retrieve_hybrid(query: str, candidate_k: int = 20, top_k: int = 10):
    """Vector + BM25 fused with RRF; return top-k."""
    v = retrieve_vector(query, top_k=candidate_k)
    b = retrieve_bm25(query, top_k=candidate_k)
    fused = rrf_fuse(v, b)[:top_k]
    return [{**docs[i], "rrf_score": s} for i, s in fused]

results = retrieve_hybrid("What is RLHF?", top_k=5)
for r in results:
    print(f"rrf {r['rrf_score']:.4f}  |  {r['source']}")
```

## Reranking with a cross-encoder

RRF gives us 10 candidates. Now we ask a more expensive model to *re-score* each `(query, chunk)` pair and pick the best 4.

### Why a separate model?

Our embedder is a **bi-encoder** — it encodes query and document independently into a fixed vector each, then compares with cosine similarity. That's fast (you embed all docs once) but loses information: the query and doc never "see" each other while being encoded.

A **cross-encoder** takes `(query, chunk)` *together* into the model, producing a single relevance score. It's much more accurate — but slow, because there's no reusable doc embedding (every query × doc pair is a fresh forward pass).

The standard pattern: bi-encoder for the cheap top-N, cross-encoder for the slow top-K. Best of both.

### The model

**`cross-encoder/ms-marco-MiniLM-L-6-v2`** — trained on the MS MARCO passage-ranking dataset, ~80 MB on disk, runs on CPU. There are bigger and better cross-encoders (Cohere Rerank API, BGE Reranker, Voyage Rerank), but this one is free and good enough for our scale.


```python
print("Loading cross-encoder reranker (first run downloads ~80MB)...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[dict], top_k: int = 4):
    """Score each (query, chunk) pair with the cross-encoder; return top-k."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_k]
    return [{**c, "rerank_score": float(s)} for c, s in ranked]

# Compare: hybrid top-4 vs hybrid top-10 reranked to top-4
query = "How does the KV cache make inference faster?"
hybrid_top4 = retrieve_hybrid(query, top_k=4)
reranked_top4 = rerank(query, retrieve_hybrid(query, top_k=10))

print("=== HYBRID TOP-4 (no rerank) ===")
for r in hybrid_top4:
    print(f"  rrf {r['rrf_score']:.4f}  |  {r['source']}  |  {r['text'][:80]!r}")
print("\n=== HYBRID TOP-10 -> RERANKED TO TOP-4 ===")
for r in reranked_top4:
    print(f"  rerank {r['rerank_score']:+.3f}  |  {r['source']}  |  {r['text'][:80]!r}")
```

## Putting it all together

Same `generate()` as notebook 10 (Sonnet 4.6, prompt cache on system prompt, asks the model to cite sources). The retrieve step is now hybrid + rerank.


```python
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a helpful assistant answering questions about LLMs and machine learning, grounded strictly in the CONTEXT provided in the user's message.

Rules:
1. Only answer based on the CONTEXT. Do NOT use parametric knowledge from your pretraining.
2. If the CONTEXT doesn't contain enough information to answer, say "I don't have enough context to answer that fully" and explain what's missing.
3. Cite specific source files in your answer. Format: "According to neural-networks.md, ..." or "(See backpropagation.md, source 2)".
4. Be concise but complete. Use markdown formatting (lists, code blocks) where helpful.
5. If multiple sources contradict each other, note it explicitly."""


def format_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        score_field = c.get("rerank_score", c.get("rrf_score", c.get("similarity", 0)))
        parts.append(
            f"--- Source {i}: {c['source']} (relevance: {score_field:.3f}) ---\n{c['text']}"
        )
    return "\n\n".join(parts)


def generate(query, chunks, model="claude-sonnet-4-6"):
    user_message = f"CONTEXT:\n\n{format_context(chunks)}\n\nQUESTION: {query}"
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


def ask_v2(query: str, candidate_k: int = 20, hybrid_k: int = 10, final_k: int = 4):
    """End-to-end RAG: hybrid retrieve -> RRF -> rerank -> generate."""
    candidates = retrieve_hybrid(query, candidate_k=candidate_k, top_k=hybrid_k)
    final = rerank(query, candidates, top_k=final_k)
    answer, usage = generate(query, final)
    print(answer)
    print(f"\n({usage.input_tokens} in, {usage.output_tokens} out tokens)")
    return answer
```

## Demo on tricky queries

We'll pick queries chosen to stress different retrieval modes:


```python
_ = ask_v2("What's the difference between greedy decoding and nucleus sampling?")
```


```python
_ = ask_v2("How does QLoRA reduce memory compared to full fine-tuning?")
```


```python
_ = ask_v2("What's the role of the constant k=60 in reciprocal rank fusion, and where does it come from?")
```

## Vector-only vs hybrid+rerank: head-to-head

Same query, both pipelines, top-4 chunks each. Watch which sources they pull and whether the answers differ in specificity / accuracy.


```python
QUERY = "What is the chunk_size parameter in our chunking function and why was it chosen?"

def vector_only(query, top_k=4):
    ids = retrieve_vector(query, top_k=top_k)
    sims = (embeddings @ embedder.encode([query], normalize_embeddings=True).T).squeeze()
    return [{**docs[i], "similarity": float(sims[i])} for i in ids]

print("=== VECTOR-ONLY (notebook 10 style) ===")
vec_chunks = vector_only(QUERY)
for c in vec_chunks:
    print(f"  sim {c['similarity']:.3f}  |  {c['source']}")

print("\n=== HYBRID + RERANK (notebook 11) ===")
h_chunks = rerank(QUERY, retrieve_hybrid(QUERY, top_k=10), top_k=4)
for c in h_chunks:
    print(f"  rerank {c['rerank_score']:+.3f}  |  {c['source']}")

print("\n=== ANSWER (vector-only) ===")
ans_v, _ = generate(QUERY, vec_chunks)
print(ans_v)

print("\n=== ANSWER (hybrid + rerank) ===")
ans_h, _ = generate(QUERY, h_chunks)
print(ans_h)
```

## Notes on real-world RAG

**Reranking is one of the highest-leverage upgrades in RAG.** On benchmarks, adding a cross-encoder reranker typically improves answer accuracy more than upgrading the embedder, more than improving chunking, and often more than upgrading the LLM. The reason is that retrieval failures cascade: if the right chunk isn't in the top-k, the LLM literally cannot answer correctly.

**Hybrid search is cheaper than it looks.** BM25 is `O(N)` per query but the constant is tiny — the indexing is in-memory and the per-query cost is in microseconds for our corpus. Even at 1M docs, it's still cheap.

**This isn't the frontier.** Newer commercial rerankers (Cohere Rerank v3, Voyage Rerank-2, BGE Reranker v2) are dramatically better than `ms-marco-MiniLM-L-6-v2`. If your retrieval is the bottleneck and you have budget, swap in a paid reranker — same code shape, paste the API call into `rerank()`.

**The next jump: agentic / iterative retrieval.** Have the LLM look at the first retrieval, decide if it has enough, and issue *follow-up* queries ("now look up X to compare with Y"). This is the architecture behind tools like Anthropic's RAG-via-tool-use and the [LangGraph](https://langchain-ai.github.io/langgraph/) agentic retriever. We don't build this here, but it's the obvious notebook 12.

## Exercises

1. **Stemming + stopwords.** Replace `tokenize` with one that removes English stopwords (a, the, of, ...) and stems (`installing` → `install`). NLTK or `sklearn.feature_extraction.text` work. Does BM25 quality improve on noisy queries?
2. **Calibrated linear fusion.** Implement linear-score fusion as an alternative to RRF: normalize each score to [0, 1] then combine with `α=0.5`. Find a query where it works better than RRF, and one where it works worse. Why?
3. **Bigger embedder, no rerank.** Swap to `BAAI/bge-large-en-v1.5` (~1.3 GB) and skip the reranker. Compare quality vs the smaller embedder + reranker combo. Which wins per dollar?
4. **LLM-as-reranker.** Replace the cross-encoder with a Sonnet/Haiku call that scores each (query, chunk) pair on a 1–10 scale. More expensive, sometimes more accurate. When does it pay off?
5. **Hybrid weights.** RRF treats vector and BM25 equally. Weight them — say 70% vector, 30% BM25 — by raising RRF contributions. For technical-term-heavy queries, increase the BM25 weight. Build a small classifier or heuristic that picks weights per query.
6. **Re-rank with citations.** Anthropic's `citations: {enabled: True}` returns *which sentences* of the retrieved doc the answer came from. Wire this in — better grounding, less hand-waving.
