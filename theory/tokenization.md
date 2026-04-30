# Tokenization

How text becomes numbers the model can process. Often-skipped, surprisingly important.

## Intuition

- Models operate on numbers, not text. Tokenization splits text into chunks (tokens) and maps each to an integer ID
- Tokens are usually **subword pieces**, not whole words: `"unbelievable"` → `["un", "believ", "able"]`
- Vocabulary size typically 30k–200k tokens
- Trade-off: smaller vocab → longer sequences and more compute per sentence; larger vocab → bigger embedding matrix and rarer tokens trained less
- Why this matters in practice:
  - Token count determines API cost and what fits in a context window
  - Weird tokenisation causes weird behaviour: digits split inconsistently, rare scripts use one byte per token, "SolidGoldMagikarp" anomalies
  - The model literally cannot see characters, only tokens — that's why "how many R's in strawberry" was historically hard

Rule of thumb for English: ~4 characters per token, ~0.75 words per token (varies by tokeniser).

## Mechanics

### Byte-Pair Encoding (BPE)

The dominant approach. Algorithm:
1. Start with the alphabet of single bytes/characters as your vocab
2. Count all adjacent symbol pairs in the training corpus
3. Merge the most frequent pair into a new token; add to vocab
4. Repeat until target vocab size is reached

- Originally a compression algorithm (Gage 1994); brought to NLP by Sennrich, Haddow & Birch 2016
- Result: common words become single tokens, rare words split into pieces, anything weird falls back to bytes

### Variants

| Tokeniser | Used by | Notes |
|---|---|---|
| BPE (char-level) | early NMT systems | Falls back to "unknown" on unseen characters |
| **Byte-level BPE** | GPT-2, GPT-3, GPT-4 | Operates on raw UTF-8 bytes — no UNK token, handles any Unicode |
| **WordPiece** | BERT | Like BPE but merges based on likelihood, not pure frequency |
| **SentencePiece** | T5, LLaMA, many multilingual models | Treats input as a raw byte stream; trains BPE or unigram LM directly without language-specific pre-tokenisation |
| Unigram LM | SentencePiece option | Probabilistic model over subwords; prunes the least-useful tokens |

### Specific tokenisers worth knowing

- **tiktoken** (OpenAI, [github.com/openai/tiktoken](https://github.com/openai/tiktoken)) — fast Rust BPE; `cl100k_base` (GPT-3.5/4), `o200k_base` (GPT-4o)
- **Anthropic** uses its own tokeniser; not publicly published
- **LLaMA** uses SentencePiece BPE with 32k vocab (LLaMA-1/2) or 128k (LLaMA-3)

### Embedding lookup

After tokenisation, each token ID indexes into an **embedding matrix** of shape `[vocab_size, d_model]`:
```
x_embed = E[token_id]
```
- This matrix is the model's first learnable layer
- Often the same matrix is reused at the output to project hidden states back to logits over the vocab — "weight tying"

### Special tokens

Every modern LLM reserves IDs for control tokens:
- Beginning of text / end of text
- Padding
- Chat role markers (e.g. `<|im_start|>user`, `<|im_end|>`) for instruction-tuned models
- Tool-use markers in some models

These are how chat templates encode "this is system, this is user, this is assistant" — the model just sees a token stream.

### Why tokenisation matters for *coding* agents

- Code is tokenised differently from prose; whitespace and indentation eat tokens
- Small models can be very sensitive to a tool-call format that splits into many tiny tokens
- Long-context performance degrades faster than parameter scaling suggests, partly because of tokeniser choice on niche material

## References

- Sennrich, Haddow & Birch 2016 — *Neural Machine Translation of Rare Words with Subword Units* (BPE for NLP) ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909))
- Kudo & Richardson 2018 — *SentencePiece* ([arXiv:1808.06226](https://arxiv.org/abs/1808.06226))
- Karpathy — *Let's build the GPT Tokenizer* (YouTube) and **minbpe** ([github.com/karpathy/minbpe](https://github.com/karpathy/minbpe))
- OpenAI — `tiktoken` ([github.com/openai/tiktoken](https://github.com/openai/tiktoken))
- Anthropic — token counting via `count_tokens` API endpoint (no public tokeniser)
