# LLM Training

How a transformer goes from random weights to ChatGPT/Claude.

## Intuition

Training a frontier LLM is three stages, each shaping the model differently:

1. **Pretraining** — feed it the internet, predict the next token. Builds raw knowledge and language ability. Months of compute on thousands of GPUs
2. **Supervised fine-tuning (SFT)** — show it tens of thousands of curated `(instruction, good response)` pairs. Teaches it to follow instructions
3. **Alignment** (RLHF or DPO) — use human preferences to shape outputs toward *helpful, harmless, honest*

Each stage is dramatically cheaper than the last but matters disproportionately for what end users feel. A pretrained-only model is impressive but unwieldy; SFT makes it usable; alignment makes it pleasant and safer.

## Mechanics

### 1. Pretraining

- **Objective**: causal language modeling — predict the next token given everything before it
- **Loss**: cross-entropy over the next-token distribution
- **Data**: trillions of tokens — Common Crawl, books, code, Wikipedia, papers, etc., aggressively filtered and deduplicated
- **Compute rule of thumb**: `~6 · N · D` FLOPs for a model with `N` parameters trained on `D` tokens (Kaplan et al. 2020)

#### Scaling laws

- Loss is a smooth power-law function of params, data, and compute (Kaplan et al. 2020)
- **Chinchilla** (Hoffmann et al. 2022): for a fixed compute budget, optimal ratio is roughly **20 tokens per parameter**
- GPT-3 (175B params, 300B tokens) was significantly undertrained by this rule
- LLaMA-3 (8B params, ~15T tokens) is dramatically *over*trained relative to Chinchilla — but inference cost matters too, so it's worth training small models on more data

#### Distributed training

- **Data parallelism** — each GPU processes a different batch shard, gradients are all-reduced
- **Tensor parallelism** — split a single layer's matmuls across GPUs
- **Pipeline parallelism** — split layers across GPUs, micro-batches flow through
- **ZeRO / FSDP** — shard optimiser state and gradients across GPUs; key to fitting large models in GPU memory

### 2. Supervised fine-tuning (SFT) — instruction tuning

- Same loss as pretraining (next-token prediction), now on `(prompt, ideal-response)` pairs
- Datasets:
  - Human-written (expensive but high-quality)
  - Model-generated then curated (Self-Instruct, Alpaca-style)
  - Distilled from a stronger model
- Loss is typically masked to only count tokens in the *response*, not the prompt
- Turns a "text completer" into an "instruction follower"

### 3. Alignment

#### RLHF — Reinforcement Learning from Human Feedback

Three sub-steps (Ouyang et al. 2022, *InstructGPT*):

1. **Collect preferences**: show humans two responses A and B for the same prompt; they pick the better one
2. **Train a reward model**: a separate model (often initialised from the SFT model) that takes a `(prompt, response)` and outputs a scalar score predicting human preference
3. **RL fine-tune the LLM** to maximise reward, with a KL penalty pulling it toward the SFT model so it doesn't drift into reward-hacking nonsense
   - Algorithm: **PPO** — Proximal Policy Optimization (Schulman et al. 2017)

Origin: Christiano et al. 2017 (preference-based RL); applied to LLMs at scale by InstructGPT.

#### Constitutional AI / RLAIF (Anthropic)

- Bai et al. 2022 — replace much of the human feedback with feedback from a model guided by a written **constitution** (a list of principles)
- Reduces dependence on human raters, more controllable safety properties
- Anthropic's models are trained with this approach

#### DPO — Direct Preference Optimization

- Rafailov et al. 2023 — derive a closed-form loss directly from RLHF's objective, **skipping the reward model and PPO entirely**
- Train the policy directly on preference pairs with what is essentially a clever cross-entropy loss
- Simpler, more stable, comparable quality. Increasingly common in open models

#### Other techniques worth knowing

- **Rejection sampling fine-tuning** — sample N completions, keep the best (judged by reward model), SFT on those. Used in LLaMA-2's training pipeline
- **KTO** (Ethayarajh et al. 2024) — alignment from binary good/bad signals, no preference pairs needed
- **Iterative refinement** — alternate SFT and preference rounds, each round trained on the previous best model's outputs

### Long-context training

- After main pretraining, continue training on longer sequences with adjusted positional encodings (e.g. RoPE base scaling, "RoPE θ" tuning)
- Critical for the 100k+ context windows in modern models — naive extrapolation degrades fast

### Model evaluation during training

- Held-out perplexity (next-token prediction loss on unseen text)
- Benchmarks: MMLU, GSM8K, HumanEval, MT-Bench, Arena ELO, etc.
- Internal A/B preferences vs the previous version

## References

- Kaplan et al. 2020 — *Scaling Laws for Neural Language Models* ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361))
- Hoffmann et al. 2022 — *Training Compute-Optimal LLMs* (Chinchilla) ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556))
- Ouyang et al. 2022 — *Training language models to follow instructions* (InstructGPT) ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155))
- Bai et al. 2022 — *Constitutional AI: Harmlessness from AI Feedback* ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073))
- Rafailov et al. 2023 — *Direct Preference Optimization* ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290))
- Schulman et al. 2017 — *PPO* ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
- Christiano et al. 2017 — *Deep RL from Human Preferences* ([arXiv:1706.03741](https://arxiv.org/abs/1706.03741))
- Karpathy — *State of GPT* (Microsoft Build 2023, YouTube) — best end-to-end overview of the pipeline
- Hugging Face — *RLHF: From zero to ChatGPT* ([huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf))
