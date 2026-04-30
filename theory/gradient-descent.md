# Gradient Descent

How a network *learns*: the algorithm that turns "this output was wrong" into "adjust these weights".

## Intuition

- We have a loss (how wrong we are right now) that depends on the network's weights
- The **gradient** points uphill in loss-space — partial derivatives of loss w.r.t. each weight
- Take a small step in the *opposite* direction → loss goes down a tiny bit
- Repeat millions or billions of times

Hill-walking analogy: blindfolded on a hillside, you feel the slope under your feet and step downhill, and again, and again. With billions of dimensions instead of two, but the principle is identical.

The two knobs that matter most:
- **Learning rate** = step size. Too big → overshoot and explode. Too small → never get there
- **Batch size** = how many examples you average over before each step. Bigger = smoother gradient but slower wall-clock progress per example

## Mechanics

### Update rule

```
w ← w - η · ∇L(w)
```
- `η` is the learning rate
- `∇L(w)` is the gradient of loss w.r.t. weights — computed by backprop

### Variants by batch size

- **Batch GD**: gradient over the whole dataset. Smooth but expensive — impossible for LLMs
- **Stochastic GD (SGD)**: gradient from a single example. Noisy but cheap
- **Mini-batch SGD**: a small batch (32 to ~4M tokens for LLM pretraining). Standard practice

### Modern optimisers

Plain SGD is rarely used directly for LLMs. Two ideas dominate:

1. **Momentum** — keep a running average of past gradients, step in that direction. Smooths out noise:
   ```
   v ← β v + ∇L
   w ← w - η v
   ```

2. **Adaptive per-parameter learning rates** — frequently-updated weights get smaller steps; rarely-updated ones get larger steps

These combine in **Adam** (Kingma & Ba 2014):
- Tracks moving average of gradient (`m`) and squared gradient (`v`)
- Per-parameter step: `w ← w - η · m̂ / (√v̂ + ε)`
- Bias-corrected by training step

**AdamW** (Loshchilov & Hutter 2019) decouples weight decay from the gradient update — used for GPT, LLaMA, Claude (unverified for Claude specifically, but standard in the field).

### Learning rate schedules

LLM training is essentially impossible without LR scheduling.
- **Warmup**: ramp from 0 to peak LR over the first ~1% of training. Avoids early-step instability
- **Decay**: cosine, linear, or inverse-sqrt to a small fraction of peak. Lets the model fine-tune at the end
- Peak LR is often `~3e-4` to `~1e-3` for transformer pretraining, but varies with model size

### Failure modes

- **LR too high** → loss spikes / NaN. Mitigation: clip gradients, lower LR, more warmup
- **LR too low** → painfully slow or stuck
- **Saddle points** → in high dimensions, true local minima are rare; saddles are the real concern (Dauphin et al. 2014). Momentum + noise help escape
- **Vanishing / exploding gradients** → see [`backpropagation.md`](backpropagation.md)

### Gradient clipping

- Cap the gradient norm (e.g. clip to 1.0) before the optimiser step
- Almost universal in transformer training; prevents single bad batches from wrecking the model

## References

- Ruder — "An overview of gradient descent optimization algorithms" ([ruder.io/optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/))
- Kingma & Ba 2014 — *Adam: A Method for Stochastic Optimization* ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980))
- Loshchilov & Hutter 2019 — *Decoupled Weight Decay Regularization* (AdamW) ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101))
- Dauphin et al. 2014 — *Identifying and attacking the saddle point problem* ([arXiv:1406.2572](https://arxiv.org/abs/1406.2572))
- Smith 2018 — *A disciplined approach to neural network hyper-parameters* ([arXiv:1803.09820](https://arxiv.org/abs/1803.09820)) — practical LR-finding
