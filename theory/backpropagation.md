# Backpropagation

The trick that makes training a network with billions of weights actually feasible.

## Intuition

- To train, we need the gradient of loss w.r.t. *every* weight in the network — possibly billions
- Naive approach: nudge each weight a tiny bit, see how loss changes, repeat. Hopelessly slow — billions of forward passes per step
- **Backprop** uses the chain rule of calculus to compute *all* those gradients in one backward pass through the network — costing about as much as a single forward pass
- Key insight: the gradient at layer `N` can be computed from the gradient at layer `N+1` plus local information. So propagate from output back to input

You can think of it as the network running in reverse, but instead of carrying activations, it carries blame: "you contributed this much to being wrong, here's how much your weights should change."

## Mechanics

### Setup

- Forward pass computes the loss and stores intermediate activations
- Backward pass walks layers in reverse, applying the chain rule

### The chain rule, applied

For a chain of operations `x → h₁ = f₁(x) → h₂ = f₂(h₁) → L`:

```
∂L/∂x = ∂L/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂x
```

At each layer we already have `∂L/∂h_{out}` from the layer above; we compute:
- `∂L/∂h_{in}` to pass back further
- `∂L/∂W` to update this layer's weights

### Computational graph view

- Every operation (add, multiply, matmul, ReLU, ...) is a node in a graph
- **Reverse-mode autodiff** walks the graph backward, multiplying local Jacobians
- All modern frameworks (PyTorch, JAX, TensorFlow) implement this automatically — you write the forward pass, gradients are computed for free
- Memory cost: must store activations from the forward pass to use during backward — this dominates training memory

### Activation checkpointing

- Recomputing some activations during the backward pass instead of storing them
- Trades compute for memory; essential for training large LLMs

### Practical issues backprop runs into

- **Vanishing gradients** — in deep nets with sigmoid/tanh, gradient signal shrinks toward the input; early layers don't learn. Mitigations:
  - ReLU / GELU (don't saturate the way sigmoid does)
  - **Residual connections** (He et al. 2015): `y = f(x) + x` — gradient flows directly through the addition. Used in *every* transformer
  - Normalisation (LayerNorm, RMSNorm)
- **Exploding gradients** — gradient magnitude grows without bound through the layers. Mitigations:
  - Gradient clipping (cap norm before optimiser step)
  - Careful weight initialisation (Xavier / He init)
- **Numerical stability** — log of small numbers, divisions by tiny variances. Frameworks handle most of this; mixed-precision (bf16) training needs care

### Why "backprop" is just one specific instance of autodiff

- Reverse-mode autodiff is general; backprop = applying it to a feedforward network with a scalar loss
- Forward-mode autodiff also exists; cheaper when output dim ≫ input dim. Not what you want for neural nets, where loss is a scalar and parameters are huge

## References

- Rumelhart, Hinton & Williams 1986 — *Learning representations by back-propagating errors* (Nature 323) — original paper applying backprop to neural nets
- Karpathy — *Yes you should understand backprop* ([karpathy.medium.com](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b))
- Karpathy — **micrograd** ([github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)) — backprop in ~150 lines of Python, builds full intuition
- 3Blue1Brown — *What is backpropagation really doing?* (YouTube)
- He et al. 2015 — *Deep Residual Learning* ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) — residual connections
- Ba, Kiros & Hinton 2016 — *Layer Normalization* ([arXiv:1607.06450](https://arxiv.org/abs/1607.06450))
