# Neural Networks

The substrate of every modern LLM. Skip this only if you already know what an MLP is.

## Intuition

- A neural network is a function: input numbers → output numbers, with billions of tunable knobs in between
- Built from "neurons": each takes weighted inputs, adds a bias, applies a non-linear "activation" function
- Layers stack: input → one or more hidden layers → output
- Training = adjusting the knobs (weights) until the function gives the right answers on examples we have, and *generalises* to ones we don't
- "Deep learning" just means many hidden layers (typically more than 2)
- Why non-linear activations? Without them, stacking layers collapses to a single linear function — no extra power

A useful mental picture: imagine a giant mixing board with billions of dials. You play it an example, observe how wrong the output is, and twiddle every dial slightly in the direction that would have made the output less wrong. Repeat trillions of times.

## Mechanics

### Single neuron

```
y = σ(w · x + b)
```
- `x` is the input vector, `w` the weight vector, `b` the bias scalar
- `σ` is the activation function (non-linear)

### Layer in matrix form

```
h = σ(W x + b)
```
- `W` is the layer's weight matrix `[output_dim, input_dim]`
- A multi-layer perceptron (MLP) is just this composed: `h2 = σ(W2 σ(W1 x + b1) + b2)`

### Common activations

| Name | Formula | Notes |
|---|---|---|
| ReLU | `max(0, x)` | Default in modern nets; cheap; "dying ReLU" if too many neurons stuck at 0 |
| GELU | `x · Φ(x)` (Φ = standard normal CDF) | Smooth ReLU; used in GPT, BERT |
| SwiGLU | gated variant with Swish | Used in LLaMA, PaLM |
| sigmoid | `1 / (1 + e^-x)` | Squashes to (0, 1); old-school, still used in gates |
| tanh | — | Squashes to (-1, 1) |
| softmax | `e^xi / Σ e^xj` | Turns logits into a probability distribution; output layer for classification / next-token prediction |

### Universal approximation

- A network with one sufficiently wide hidden layer can approximate any continuous function (Cybenko 1989, Hornik 1991)
- But required width can grow exponentially. Depth is exponentially more efficient for many functions — *why* deep nets work in practice

### Loss functions

- Measure how wrong the output is, in a single number to minimise
- **Cross-entropy** for classification / next-token prediction: `L = -Σ y_i log(ŷ_i)`
- **MSE** for regression: `L = (y - ŷ)²`
- LLMs are trained with cross-entropy on the next-token distribution

### Forward vs backward pass

- **Forward pass**: compute output from input, layer by layer
- **Backward pass**: compute gradients of loss w.r.t. every parameter (see [`backpropagation.md`](backpropagation.md))
- Optimiser then nudges weights using those gradients (see [`gradient-descent.md`](gradient-descent.md))

### Regularisation (briefly)

- **Dropout** (Srivastava et al. 2014): randomly zero activations during training; prevents over-reliance on any one neuron
- **Weight decay**: penalise large weights via L2 term in loss
- **Layer normalisation** / RMSNorm: stabilise activation scale across the layer
- Modern transformers usually ditch dropout in favour of normalisation + scale

## References

- 3Blue1Brown — *Neural Networks* playlist ([youtube.com/@3blue1brown](https://www.youtube.com/@3blue1brown)) — best visual intro
- Goodfellow, Bengio, Courville — *Deep Learning* (deeplearningbook.org) — free online textbook
- Nielsen — *Neural Networks and Deep Learning* (neuralnetworksanddeeplearning.com) — free, hands-on
- Cybenko 1989 — universal approximation theorem (sigmoid)
- Hornik 1991 — universal approximation, general activations
- Srivastava et al. 2014 — *Dropout* (jmlr.org/papers/v15/srivastava14a.html)
