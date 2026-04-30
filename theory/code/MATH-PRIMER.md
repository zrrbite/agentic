# Math primer for the LLM theory notebooks

If parts of the math feel rusty, these are the canonical free references. Pick one per topic — you don't need them all.

## Calculus (derivatives, chain rule, gradients)

The math behind notebooks 02 (numerical gradients) and 03 (backprop).

- **3Blue1Brown — *Essence of Calculus*** (free, ~3hrs total): [youtube playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) — visual intuition for what derivatives, integrals, and the chain rule actually mean. Watch episodes 1–4 if you only have an hour
- **Khan Academy — Differential Calculus**: [khanacademy.org/math/differential-calculus](https://www.khanacademy.org/math/differential-calculus) — drill-style with exercises; best if you want to actually *compute* things

## Linear algebra (matrices, matrix calculus)

Used everywhere — every layer is a matrix multiply. Matrix calculus (e.g. "what is `dL/dW` when `y = x @ W.T + b`?") shows up explicitly in notebook 03.

- **3Blue1Brown — *Essence of Linear Algebra*** (free, ~3hrs total): [youtube playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — visual intuition for vectors, matrices, eigenvalues, what `A @ x` *means* geometrically. **Watch this before everything else** if linear algebra is rusty
- **The Matrix Calculus You Need For Deep Learning** (Parr & Howard): [explained.ai/matrix-calculus](https://explained.ai/matrix-calculus/) — exactly what the title says. Reference for the Jacobian / gradient derivations in backprop

## Probability and information theory (softmax, cross-entropy, KL divergence)

The output side of every classifier — softmax + cross-entropy is the loss in notebooks 01–06.

- **Goodfellow, Bengio, Courville — *Deep Learning* book, chapter 3** (free): [deeplearningbook.org/contents/prob.html](https://www.deeplearningbook.org/contents/prob.html) — covers everything you need
- **Christopher Olah — *Visual Information Theory***: [colah.github.io/posts/2015-09-Visual-Information](https://colah.github.io/posts/2015-09-Visual-Information/) — visual, accessible explanation of why cross-entropy and KL divergence look the way they do

## Neural network math specifically

- **CS231n notes** (Stanford CNN course) — the gold standard:
  - [Backpropagation, intuitions](https://cs231n.github.io/optimization-2/) — algorithm with worked examples
  - [Softmax + cross-entropy derivation](https://cs231n.github.io/neural-networks-case-study/#grad)
  - [Gradient checking and the ReLU-kink issue](https://cs231n.github.io/neural-networks-3/#gradcheck) (you ran into this in notebook 03)
- **Karpathy — *Yes you should understand backprop***: [karpathy.medium.com](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — short, memorable, on *why* the math matters in practice
- **Karpathy — *Neural Networks: Zero to Hero***: [youtube playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — code-along videos that build a tiny autograd engine, then nanoGPT. Mirrors what we're doing in notebooks 03 and 06

## Comprehensive textbook

- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong): [mml-book.github.io](https://mml-book.github.io/) — free PDF. Linear algebra, calculus, probability, optimisation, all in one place. The book to work through if you want all of this *properly*

## Fastest path if you remember none of this

Watch in order:

1. 3Blue1Brown's *Essence of Linear Algebra* (~3hrs)
2. 3Blue1Brown's *Essence of Calculus* (~3hrs)
3. 3Blue1Brown's *Neural Networks* playlist (~1hr)

That's ~7 hours and gets you working intuition for everything in `theory/`.
