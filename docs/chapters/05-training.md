# Chapter 5: Training

## Intro

This chapter covers the training loop — the process that turns random parameters into a model that understands the structure of names. Training is where the three ingredients from Chapter 1 come together: the model (Chapter 4) makes predictions, the loss measures how wrong they are, and the optimizer adjusts parameters to do better next time. Each training step builds a computation graph through the forward pass, flows gradients backward through it, and updates every parameter. After 1,000 steps, the model has learned patterns it was never explicitly taught.

## Sections

### 5.1 The Training Loop Structure
The outer loop: 1,000 steps, each processing one document. Why one document at a time (no batching) — it's the simplest correct approach. The `step % len(docs)` cycling through the dataset. What changes per step and what stays fixed.

### 5.2 Tokenizing a Training Example
Taking a name, wrapping it with BOS tokens, and truncating to `block_size`. The input-target pairs: for each position, the input is `tokens[pos_id]` and the target is `tokens[pos_id + 1]`. This is the fundamental structure of next-token prediction — the model sees a prefix and must predict what follows.

### 5.3 The Forward Pass
Calling `gpt()` for each position in the sequence, accumulating key-value pairs for attention. The forward pass does two things simultaneously: it computes the model's predictions AND it builds the computation graph that `backward()` will traverse. Every `Value` operation in Chapter 4 is silently recording its children and local gradients.

### 5.4 Cross-Entropy Loss
Converting logits to probabilities with `softmax`, then computing `-log(probability of the correct token)`. Why negative log probability is the right loss — it's zero when the model is certain and correct, infinite when it assigns zero probability to the truth. Averaging over the sequence: `(1/n) * sum(losses)`.

### 5.5 Backpropagation
Calling `loss.backward()`: the single line that triggers reverse-mode automatic differentiation through the entire computation graph. After this call, every `Value` in every parameter matrix has a `.grad` that says "increasing this value by a tiny amount would change the loss by this much." This is the gradient — the direction of steepest ascent.

### 5.6 The Adam Optimizer
Why not just subtract the gradient (vanilla SGD): raw gradients are noisy and don't account for curvature. Adam maintains two running averages per parameter — the first moment (momentum, smoothing out noise) and the second moment (adaptive learning rate, scaling by historical gradient magnitude). Bias correction for the early steps when the running averages haven't converged.

### 5.7 Learning Rate Decay
Linear decay from `0.01` to `0` over the training run. Why decay: large steps early for fast initial progress, small steps late for fine-tuning. Why linear (simplest schedule that works). The interaction between base learning rate and Adam's adaptive scaling.

### 5.8 Gradient Zeroing
The `p.grad = 0` at the end of each step. Why it's necessary — `backward()` accumulates gradients with `+=`, so without zeroing, gradients from previous steps would contaminate the current step. Why zeroing happens after the update, not before the forward pass (convention; same effect).

## Conclusion

The reader now understands the complete training process: tokenize a document into input-target pairs, forward each position through the model to build a computation graph, compute the cross-entropy loss, backpropagate to get gradients for every parameter, and update parameters with Adam. They understand why each component exists: the loss gives direction, backpropagation distributes that direction to every parameter, and Adam follows that direction intelligently. The model is now trained. The next chapter uses it.

## Cross-Chapter Coordination

- **Introduces**: Cross-entropy loss, training step, Adam optimizer (first/second moments, bias correction), learning rate decay, gradient zeroing, the training loop itself
- **Referenced by**: Chapter 6 (uses the trained parameters), Chapter 7 (discusses what the training loop omits — batching, warmup, evaluation)
- **Depends on**: Chapter 2 (tokenization of documents, BOS wrapping), Chapter 3 (`backward()` and the `.grad` attribute), Chapter 4 (`gpt()` function and `softmax`)
- **Key connection**: This chapter is where Chapters 3 and 4 meet. The forward pass (Chapter 4) builds the graph; `backward()` (Chapter 3) traverses it. Training is the connective tissue.
