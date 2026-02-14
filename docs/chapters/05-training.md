# Chapter 5: Training

## 5.1 The Training Loop Structure

Training is a loop. Each iteration — called a *step* — takes one name from the dataset, measures how well the model predicts it, and adjusts the model's parameters to predict it better.

```python
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
```

The loop runs for 1,000 steps. Each step selects one document (name) from the shuffled dataset using modular indexing — `step % len(docs)` cycles through the dataset, wrapping around to the beginning after exhausting all names. With approximately 32,000 names and 1,000 steps, the model sees about 3% of the dataset during training. This is enough to learn basic character-level patterns, though more steps would improve the model.

There is no batching — each step processes a single name. In production training, you process multiple examples simultaneously (a "batch") to get smoother gradient estimates and better GPU utilization. microgpt processes one example at a time because it is the simplest correct approach: compute the loss for one example, get the gradients, update the parameters. Batching computes the average gradient across multiple examples before updating — mathematically equivalent to running multiple single-example steps and averaging, but faster on parallel hardware.

## 5.2 Tokenizing a Training Example

Each training step begins by converting a name into a token sequence:

```python
doc = docs[step % len(docs)]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = min(block_size, len(tokens) - 1)
```

The name is wrapped with BOS tokens on both sides, as described in Chapter 2. `uchars.index(ch)` converts each character to its integer ID. The result for a name like "ada" is `[27, 0, 3, 0, 27]` — BOS, 'a', 'd', 'a', BOS.

The variable `n` is the number of prediction positions in this sequence. It is `len(tokens) - 1` because each position predicts the *next* token (so the last token has no target to predict). The `min(block_size, ...)` caps this at the model's maximum sequence length. For most names (shorter than 16 characters plus 2 BOS tokens), the full name is used.

The inner training loop iterates over positions:

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
losses = []
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)
```

For each position, the input is the current token and the target is the next token. The model produces logits (raw scores for each vocabulary token), softmax converts them to probabilities, and the loss at this position is the negative log probability of the correct target.

The KV cache (`keys` and `values`) is initialized fresh for each training example. As the loop progresses through positions, each call to `gpt()` appends the new key and value vectors to the cache. When processing position 5, the model has access to the keys and values from positions 0 through 5, allowing it to attend to all previous tokens.

## 5.3 The Forward Pass

The line `logits = gpt(token_id, pos_id, keys, values)` calls the model defined in Chapter 4. But something important is happening beneath the surface: because every number is a `Value` object, and every operation on `Value` objects records its children and local gradients, the forward pass is *simultaneously* computing the model's prediction *and* building the computation graph that will be traversed during backpropagation.

Consider what the computation graph looks like for a single position. The token embedding lookup selects 16 `Value` objects from the embedding table. Position embedding lookup selects 16 more. Element-wise addition creates 16 new `Value` nodes. RMSNorm creates a chain of nodes for the sum-of-squares, the scale factor, and the normalized values. The attention block creates nodes for Q, K, V projections (each 16 nodes), attention scores, softmax weights, weighted value aggregation, and the output projection. The MLP creates nodes for the expansion, ReLU, contraction, and residual addition. The output head creates 28 logit nodes.

For a single position, this is hundreds of `Value` nodes, each pointing to its children with stored local gradients. Across all positions of a training example, the graph grows to thousands of nodes. This entire graph is held in memory, waiting for the backward pass.

This is the fundamental trade-off of automatic differentiation: you pay with memory (storing the entire computation graph) to gain the ability to compute all gradients in a single backward pass. In production systems, this memory cost is a primary constraint — it determines how large a batch you can process on a given GPU.

## 5.4 Cross-Entropy Loss

After the forward pass produces a probability distribution over next tokens, the loss measures how wrong the prediction was:

```python
probs = softmax(logits)
loss_t = -probs[target_id].log()
```

This is the *cross-entropy loss*, also called *negative log-likelihood*. It takes the probability the model assigned to the correct answer and computes its negative logarithm.

Why negative log probability? Consider the behavior:
- If the model assigns probability 1.0 to the correct token: loss = -log(1.0) = 0. Perfect prediction, zero loss.
- If the model assigns probability 0.5: loss = -log(0.5) ≈ 0.693. Uncertain, moderate loss.
- If the model assigns probability 0.01: loss = -log(0.01) ≈ 4.605. Nearly missed the correct answer, high loss.
- If the model assigns probability approaching 0: loss approaches infinity.

This loss function has several desirable properties. It is zero only when the prediction is perfect. It increases without bound as the prediction worsens. And its gradient through softmax has a clean form: the gradient of the loss with respect to the logit of token `i` is `(predicted_probability_i - 1)` if `i` is the correct token, and `predicted_probability_i` otherwise. This means incorrect tokens that received high probability get pushed down, and the correct token gets pushed up, with strength proportional to the error.

The per-position losses are averaged to get the overall loss for the training example:

```python
loss = (1 / n) * sum(losses)
```

Averaging by `n` (the number of positions) means the loss magnitude is independent of sequence length. A 3-character name and a 10-character name produce losses on the same scale, so the optimizer doesn't need to adjust its step size based on name length.

At the start of training with random parameters, the model assigns roughly uniform probability to all tokens. With a vocabulary of 28, each token gets probability ≈ 1/28, giving a loss of -log(1/28) ≈ 3.33 per position. This matches what you see in the first training step: `loss 3.3184`. The loss is close to the theoretical random-guessing baseline — the model knows nothing yet.

## 5.5 Backpropagation

After the forward pass builds the computation graph and computes the loss, a single line triggers the backward pass:

```python
loss.backward()
```

This calls the `backward()` method defined in Chapter 3. Starting from the loss node (whose gradient is set to 1), it traverses the computation graph in reverse topological order, applying the chain rule at each node to accumulate gradients.

After `backward()` completes, every `Value` object that participated in computing the loss — every parameter in every weight matrix — has a `.grad` attribute containing the derivative of the loss with respect to that parameter. This gradient answers a specific question: "if I increase this parameter by a tiny amount epsilon, the loss changes by approximately `epsilon * gradient`."

The sign of the gradient tells you the direction:
- Positive gradient: increasing this parameter increases the loss (bad). Decrease it.
- Negative gradient: increasing this parameter decreases the loss (good). Increase it.

The magnitude tells you the sensitivity: a large gradient means the loss is very sensitive to this parameter; a small gradient means it barely matters.

For the model's 4,224 parameters, `backward()` computes 4,224 gradients in one pass. This is the power of reverse-mode automatic differentiation — regardless of the number of parameters, the cost of the backward pass is proportional to the cost of the forward pass (roughly 2-3x, accounting for the chain rule multiplications).

## 5.6 The Adam Optimizer

The gradients tell us *which direction* to adjust each parameter. The optimizer determines *how much* to adjust it. microgpt uses Adam (Adaptive Moment Estimation), a popular optimizer that is more sophisticated than simply subtracting the gradient:

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # first moment buffer
v = [0.0] * len(params)  # second moment buffer
```

Adam maintains two running averages per parameter, initialized to zero:

**First moment (`m`)**: An exponential moving average of the gradient. This is *momentum* — it smooths out the noise in individual gradient estimates by averaging over recent steps. If the gradient has been consistently positive across recent steps, the first moment will be positive and relatively large, giving a stronger signal than any single noisy gradient. The decay factor `beta1 = 0.85` means each new gradient gets 15% weight and the running average gets 85% weight.

**Second moment (`v`)**: An exponential moving average of the *squared* gradient. This tracks the magnitude of recent gradients for each parameter. Parameters with consistently large gradients get smaller effective learning rates (the update is divided by the square root of this value), while parameters with small gradients get larger effective learning rates. This *adaptive* learning rate is what makes Adam work well across parameters with very different gradient scales — embedding parameters and attention weights might have gradients that differ by orders of magnitude, and Adam handles this automatically.

The update rule:

```python
lr_t = learning_rate * (1 - step / num_steps)  # linear LR decay
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    p.grad = 0
```

Line by line:
1. **Update first moment**: blend old average (85%) with new gradient (15%)
2. **Update second moment**: blend old average (99%) with new squared gradient (1%)
3. **Bias correction for first moment**: divide by `(1 - beta1^t)`. In early steps, the running average is biased toward zero (it started at zero). This correction compensates — at step 1, it divides by 0.15, effectively amplifying the first gradient to compensate for having no history.
4. **Bias correction for second moment**: same logic, dividing by `(1 - beta2^t)`
5. **Parameter update**: subtract the learning rate times the corrected first moment divided by the square root of the corrected second moment (plus epsilon to prevent division by zero)
6. **Zero the gradient**: reset for the next step

The update formula `lr * m_hat / sqrt(v_hat)` can be understood as: take the smoothed gradient direction (`m_hat`), scale it by the inverse of the smoothed gradient magnitude (`1/sqrt(v_hat)`), and multiply by the learning rate. The result is a step that follows the recent gradient direction but is normalized by recent gradient magnitude — parameters with large gradients take small steps, parameters with small gradients take large steps.

## 5.7 Learning Rate Decay

The learning rate decreases linearly from 0.01 to 0 over the 1,000 training steps:

```python
lr_t = learning_rate * (1 - step / num_steps)
```

At step 0: `lr_t = 0.01 * 1.0 = 0.01` (full learning rate).
At step 500: `lr_t = 0.01 * 0.5 = 0.005` (half learning rate).
At step 999: `lr_t = 0.01 * 0.001 = 0.00001` (nearly zero).

Why decay? Early in training, the parameters are random and the model needs to make large adjustments to move toward the right region of parameter space. Large learning rates enable this rapid initial progress. Later in training, the model has found a reasonable solution and needs to make small, precise adjustments to fine-tune. Large steps at this point would overshoot the optimum, causing the loss to bounce around instead of converging.

Linear decay is the simplest schedule that captures this intuition. More sophisticated schedules — cosine decay, warmup-then-decay — are common in production training. The choice of schedule matters more for large models trained for many epochs; for microgpt's 1,000 steps on a small model, linear decay is sufficient.

Note the interaction between learning rate decay and Adam's adaptive scaling. Adam already adjusts the effective step size per parameter based on gradient history. The learning rate is a global multiplier on top of Adam's per-parameter adjustment. Together, they provide both local adaptation (different parameters get different effective rates via Adam) and global scheduling (all parameters ramp down over time via the learning rate).

## 5.8 Gradient Zeroing

The final line of the optimizer loop resets each parameter's gradient to zero:

```python
p.grad = 0
```

Why is this necessary? Recall from Chapter 3 that `backward()` *accumulates* gradients using `+=`. This means if you call `backward()` twice without zeroing, the gradients from the second call are added to the gradients from the first. For standard training where each step should use only the current example's gradient, this would contaminate the signal — the optimizer would be updating based on a sum of gradients from this step and previous steps, which is not what you want.

Zeroing after the update (rather than before the forward pass) is a convention. The effect is the same: when the next step's `backward()` runs, all gradients start at 0. Some frameworks zero before the forward pass; microgpt zeros after the update. Both produce correct results.

There are situations where gradient accumulation without zeroing is useful — it allows you to simulate a larger batch size by accumulating gradients over multiple forward-backward passes before updating. But microgpt doesn't use this technique.

---

A complete training step, in summary:

1. Pick a name from the dataset
2. Tokenize it: wrap with BOS, convert to integer IDs
3. Forward pass: process each position through the model, building the computation graph
4. Loss: measure how wrong the predictions were (cross-entropy)
5. Backward pass: propagate gradients through the graph to every parameter
6. Optimizer: update each parameter using Adam with decaying learning rate
7. Zero gradients: reset for the next step

After 1,000 repetitions of this cycle, the model's parameters have moved from random initialization to a configuration that captures the character-level patterns of English names. The loss has dropped from ~3.3 (random guessing) to ~2.0 (meaningful predictions). The model is now ready to generate.
