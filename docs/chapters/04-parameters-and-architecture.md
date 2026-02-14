# Chapter 4: Parameters and Architecture

## 4.1 Parameter Initialization

Before the model can make predictions, it needs parameters — the learnable numbers that store its knowledge. At the start of training, these parameters are random. By the end, they encode the statistical patterns of English names.

```python
n_embd = 16     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 1     # number of layers
block_size = 16 # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head
```

These hyperparameters define the model's shape. The embedding dimension (`n_embd = 16`) determines the size of the vector that represents each token at each position — every token is described by 16 numbers. The number of attention heads (`n_head = 4`) splits the attention mechanism into 4 independent subspaces, each of dimension `head_dim = 4`. The model has a single transformer layer (`n_layer = 1`) and can process sequences up to 16 tokens (`block_size = 16`).

These are small values. GPT-2 uses an embedding dimension of 768, 12 attention heads, 12 layers, and a block size of 1024. But the algorithm is the same — microgpt is a miniature GPT, not a different kind of model.

The parameters are created using a helper function:

```python
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]
```

Each weight matrix is a list of lists of `Value` objects, initialized from a Gaussian distribution with mean 0 and standard deviation 0.08. The small standard deviation keeps initial weights close to zero, which gives the model a gentle starting point — predictions start close to uniform rather than wildly confident in random directions. The exact value (0.08 vs. 0.05 vs. 0.1) is not critical; what matters is that it's small enough to avoid large initial logits that would saturate the softmax, but not so small that the gradients vanish at the start of training.

The `state_dict` collects all the weight matrices:

```python
state_dict = {
    'wte': matrix(vocab_size, n_embd),   # token embedding: vocab_size x 16
    'wpe': matrix(block_size, n_embd),   # position embedding: 16 x 16
    'lm_head': matrix(vocab_size, n_embd) # output head: vocab_size x 16
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # query projection: 16 x 16
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # key projection: 16 x 16
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # value projection: 16 x 16
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # output projection: 16 x 16
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) # MLP expand: 64 x 16
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd) # MLP contract: 16 x 64
```

With `vocab_size = 28`, `n_embd = 16`, and `n_layer = 1`, the total parameter count is:

- Token embedding: 28 × 16 = 448
- Position embedding: 16 × 16 = 256
- Language model head: 28 × 16 = 448
- Attention Q, K, V, O projections: 4 × (16 × 16) = 1,024
- MLP layers: (64 × 16) + (16 × 64) = 2,048

Total: 4,224 parameters. Each one is a `Value` object that tracks its gradient during training.

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
```

This line flattens all the weight matrices into a single list. The optimizer iterates over this list to update every parameter after each backward pass.

## 4.2 Linear Transformation

The most fundamental operation in the model is the linear transformation — matrix-vector multiplication:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

This takes a vector `x` (a list of `Value` objects) and a weight matrix `w` (a list of lists of `Value` objects), and returns a new vector where each element is the dot product of one row of `w` with `x`.

If `x` has 16 elements and `w` has 64 rows of 16 elements each, the output has 64 elements. Each output element is a weighted sum of all input elements, where the weights are learned parameters. This is matrix-vector multiplication written as explicit loops.

In NumPy, this would be `w @ x` — one line, executed in optimized C. Here, it is a Python list comprehension with an inner generator expression. The output is identical; the performance differs by orders of magnitude. But the computation is visible: every multiplication between a weight and an input, every summation, is a `Value` operation that records itself in the computation graph.

Linear transformations appear everywhere in the model: projecting inputs to queries, keys, and values for attention; the two layers of the MLP; and the final output head that produces logits. The `linear` function is called 6 times per transformer layer (Q, K, V, O projections plus two MLP layers), plus once for the output head — 7 times total for each token processed.

What does a linear transformation *do*, conceptually? It applies a learned rotation and scaling to the input vector. The weight matrix defines a new coordinate system, and the output is the input vector expressed in that system. By learning the right coordinate system, the model learns to extract useful features from its input — "this looks like the beginning of a name" or "a vowel tends to follow here."

## 4.3 Softmax

The softmax function converts a vector of arbitrary real numbers into a probability distribution — a vector of non-negative numbers that sum to 1:

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

The function exponentiates each value (making everything positive), then divides by the total (making them sum to 1). The result is that larger input values get larger probabilities, but the relationship is *exponential* — a value that's 2 units larger than another gets approximately `e^2 ≈ 7.4` times as much probability.

The `max_val` subtraction is a numerical stability trick. Exponentiating large numbers can cause overflow (numbers too large for floating point). Subtracting the maximum value shifts all inputs so the largest is 0, which means the largest exponentiated value is `exp(0) = 1`. This doesn't change the output — subtracting a constant from all inputs before softmax produces the same probabilities — but it prevents overflow.

Note that `max_val` is extracted with `.data` (a raw float), not kept as a `Value`. The subtraction `val - max_val` still creates a `Value` computation graph node (because `val` is a `Value` and `max_val` is a float that gets wrapped). The maximum itself doesn't need gradients — it's a constant used for numerical stability.

Softmax appears in two roles in microgpt. In the attention mechanism (Section 4.6), it converts attention scores into attention weights — "how much should this token attend to each previous token?" In the loss computation and inference (Chapters 5 and 6), it converts the model's output logits into token probabilities — "what does the model think the next character is?"

## 4.4 RMSNorm

Normalization keeps the scale of activations stable as data flows through the network:

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

RMSNorm — Root Mean Square Normalization — divides each element of the vector by the root-mean-square of the whole vector. The result is a vector with an RMS magnitude close to 1, regardless of what the input magnitude was.

The computation: `ms` is the mean of squared elements (the mean square). `scale` is `1 / sqrt(ms + epsilon)`, where the tiny epsilon (`1e-5`) prevents division by zero when all elements are near zero. Multiplying each element by `scale` normalizes the vector.

Why is normalization necessary? Without it, the magnitude of activations can grow or shrink as data passes through successive layers of the network. Growing activations cause numerical overflow and exploding gradients. Shrinking activations cause the signal to vanish. Normalization after each major operation keeps everything in a stable range.

microgpt uses RMSNorm rather than the LayerNorm used in the original GPT-2. LayerNorm does two things: it normalizes the magnitude (like RMSNorm) and it centers the mean to zero. RMSNorm skips the mean-centering step. Research shows that mean centering contributes relatively little to training stability, while the magnitude normalization is what actually matters. RMSNorm is simpler (fewer operations, simpler gradient) and works just as well in practice — a trade-off in favor of clarity that doesn't sacrifice effectiveness.

RMSNorm appears three times in the `gpt()` function: once after combining token and position embeddings, once before the attention block, and once before the MLP block. Each application stabilizes the activations before the next major computation.

## 4.5 Embeddings

The `gpt()` function begins by converting a token ID and position ID into a continuous vector representation:

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]  # token embedding
    pos_emb = state_dict['wpe'][pos_id]    # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # joint embedding
    x = rmsnorm(x)
```

**Token embedding** (`wte`): The token embedding table has one row per vocabulary entry — 28 rows of 16 values each. Looking up `state_dict['wte'][token_id]` retrieves the 16-dimensional vector associated with this token. These vectors are learned parameters: during training, the model adjusts them so that tokens used in similar contexts get similar embeddings.

**Position embedding** (`wpe`): The position embedding table has one row per position — 16 rows of 16 values each (one for each possible position up to `block_size`). Looking up `state_dict['wpe'][pos_id]` retrieves the vector associated with this position. These vectors are also learned parameters.

**Why two separate embeddings?** Tokens carry identity ("this is the letter 'e'") and positions carry location ("this is the 3rd character"). The model needs both pieces of information. The letter 'e' as the first character of a name carries different predictive information than 'e' as the fifth character.

**Why addition?** The two embeddings are combined by element-wise addition: `[t + p for t, p in zip(tok_emb, pos_emb)]`. An alternative would be concatenation — stacking the two vectors into a 32-dimensional vector. Addition has a practical advantage: it keeps the vector the same dimension (16), so every downstream operation works with 16-dimensional inputs. Concatenation would require adjusting all weight matrix dimensions. Addition also has an interesting representational property: it creates a shared space where token identity and positional information interact immediately, rather than keeping them in separate halves of the vector.

The combined embedding is then passed through RMSNorm to stabilize its magnitude before entering the transformer layer.

## 4.6 Multi-Head Self-Attention

Attention is the mechanism that allows each token to gather information from other tokens in the sequence. It answers the question: "to predict what comes after this token, which previous tokens are most relevant, and what information should I take from them?"

The attention block starts by projecting the input into three different representations:

```python
x_residual = x
x = rmsnorm(x)
q = linear(x, state_dict[f'layer{li}.attn_wq'])  # query
k = linear(x, state_dict[f'layer{li}.attn_wk'])  # key
v = linear(x, state_dict[f'layer{li}.attn_wv'])  # value
keys[li].append(k)
values[li].append(v)
```

**Queries, keys, and values** are three different linear projections of the same input. The analogy to information retrieval is useful: the *query* is "what am I looking for?", the *key* is "what do I contain?", and the *value* is "what information do I provide if you attend to me?" The query of the current token is compared against the keys of all previous tokens to determine relevance, and then the values of those tokens are aggregated weighted by relevance.

The keys and values are appended to growing lists (`keys[li]` and `values[li]`). This is the **key-value (KV) cache**: when processing a sequence position by position, the keys and values from previous positions are stored so they don't need to be recomputed. Each new position computes its own query, key, and value, but it compares its query against all accumulated keys.

The attention computation is split across multiple heads:

```python
x_attn = []
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

With `n_embd = 16` and `n_head = 4`, each head operates on a 4-dimensional slice of the 16-dimensional vectors. Head 0 uses dimensions 0-3, head 1 uses dimensions 4-7, and so on. Each head has its own query, key, and value subvectors.

**Why multiple heads?** A single attention head can only compute one notion of "relevance" — it learns one pattern of which tokens should attend to which. Multiple heads allow the model to simultaneously attend to different types of relationships. One head might learn "attend to the previous vowel," while another learns "attend to the first character of the name." With 4 heads, the model can maintain 4 parallel attention patterns.

For each head, the attention scores are computed:

```python
attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
]
attn_weights = softmax(attn_logits)
```

Each attention score is the dot product of the query with a key, divided by `sqrt(head_dim)`. The dot product measures similarity — if the query and key point in similar directions, the score is high. The `1/sqrt(head_dim)` scaling prevents the dot products from growing proportionally with dimension, which would push softmax into regions where it has very small gradients (saturating at near-0 or near-1 probabilities).

Softmax converts the scores into weights that sum to 1. These weights determine how much information flows from each previous position to the current position.

The weighted aggregation:

```python
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
x_attn.extend(head_out)
```

Each element of the head output is a weighted sum of the corresponding element across all value vectors. If attention assigns weight 0.7 to position 2 and 0.3 to position 0, the output is 70% of position 2's value and 30% of position 0's value. The head outputs are concatenated back into a 16-dimensional vector.

Finally, the concatenated attention output is projected through one more linear transformation and added back to the residual:

```python
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
x = [a + b for a, b in zip(x, x_residual)]
```

The output projection mixes information across heads — it allows the model to combine the insights from different attention patterns into a coherent representation.

## 4.7 The Feed-Forward MLP

After attention, each token's representation passes through a two-layer feed-forward network:

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # expand: 16 → 64
x = [xi.relu() for xi in x]                       # nonlinearity
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # contract: 64 → 16
x = [a + b for a, b in zip(x, x_residual)]       # residual connection
```

The MLP has a characteristic "expand-and-contract" shape. The first linear layer projects from 16 dimensions to 64 dimensions (4× expansion). ReLU applies a nonlinearity — setting negative values to zero. The second linear layer projects back from 64 to 16 dimensions.

**Why the expansion?** The 4× expansion creates a wider hidden layer where the model can compute more complex transformations. Think of it as a workspace: the model temporarily expands into a higher-dimensional space where it has more room to manipulate the representation, then compresses back to the original dimension. The expansion factor of 4 is a convention from the original transformer paper that has persisted because it works well.

**Why ReLU?** Attention is a weighted average — a linear operation. The MLP is a sequence of linear operations with ReLU in between. Without the nonlinearity, two successive linear layers would collapse into a single linear layer (matrix multiplication is associative). ReLU introduces the ability to "turn off" certain dimensions (by clamping negatives to zero), which gives the network the capacity to learn nonlinear functions. The original GPT-2 uses GeLU (Gaussian Error Linear Unit), which is a smooth approximation of ReLU. microgpt uses ReLU because its derivative is simpler (a step function vs. a formula involving the Gaussian cumulative distribution function), and at this scale the difference is negligible.

**What does the MLP do, conceptually?** If attention's job is to move information between tokens ("what did I learn from the previous characters?"), the MLP's job is to transform information within a token ("given what I now know, what features should I compute?"). Attention is communication; MLP is computation.

## 4.8 Residual Connections

The pattern `x = [a + b for a, b in zip(x, x_residual)]` appears after both the attention block and the MLP block. This is the *residual connection* — adding the block's output to its input.

Without residual connections, each layer would completely transform its input. The output of layer 1 becomes the input of layer 2, which becomes the input of layer 3, and so on. If any layer learned a poor transformation, the damage would propagate — later layers would receive corrupted input and have to compensate.

With residual connections, each layer learns a *correction* rather than a complete transformation. The input flows directly through to the output (via the addition), and the layer adds an adjustment. If a layer has nothing useful to contribute, it can learn to output near-zero values and the residual connection passes the input through unchanged. This makes training much more stable.

There is another crucial benefit for gradient flow. During backpropagation, gradients must flow from the loss back through every layer. Without residual connections, the gradient at each layer is multiplied by that layer's weights — and if many layers multiply gradients by values less than 1, the gradient shrinks exponentially (the "vanishing gradient" problem). The residual connection creates a shortcut: the gradient of `x + correction` with respect to `x` is always 1 (plus whatever gradient flows through the correction path). This means gradients can flow directly from the loss to early layers without attenuation.

In microgpt with its single layer, the residual connections are not strictly necessary for avoiding vanishing gradients. But they are part of the GPT-2 architecture that microgpt follows, and they illustrate a pattern that becomes essential when you scale to dozens of layers.

The combination of attention, MLP, and residual connections forms one transformer block. In microgpt, there is one block. In GPT-2, there are 12. In GPT-3, there are 96. The blocks are identical in structure — only the weight values differ — and they are stacked sequentially, each one reading from and writing to the residual stream.

After the final transformer block, the model produces its output:

```python
logits = linear(x, state_dict['lm_head'])
return logits
```

The language model head projects the 16-dimensional representation back to `vocab_size` dimensions — one score (logit) per token in the vocabulary. These logits are the model's raw predictions: higher logits mean the model considers that token more likely as the next token. The softmax function (applied outside `gpt()`, in the training loop or inference code) converts these logits into probabilities.

This completes the forward pass. An integer token ID entered the function, became a 16-dimensional embedding, passed through attention and an MLP with residual connections, and emerged as a vector of 28 logits — one for each possible next token. Every operation along the way was performed on `Value` objects, so the entire computation is recorded in a graph that `backward()` can traverse to compute gradients.
