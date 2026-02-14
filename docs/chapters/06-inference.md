# Chapter 6: Inference

## 6.1 Autoregressive Generation

The model has been trained. Its parameters now encode the statistical patterns of English names — which characters tend to follow which, how names typically start and end, what lengths are common. To generate a new name, the model uses these learned patterns to predict one character at a time:

```python
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

The generation process is *autoregressive*: each prediction depends on all previous predictions. The model doesn't generate an entire name at once — it generates one character, feeds that character back as input, generates the next character, and so on.

The loop starts with `token_id = BOS` — the beginning-of-sequence token. This tells the model "a new name is starting; predict the first character." The model produces logits (one score per vocabulary token), which are converted to probabilities and sampled. The sampled character becomes the input for the next iteration.

Generation stops when either:
- The model predicts BOS (the end-of-sequence signal), or
- The sequence reaches `block_size` (16 tokens), the maximum length the model can handle

A fresh KV cache (`keys` and `values`) is initialized for each name. This is important: each generated name is an independent sequence. The attention mechanism should only attend to tokens within the current name, not to tokens from previously generated names.

This is the same `gpt()` function used during training, called in the same way — one token at a time, with the KV cache accumulating. The only difference is what happens with the output: during training, the output is compared to a known target to compute loss; during inference, the output is sampled to produce the next character.

## 6.2 Temperature

Before applying softmax, the logits are divided by a temperature parameter:

```python
probs = softmax([l / temperature for l in logits])
```

Temperature controls the "sharpness" of the probability distribution. To understand why, consider what softmax does to a vector of logits `[2.0, 1.0, 0.5]`:

- **Temperature 1.0** (no scaling): softmax produces approximately `[0.51, 0.24, 0.11, ...]`. The highest logit gets about twice the probability of the second highest. This is the raw learned distribution.

- **Temperature 0.5** (the code's setting): dividing by 0.5 doubles the logits to `[4.0, 2.0, 1.0]`. Softmax produces approximately `[0.84, 0.11, 0.04, ...]`. The highest logit now dominates — the distribution is *sharper*. The model more strongly favors its best guesses.

- **Temperature 0.1** (very low): the logits become `[20.0, 10.0, 5.0]`. Softmax produces nearly `[1.0, 0.0, 0.0, ...]`. The distribution is almost a hard argmax — the model always picks its most likely token. This produces the most predictable output but loses diversity.

- **Temperature 2.0** (high): the logits become `[1.0, 0.5, 0.25]`. Softmax produces approximately `[0.39, 0.24, 0.18, ...]`. The distribution is *flatter* — even low-probability tokens have a reasonable chance of being selected. This produces more diverse and creative output but also more errors.

The mathematical intuition: dividing logits by `T` before softmax is equivalent to raising each probability to the power `1/T` and renormalizing. Low temperature exponentiates probabilities (sharpening), high temperature roots them (flattening).

microgpt uses `temperature = 0.5`, a moderately conservative setting. The model favors its best predictions while still allowing enough variation that different runs produce different names. This is a good default for name generation — you want plausible names, not identical copies of the most common name in the dataset.

## 6.3 Sampling

Given the probability distribution over next tokens, the model selects one by sampling:

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

`random.choices` performs weighted random selection. Each token ID (from 0 to `vocab_size - 1`) is selected with probability proportional to its weight. The `[0]` extracts the single selected element from the one-element list that `random.choices` returns.

Note that `weights=[p.data for p in probs]` extracts the raw float values from the `Value` objects. During inference, we don't need gradient tracking — we just need the numbers. The `.data` attribute gives us the underlying float without the autograd overhead.

Why sample instead of always picking the highest-probability token? Deterministic selection (argmax) would produce the same output every time. If the model thinks 'a' is the most likely first character, every generated name would start with 'a'. Sampling introduces controlled randomness: 'a' might be selected 40% of the time, 'j' 15%, 'm' 12%, and so on. This produces diverse outputs that collectively reflect the learned distribution.

The interaction between temperature and sampling is important:
- **Temperature shapes the distribution**: it determines how peaked or flat the probabilities are
- **Sampling draws from the distribution**: it selects a token according to those probabilities

At very low temperature, sampling becomes effectively deterministic (one probability is nearly 1.0). At high temperature, sampling becomes effectively uniform (all probabilities are similar). The temperature parameter lets you navigate this spectrum without changing the sampling code.

## 6.4 BOS as Stop Signal

The generation loop includes:

```python
if token_id == BOS:
    break
```

As described in Chapter 2, BOS serves as both the start and stop token. During training, the model learned that BOS follows the last character of a name. When it generates BOS during inference, it's predicting "this name is over" — a learned behavior, not a hard-coded rule.

The important consequence for generation is that names have *variable, learned lengths*. The model doesn't generate a fixed number of characters; it generates until it predicts BOS, which depends on the character sequence so far. A name that starts with common short-name patterns ("ed", "al") might end after 2-3 characters, while one that starts with longer-name patterns ("chris", "alex") might continue for 5-7. Without this stop signal, generation would continue to `block_size` (16 tokens), producing gibberish after the name's natural end.

## 6.5 The KV Cache in Generation

The KV cache introduced in Chapter 4 takes on particular importance during generation. In training, the full sequence is known in advance and the cache is a convenience. In generation, the sequence is being constructed one token at a time, and the cache is what makes autoregressive processing efficient.

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
```

At the start of generation, the cache is empty. With each generated token, the cache grows by one key-value pair per layer. By the time the model is predicting the 8th character, the cache contains 8 key-value pairs, and the attention mechanism computes scores against all 8 previous positions — without reprocessing any of them through the Q, K, V projections.

For microgpt's short sequences, the efficiency gain is modest. But this is the same pattern used in production LLM inference: when generating a 1,000-token response with a 96-layer model, the savings from caching are enormous.

The cache also reveals a design choice worth noting: `gpt()` *mutates* the cache by appending. This is a departure from the stateless style of the model's other components (`linear`, `softmax`, `rmsnorm` are all pure functions). The mutation is contained and intentional — it is the mechanism that allows autoregressive processing without redundant computation.

---

The complete inference pipeline, for each generated name:

1. Start with BOS as the first token
2. Forward pass through `gpt()`: embed the token, compute attention over all cached positions, apply MLP, produce logits
3. Scale logits by temperature, apply softmax to get probabilities
4. Sample a token from the probability distribution
5. If the sampled token is BOS, the name is complete — stop
6. Otherwise, append the character to the output and go to step 2 with the new token

After 20 repetitions, the model has produced 20 new names. Some will be recognizable as plausible names; others will be unusual but follow English phonotactic patterns. The quality depends on how well training went — a model trained for more steps on more data produces better names. But the algorithm is the same regardless of scale.
