# Chapter 6: Inference

## Intro

This chapter covers generation — using the trained model to produce new names it has never seen. Inference reuses the same `gpt()` function from Chapter 4, but without the training machinery: no loss computation, no backward pass, no optimizer. Instead, the model predicts one token at a time, samples from the prediction, and feeds the sampled token back as input. Temperature controls whether the model plays it safe or takes creative risks. The result: names that sound plausible but never existed in the training data.

## Sections

### 6.1 Autoregressive Generation
The generation loop: start with BOS, call `gpt()` to get logits, convert to probabilities, sample a token, repeat until BOS appears again or the sequence reaches `block_size`. Why this is called "autoregressive" — each prediction depends on all previous predictions. The model generates one token at a time, not an entire name at once.

### 6.2 Temperature
Dividing logits by temperature before applying softmax: `softmax([l / temperature for l in logits])`. What temperature does mathematically — it sharpens (temperature < 1) or flattens (temperature > 1) the probability distribution. At temperature 0.5 (the code's setting): the model is moderately conservative, favoring its best guesses while still allowing variation. At temperature 1.0: the raw learned distribution. Near temperature 0: greedy decoding, always picking the most likely token.

### 6.3 Sampling
Using `random.choices` with probability weights to sample from the distribution. Why sampling instead of always taking the argmax — deterministic decoding produces repetitive output. Sampling introduces controlled randomness that makes each generation unique. The interaction between temperature and sampling: temperature shapes the distribution, sampling draws from it.

### 6.4 BOS as Stop Signal
The same BOS token that starts a sequence also ends it. When the model generates BOS, generation stops. Why this works: during training, the model learned that BOS follows the last character of a name. It learned sequence boundaries from the data itself, not from a separate "end of sequence" token. An elegant design choice that saves one vocabulary slot.

### 6.5 The KV Cache in Generation
During inference, `keys` and `values` grow with each generated token — this is the KV cache. Each new token attends to all previous tokens without recomputing their key and value projections. Why this matters for efficiency even in this small implementation: without the cache, generating a 10-character name would require recomputing attention for all previous positions at each step.

## Conclusion

The reader now understands the complete generation process: start with BOS, predict the next token, sample from the temperature-adjusted distribution, and repeat. They understand how temperature controls the quality-diversity trade-off and why the KV cache avoids redundant computation. With this chapter, the full pipeline is complete: data (Chapter 2) trains a model (Chapters 4-5) that generates new data (this chapter). The final chapter steps back to see the whole picture.

## Cross-Chapter Coordination

- **Introduces**: Autoregressive generation, temperature, sampling, the inference loop, KV cache as an inference optimization
- **Referenced by**: Chapter 7 (discusses generation quality and what more sophisticated sampling strategies exist)
- **Depends on**: Chapter 2 (BOS token, `uchars` for decoding back to characters), Chapter 4 (`gpt()` function, `softmax`), Chapter 5 (the trained parameters that make generation meaningful)
- **Note on softmax**: `softmax` was introduced in Chapter 4. Here it appears with temperature scaling — this is a new use, not a new concept. The chapter explains the temperature modification without re-deriving softmax.
