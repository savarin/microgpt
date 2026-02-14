# Chapter 4: Parameters and Architecture

## Intro

This chapter covers the model itself: the learnable parameters that store its knowledge and the stateless functions that compose into a GPT. The architecture follows GPT-2 with deliberate simplifications — RMSNorm instead of LayerNorm, ReLU instead of GeLU, no biases. Each simplification removes complexity without changing the fundamental algorithm, and each is explained as a trade-off. By the end of this chapter, you can trace a token's journey from integer ID to logit vector — the complete forward pass.

## Sections

### 4.1 Parameter Initialization
The `state_dict`: a dictionary of weight matrices, each a list of lists of `Value` objects. Token embeddings (`wte`), position embeddings (`wpe`), and the language model head (`lm_head`) are created first, then per-layer attention and MLP weights. Why Gaussian initialization with `std=0.08`. Why the `matrix` lambda. The total parameter count and what it means for this small model.

### 4.2 Linear Transformation
The `linear(x, w)` function: matrix-vector multiplication implemented as nested dot products. This is the fundamental building block — attention projections, MLP layers, and the output head all use it. What `linear` computes geometrically: a learned rotation and scaling of the input vector.

### 4.3 Softmax
The `softmax(logits)` function: turning a vector of arbitrary scores into a probability distribution. The numerical stability trick of subtracting the maximum value before exponentiating. Why softmax appears twice in the code — once in attention (to weight values by relevance) and once in the loss/inference (to get token probabilities).

### 4.4 RMSNorm
The `rmsnorm(x)` function: normalizing a vector by its root-mean-square magnitude. Why normalization matters — without it, activations can grow or shrink through layers, destabilizing training. Why RMSNorm over LayerNorm: it skips the mean-centering step, which is simpler and empirically works just as well. Where it appears in the architecture (before attention, before MLP, after initial embedding).

### 4.5 Embeddings
The first operations inside `gpt()`: looking up the token embedding and position embedding, adding them element-wise. Why two embeddings — tokens carry identity ("this is the letter e"), positions carry location ("this is the 3rd token"). Why addition rather than concatenation. The initial RMSNorm that stabilizes the combined embedding.

### 4.6 Multi-Head Self-Attention
The attention mechanism inside the transformer layer: project the input to queries, keys, and values; split into heads; compute attention scores as scaled dot products; apply softmax to get attention weights; use weights to aggregate values; project back. Why multi-head (each head learns different relationships). Why the `1/sqrt(head_dim)` scaling factor. The KV cache: why `keys` and `values` are lists that grow over positions.

### 4.7 The Feed-Forward MLP
The two-layer feed-forward network after attention: expand to 4x the embedding dimension, apply ReLU, project back. Why the expansion factor of 4. Why ReLU instead of GeLU — simpler derivative, adequate for this scale. The residual connection that adds the MLP output back to its input.

### 4.8 Residual Connections
The `x = [a + b for a, b in zip(x, x_residual)]` pattern that appears after both attention and MLP blocks. Why residual connections exist — they let gradients flow directly through the network without being attenuated by every layer. The key insight: each layer learns a *correction* to the residual stream, not a complete transformation.

## Conclusion

The reader can now trace the complete forward pass: an integer token ID enters the `gpt()` function, gets embedded with position information, passes through attention (which lets it gather information from earlier tokens), passes through an MLP (which transforms that information), and exits as a vector of logits — one score per vocabulary token. Every operation along the way is a `Value` computation tracked by the autograd engine from Chapter 3. The next chapter connects this forward pass to learning: how the logits become a loss, how the loss becomes gradients, and how gradients become better parameters.

## Cross-Chapter Coordination

- **Introduces**: `state_dict`, embeddings (token and position), `linear`, `softmax`, `rmsnorm`, attention (Q/K/V, heads, KV cache), MLP, residual connections, logits, the `gpt()` function
- **Referenced by**: Chapter 5 (training calls `gpt()` and uses `softmax` to compute loss), Chapter 6 (inference calls `gpt()` with temperature-scaled `softmax`), Chapter 7 (discusses what the architecture omits vs GPT-2)
- **Depends on**: Chapter 2 (`vocab_size` and `block_size` determine embedding dimensions), Chapter 3 (all parameters and operations are `Value` objects)
- **First introduced here**: `softmax` is introduced in this chapter (Section 4.3) even though it also appears in training (loss computation) and inference (sampling). The definition lives here because it's an architectural building block; Chapters 5 and 6 reference it.
