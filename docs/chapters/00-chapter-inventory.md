# Chapter Inventory

## Chapter List

### 1. Introduction
The philosophy behind microgpt: 201 lines, zero dependencies, the complete algorithm. What the code does, why it exists, and how to read this book. Sets the frame: "this file is the complete algorithm — everything else is just efficiency."

### 2. Data and Tokens
From a text file of names to a list of integer sequences. How raw text becomes the discrete symbols a model can process — character-level tokenization, the vocabulary, and the special BOS token that marks sequence boundaries.

### 3. Autograd
The `Value` class: a scalar that remembers how it was computed. Operator overloading builds a computation graph during the forward pass; topological sort and the chain rule propagate gradients backward. This is the engine that makes learning possible.

### 4. Parameters and Architecture
Initializing the model's learnable weights, then defining the stateless functions that compose into a GPT: `linear` (matrix-vector multiplication), `softmax` (turning scores into probabilities), `rmsnorm` (stabilizing activations), and the `gpt` function itself — embeddings, multi-head self-attention, feed-forward MLP, and residual connections.

### 5. Training
The training loop: tokenize a document, forward it through the model one token at a time to build a computation graph, compute cross-entropy loss, backpropagate gradients, and update parameters with Adam. One thousand steps from random noise to a model that has learned the structure of names.

### 6. Inference
Using the trained model to generate new names. Autoregressive sampling: predict the next token, sample from the distribution, feed it back. Temperature as the knob between conservative and creative generation. The BOS token as both start and stop signal.

### 7. The Complete Picture
Stepping back to see the full data flow from raw text to generated names. What this implementation includes, what it deliberately omits (batching, GPU, LayerNorm, GeLU, biases, dropout, learning rate warmup), and why those omissions don't change the fundamental algorithm. Where to go from here.

## Natural Groupings

The chapters follow the code's own structure — each chapter maps to a clearly delimited section of `microgpt.py`. Three natural clusters emerge:

- **Foundation** (Chapters 1-3): Setup, data, and the computational engine
- **The Model** (Chapters 4-5): Architecture and how it learns
- **Using It** (Chapters 6-7): Generation and perspective

## Reflection

**Does this cover the back cover promises?**

| Back Cover Takeaway | Chapter Coverage |
|---------------------|-----------------|
| How autograd works | Chapter 3 (dedicated) |
| How a transformer processes language | Chapter 4 (architecture) |
| How training actually works | Chapter 5 (dedicated) |
| How generation works | Chapter 6 (dedicated) |
| Why every design choice exists | Threaded throughout, consolidated in Chapter 7 |

All five takeaways are covered. No gaps. The "why every design choice exists" promise is distributed across chapters (each explains its own trade-offs) with Chapter 7 providing the consolidated view.

**Potential concerns:**
- Chapter 4 is the densest — it covers parameter initialization AND four functions AND the full GPT forward pass. This is deliberate: the reader should see how building blocks compose into the complete model. Splitting would break that arc.
- Chapter 7 could feel thin if it's just "here's what we skipped." It needs to earn its place by connecting the pieces in a way earlier chapters couldn't (full data flow diagram, the relationship between all components seen together).
