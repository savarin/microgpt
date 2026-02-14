# Chapter 1: Introduction

## 1.1 The Complete Algorithm

Open the file `src/microgpt.py` and scroll to the top. The docstring reads:

```
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
```

This is a strong claim. GPT-style language models power systems that generate essays, translate languages, and write code. The models behind these systems have billions of parameters, train on clusters of thousands of GPUs, and are built on top of deep software stacks — PyTorch, CUDA, custom kernels, distributed training frameworks. How can 201 lines of Python with no dependencies constitute "the complete algorithm"?

The answer lies in the distinction between an algorithm and its implementation. The algorithm for training a GPT is a sequence of mathematical operations: embed tokens, compute attention, transform through feed-forward layers, measure loss, propagate gradients, update parameters. These operations are the same whether you execute them on a single CPU with scalar arithmetic or on a thousand GPUs with tensor parallelism. The GPU version computes the same multiplications — it just computes many of them simultaneously.

microgpt implements every one of these operations explicitly, using nothing but Python's built-in arithmetic. There is no hidden complexity behind a library call, no `torch.nn.Linear` that quietly performs a matrix multiply you can't see into. Every multiplication, every addition, every gradient flows through code you can read on a single screen.

This book walks through that code line by line.

## 1.2 What the Code Does

When you run `python src/microgpt.py`, three things happen.

**First**, the program downloads a dataset of approximately 32,000 human names — one per line — and shuffles them into a random order:

```
num docs: 32033
```

**Second**, it trains a language model for 1,000 steps. Each step takes one name from the dataset, feeds it through the model, measures how well the model predicted each character, and adjusts the model's parameters to do better next time. You watch the loss — a number measuring how wrong the model is — decrease over time:

```
step    1 / 1000 | loss 3.3184
step    2 / 1000 | loss 3.3077
...
step  999 / 1000 | loss 1.8551
step 1000 / 1000 | loss 2.2895
```

The loss starts around 3.3, which is roughly what you'd expect from random guessing across 27 possible characters (ln(27) ≈ 3.30). By the end, it has dropped to around 2.0. Individual steps vary — each step processes a different name, so the loss bounces around — but the trend is clearly downward. The model has learned something about the structure of names — which characters tend to follow which — without being told any rules.

**Third**, the trained model generates 20 new names that never appeared in the dataset:

```
--- inference (new, hallucinated names) ---
sample  1: mede
sample  2: wede
sample  3: lede
...
```

These names are plausible — they look and sound like they could be real names — because the model learned patterns from the training data and now reproduces those patterns in novel combinations. This is language generation at its most basic: learn the statistical structure of text, then sample from that learned structure to produce new text.

The entire process — data loading, tokenization, model definition, training, and generation — lives in a single file with no dependencies beyond Python's standard library.

## 1.3 The Three Ingredients

Every neural network training system, from the simplest perceptron to GPT-4, is built from three components:

**The model** is a function that takes an input and produces a prediction. In microgpt, the model takes a sequence of character tokens and predicts a probability distribution over what character comes next. The model has *parameters* — numbers that determine its behavior — and the values of these parameters are what the model "knows." At the start of training, the parameters are random, so the predictions are random. By the end, the parameters encode the patterns of English names.

**The loss** is a function that takes the model's prediction and the correct answer, and produces a single number measuring how wrong the prediction was. In microgpt, the loss is the negative log probability of the correct next character. If the model assigns high probability to the right character, the loss is low. If it assigns low probability, the loss is high. The loss gives training a direction: reduce this number.

**The optimizer** is the rule for adjusting the model's parameters to reduce the loss. It uses *gradients* — the derivative of the loss with respect to each parameter — to determine which direction to adjust each parameter and by how much. In microgpt, the optimizer is Adam, which maintains running averages of gradients to make intelligent updates.

These three components map directly onto the code. The model is the `gpt()` function (Chapter 4). The loss is the cross-entropy computation in the training loop (Chapter 5). The optimizer is the Adam update at the bottom of each training step (Chapter 5). And the glue that connects them — the mechanism that computes how each parameter affects the loss — is the autograd engine (Chapter 3).

## 1.4 Why Zero Dependencies

The imports at the top of `microgpt.py` are:

```python
import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
```

That's it. No NumPy, no PyTorch, no TensorFlow, no JAX. The comments even explain what each import is used for — there are no hidden capabilities.

This constraint has a cost. Without NumPy, matrix multiplication is a nested Python loop over individual scalar operations. A single matrix-vector multiply that would take one line in NumPy (`w @ x`) becomes a list comprehension with an inner sum. Without PyTorch, the autograd engine — the system that tracks operations and computes gradients — is written by hand as a Python class. These are slower by orders of magnitude than their library equivalents.

But the constraint also has a profound benefit: *nothing is hidden*. When you read `linear(x, w)`, you see every multiplication and addition that constitutes a matrix-vector product. When you read `Value.__add__`, you see exactly how the computation graph records an addition and its local gradient. When you read `loss.backward()`, you see the topological sort and chain rule that make up backpropagation. There is no function call that disappears into a C library you can't inspect.

This is the trade-off microgpt makes deliberately. It is too slow to be useful for real training — but it is the clearest possible expression of what training *is*.

## 1.5 How to Read This Book

This book follows the code bottom-up.

Chapter 2 starts with the data: how a text file of names becomes a list of integer sequences. Chapter 3 introduces the `Value` class — the autograd engine that tracks computations and propagates gradients. Chapter 4 builds the model architecture on top of `Value`: embeddings, attention, normalization, feed-forward layers. Chapter 5 puts it all together in the training loop: forward pass, loss, backward pass, optimizer. Chapter 6 covers inference — using the trained model to generate new names. Chapter 7 steps back to see the complete picture and discuss what microgpt omits.

Each chapter corresponds to a section of `microgpt.py`, and the code is reproduced in full with explanation. You are reading a walkthrough of a real, runnable program.

The most valuable thing you can do alongside reading is to run the code yourself:

```bash
python src/microgpt.py
```

Watch the loss decrease. Read the generated names. Then come back to the book and understand *why* the loss decreased and *how* those names were generated. The gap between "I saw it work" and "I understand why it works" is exactly what this book is designed to close.
