# Inside microgpt

A bottom-up walkthrough of a GPT-style language model in 201 lines of Python — no PyTorch, no NumPy, no hidden complexity.

## What This Is

This book explains every computation inside a working GPT implementation: how a scalar value tracks its own gradient, how scalars compose into attention, and how attention composes into a language model that learns to generate names. Each chapter maps to a section of `microgpt.py`, and the code is reproduced in full.

## Chapters

1. **Introduction** — The complete algorithm in one file, and how to read this book
2. **Data and Tokens** — From a text file of names to integer sequences
3. **Autograd** — The `Value` class: a scalar that remembers how it was computed
4. **Parameters and Architecture** — Embeddings, attention, feed-forward layers, and how they compose into a GPT
5. **Training** — Forward pass, cross-entropy loss, backpropagation, and Adam
6. **Inference** — Autoregressive generation and temperature sampling
7. **The Complete Picture** — Full data flow, what the code omits, and where to go next

## Who This Is For

Programmers who can read Python and want to understand GPT models from the ground up. No linear algebra, calculus, or framework experience required — everything is built from scratch in front of you.
