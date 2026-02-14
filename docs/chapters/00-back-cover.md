# Back Cover

## The Problem

You've used GPT models. You've read blog posts about attention mechanisms. You've maybe even fine-tuned a model using PyTorch or Hugging Face. But when someone asks "how does it actually work?" you find yourself reaching for abstractions and hand-waves. The gap between *using* a transformer and *understanding* one remains wide — not because the ideas are impossibly hard, but because every explanation you've found either skips the math, hides behind frameworks, or drowns you in implementation details that obscure the core algorithm.

## The Reader

This book is for programmers who want to understand GPT-style language models from the ground up. You should be comfortable reading Python and have a basic sense of what neural networks do (they learn functions from data by adjusting parameters). You do not need linear algebra, calculus, or prior experience with deep learning frameworks. Everything you need is built from scratch in front of you.

You might be:
- A software engineer who uses LLMs at work and wants to understand what's happening beneath the API
- A student who finds textbook treatments too abstract and framework tutorials too hand-wavy
- An experienced ML practitioner who wants to see the entire algorithm laid bare in one place, without the ceremony of production code

## What You Will Learn

1. **How autograd works** — how a scalar-level computation graph tracks operations and propagates gradients backward through the chain rule, turning "adjust parameters to reduce loss" from a concept into a concrete mechanism.

2. **How a transformer processes language** — how token and position embeddings, multi-head self-attention, and feed-forward layers compose to transform a sequence of symbols into a prediction about what comes next.

3. **How training actually works** — how a single forward pass builds a computation graph, how backpropagation flows gradients through that graph, and how the Adam optimizer uses those gradients to update parameters step by step.

4. **How generation works** — how a trained model produces new text by repeatedly predicting and sampling the next token, and how temperature controls the trade-off between coherence and creativity.

5. **Why every design choice exists** — why RMSNorm instead of LayerNorm, why residual connections, why multi-head attention splits the embedding space, why Adam over vanilla SGD. Each choice is a trade-off, and you will understand both sides.

## Why This Book

The entire GPT implementation this book explains is 201 lines of Python using only the standard library. No PyTorch. No NumPy. No hidden complexity. Every multiplication, every gradient, every attention score is a line of code you can read, trace, and modify.

Most resources teach transformers top-down: here's the architecture diagram, here are the building blocks, here's the framework code. This book teaches bottom-up: here's a scalar value that tracks its own gradient, here's how scalars compose into matrix operations, here's how matrix operations compose into attention, here's how attention composes into a language model. Each layer is fully understood before the next begins.

By the end, you won't just know *what* a GPT does — you'll know *why* it does it, because you'll have seen every computation that makes it work.
