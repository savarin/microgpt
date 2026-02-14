# Chapter 1: Introduction

## Intro

This chapter sets the stage for everything that follows. It answers three questions: What is microgpt? Why does it exist? And how should you read this book? The key insight is that a GPT-style language model is not inherently complex — it is a composition of simple operations, and you can understand every one of them by reading 201 lines of Python that depend on nothing but the standard library.

## Sections

### 1.1 The Complete Algorithm
What microgpt is: a single Python file that trains a GPT-style language model on a dataset of names and generates new ones. The file's docstring — "this file is the complete algorithm; everything else is just efficiency" — as the organizing principle of the book.

### 1.2 What the Code Does
A high-level walkthrough of the program's output: it downloads a list of names, trains a model for 1,000 steps (watching the loss decrease), and then generates 20 new names that look plausible but never existed. This is the end-to-end result we're building toward.

### 1.3 The Three Ingredients
Every neural network training system has three parts: a model (function from inputs to predictions), a loss (measure of how wrong the predictions are), and an optimizer (rule for adjusting parameters to reduce the loss). This triad structures the entire book and maps directly to sections of the code.

### 1.4 Why Zero Dependencies
What it means to use only `os`, `math`, and `random`. No NumPy means every matrix multiply is nested loops over scalars. No PyTorch means autograd is hand-written. This is deliberately slow and deliberately clear — the goal is understanding, not performance.

### 1.5 How to Read This Book
The bottom-up approach: each chapter builds on the previous one, and you should read sequentially. The code is reproduced in full within each chapter alongside explanation. You are encouraged to run `microgpt.py` yourself to see training progress and generated output.

## Conclusion

By the end of this chapter, the reader understands what they're building toward (a name-generating language model), the three conceptual pillars they'll encounter (model, loss, optimizer), and the philosophy of the implementation (completeness over efficiency). They're ready to start with the data.

## Cross-Chapter Coordination

- **Introduces**: The model/loss/optimizer triad, the "everything else is efficiency" frame, the bottom-up approach
- **Referenced by**: Every subsequent chapter refers back to the triad; Chapter 7 returns to "everything else is efficiency" to discuss what "efficiency" means in practice
- **Depends on**: Nothing — this is the entry point
