# Chapter 3: Autograd

## Intro

This chapter covers the computational engine that makes learning possible. The `Value` class wraps a single floating-point number with the machinery to track how it was computed and to propagate gradients backward through the chain rule. This is autograd — automatic differentiation — and it is the mechanism that turns "adjust parameters to reduce loss" from a vague aspiration into a concrete algorithm. Every operation in the model (Chapter 4) and every gradient in training (Chapter 5) flows through this class.

## Sections

### 3.1 The Core Idea
A `Value` is a number that remembers its history. When you write `c = a + b`, the result `c` knows it came from adding `a` and `b`, and it knows the local derivative of addition with respect to each input (both are 1). This is the atomic unit of the computation graph.

### 3.2 The Value Class
The four fields: `data` (the scalar value), `grad` (the accumulated gradient), `_children` (parent nodes in the computation graph), and `_local_grads` (local derivatives). Why `__slots__` exists — memory optimization for the thousands of `Value` objects the model creates.

### 3.3 Arithmetic Operations
How operator overloading (`__add__`, `__mul__`, `__pow__`) lets `Value` objects participate in normal Python arithmetic while silently building the computation graph. Each operation creates a new `Value` node that records its inputs and the local gradient of that operation. The coercion pattern: `other if isinstance(other, Value) else Value(other)`.

### 3.4 Mathematical Functions
`log`, `exp`, and `relu` — the non-arithmetic operations the model needs. Each stores its local derivative: `1/x` for log, `exp(x)` for exp, and the step function `float(x > 0)` for ReLU. Why these specific functions: log appears in the loss, exp appears in softmax, ReLU is the activation function.

### 3.5 Reverse-Mode Automatic Differentiation
The `backward` method: topological sort of the computation graph, then reverse traversal accumulating gradients via the chain rule. Why reverse mode (one backward pass gives gradients for *all* parameters) vs forward mode (one pass per parameter). Why gradients accumulate with `+=` rather than being assigned.

### 3.6 The Helper Operations
The convenience methods: `__neg__`, `__sub__`, `__rsub__`, `__rmul__`, `__truediv__`, `__rtruediv__`, `__radd__`. These are all defined in terms of `__add__`, `__mul__`, and `__pow__` — no new gradient logic needed. The `r`-prefix methods handle the case where the `Value` appears on the right side of an operator.

## Conclusion

The reader now understands how automatic differentiation works at the scalar level: operations build a graph forward, and `backward()` propagates gradients in reverse. Every number in the model will be a `Value`, so every computation is tracked. The next chapter builds the model's architecture on top of this foundation — and every matrix multiply, every attention score, every normalization step will flow through the `Value` operations defined here.

## Cross-Chapter Coordination

- **Introduces**: `Value`, computation graph, forward pass (graph construction), backward pass (gradient propagation), chain rule, topological sort, `grad` attribute
- **Referenced by**: Chapter 4 (all model parameters are `Value` objects; all model operations use `Value` arithmetic), Chapter 5 (training calls `loss.backward()` and reads `.grad` from parameters)
- **Depends on**: Chapter 1 (the "three ingredients" — autograd is what connects the model to the optimizer)
