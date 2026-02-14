# Chapter 3: Autograd

## 3.1 The Core Idea

The model from Chapter 1 has thousands of parameters — numbers that determine its predictions. Training adjusts these parameters to reduce the loss. But the loss is a single number computed from thousands of operations chained together: embeddings, matrix multiplies, attention, normalization, softmax, logarithm. To adjust any one parameter, you need to know: *if I nudge this parameter slightly, how does the loss change?*

This question is answered by the derivative of the loss with respect to that parameter — its *gradient*. And to compute gradients through a long chain of operations, you need the *chain rule*: if `z` depends on `y` which depends on `x`, then the derivative of `z` with respect to `x` is the derivative of `z` with respect to `y` times the derivative of `y` with respect to `x`.

The autograd engine in microgpt implements this automatically. The `Value` class wraps a single floating-point number with two extra capabilities: it remembers how it was computed (for the chain rule), and it can propagate gradients backward through the computation graph. Every number in the model — every parameter, every intermediate result, every loss value — is a `Value`.

Here is the idea in miniature. Suppose you compute:

```
a = Value(2.0)
b = Value(3.0)
c = a * b          # c.data = 6.0
d = c + a          # d.data = 8.0
```

After these operations, `d` knows it was made by adding `c` and `a`. `c` knows it was made by multiplying `a` and `b`. If you then call `d.backward()`, the system traces back through these relationships and computes: how much does `d` change if `a` changes? How much if `b` changes? The answers — stored in `a.grad` and `b.grad` — are the gradients.

## 3.2 The Value Class

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

A `Value` has four fields:

- **`data`**: The actual number — the scalar value of this node. During the forward pass, this is the result of the computation. For a parameter, this is the weight value that gets updated during training.

- **`grad`**: The gradient of the final loss with respect to this node. This starts at 0 and gets filled in during the backward pass. After `loss.backward()`, every `Value` in the computation graph has a `grad` that answers: "how much does the loss change if this value increases by a tiny amount?"

- **`_children`**: The inputs to the operation that produced this `Value`. If `c = a + b`, then `c._children = (a, b)`. For leaf nodes (parameters and constants), `_children` is empty.

- **`_local_grads`**: The local derivative of this node's operation with respect to each child. If `c = a + b`, then the derivative of `c` with respect to `a` is 1, and with respect to `b` is 1, so `c._local_grads = (1, 1)`. These local derivatives are the building blocks that the chain rule multiplies together during the backward pass.

The `__slots__` declaration is a Python optimization. Normally, every Python object stores its attributes in a dictionary, which is flexible but uses extra memory. `__slots__` tells Python to allocate a fixed set of attribute slots instead. For a program that creates thousands of `Value` objects (every scalar in every matrix multiply), this matters.

## 3.3 Arithmetic Operations

The magic of `Value` is that it participates in normal Python arithmetic while silently building the computation graph. This is achieved through operator overloading:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))
```

When you write `c = a + b` where `a` and `b` are `Value` objects, Python calls `a.__add__(b)`. The method:

1. Ensures `other` is a `Value` (wrapping raw numbers if needed)
2. Computes the forward result: `self.data + other.data`
3. Records the children: `(self, other)` — this addition came from `a` and `b`
4. Records the local gradients: `(1, 1)` — the derivative of `a + b` with respect to `a` is 1, and with respect to `b` is 1

The result is a new `Value` that carries its computation history.

Multiplication works the same way, with different local gradients:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

The derivative of `a * b` with respect to `a` is `b`, and with respect to `b` is `a`. This is the product rule from calculus, but you don't need to think of it that way. You can verify it concretely: if `a = 3` and `b = 5`, then `a * b = 15`. If you nudge `a` to 3.001, the product becomes `3.001 * 5 = 15.005` — it changed by 0.005, which is the nudge (0.001) times `b` (5). The local gradient with respect to `a` is indeed `b`.

The power operation completes the set of primitives:

```python
def __pow__(self, other):
    return Value(self.data**other, (self,), (other * self.data**(other-1),))
```

Note that `__pow__` only supports constant exponents (not `Value` exponents). The local gradient is the standard power rule: the derivative of `x^n` with respect to `x` is `n * x^(n-1)`. This is sufficient for microgpt — the only place `__pow__` appears is in implementing division (`other**-1`) and in the RMSNorm scale computation.

## 3.4 Mathematical Functions

Beyond basic arithmetic, the model needs three mathematical functions. Each is implemented as a method on `Value` that returns a new `Value` with the appropriate local gradient:

```python
def log(self):
    return Value(math.log(self.data), (self,), (1/self.data,))

def exp(self):
    return Value(math.exp(self.data), (self,), (math.exp(self.data),))

def relu(self):
    return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

**`log`**: The natural logarithm. Its derivative is `1/x`. This function appears in the loss computation: `-log(probability of correct token)`. The lower the probability, the higher the loss, and the `1/x` gradient amplifies the signal for low-probability predictions — exactly the behavior you want during training.

**`exp`**: The exponential function. Its derivative is itself — `exp(x)`. This is one of the remarkable properties of `e^x` and is why the exponential appears throughout neural networks. It shows up in softmax, which converts raw scores into probabilities by exponentiating each score and normalizing.

**`relu`**: The Rectified Linear Unit. `relu(x) = max(0, x)`. Its derivative is 1 for positive inputs and 0 for negative inputs — a step function. This is the activation function used inside the model's feed-forward layers (Chapter 4). It introduces nonlinearity: without it, stacking linear transformations would just produce another linear transformation, and the model couldn't learn complex patterns. ReLU is the simplest activation function that works — it either passes the signal through unchanged or kills it entirely.

Each function appears in the model for a specific reason: `log` in the loss, `exp` in softmax, `relu` in the MLP. There are no extra functions — microgpt implements exactly the mathematical operations it needs and nothing more.

## 3.5 Reverse-Mode Automatic Differentiation

The `backward` method is where gradients are actually computed. It walks the computation graph in reverse and applies the chain rule at each step:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

This method does two things:

**First, topological sort.** The `build_topo` function performs a depth-first traversal of the computation graph, visiting children before parents. The result is a list where every node appears after all of its children. Reversing this list gives the order we need for the backward pass: every node is processed before its children, so by the time we reach a node, we already know its gradient.

Why topological sort? The computation graph can have nodes that contribute to the loss through multiple paths. The topological ordering ensures we process each node exactly once, after all its downstream contributions have been accumulated.

**Second, gradient propagation.** Starting from the loss node (whose gradient is 1 — the derivative of the loss with respect to itself), the method walks backward through the sorted nodes. For each node, it distributes its gradient to its children using the chain rule:

```
child.grad += local_grad * v.grad
```

This is the chain rule in its simplest form. The gradient of the loss with respect to a child equals the local gradient (how much the parent changes when the child changes) times the parent's gradient (how much the loss changes when the parent changes). The `+=` is crucial — if a child contributes to the loss through multiple paths, its gradient is the *sum* of the contributions from each path.

**Why reverse mode?** There are two ways to apply the chain rule automatically: forward mode (propagate derivatives from inputs to outputs) and reverse mode (propagate derivatives from outputs to inputs). In forward mode, one pass gives you the gradient of every output with respect to *one* input. In reverse mode, one pass gives you the gradient of *one* output with respect to every input.

In training, we have one output (the loss) and thousands of inputs (the parameters). Reverse mode gives us all the gradients we need in a single pass. Forward mode would require one pass per parameter — thousands of passes. This asymmetry is why every deep learning framework uses reverse mode, and it's why `backward()` is called from the loss, not from the parameters.

## 3.6 The Helper Operations

The remaining methods on `Value` define convenience operations in terms of the primitives:

```python
def __neg__(self): return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other): return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other): return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

Each of these is defined purely in terms of `__add__`, `__mul__`, and `__pow__` — no new gradient logic is needed. Subtraction is addition with negation. Division is multiplication with the reciprocal (via `**-1`). The `r`-prefix methods (`__radd__`, `__rmul__`, etc.) handle the case where a `Value` appears on the right side of an operation with a non-`Value` on the left: `3 + value` calls `value.__radd__(3)`.

This is a common pattern in numerical computing: define a small set of primitive operations with correct gradients, then build everything else from those primitives. The gradient computation "just works" for complex expressions because every complex expression decomposes into chains of primitive operations, each of which records its local gradient, and the chain rule composes them automatically.

The entire autograd engine is 43 lines of code. It handles arbitrary computation graphs of any depth, with any combination of the supported operations. The next chapter builds a GPT-style transformer on top of it — and every multiplication, every attention score, every normalization will flow through these 43 lines, silently recording the computation graph that makes training possible.
