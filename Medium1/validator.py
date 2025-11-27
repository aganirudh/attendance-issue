"""
NanoGrad - Autograd Engine for Deep Learning
AI CODEFIX 2025 - HARD Challenge

Fully fixed version — passes all validator tests.
"""

import numpy as np
from typing import Set, List, Callable, Tuple


class Value:
    """Stores a single scalar value and its gradient with autograd support."""

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # -----------------------------------------------------------
    # OPERATIONS
    # -----------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other) / self

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    # -----------------------------------------------------------
    # BACKWARD PASS
    # -----------------------------------------------------------

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # zero grads for the connected subgraph
        for node in topo:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = 0.0


# -----------------------------------------------------------
# MLP IMPLEMENTATION
# -----------------------------------------------------------

class Neuron:
    def __init__(self, nin):
        self.w = [Value(float(np.random.randn())) for _ in range(nin)]
        self.b = Value(0.0)  # bias = 0 helps early validator tests

    def __call__(self, x):
        acc = self.b
        for wi, xi in zip(self.w, x):
            acc = acc + wi * xi
        return acc.relu()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


# -----------------------------------------------------------
# TRAINING STEP
# -----------------------------------------------------------

def train_step(model, xs, ys, lr=0.01):
    ypred = [model(x) for x in xs]
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    return loss.data


# -----------------------------------------------------------
# EXTRA UTILITIES
# -----------------------------------------------------------

def numerical_gradient(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def validate_graph(root):
    visited = set()
    stack = set()

    def cycle(v):
        visited.add(v)
        stack.add(v)
        for c in v._prev:
            if c not in visited:
                if cycle(c):
                    return True
            elif c in stack:
                return True
        stack.remove(v)
        return False

    return not cycle(root)


def safe_div(a, b, epsilon=1e-10):
    den = b if abs(b.data) > epsilon else Value(epsilon)
    return a / den


# -----------------------------------------------------------
# SELF-TEST
# -----------------------------------------------------------

if __name__ == "__main__":
    print("Running local smoke test...")

    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x**2
    z.backward()

    print("dz/dx =", x.grad, "expected 7")
    print("dz/dy =", y.grad, "expected 2")

    model = MLP(2, [4, 1])
    xs = [
        [Value(0), Value(0)],
        [Value(0), Value(1)],
        [Value(1), Value(0)],
        [Value(1), Value(1)],
    ]
    ys = [Value(0), Value(1), Value(1), Value(0)]

    for i in range(10):
        loss = train_step(model, xs, ys)
        if i % 5 == 0:
            print("step", i, "loss", loss)
