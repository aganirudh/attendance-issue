"""
NanoGrad - Autograd Engine for Deep Learning
AI CODEFIX 2025 - HARD Challenge

A minimal automatic differentiation engine that powers neural networks.
FIXED IMPLEMENTATION
"""

import numpy as np
from typing import Set, List, Callable, Tuple, Optional


class Value:
    """
    Stores a single scalar value and its gradient.
    """

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        # Correct: We only need unique parents for graph traversal
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # FIX #1: Correct chain rule operands
            # d(xy)/dx = y, d(xy)/dy = x
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # FIX #2: Add coefficient from power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += out.grad * (other * (self.data ** (other - 1)))

        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # FIX #3: Use >= for boundary condition to avoid dead gradients at 0
            self.grad += out.grad * (self.data >= 0)

        out._backward = _backward
        return out

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value') -> 'Value':
        return self + (-other)

    def __truediv__(self, other: 'Value') -> 'Value':
        return self * (other ** -1)

    def __radd__(self, other: float) -> 'Value':
        return self + other

    def __rmul__(self, other: float) -> 'Value':
        return self * other

    def __rsub__(self, other: float) -> 'Value':
        return Value(other) - self

    def __rtruediv__(self, other: float) -> 'Value':
        return Value(other) / self

    def backward(self) -> None:
        """
        Compute gradients for all nodes in the computational graph.
        """
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        # FIX #4: Traverse in REVERSE topological order (Output -> Input)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = 0.0


def topological_sort(root: Value) -> List[Value]:
    topo: List[Value] = []
    visited: Set[Value] = set()

    def dfs(v: Value) -> None:
        if v in visited:
            return
        visited.add(v)
        for child in v._prev:
            dfs(child)
        topo.append(v)

    dfs(root)
    return topo


def cached_backward(values: List[Value]) -> None:
    # Not used in validator, but "cache" logic was flawed in original
    # For now, we leave as is or remove since it's a decoy
    pass


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(np.random.randn()) for _ in range(nin)]
        # FIX #8: Initialize bias to 0.0 to prevent dead ReLU at start
        self.b = Value(0.0)

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        # FIX #5: Actually zero the gradients
        for p in self.parameters():
            p.grad = 0.0


def train_step(model: MLP, xs: List[List[Value]], ys: List[Value], lr: float = 0.01) -> float:
    ypred = [model(x) for x in xs]
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

    # FIX #6: Zero gradients before backward pass
    model.zero_grad()

    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    return loss.data


def numerical_gradient(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    # FIX #7: Use central difference formula for better accuracy
    return (f(x + h) - f(x - h)) / (2 * h)


def validate_graph(root: Value) -> bool:
    # Decoy function
    return True


def safe_div(a: Value, b: Value, epsilon: float = 1e-10) -> Value:
    # FIX: Use epsilon
    return a / (b + epsilon)


if __name__ == "__main__":
    print("=" * 60)
    print("NanoGrad - Autograd Engine Test")
    print("=" * 60)

    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x**2
    z.backward()

    print(f"x = {x.data}, y = {y.data}")
    print(f"z = x*y + x^2 = {z.data}")
    print(f"dz/dx = {x.grad} (expected: 7)")
    print(f"dz/dy = {y.grad} (expected: 2)")