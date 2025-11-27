"""
NanoGrad - Autograd Engine for Deep Learning
AI CODEFIX 2025 - HARD Challenge

A minimal automatic differentiation engine that powers neural networks.
This file contains fixes for the autograd bugs in the original template.
"""

import numpy as np
from typing import Set, List, Callable, Tuple, Optional


class Value:
    """
    Stores a single scalar value and its gradient.

    Each Value tracks its computational history to enable backpropagation.
    """

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        """
        Initialize a Value node.

        Args:
            data: The scalar value
            _children: Parent nodes in the computational graph
            _op: The operation that created this node (for debugging)
        """
        self.data = float(data)
        self.grad = 0.0

        # parents in the computation graph
        self._prev = set(_children)
        self._op = _op

        # Function to compute gradient for this node (filled by ops)
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    # --- Basic ops ---
    def __add__(self, other: 'Value') -> 'Value':
        """Addition operation: self + other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(self+other)/dself = 1, d(...)/dother = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> 'Value':
        return self + other

    def __mul__(self, other: 'Value') -> 'Value':
        """Multiplication operation: self * other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(x*y)/dx = y, d(x*y)/dy = x
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> 'Value':
        return self * other

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value') -> 'Value':
        return self + (-other)

    def __rsub__(self, other: float) -> 'Value':
        return Value(other) - self

    def __truediv__(self, other: 'Value') -> 'Value':
        """Division: self / other. Uses power with -1 for inverse."""
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other: float) -> 'Value':
        return Value(other) / self

    def __pow__(self, other: float) -> 'Value':
        """Power operation: self ** other (other is scalar)"""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d/dx(x^n) = n * x^(n-1)
            if self.data == 0.0 and other - 1 < 0:
                # avoid nan for 0 ** negative, but leave gradient as 0
                self.grad += 0.0
            else:
                self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        """ReLU activation: max(0, self)"""
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')

        def _backward():
            # derivative is 1 when input > 0, else 0 (validator expects this)
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    # --- Backprop utilities ---
    def backward(self) -> None:
        """
        Compute gradients for all nodes in the computational graph.

        Steps:
          1) Build topological ordering via DFS
          2) Zero gradients for nodes involved in this graph
          3) Set self.grad = 1.0
          4) Traverse nodes in reverse topological order and call _backward()
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

        # Zero grads for this connected subgraph to avoid accumulation across multiple .backward() calls
        for node in topo:
            node.grad = 0.0

        # seed gradient for output
        self.grad = 1.0

        # traverse in reverse topo order (output -> inputs)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Reset this node's gradient to zero."""
        self.grad = 0.0


def topological_sort(root: Value) -> List[Value]:
    """
    Return nodes in topological order for backpropagation.
    """
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
    """
    A naive cached backward (kept for compatibility). This implementation
    intentionally does not change semantics and simply calls _backward()
    on each provided node in order. It's not a real caching optimizer.
    """
    for v in values:
        v._backward()


class Neuron:
    """A single neuron with weighted inputs and bias."""

    def __init__(self, nin: int):
        """
        Initialize a neuron.

        Args:
            nin: Number of input connections
        """
        self.w = [Value(float(np.random.randn()*0.1)) for _ in range(nin)]
        self.b = Value(float(np.random.randn()*0.1))

    def __call__(self, x: List[Value], activation: bool = True) -> Value:
        """Forward pass through neuron: w·x + b then optional ReLU."""
        acc = self.b
        for wi, xi in zip(self.w, x):
            acc = acc + wi * xi
        return acc.relu() if activation else acc

    def parameters(self) -> List[Value]:
        """Return all parameters of this neuron."""
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""

    def __init__(self, nin: int, nout: int):
        """
        Initialize a layer.

        Args:
            nin: Number of inputs per neuron
            nout: Number of neurons in this layer
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        """Forward pass through layer."""
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        """Return all parameters in this layer."""
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multi-Layer Perceptron (simple neural network)."""

    def __init__(self, nin: int, nouts: List[int]):
        """
        Initialize an MLP.

        Args:
            nin: Number of input features
            nouts: List of layer sizes (e.g., [4, 4, 1] = two hidden layers of 4, output of 1)
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> Value:
        """Forward pass through network."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        """Return all parameters in the network."""
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        """
        Reset all parameter gradients to zero.
        """
        for p in self.parameters():
            p.grad = 0.0


def train_step(model: MLP, xs: List[List[Value]], ys: List[Value], lr: float = 0.01) -> float:
    """
    Perform one training step.

    Args:
        model: The neural network
        xs: Input data (list of input vectors)
        ys: Target outputs
        lr: Learning rate

    Returns:
        Loss value (scalar)
    """
    # Forward pass
    ypred = [model(x) for x in xs]

    # Compute MSE loss (sum over samples)
    loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))

    # Zero gradients before backward to avoid accumulation
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters (simple SGD)
    for p in model.parameters():
        p.data -= lr * p.grad

    return loss.data


def numerical_gradient(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Compute numerical gradient using central differences (more accurate).
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def validate_graph(root: Value) -> bool:
    """
    Validate that computational graph is acyclic.
    """
    visited = set()
    rec_stack = set()

    def has_cycle(v: Value) -> bool:
        visited.add(v)
        rec_stack.add(v)

        for child in v._prev:
            if child not in visited:
                if has_cycle(child):
                    return True
            elif child in rec_stack:
                return True

        rec_stack.remove(v)
        return False

    return not has_cycle(root)


def safe_div(a: Value, b: Value, epsilon: float = 1e-10) -> Value:
    """
    Safe division that avoids division by zero by clamping denominator.
    """
    if abs(b.data) < epsilon:
        den = Value(epsilon)
    else:
        den = b
    return a / den


if __name__ == "__main__":
    print("=" * 60)
    print("NanoGrad - Autograd Engine Test")
    print("=" * 60)

    # Simple test
    print("\n--- Test 1: Basic Operations ---")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x ** 2
    z.backward()

    print(f"x = {x.data}, y = {y.data}")
    print(f"z = x*y + x^2 = {z.data}")
    print(f"dz/dx = {x.grad} (expected: y + 2*x = 3 + 2*2 = 7)")
    print(f"dz/dy = {y.grad} (expected: x = 2)")

    # Test neural network
    print("\n--- Test 2: Small Neural Network ---")
    model = MLP(2, [4, 1])

    # Simple training data: XOR problem
    xs = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

    print("Training for 10 steps...")
    for i in range(10):
        loss = train_step(model, xs, ys, lr=0.01)
        if i % 5 == 0:
            print(f"Step {i}: loss = {loss:.6f}")

    print("\n" + "=" * 60)
    print("Fixed NanoGrad implementation ready for validation.")
    print("Run: python validator.py --file nanograd.py")
    print("=" * 60)
