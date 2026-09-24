"""Complete small NumPy teaching engine and offline experiments, not app code."""
from pathlib import Path
import json
import numpy as np


def reduce_to_shape(gradient, shape):
    """Undo additive/multiplicative NumPy broadcasting."""
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, width in enumerate(shape):
        if width == 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    """Float64, immutable forward values; fresh gradients per backward call."""
    def __init__(self, value, requires_grad=False, parents=(), operation="leaf"):
        self._data = np.array(value, dtype=np.float64, copy=True)
        self._data.setflags(write=False)
        self.requires_grad = requires_grad
        self.parents = tuple(parents)
        self.operation = operation
        self.grad = np.zeros_like(self._data)
        self.pullback = lambda: None

    @property
    def data(self):
        return self._data

    @staticmethod
    def wrap(value):
        return value if isinstance(value, Tensor) else Tensor(value)

    @staticmethod
    def result(value, parents, operation):
        return Tensor(value, any(p.requires_grad for p in parents), parents, operation)

    def add_gradient(self, value):
        if self.requires_grad:
            self.grad += reduce_to_shape(np.asarray(value), self.data.shape)

    def __add__(self, other):
        other = Tensor.wrap(other)
        result = Tensor.result(self.data + other.data, (self, other), "add")
        def pullback():
            self.add_gradient(result.grad)
            other.add_gradient(result.grad)
        result.pullback = pullback
        return result

    __radd__ = __add__

    def __mul__(self, other):
        other = Tensor.wrap(other)
        result = Tensor.result(self.data * other.data, (self, other), "multiply")
        def pullback():
            self.add_gradient(other.data * result.grad)
            other.add_gradient(self.data * result.grad)
        result.pullback = pullback
        return result

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -Tensor.wrap(other)

    def __rsub__(self, other):
        return Tensor.wrap(other) + -self

    def __matmul__(self, other):
        other = Tensor.wrap(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("This teaching matmul supports two matrices only.")
        result = Tensor.result(self.data @ other.data, (self, other), "matmul")
        def pullback():
            self.add_gradient(result.grad @ other.data.T)
            other.add_gradient(self.data.T @ result.grad)
        result.pullback = pullback
        return result

    def sum(self):
        result = Tensor.result(self.data.sum(), (self,), "sum")
        result.pullback = lambda: self.add_gradient(np.ones_like(self.data) * result.grad)
        return result

    def mean(self):
        if self.data.size == 0:
            raise ValueError("Mean needs a nonempty tensor.")
        return self.sum() * (1 / self.data.size)

    def tanh(self):
        values = np.tanh(self.data)
        result = Tensor.result(values, (self,), "tanh")
        result.pullback = lambda: self.add_gradient((1 - values**2) * result.grad)
        return result

    def relu(self):
        mask = self.data > 0
        result = Tensor.result(np.maximum(self.data, 0), (self,), "relu")
        result.pullback = lambda: self.add_gradient(mask * result.grad)
        return result

    def exp(self):
        values = np.exp(self.data)
        result = Tensor.result(values, (self,), "exp")
        result.pullback = lambda: self.add_gradient(values * result.grad)
        return result

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("Log needs strictly positive inputs.")
        result = Tensor.result(np.log(self.data), (self,), "log")
        result.pullback = lambda: self.add_gradient(result.grad / self.data)
        return result

    def cross_entropy(self, labels):
        labels = np.asarray(labels)
        if self.data.ndim != 2 or len(self.data) == 0:
            raise ValueError("Cross-entropy needs a nonempty (N,C) matrix.")
        count, classes = self.data.shape
        if (labels.shape != (count,) or labels.dtype.kind not in "iu"
                or np.any(labels < 0) or np.any(labels >= classes)):
            raise ValueError("Labels must be integer class IDs of shape (N,).")
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        normalizer = np.exp(shifted).sum(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(normalizer)
        probabilities = np.exp(shifted) / normalizer
        value = -log_probabilities[np.arange(count), labels].mean()
        local_gradient = probabilities.copy()
        local_gradient[np.arange(count), labels] -= 1
        local_gradient /= count
        result = Tensor.result(value, (self,), "cross_entropy")
        result.pullback = lambda: self.add_gradient(result.grad * local_gradient)
        return result

    def backward(self, seed=None):
        if seed is None:
            if self.data.ndim != 0:
                raise ValueError("A nonscalar output needs an explicit seed.")
            seed = np.ones_like(self.data)
        seed = np.asarray(seed, dtype=np.float64)
        if seed.shape != self.data.shape:
            raise ValueError("Seed shape must match output shape.")
        ordered, visited = [], set()
        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            ordered.append(node)
        visit(self)
        for node in ordered:
            node.grad = np.zeros_like(node.data)
        self.grad = seed.copy()
        for node in reversed(ordered):
            node.pullback()


def make_parameters(seed, dimensions):
    rng = np.random.default_rng(seed)
    input_width, hidden_width, output_width = dimensions
    return [
        Tensor(rng.normal(0, .5, (input_width, hidden_width)), True),
        Tensor(np.zeros(hidden_width), True),
        Tensor(rng.normal(0, .5, (hidden_width, output_width)), True),
        Tensor(np.zeros(output_width), True),
    ]


def predict(features, parameters, activation="tanh"):
    first_weight, first_bias, last_weight, last_bias = parameters
    hidden = Tensor(features) @ first_weight + first_bias
    hidden = hidden.tanh() if activation == "tanh" else hidden.relu()
    return hidden @ last_weight + last_bias


def train_xor():
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    targets = np.array([[0], [1], [1], [0]], dtype=float)
    parameters = make_parameters(3, (2, 4, 1))
    trace = []
    for step in range(2001):
        outputs = predict(features, parameters)
        errors = outputs - targets
        loss = (errors * errors).mean()
        if step in [0, 1, 10, 100, 500, 2000]:
            trace.append({"step":step, "mse":float(loss.data), "outputs":outputs.data.tolist()})
        if step == 2000:
            break
        loss.backward()
        parameters = [Tensor(p.data - .1*p.grad, True) for p in parameters]
    return trace


def train_digits():
    from sklearn.model_selection import train_test_split
    directory = Path(__file__).resolve().parent
    data = np.genfromtxt(directory/"digits-400.csv", delimiter=",", names=True)
    features = np.column_stack([data[f"pixel_{j}"] for j in range(64)])/16
    labels = data["digit"].astype(int)
    train, valid = train_test_split(np.arange(len(data)), test_size=120,
                                   stratify=labels, random_state=22)
    parameters = make_parameters(4, (64, 16, 10))
    trace = []
    for step in range(501):
        logits = predict(features[train], parameters)
        loss = logits.cross_entropy(labels[train])
        if step in [0, 1, 10, 100, 250, 500]:
            predictions = predict(features[valid], parameters).data.argmax(axis=1)
            trace.append({"step":step, "trainLoss":float(loss.data),
                          "validationCorrect":int((predictions == labels[valid]).sum())})
        if step == 500:
            break
        loss.backward()
        parameters = [Tensor(p.data - .2*p.grad, True) for p in parameters]
    return trace


if __name__ == "__main__":
    print("XOR", json.dumps(train_xor()))
    print("Digits", json.dumps(train_digits()))
