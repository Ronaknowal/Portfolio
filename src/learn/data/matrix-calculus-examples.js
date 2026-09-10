export const matrixCalculusExamples = {
  local: {
    title: 'Check a local prediction against the actual change',
    code: `import numpy as np

def f(x):
    return np.array([x[0] ** 2 + x[1], x[0] * x[1]])

x = np.array([2.0, 3.0])
delta = np.array([0.01, -0.02])
J = np.array([[2 * x[0], 1.0], [x[1], x[0]]])
predicted = J @ delta
actual = f(x + delta) - f(x)
print("Jacobian:", J.tolist())
print("predicted:", np.round(predicted, 6).tolist())
print("actual:", np.round(actual, 6).tolist())
print("remainder:", np.round(actual - predicted, 6).tolist())
half = f(x + delta / 2) - f(x) - J @ (delta / 2)
print("half-step remainder:", np.round(half, 6).tolist())`,
    expected: `Jacobian: [[4.0, 1.0], [3.0, 2.0]]
predicted: [0.02, -0.01]
actual: [0.0201, -0.0102]
remainder: [0.0001, -0.0002]
half-step remainder: [2.5e-05, -5e-05]`,
  },
  chain: {
    title: 'Compute forward sensitivity and reverse gradients separately',
    code: `import numpy as np

A = np.array([[1.0, 2.0], [-1.0, 1.0]])
x = np.array([1.0, 2.0])
v = np.array([1.0, 0.0])
q = np.array([1.0, -1.0])
y = A @ x
z = y * y
L = q @ z
J_square = np.diag(2 * y)
J_z = J_square @ A
tangent_y = A @ v
tangent_z = J_square @ tangent_y
rate = q @ tangent_z
gradient_y = J_square.T @ q
gradient_x = A.T @ gradient_y
print("y, z, L:", y.tolist(), z.tolist(), float(L))
print("Jacobian of z:", J_z.tolist())
print("forward rate:", float(rate))
print("gradient x:", gradient_x.tolist())
print("same directional rate:", float(gradient_x @ v))`,
    expected: `y, z, L: [5.0, 1.0] [25.0, 1.0] 24.0
Jacobian of z: [[10.0, 20.0], [-2.0, 2.0]]
forward rate: 12.0
gradient x: [12.0, 18.0]
same directional rate: 12.0`,
  },
  affine: {
    title: 'Run a complete four-observation affine backward pass',
    code: `import numpy as np

# Rows are observations. The three columns are input features.
X = np.array([[1., 2., 0.], [0., 1., 1.],
              [2., 0., 1.], [1., 1., 1.]])
W = np.array([[1., 0.], [0., 2.], [1., -1.]])
b = np.array([0.5, -0.5])
target = np.array([[1., 1.], [0., 0.], [2., -1.], [1., 2.]])
N = X.shape[0]
Y = X @ W + b
residual = Y - target
# Mean over observations, SUM over output coordinates, and a factor 1/2.
loss = np.sum(residual ** 2) / (2 * N)
G = residual / N
grad_W = X.T @ G
grad_b = G.sum(axis=0)
grad_X = G @ W.T
print("Y:", Y.tolist())
print("loss:", float(loss))
print("grad W:", grad_W.tolist())
print("grad b:", grad_b.tolist())
print("grad X:", grad_X.tolist())
print("shapes:", grad_W.shape, grad_b.shape, grad_X.shape)`,
    expected: `Y: [[1.5, 3.5], [1.5, 0.5], [3.5, -1.5], [2.5, 0.5]]
loss: 2.0
grad W: [[1.25, 0.0], [1.0, 1.0], [1.125, -0.375]]
grad b: [1.25, 0.25]
grad X: [[0.125, 1.25, -0.5], [0.375, 0.25, 0.25], [0.375, -0.25, 0.5], [0.375, -0.75, 0.75]]
shapes: (3, 2) (2,) (4, 3)`,
  },
  gradientCheck: {
    title: 'Check an analytic gradient and expose an extra-mean bug',
    code: `import numpy as np

X = np.array([[1., 2.], [3., -1.]])
W = np.array([[1., -1.], [2., 1.]])
b = np.array([0., 1.])
target = np.array([[1., 0.], [0., 2.]])
N = X.shape[0]

def loss(weights, bias):
    residual = X @ weights + bias - target
    return np.sum(residual * residual) / (2 * N)

G = (X @ W + b - target) / N
grad_W, grad_b = X.T @ G, G.sum(axis=0)
h = 1e-5
numeric_W = np.zeros_like(W)
numeric_b = np.zeros_like(b)
for index in np.ndindex(W.shape):
    E = np.zeros_like(W)
    E[index] = 1
    numeric_W[index] = (loss(W + h * E, b) - loss(W - h * E, b)) / (2 * h)
for index in range(b.size):
    e = np.zeros_like(b)
    e[index] = 1
    numeric_b[index] = (loss(W, b + h * e) - loss(W, b - h * e)) / (2 * h)

print("analytic W:", grad_W.tolist())
print("analytic b:", grad_b.tolist())
print("checks:", bool(np.allclose(grad_W, numeric_W, atol=1e-7)),
      bool(np.allclose(grad_b, numeric_b, atol=1e-7)))
wrong_b = G.mean(axis=0)  # BUG: the objective already divided by N
print("wrong extra mean:", wrong_b.tolist())
print("bug detected:", not np.allclose(wrong_b, numeric_b, atol=1e-7))`,
    expected: `analytic W: [[3.5, -6.5], [3.5, 4.5]]
analytic b: [2.5, -1.5]
checks: True True
wrong extra mean: [1.25, -0.75]
bug detected: True`,
  },
  dual: {
    title: 'Build a tiny forward-mode engine for addition and multiplication',
    code: `from dataclasses import dataclass

@dataclass(frozen=True)
class Dual:
    value: float
    tangent: float

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.value + other.value, self.tangent + other.tangent)

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.value * other.value,
                    self.tangent * other.value + self.value * other.tangent)

    __rmul__ = __mul__

def program(x1, x2):
    return x1 * x1 + x2, x1 * x2

for direction in [(1.0, 0.0), (0.0, 1.0), (1.0, -2.0)]:
    outputs = program(Dual(2.0, direction[0]), Dual(3.0, direction[1]))
    print("direction:", direction,
          "values:", [item.value for item in outputs],
          "Jv:", [item.tangent for item in outputs])`,
    expected: `direction: (1.0, 0.0) values: [7.0, 6.0] Jv: [4.0, 3.0]
direction: (0.0, 1.0) values: [7.0, 6.0] Jv: [1.0, 2.0]
direction: (1.0, -2.0) values: [7.0, 6.0] Jv: [2.0, -1.0]`,
  },
  differences: {
    title: 'Use exact decimal arithmetic to separate truncation from rounding',
    code: `from decimal import Decimal, localcontext

with localcontext() as context:
    context.prec = 70
    x = Decimal(2)
    exact = 3 * x * x
    for text in ["0.1", "0.0001", "0.00000001"]:
        h = Decimal(text)
        central = ((x + h) ** 3 - (x - h) ** 3) / (2 * h)
        print("h:", text, "central error:", format(central - exact, ".16f"))

for h in [1.0, 0.1, 0.0001]:
    right = (abs(h) - abs(0.0)) / h
    left = (abs(0.0) - abs(-h)) / h
    central = (abs(h) - abs(-h)) / (2 * h)
    print("kink h:", h, "left/right/central:", left, right, central)`,
    expected: `h: 0.1 central error: 0.0100000000000000
h: 0.0001 central error: 0.0000000100000000
h: 0.00000001 central error: 0.0000000000000001
kink h: 1.0 left/right/central: -1.0 1.0 0.0
kink h: 0.1 left/right/central: -1.0 1.0 0.0
kink h: 0.0001 left/right/central: -1.0 1.0 0.0`,
  },
  square: {
    title: 'Differentiate a matrix square without assuming commutativity',
    code: `import numpy as np

A = np.array([[1., 2.], [0., 1.]])
E = np.array([[0., 0.], [1., 0.]])
correct = A @ E + E @ A
wrong = 2 * A @ E
h = 1e-5
numeric = ((A + h * E) @ (A + h * E)
           - (A - h * E) @ (A - h * E)) / (2 * h)
print("AE + EA:", correct.tolist())
print("2AE:", wrong.tolist())
print("correct matches:", bool(np.allclose(correct, numeric, atol=1e-7)))
print("shortcut matches:", bool(np.allclose(wrong, numeric, atol=1e-7)))`,
    expected: `AE + EA: [[2.0, 0.0], [2.0, 2.0]]
2AE: [[4.0, 0.0], [2.0, 0.0]]
correct matches: True
shortcut matches: False`,
  },
  solve: {
    title: 'Predict how a solved system changes when its coefficients move',
    code: `import numpy as np

A = np.array([[2., 1.], [1., 3.]])
b = np.array([1., 2.])
E = np.array([[1., 0.], [0., 0.]])  # A changes along E
c = np.array([0., 1.])              # b changes along c
u = np.linalg.solve(A, b)
rate = np.linalg.solve(A, c - E @ u)
h = 1e-5
numeric = (np.linalg.solve(A + h * E, b + h * c)
           - np.linalg.solve(A - h * E, b - h * c)) / (2 * h)
print("u:", np.round(u, 6).tolist())
print("du/dt:", np.round(rate, 6).tolist())
print("independent check:", bool(np.allclose(rate, numeric, atol=1e-7)))
print("linearized equation:", bool(np.allclose(A @ rate + E @ u, c)))`,
    expected: `u: [0.2, 0.6]
du/dt: [-0.32, 0.44]
independent check: True
linearized equation: True`,
  },
  softmax: {
    title: 'Explain why a common score offset disappears',
    code: `import numpy as np

def softmax(scores):
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    return weights / weights.sum()

scores = np.array([0., np.log(2), np.log(3)])
p = softmax(scores)
J = np.diag(p) - np.outer(p, p)
v = np.array([1., 0., -1.])
rate = p * (v - p @ v)  # Jv without storing a Jacobian
h = 1e-5
numeric = (softmax(scores + h * v) - softmax(scores - h * v)) / (2 * h)
print("probabilities:", np.round(p, 6).tolist())
print("common shift unchanged:", bool(np.allclose(softmax(scores + 100), p)))
print("J times ones:", np.round(J @ np.ones(3), 12).tolist())
print("directional rate:", np.round(rate, 6).tolist())
print("rate sums to zero:", bool(np.isclose(rate.sum(), 0)))
print("check:", bool(np.allclose(rate, numeric, atol=1e-7)))`,
    expected: `probabilities: [0.166667, 0.333333, 0.5]
common shift unchanged: True
J times ones: [0.0, 0.0, 0.0]
directional rate: [0.222222, 0.111111, -0.333333]
rate sums to zero: True
check: True`,
  },
  hessian: {
    title: 'Compare a gradient with how that gradient changes',
    code: `import numpy as np

# L(x) = x1^2 + x1*x2 + 2*x2^2
def loss(x):
    return x[0] ** 2 + x[0] * x[1] + 2 * x[1] ** 2

def gradient(x):
    return np.array([2 * x[0] + x[1], x[0] + 4 * x[1]])

x = np.array([1., -1.])
v = np.array([2., 1.])
H = np.array([[2., 1.], [1., 4.]])
h = 0.01
first = loss(x) + h * (gradient(x) @ v)
second = first + 0.5 * h * h * (v @ H @ v)
print("gradient:", gradient(x).tolist())
print("Hessian times v:", (H @ v).tolist())
print("first order:", round(float(first), 6))
print("second order:", round(float(second), 6))
print("actual:", round(float(loss(x + h * v)), 6))
print("gradient-change check:", bool(np.allclose(
    (gradient(x + h * v) - gradient(x)) / h, H @ v)))`,
    expected: `gradient: [1.0, -3.0]
Hessian times v: [5.0, 6.0]
first order: 1.99
second order: 1.9908
actual: 1.9908
gradient-change check: True`,
  },
};
