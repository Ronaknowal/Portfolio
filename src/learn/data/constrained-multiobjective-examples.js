export const constrainedExamples = {
  selection: {
    title: "Filter the original three deployment choices before comparing error",
    code: String.raw`models = [
    {"name": "Small", "error": 0.12, "latency_ms": 8},
    {"name": "Medium", "error": 0.08, "latency_ms": 18},
    {"name": "Large", "error": 0.07, "latency_ms": 70},
]

def choose(models, latency_limit):
    feasible = [m for m in models if m["latency_ms"] <= latency_limit]
    if not feasible:
        return None
    return min(feasible, key=lambda m: (m["error"], m["latency_ms"]))

print(choose(models, 25))
print("5 ms budget:", choose(models, 5))`,
    expected: String.raw`{'name': 'Medium', 'error': 0.08, 'latency_ms': 18}
5 ms budget: None`,
  },
  projection: {
    title: "Compare both projection orders with the true intersection projection",
    code: String.raw`import numpy as np

def checked(v, budget):
    v = np.asarray(v, dtype=float)
    if v.shape != (2,) or not np.isfinite(v).all():
        raise ValueError("Expected two finite coordinates")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Budget must be finite and positive")
    return v

def line_projection(v, budget):
    v = checked(v, budget)
    return v - (v.sum() - budget) / 2

def simplex_projection(v, budget):
    v = checked(v, budget)
    first = np.clip((v[0] - v[1] + budget) / 2, 0, budget)
    return np.array([first, budget - first])

c = np.array([2.0, -0.6])
first_order = line_projection(np.maximum(c, 0), 1)
reverse_order = np.maximum(line_projection(c, 1), 0)
exact = simplex_projection(c, 1)
for name, point in [("C then D", first_order),
                    ("D then C", reverse_order), ("Intersection", exact)]:
    print(name, np.round(point, 5).tolist(),
          "sum", round(float(point.sum()), 5),
          "minimum", round(float(point.min()), 5))`,
    expected: String.raw`C then D [1.5, -0.5] sum 1.0 minimum -0.5
D then C [1.8, 0.0] sum 1.8 minimum 0.0
Intersection [1.0, 0.0] sum 1.0 minimum 0.0`,
  },
  projectedGradient: {
    title: "A nonzero gradient at a constrained optimum",
    code: String.raw`import numpy as np

def project(v):
    t = np.clip((v[0] - v[1] + 1) / 2, 0, 1)
    return np.array([t, 1 - t])

def gradient(x):
    return np.array([x[0] - 2, 4 * (x[1] - 0.5)])

eta = 0.2
x = np.array([0.5, 0.5])
for _ in range(30):
    x = project(x - eta * gradient(x))
g = gradient(x)
mapping = (x - project(x - eta * g)) / eta
print("Decision:", np.round(x, 6).tolist())
print("Ordinary gradient:", np.round(g, 6).tolist())
print("Mapping norm < 1e-8:", bool(np.linalg.norm(mapping) < 1e-8))
print("Feasible:", bool(x.min() >= 0 and abs(x.sum() - 1) < 1e-12))`,
    expected: String.raw`Decision: [0.8, 0.2]
Ordinary gradient: [-1.2, -1.2]
Mapping norm < 1e-8: True
Feasible: True`,
  },
  penalties: {
    title: "Finite quadratic penalty, exact hinge threshold and interior barrier",
    code: String.raw`import math

# Objective: (x-center)^2 / 2. Hard constraint: x <= bound.
def softened_optima(center, bound, strength):
    if not all(math.isfinite(v) for v in (center, bound, strength)):
        raise ValueError("Inputs must be finite")
    if center <= bound or strength <= 0:
        raise ValueError("This formula assumes center > bound and strength > 0")
    gap = center - bound
    quadratic = bound + gap / (1 + strength)
    hinge = max(bound, center - strength)
    # Stable positive root of d^2 + gap*d - strength = 0.
    d = 2 * strength / (math.sqrt(gap * gap + 4 * strength) + gap)
    barrier = bound - d
    return quadratic, hinge, barrier

for strength in (1, 2, 10):
    values = softened_optima(3, 1, strength)
    print(strength, [round(x, 6) for x in values])`,
    expected: String.raw`1 [2.0, 2, 0.585786]
2 [1.666667, 1, 0.267949]
10 [1.181818, 1, -1.316625]`,
  },
  admm: {
    title: "Run exact two-block ADMM and check both residuals",
    code: String.raw`import numpy as np

def consensus(c, budget=1.0, rho=1.0, max_steps=500,
              abs_tol=1e-4, rel_tol=1e-3):
    c = np.asarray(c, dtype=float)
    if c.shape != (2,) or not np.isfinite(c).all():
        raise ValueError("Expected a finite two-coordinate target")
    if not all(np.isfinite(v) and v > 0 for v in
               (budget, rho, abs_tol, rel_tol)):
        raise ValueError("Budget, rho and tolerances must be positive")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    x = np.full(2, budget / 2)
    z = x.copy()
    u = np.zeros(2)
    history = []
    for step in range(1, max_steps + 1):
        previous_z = z.copy()
        x = np.maximum(0, (c + rho * (z - u)) / (1 + rho))
        temporary = x + u
        z = temporary - (temporary.sum() - budget) / 2
        u = u + x - z
        primal = np.linalg.norm(x - z)
        dual = np.linalg.norm(-rho * (z - previous_z))
        eps_primal = np.sqrt(2) * abs_tol + rel_tol * max(
            np.linalg.norm(x), np.linalg.norm(z))
        eps_dual = np.sqrt(2) * abs_tol + rel_tol * np.linalg.norm(rho * u)
        passed = primal <= eps_primal and dual <= eps_dual
        history.append((step, x.copy(), z.copy(), u.copy(), primal, dual, passed))
        if passed:
            break
    return history

history = consensus([2, -0.6])
first, final = history[0], history[-1]
print("First x:", first[1].tolist(), "z:", first[2].tolist())
print("First residual norms:", round(first[4], 6), round(first[5], 6))
print("Both stopping tests:", bool(final[6]))
print("Approximate x:", np.round(final[1], 3).tolist())
print("Original x sum residual:", round(float(final[1].sum() - 1), 6))
# A passing numerical test still leaves a small original-constraint residual.`,
    expected: String.raw`First x: [1.25, 0.0] z: [1.125, -0.125]
First residual norms: 0.176777 0.883883
Both stopping tests: True
Approximate x: [1.001, 0.0]
Original x sum residual: 0.001221`,
  },
  pareto: {
    title: "Keep equivalent choices; remove strictly dominated metrics",
    code: String.raw`import math

def pareto_indices(points):
    points = [tuple(point) for point in points]
    if not points:
        return []
    width = len(points[0])
    if width == 0 or any(len(p) != width for p in points):
        raise ValueError("Objective dimensions must match and be positive")
    if any(not math.isfinite(v) for point in points for v in point):
        raise ValueError("Objectives must be finite")
    def dominates(a, b):
        return all(x <= y for x, y in zip(a, b)) and any(
            x < y for x, y in zip(a, b))
    return [i for i, p in enumerate(points)
            if not any(dominates(q, p) for q in points)]

names = ["S", "C", "M", "L", "G", "M-copy"]
points = [(12, 8), (11, 15), (8, 18), (7, 70), (13, 22), (8, 18)]
print("Nondominated:", [names[i] for i in pareto_indices(points)])
print("Empty input:", pareto_indices([]))`,
    expected: String.raw`Nondominated: ['S', 'C', 'M', 'L', 'M-copy']
Empty input: []`,
  },
  unsupported: {
    title: "Use a constraint to recover an unsupported Pareto point",
    code: String.raw`from fractions import Fraction

choices = [("S", 12, 8), ("C", 11, 15), ("M", 8, 18), ("L", 7, 70)]
# For C to beat S: 11 + 15*p <= 12 + 8*p.
# For C to beat M: 11 + 15*p <= 8 + 18*p.
upper = Fraction(12 - 11, 15 - 8)
lower = Fraction(11 - 8, 18 - 15)
print("C requires price <=", upper, "and price >=", lower)
print("Compatible:", lower <= upper)
feasible = [row for row in choices if row[2] <= 15]
answer = min(feasible, key=lambda row: (row[1], row[2]))
print("Error first, latency <= 15:", answer[0])`,
    expected: String.raw`C requires price <= 1/7 and price >= 1
Compatible: False
Error first, latency <= 15: C`,
  },
  units: {
    title: "Changing units must also change the exchange rate",
    code: String.raw`models = [("S", 12, 8), ("M", 8, 18), ("L", 7, 70)]
for label, divisor, price in [("ms", 1, 0.2),
                               ("seconds, wrong price", 1000, 0.2),
                               ("seconds, converted price", 1000, 200)]:
    scores = [(name, error + price * latency / divisor)
              for name, error, latency in models]
    print(label, [(name, round(score, 4)) for name, score in scores],
          "winner", min(scores, key=lambda row: row[1])[0])`,
    expected: String.raw`ms [('S', 13.6), ('M', 11.6), ('L', 21.0)] winner M
seconds, wrong price [('S', 12.0016), ('M', 8.0036), ('L', 7.014)] winner L
seconds, converted price [('S', 13.6), ('M', 11.6), ('L', 21.0)] winner M`,
  },
  continuous: {
    title: "Map a weight or a requirement back to an actual decision",
    code: String.raw`import math

def weighted_choice(alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    return 2 * (1 - alpha)

def epsilon_choice(epsilon):
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 4:
        raise ValueError("This example accepts epsilon from zero to four")
    return max(0, 2 - math.sqrt(epsilon))

for alpha in (0, 0.25, 0.5, 1):
    x = weighted_choice(alpha)
    print("alpha", alpha, "x", x, "objectives", (x*x, (x-2)**2))
for epsilon in (0, 0.25, 4):
    x = epsilon_choice(epsilon)
    print("epsilon", epsilon, "x", x, "objectives", (x*x, (x-2)**2))`,
    expected: String.raw`alpha 0 x 2 objectives (4, 0)
alpha 0.25 x 1.5 objectives (2.25, 0.25)
alpha 0.5 x 1.0 objectives (1.0, 1.0)
alpha 1 x 0 objectives (0, 4)
epsilon 0 x 2.0 objectives (4.0, 0.0)
epsilon 0.25 x 1.5 objectives (2.25, 0.25)
epsilon 4 x 0 objectives (0, 4)`,
  },
  commonDescent: {
    title: "Find a direction that decreases two differentiable objectives",
    code: String.raw`import numpy as np

def common_direction(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    if (first.ndim != 1 or first.size == 0 or first.shape != second.shape
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("Expected equal finite nonempty gradient vectors")
    edge = first - second
    denominator = float(edge @ edge)
    alpha = 0.0 if denominator == 0 else float(
        np.clip(-(second @ edge) / denominator, 0, 1))
    v = alpha * first + (1-alpha) * second
    return alpha, -v

for first, second in [([2, 0], [0, 2]), ([2], [-2]), ([6], [2])]:
    alpha, direction = common_direction(first, second)
    rates = [float(np.dot(g, direction)) for g in (first, second)]
    print("weight", alpha, "direction", direction.tolist(), "rates", rates)
# Zero direction is only a first-order obstruction, not a global certificate.`,
    expected: String.raw`weight 0.5 direction [-1.0, -1.0] rates [-2.0, -2.0]
weight 0.5 direction [-0.0] rates [-0.0, 0.0]
weight 0.0 direction [-2.0] rates [-12.0, -4.0]`,
  },
  measurement: {
    title: "Audit the metric before calling a decision acceptable",
    code: String.raw`import math
from statistics import mean

def nearest_rank(values, probability):
    values = sorted(values)
    if not values or not 0 < probability <= 1:
        raise ValueError("Need data and a probability in (0, 1]")
    return values[math.ceil(probability * len(values)) - 1]

latencies = [10] * 90 + [100] * 10
print("Mean ms:", mean(latencies))
print("Nearest-rank p95 ms:", nearest_rank(latencies, 0.95))
for name, groups in [("A", [0.95, 0.55]), ("B", [0.80, 0.68])]:
    print(name, "equal-size mean accuracy", round(mean(groups), 2),
          "worst-group accuracy", min(groups))`,
    expected: String.raw`Mean ms: 19
Nearest-rank p95 ms: 100
A equal-size mean accuracy 0.75 worst-group accuracy 0.55
B equal-size mean accuracy 0.74 worst-group accuracy 0.68`,
  }
};
