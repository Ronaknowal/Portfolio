"""Complementary exact design checks; this does not verify unwritten production code."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sympy as s

ROOT = Path(__file__).resolve().parents[1]
x = s.symbols("x", real=True)
counts = {}


def equal(actual, expected, name):
    assert s.simplify(actual - expected) == 0, (name, actual, expected)
    counts[name] = counts.get(name, 0) + 1


for length in [s.Rational(3, 2), s.Rational(2)]:
    for intervals in [3, 5, 9]:
        for scale in [s.Rational(1, 3), s.Rational(2)]:
            h = length / intervals
            t = x / length
            target = scale * length**2 * (t - 2*t**3 + t**4) + 2 - 3*t
            force = -s.diff(target, x, 2)
            operator = s.zeros(intervals - 1)
            for i in range(intervals - 1):
                operator[i, i] = 2 / h**2
                if i:
                    operator[i, i - 1] = -1 / h**2
                    operator[i - 1, i] = -1 / h**2
            rhs = s.Matrix([force.subs(x, j*h) for j in range(1, intervals)])
            rhs[0] += target.subs(x, 0) / h**2
            rhs[-1] += target.subs(x, length) / h**2
            stored = operator.inv() * rhs
            samples = s.Matrix([target.subs(x, j*h) for j in range(1, intervals)])
            # Odd meshes: the largest barrier value is strictly below L^2/8.
            inverse_norm = max(sum(operator.inv().row(i)) for i in range(intervals-1))
            equal(inverse_norm, length**2 * (1 - s.Rational(1, intervals**2)) / 8, "odd_grid_inverse_norm")
            for j in range(1, intervals):
                equal(stored[j-1] - samples[j-1], scale*h**2*(j*h/length)*(1-j*h/length), "scaled_nonzero_boundary_nodal_error")
            # Exact polynomial candidates, not dense graph samples, bound the reconstruction.
            all_values = [target.subs(x, 0), *stored, target.subs(x, length)]
            field_bound = s.Rational(5, 8) * scale * h**2
            for j in range(intervals):
                left, right = j*h, (j+1)*h
                line = all_values[j]*(right-x)/h + all_values[j+1]*(x-left)/h
                error = s.expand(line-target)
                candidates = [left, right]
                for root in s.nroots(s.diff(error, x)):
                    if abs(s.im(root)) < 1e-14 and float(left) < float(s.re(root)) < float(right):
                        candidates.append(s.re(root))
                assert max(abs(float(error.subs(x, point))) for point in candidates) <= float(field_bound) + 1e-12
                counts["between_node_polynomial_extrema"] = counts.get("between_node_polynomial_extrema", 0) + 1

# Fixed-grid spectral stability can coexist with lost positivity.
step = s.Matrix([[-s.Rational(1, 5), s.Rational(3, 5)], [s.Rational(3, 5), -s.Rational(1, 5)]])
assert set(step.eigenvals()) == {s.Rational(2, 5), -s.Rational(4, 5)}
assert (step*s.Matrix([1, 0]))[0] < 0
counts["spectral_stable_nonpositive_ftcs"] = 1

# Known outward Robin load: -u''=2, q_right=0, beta=3, bath=5.
nodes = [s.Rational(0), s.Rational(1, 5), s.Rational(3, 5), s.Rational(1)]
stiffness, load = s.zeros(4), s.zeros(4, 1)
for i, (left, right) in enumerate(zip(nodes, nodes[1:])):
    width = right-left
    local = s.Matrix([[1, -1], [-1, 1]]) / width
    for a in range(2):
        load[i+a] += width
        for b in range(2):
            stiffness[i+a, i+b] += local[a, b]
stiffness[0, 0] += 3
load[0] += 15
solution = stiffness.inv()*load
for value, point in zip(solution, nodes):
    equal(value, 2*point-point**2+s.Rational(17, 3), "robin_outward_sign_nodal_solution")
equal(3*(solution[0]-5), 2, "robin_outward_balance")

# Direct affine coefficients versus transformed reference gradients on changed triangles.
reference_gradients = s.Matrix([[-1, 1, 0], [-1, 0, 1]])
for vertices in [[(1, 2), (3, 2), (1, 5)], [(2, -1), (2, -3), (5, -1)], [(0, 0), (2, 1), (1, 3)]]:
    coordinate_matrix = s.Matrix([[1, *point] for point in vertices])
    affine_coefficients = coordinate_matrix.inv()
    gradients = affine_coefficients[1:, :]
    origin = s.Matrix(vertices[0])
    jacobian = s.Matrix.hstack(s.Matrix(vertices[1])-origin, s.Matrix(vertices[2])-origin)
    transformed = jacobian.inv().T*reference_gradients
    assert gradients == transformed
    area = abs(jacobian.det())/2
    element = area*gradients.T*gradients
    assert element*s.ones(3, 1) == s.zeros(3, 1)
    values = s.Matrix([3+2*a-4*b for a, b in vertices])
    equal((values.T*element*values)[0], 20*area, "triangle_affine_energy")

# A Neumann pin can hide a violated original row: retain the omitted equation.
neumann = s.Matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
incompatible = s.Matrix([1, 0, 0])
pin_solution = s.Matrix([0, 0, 0])
assert (neumann*pin_solution-incompatible)[0] == -1
assert sum(incompatible) != 0
counts["pin_does_not_repair_neumann_compatibility"] = 1

paths = ["docs/teaching/NUMERICAL-PDES-LESSON-DESIGN.md", "src/learn/data/curriculum/blueprints/numerical-pdes-grids-finite-elements-stability.js"]
record = {
    "checkedAt": datetime.now(timezone.utc).isoformat(),
    "passed": True,
    "scope": "Independent assessment of root-authored proposed fixtures; production implementation does not exist yet.",
    "counts": counts,
    "sourceFiles": [{"path": path, "sha256": hashlib.sha256((ROOT/path).read_bytes()).hexdigest()} for path in paths],
    "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "sympyVersion": s.__version__,
}
destination = ROOT/"scratch/numerical-pde-design/independent-results.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
print(json.dumps(record, indent=2))
