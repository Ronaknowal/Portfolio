"""Run the actual exported programs and test fresh inputs independently."""
import contextlib
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/dynamical-systems-review"
export = subprocess.run(["node", "--input-type=module", "-e",
    "import {dynamicalSystemsExamples as e} from './src/learn/data/dynamical-systems-examples.js';console.log(JSON.stringify(e));"],
    cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=True)
examples = json.loads(export.stdout)
spaces = {}
comparisons = 0
cases = {}

def close(actual, expected, tolerance=1e-10):
    global comparisons
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    comparisons += int(np.asarray(actual).size)

for key, example in examples.items():
    program = OUT / ("displayed-" + key + ".py")
    program.write_text(example["code"], encoding="utf-8")
    run = subprocess.run([sys.executable, str(program)], text=True, encoding="utf-8",
                         capture_output=True, timeout=30, check=True)
    assert run.stdout.rstrip("\n") == example["expected"], key
    assert run.stderr == "", (key, run.stderr)
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], str(program), "exec"), namespace)
    spaces[key] = namespace
original = json.loads((ROOT / "docs/teaching/evidence/dynamical-systems-original-content.json").read_text(encoding="utf-8"))
assert examples["original"]["code"] == original["blocks"][0]["code"]
assert examples["original"]["expected"] == original["blocks"][1]["code"]

cases["cooling"] = 0
for rate, start, step in itertools.product([0, 0.125, 1, 3], [-2, 0, 0.7], [0.125, 0.5, 1]):
    count, numerical, exact = spaces["cooling"]["cooling_comparison"](rate, start, 4, step)
    close(numerical, float(Fraction(start) * (1 - Fraction(rate) * Fraction(step))**count), 2e-10)
    close(exact, start * math.exp(-4 * rate))
    cases["cooling"] += 1

cases["cubic"] = 0
for start in [-1.5, -0.1, 0, 0.7]:
    for time in [0.1, 0.7, 2]:
        actual = spaces["scalar"]["cubic_solution"](start, time)
        reference = solve_ivp(lambda t, y: -y**3, [0, time], [start], method="DOP853", rtol=1e-12, atol=1e-14)
        close(actual, reference.y[0, -1], 2e-10)
        cases["cubic"] += 1

cases["linear"] = 0
for matrix in [np.array([[-0.3, -1], [1, -0.3]]), np.array([[-2, 7], [0, -0.5]]), np.array([[0.3, 0], [0, -0.4]])]:
    for start in [[0.2, -0.7], [1, 0], [0, 0]]:
        actual = spaces["transient"]["linear_state"](matrix, start, 2.7)
        reference = solve_ivp(lambda t, state: matrix @ state, [0, 2.7], start, method="DOP853", rtol=2e-13, atol=1e-14)
        close(actual, reference.y[:, -1], 3e-10)
        cases["linear"] += 1

cases["hopf"] = 0
for parameter, radius, time in itertools.product([-0.7, -1e-9, -5e-324, 0, 5e-324, 1e-9, 0.4, 1], [0, 0.13, 1.7], [0.3, 2, 12]):
    actual = spaces["hopf"]["radius_at"](parameter, radius, time)
    reference = solve_ivp(lambda t, state: parameter * state - state**3, [0, time], [radius], method="DOP853", rtol=2e-13, atol=1e-14)
    close(actual, reference.y[0, -1], 3e-10)
    cases["hopf"] += 1

cases["logistic"] = 0
for growth, start in itertools.product([0, 0.5, 1, 2.7, 3.2, 3.83, 4], [0, 0.125, 0.37, 1]):
    values = spaces["cycles"]["logistic_values"](growth, start, 8)
    exact = Fraction(start)
    for value in values:
        close(value, float(exact), 1e-11)
        exact = Fraction(growth) * exact * (1 - exact)
    cases["logistic"] += 1
for growth in [3.01, 3.2, 3.7, 4]:
    lower, upper = spaces["cycles"]["two_cycle"](growth)
    close(growth * lower * (1 - lower), upper)
    close(growth * upper * (1 - upper), lower)
    close(growth**2 * (1 - 2 * lower) * (1 - 2 * upper), 4 + 2 * growth - growth**2)
assert spaces["exponent"]["finite_exponent"](4, 0.5, 0, 6) == -math.inf
close(spaces["exponent"]["finite_exponent"](4, 0.75, 0, 57), math.log(2))
for count in [4, 8, 16]:
    start, growth = 0.31, 3.2
    values = spaces["cycles"]["logistic_values"](growth, start, count)
    derivative = math.prod(growth * (1 - 2 * value) for value in values[:-1])
    close(spaces["exponent"]["finite_exponent"](growth, start, 0, count), math.log(abs(derivative)) / count)

cases["tent_words"] = 0
for length in range(1, 7):
    intervals = []
    for word in itertools.product("LR", repeat=length):
        word = "".join(word)
        interval, periodic = spaces["tent"]["inverse_word"](word)
        intervals.append(interval)
        assert interval[1] - interval[0] == Fraction(1, 2**length)
        state = periodic
        for _ in range(length):
            state = 2 * min(state, 1 - state)
        assert state == periodic
        assert interval[0] <= periodic <= interval[1]
        cases["tent_words"] += 1
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == 1
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))

cases["lorenz"] = 0
for rho, initial in itertools.product([0.7, 8, 28], [[-2, 3, 7], [0, 0, 0], [3, -1, 12]]):
    actual = spaces["lorenz"]["solve_lorenz"](initial, rho, 0.6, 1e-10)
    def independent_field(time, state):
        matrix = np.array([[-10, 10, 0], [rho - state[2], -1, 0], [state[1], 0, -8/3]])
        return matrix @ state
    reference = solve_ivp(independent_field, [0, 0.6], initial, t_eval=actual.t, method="Radau", rtol=3e-12, atol=1e-13, max_step=0.002)
    close(actual.y, reference.y, 2e-8)
    cases["lorenz"] += 1

cases["oscillator"] = 0
for step, count, start in itertools.product([0.03, 0.125, 0.4, 0.5], [0, 7, 40, 100], [[1, 0], [0.2, -0.7], [0, 0]]):
    actual = spaces["energy"]["integrate_oscillator"](step, count, start)
    forward = np.linalg.matrix_power(np.array([[1, step], [-step, 1]]), count) @ start
    symplectic = np.linalg.matrix_power(np.array([[1-step**2, step], [-step, 1]]), count) @ start
    metric = np.array([[1, -step/2], [-step/2, 1]])
    expected = [forward @ forward / 2, symplectic @ symplectic / 2, np.asarray(start) @ metric @ start / 2]
    close(actual, expected, 2e-10)
    cases["oscillator"] += 1

invalid = [
    lambda: spaces["cooling"]["cooling_comparison"](-1, 1, 1, 0.1),
    lambda: spaces["cooling"]["cooling_comparison"](1, 1, 1, 0.3),
    lambda: spaces["scalar"]["cubic_solution"](1, 1, False),
    lambda: spaces["transient"]["linear_state"]([[1]], [1], 1),
    lambda: spaces["hopf"]["radius_at"](2, 1, 1),
    lambda: spaces["cycles"]["logistic_values"](5, 0.2, 5),
    lambda: spaces["cycles"]["two_cycle"](3),
    lambda: spaces["exponent"]["finite_exponent"](3.2, 0.2, 0, 0),
    lambda: spaces["tent"]["inverse_word"]("BAD"),
    lambda: spaces["tent"]["tent"](Fraction(3, 2)),
    lambda: spaces["lorenz"]["solve_lorenz"]([1, 1, 1], 29, 1, 1e-9),
    lambda: spaces["energy"]["integrate_oscillator"](0.6, 10),
    lambda: spaces["cooling"]["cooling_comparison"](1, 1, 1, 1e-300),
    lambda: spaces["scalar"]["cubic_solution"](1e200, 0),
    lambda: spaces["transient"]["linear_state"]([[1e300, 0], [0, 1]], [1, 0], 1),
]
for fail in invalid:
    try:
        fail()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a domain rejection")

# The atlas is a computed finite floating-point trace: independently reproduce
# its declared grid, burn and retained counts, not an infinite-time orbit claim.
models = json.loads((OUT / "model-fixtures.json").read_text())
atlas = models["atlas"]
for index, row in enumerate(atlas["rows"]):
    growth = atlas["minimum"] + (atlas["maximum"] - atlas["minimum"]) * index / (atlas["columns"] - 1)
    close(row["growth"], growth)
    values = spaces["cycles"]["logistic_values"](growth, atlas["initial"], atlas["burn"] + atlas["retained"])
    close(row["values"], values[atlas["burn"] + 1:], 1e-10)

result = dict(verifiedAt=datetime.now(timezone.utc).isoformat(),
    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
    displayedPrograms=len(examples), originalCodeAndOutputPreserved=True,
    comparisons=comparisons, cases=cases, invalidCases=len(invalid), atlasColumns=len(atlas["rows"]),
    examplesSha256=hashlib.sha256((ROOT / "src/learn/data/dynamical-systems-examples.js").read_bytes()).hexdigest(),
    limits="Fresh bounded native inputs and short-horizon solvers; finite atlas reproduction, not an asymptotic chaos theorem or arbitrary-input certification.")
(OUT / "native-independent-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
