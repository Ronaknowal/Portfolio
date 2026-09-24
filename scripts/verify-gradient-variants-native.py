"""Independent explicit-history, finite-population and matrix-power oracles."""
import ast
import contextlib
import datetime
import io
import itertools
import json
import math
from fractions import Fraction
import sys
from pathlib import Path
import numpy as np

folder = Path(sys.argv[1])
cases = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
programs = json.loads((folder / "programs.json").read_text(encoding="utf-8"))
spaces = {}
for name, example in programs.items():
    stream, space = io.StringIO(), {}
    with contextlib.redirect_stdout(stream):
        exec(compile(example["code"], name, "exec"), space)
    assert stream.getvalue().strip() == example["expected"].strip(), name
    spaces[name] = space

counts = {"programs": len(programs), "batch": 0, "momentum_frames": 0,
          "adaptive_frames": 0, "decay_frames": 0, "layer_blocks": 0,
          "actual_optimizer_steps": 0, "actual_block_steps": 0,
          "invalid_native": 0, "practice_groups": 6}


def close(actual, expected, tolerance=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def weighted(history, beta):
    """Direct geometric weights, independent of recursive EMA implementation."""
    values = np.array(history, dtype=float)
    powers = np.arange(len(values) - 1, -1, -1)
    return np.sum((1 - beta) * beta**powers[:, None] * values, axis=0)


measurements = np.array([-3., -1., 1., 3.])
for case in cases["batch"]:
    theta = case["theta"]
    individual = theta - measurements
    selected = case["selected"]
    batch = sum(individual[selected]) / len(selected)
    means = [np.mean(individual[list(indices)]) for indices in itertools.combinations(range(4), len(selected))]
    used = batch if case["reduction"] == "mean" else batch * len(selected)
    updated = theta - case["rate"] * used
    close(case["gradients"], individual)
    close([case["meanGradient"], case["expectedMean"], case["variance"]], [batch, np.mean(means), np.var(means)])
    close([case["next"], case["nextFullLoss"]], [updated, np.mean((updated - measurements)**2) / 2])
    counts["batch"] += 1

for case in cases["momentum"]:
    rate, beta = case["rate"], case["beta"]
    for frame in case["frames"]:
        step = frame["step"]
        wanted = []
        for curvature, initial in ((1, 2), (case["curvature"], 1)):
            if case["method"] == "sgd":
                wanted.append((1 - rate*curvature)**step * initial)
            else:
                factor = 1 - rate*curvature if case["method"] == "nesterov" else 1
                matrix = np.array([[1 - rate*curvature, -rate*beta*factor], [curvature, beta*factor]])
                wanted.append((np.linalg.matrix_power(matrix, step) @ [initial, 0])[0])
        close(frame["theta"], wanted, 3e-9)
        close(frame["loss"], (wanted[0]**2 + case["curvature"]*wanted[1]**2)/2, 3e-9)
        counts["momentum_frames"] += 1
    assert max(abs(value) for frame in case["frames"] for value in frame["theta"]) <= case["bound"]

for case in cases["adaptive"]:
    for index, frame in enumerate(case["frames"]):
        history = np.array(case["gradients"][:index + 1])
        squares = history**2
        second = squares.sum(axis=0) if case["method"] == "adagrad" else weighted(squares, case["beta2"])
        numerator = history[-1]
        scaled_second = second
        if case["method"] == "adam":
            first = weighted(history, case["beta1"])
            close(frame["first"], first)
            numerator = first / (1 - case["beta1"]**len(history)) if case["correction"] else first
            if case["correction"]:
                scaled_second = second / (1 - case["beta2"]**len(history))
        denominator = np.sqrt(scaled_second) + case["epsilon"]
        close(frame["second"], second)
        close(frame["numerator"], numerator)
        close(frame["denominator"], denominator)
        close(frame["displacement"], -case["rate"] * numerator / denominator)
        counts["adaptive_frames"] += 1

for case in cases["decay"]:
    initial = np.array(case["initial"]["theta"], dtype=float)
    data_gradient = np.array(case["initial"]["gradient"], dtype=float)
    for method in case["methods"]:
        history = []
        theta = initial.copy()
        for frame in method["frames"]:
            current = data_gradient + case["decay"] * theta if method["method"] == "coupled" else data_gradient.copy()
            history.append(current)
            first = weighted(history, 0.9) / (1 - 0.9**len(history))
            second = weighted(np.array(history)**2, 0.99) / (1 - 0.99**len(history))
            direction = first / (np.sqrt(second) + 1e-8)
            if method["method"] == "adamw":
                theta = theta * (1 - case["rate"] * case["decay"])
            theta = theta - case["rate"] * direction
            close(frame["theta"], theta)
            counts["decay_frames"] += 1

for case in cases["layer"]:
    for block in case["blocks"]:
        theta, gradient = map(np.array, (block["theta"], block["gradient"]))
        size = math.sqrt(sum(value*value for value in theta))
        base = gradient / (abs(gradient) + 1e-8) if case["method"] == "lamb" else gradient
        direction = base + case["decay"] * theta
        denominator = (math.sqrt(sum(value*value for value in gradient)) + case["decay"] * size
                       if case["method"] == "lars" else math.sqrt(sum(value*value for value in direction)))
        ratio = 1 if case["method"] == "sgd" or not size or not denominator else size / denominator
        if case["method"] == "lars" and size and denominator:
            ratio *= case["trustCoefficient"]
        update = -case["rate"] * ratio * direction
        close(block["ratio"], ratio)
        close(block["displacement"], update)
        close(block["next"], theta + update)
        if size:
            close(block["relativeUpdate"], np.linalg.norm(update)/size)
        else:
            assert block["relativeUpdate"] is None
        counts["layer_blocks"] += 1

# Actual helper definitions must be identical in every independently runnable copy.
classes = []
for name in ("adaptive", "decay", "experiment", "resume"):
    definition = next(node for node in ast.parse(programs[name]["code"]).body if isinstance(node, ast.ClassDef))
    classes.append(ast.dump(definition, include_attributes=False))
assert len(set(classes)) == 1
Optimizer = spaces["adaptive"]["SmallOptimizer"]
rng = np.random.default_rng(90113)
for method in ("sgd", "momentum", "adagrad", "rmsprop", "adam", "adamw"):
    for _ in range(35):
        initial = rng.normal(size=5)
        rate, beta1, beta2, epsilon, decay = 0.03, 0.7, 0.93, 0.0001, 0.04
        optimizer = Optimizer(initial, method, rate, beta1, beta2, epsilon, decay)
        theta = initial.copy()
        history = []
        for step in range(1, 11):
            data = rng.choice([-3.0, 0.0, 0.0, 1.0, 5.0], size=5)
            gradient = data if method == "adamw" else data + decay * theta
            history.append(gradient.copy())
            if method == "momentum":
                direction = weighted(history, beta1) / (1 - beta1)
            elif method == "adagrad":
                direction = gradient / (np.sqrt(np.sum(np.array(history)**2, axis=0)) + epsilon)
            elif method == "rmsprop":
                direction = gradient / (np.sqrt(weighted(np.array(history)**2, beta2)) + epsilon)
            elif method in ("adam", "adamw"):
                first = weighted(history, beta1)/(1 - beta1**step)
                second = weighted(np.array(history)**2, beta2)/(1 - beta2**step)
                direction = first/(np.sqrt(second) + epsilon)
            else:
                direction = gradient
            if method == "adamw":
                theta *= 1 - rate * decay
            theta -= rate * direction
            close(optimizer.step(data), theta)
            assert optimizer.t == step
            counts["actual_optimizer_steps"] += 1

block_step = spaces["blocks"]["block_step"]
for method in ("lars", "lamb"):
    for _ in range(100):
        theta, gradient, first, velocity = rng.normal(size=(4, 7))
        second = rng.uniform(0, 3, size=7)
        actual = block_step(theta, gradient, first, second, velocity, 4, method,
                            lr=.2, beta1=.6, beta2=.8, decay=.3, coefficient=.07, epsilon=.001)
        if method == "lars":
            ratio = .07*np.linalg.norm(theta)/(np.linalg.norm(gradient) + .3*np.linalg.norm(theta))
            expected = theta - .6*velocity - .2*ratio*(gradient + .3*theta)
        else:
            first_next = .6*first + .4*gradient
            second_next = .8*second + .2*gradient**2
            direction = first_next/(1 - .6**4)/(np.sqrt(second_next/(1 - .8**4)) + .001) + .3*theta
            expected = theta - .2*np.linalg.norm(theta)*direction/np.linalg.norm(direction)
        close(actual[0], expected)
        counts["actual_block_steps"] += 1

invalid = [lambda: Optimizer([], "adam"), lambda: Optimizer([1], "unknown"),
           lambda: Optimizer([1], lr=0), lambda: Optimizer([1], beta1=1),
           lambda: Optimizer([1], beta2=-1), lambda: Optimizer([1], epsilon=0),
           lambda: Optimizer([1], decay=-1), lambda: Optimizer([math.nan]),
           lambda: Optimizer([[1]]), lambda: Optimizer([1]).step([1, 2]),
           lambda: Optimizer([1]).step([math.inf]),
           lambda: block_step([1], [1], [0], [0], [0], True, "lars"),
           lambda: block_step([1], [1], [0], [-1], [0], 1, "lamb"),
           lambda: block_step([1], [1, 2], [0], [0], [0], 1, "lamb")]
for function in invalid:
    try:
        function()
        raise AssertionError("Expected invalid input to fail")
    except ValueError:
        counts["invalid_native"] += 1

for method in ("sgd", "momentum", "adagrad", "rmsprop", "adam", "adamw"):
    optimizer = Optimizer([1.0], method, lr=10.0)
    saved = (optimizer.theta.copy(), optimizer.first.copy(), optimizer.second.copy(), optimizer.t)
    try:
        optimizer.step([1e308])
        raise AssertionError("Extreme float64 arithmetic must fail visibly")
    except FloatingPointError:
        for current, previous in zip((optimizer.theta, optimizer.first, optimizer.second), saved[:3]):
            np.testing.assert_array_equal(current, previous)
        assert optimizer.t == saved[3]
        counts["invalid_native"] += 1

# Hand-practice calculations are independently recomputed, not copied outputs.
loss = lambda theta: sum((theta - value)**2 for value in [0, 2, 7])/6
close([loss(2), loss(1.8)], [29/6, 30.32/6])
theta, buffer = 2.0, 0.0
for _ in range(2):
    buffer = .5*buffer + 3*theta
    theta -= .1*buffer
close(theta, .68)
close(-.1 * (weighted([[3], [0]], .5)/.75)/(np.sqrt(weighted([[9], [0]], .5)/.75)), [-.1/math.sqrt(3)])
close(.95*np.array([1, 4]), [.95, 3.8])
assert (2*Fraction(3) + Fraction(-2))/3 == Fraction(4, 3)
result = {"checkedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "python": sys.version.split()[0], "numpy": np.__version__, "status": "passed", "counts": counts}
(folder / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
