"""Independent exhaustive, constrained-optimization and scalar-allocation oracles."""
import itertools
import contextlib
import io
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/rate-distortion-review"
fixtures = json.loads((DIRECTORY / "model-fixtures.json").read_text())
comparisons = 0

def near(actual, expected, tolerance=2e-10):
    global comparisons
    comparisons += 1
    assert abs(actual-expected) <= tolerance * max(1, abs(expected)), (actual, expected, tolerance)

def entropy(probabilities):
    return -sum(p*math.log2(p) for p in probabilities if p)

for case in fixtures["binary"]:
    joint = np.array(case["joint"])
    source = joint.sum(axis=1)
    output = joint.sum(axis=0)
    near(source[1], case["probability"])
    information = entropy(source) + entropy(output) - entropy(joint.ravel())
    near(information, case["rate"])
    near(joint[0, 1] + joint[1, 0], case["distortion"])
    assert case["distortion"] <= case["budget"] + 1e-12

words = list(itertools.product((0, 1), repeat=3))
for case in fixtures["blocks"]:
    codebook = [words[index] for index in case["codewords"]]
    p = case["probability"]
    expected = 0
    output_mass = [0]*len(codebook)
    for index, word in enumerate(words):
        probability = math.prod(p if bit else 1-p for bit in word)
        distances = [sum(a != b for a, b in zip(word, candidate)) for candidate in codebook]
        winner = distances.index(min(distances))
        near(case["rows"][index]["mass"], probability)
        assert case["rows"][index]["index"] == winner
        expected += probability * distances[winner]/3
        output_mass[winner] += probability
    near(case["distortion"], expected)
    near(case["indexEntropy"], entropy(output_mass))
    near(case["rate"], math.ceil(math.log2(len(codebook)))/3)
    assert case["rate"] >= case["lowerBound"] - 1e-12

optimizer_checks = []
for case in fixtures["optimizers"]:
    p = np.array(case["source"])
    cost = np.array(case["costs"])
    weight = case["lambda"]
    channel = np.array(case["conditional"])
    assert np.all(channel >= 0)
    assert np.max(abs(channel.sum(axis=1)-1)) < 1e-9
    joint = p[:, None] * channel
    marginal = joint.sum(axis=0)
    information = entropy(p)+entropy(marginal)-entropy(joint.ravel())
    near(information, case["information"], 2e-8)
    near(float((joint*cost).sum()), case["distortion"])
    near(information+weight*case["distortion"], case["upper"], 2e-8)
    # Optimize the convex output-mixture objective using a distinct native solver.
    # Remove row constants before exponentiation; they do not affect its minimizer.
    offsets = cost.min(axis=1)
    kernel = np.exp(-weight*math.log(2)*(cost-offsets[:, None]))
    constant = weight*float(p@offsets)
    def objective(output):
        sums = kernel@output
        if np.any(sums[p > 0] <= 0):
            return 1e100
        return -float(p[p > 0]@np.log2(sums[p > 0]))+constant
    starts = [np.full(cost.shape[1], 1/cost.shape[1]), np.array(case["output"])]
    candidates = []
    for start in starts:
        result = minimize(objective, start, method="SLSQP",
                          bounds=[(1e-14, 1)]*cost.shape[1],
                          constraints={"type": "eq", "fun": lambda output: output.sum()-1},
                          options={"ftol": 1e-12, "maxiter": 1000})
        if result.success:
            candidates.append(float(result.fun))
    # Direct endpoint evaluations also cover boundary optima.
    candidates.extend(objective(row) for row in np.eye(cost.shape[1]))
    optimum = min(candidates)
    assert math.isfinite(optimum) and optimum < 1e90
    assert case["lower"] <= optimum + 2e-7, (case["case"], case["lower"], optimum)
    assert optimum <= case["upper"] + 2e-7, (case["case"], optimum, case["upper"])
    if case["converged"]:
        near(case["upper"], optimum, 2e-7)
    optimizer_checks.append({"case": case["case"], "lambda": weight, "scipyObjective": optimum, "reportedGap": case["gap"], "converged": case["converged"]})

allocation_checks = 0
for case in fixtures["allocations"]:
    variances = np.array(case["variances"])
    errors = np.array([component["distortion"] for component in case["components"]])
    near(float(errors.sum()), case["distortion"])
    assert np.all(errors >= 0) and np.all(errors <= variances + 1e-12)
    positive = variances[variances > 0]
    if not len(positive) or case["distortion"] >= variances.sum():
        near(case["rate"], 0)
    elif case["distortion"] == 0:
        assert case["rate"] is None  # JSON represents the intentional infinity as null.
    else:
        budget = case["distortion"]
        def objective(distortions):
            return float(0.5*np.log2(positive/distortions).sum())
        start = np.array([component["distortion"] for component in case["components"] if component["variance"] > 0])
        result = minimize(objective, start, method="SLSQP", bounds=[(1e-12, variance) for variance in positive],
                          constraints={"type": "eq", "fun": lambda d: d.sum()-budget},
                          options={"ftol": 1e-11, "maxiter": 1000})
        assert result.success, result.message
        near(case["rate"], result.fun, 3e-7)
        allocation_checks += 1

for case in fixtures["fidelity"]:
    near(sum(mass*(x-y)**2 for x, y, mass in case["pairs"]), case["distortion"])

examples = json.loads((DIRECTORY / "native/verify-input.json").read_text())
for name, example in examples.items():
    result = subprocess.run([sys.executable, "-c", example["code"]], check=True, capture_output=True, text=True, timeout=30)
    assert result.stdout.strip() == example["expected"], (name, result.stdout)
    assert not result.stderr, (name, result.stderr)

# Exercise the actual displayed functions on independent inputs, not only their stdout fixtures.
namespaces = {}
for name in ["bitstream", "gaussianAllocation"]:
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(examples[name]["code"], name, "exec"), namespace)
    namespaces[name] = namespace
encode = namespaces["bitstream"]["compress"]
decode = namespaces["bitstream"]["decompress"]
codec_cases = 0
for length in range(11):
    for bits in itertools.product("01", repeat=length):
        original = "".join(bits)
        wire = encode(original)
        restored = decode(wire)
        assert len(restored) == length
        assert len(wire) == 4 + math.ceil(math.ceil(length/3)/8)
        assert sum(a != b for a, b in zip(original, restored)) <= math.ceil(length/3)
        codec_cases += 1
nonzero_padding = bytearray(encode("011"))
nonzero_padding[-1] |= 1
for malformed in [bytes(nonzero_padding), encode("011")[:-1], encode("011")+b"\0"]:
    try:
        decode(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed byte format accepted")
allocate = namespaces["gaussianAllocation"]["allocate"]
assert allocate([9, 1], 0) == ([0.0, 0.0], [math.inf, math.inf])
assert allocate([0, 0], 0) == ([0.0, 0.0], [0.0, 0.0])
assert allocate([0, 4], 9) == ([0, 4], [0.0, 0.0])
for bad in [([], 1), ([1], -1), ([1, -1], 1), ([1, 1], math.ulp(0.0)), ([1, 1], 3*math.ulp(0.0))]:
    try:
        allocate(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid Gaussian inputs accepted")
for variances, budget in [([1, 1], 1e-250), ([0, 1], math.ulp(0.0)), ([1, 1], 2*math.ulp(0.0))]:
    errors, rates = allocate(variances, budget)
    assert math.isfinite(sum(rates))
    assert math.fsum(errors) == budget
# Independent one-bit Gaussian quantizer integral and two changed practice answers.
sigma = 3
reconstruction = sigma*math.sqrt(2/math.pi)
density = lambda x: math.exp(-x*x/(2*sigma*sigma))/(sigma*math.sqrt(2*math.pi))
quantizer_error = 2*quad(lambda x: (x-reconstruction)**2*density(x), 0, math.inf, epsabs=1e-11)[0]
near(quantizer_error, 9*(1-2/math.pi))
near(entropy([.7,.3])-entropy([.9,.1]), .41229530564141126)
near(sum(allocate([4, 1], 1.5)[1]), 1.4150374992788437)
codebook = ("000", "001", "110", "111")
source = "001011101111"
indices = [min(range(4), key=lambda j: sum(a != b for a,b in zip(source[start:start+3], codebook[j]))) for start in range(0,len(source),3)]
assert indices == [1,1,1,3]
assert int("".join(f"{index:02b}" for index in indices), 2) == 0x57
assert "".join(codebook[index] for index in indices) == "001001001111"
report = {"at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "numericComparisons": comparisons, "independentOptimizerCases": len(optimizer_checks), "optimizerChecks": optimizer_checks, "independentAllocationCases": allocation_checks, "actualCodecCases": codec_cases, "actualGaussianEndpointCases": 6, "gaussianQuantizerQuadrature": quantizer_error, "completePrograms": len(examples)}
report["actualGaussianEndpointCases"] = 11
(DIRECTORY / "native-results.json").write_text(json.dumps(report, indent=2)+"\n")
print(f"Independent oracle passed: {comparisons} numeric checks, {len(optimizer_checks)} finite optimizations, {allocation_checks} Gaussian allocations, {len(examples)} programs.")
