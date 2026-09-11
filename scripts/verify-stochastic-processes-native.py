"""Independent finite enumeration, linear algebra, quadrature and native-program checks."""
import contextlib
import io
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import ndtr
from scipy.stats import norm, poisson, binom

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/stochastic-processes-native-verification"
data = json.loads((DIRECTORY / "fixtures.json").read_text(encoding="utf-8"))
errors = []


def close(actual, expected, atol=2e-10, rtol=2e-9):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)
    errors.extend(np.abs(actual-expected).flat)


spaces = {}
for key, example in data["examples"].items():
    file = DIRECTORY / (key + ".py")
    file.write_text(example["code"], encoding="utf-8")
    run = subprocess.run([sys.executable, str(file)], capture_output=True, text=True, check=True)
    assert run.stdout.rstrip() == example["expected"].rstrip(), (key, run.stdout, example["expected"])
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], str(file), "exec"), namespace)
    spaces[key] = namespace

for law in data["laws"]:
    length = law["length"]
    paths = list(itertools.product([0, 1], repeat=length))
    if law["kind"] == "frozen":
        paths = [tuple([start]*length) for start in [0, 1]]
    elif law["kind"] == "alternating":
        paths = [tuple((start+i) % 2 for i in range(length)) for start in [0, 1]]
    assert [tuple(row["values"]) for row in law["paths"]] == sorted(paths)
    close(law["marginalOne"], np.mean(paths, axis=0))
    close(law["adjacentEqual"], sum(path[0] == path[1] for path in paths)/len(paths))

for state in data["markov"]:
    a, b = state["a"], state["b"]
    matrix = np.array([[1-a, a], [b, 1-b]])
    initial = [state["initialSunny"], 1-state["initialSunny"]]
    for row in state["history"]:
        close(row["distribution"], np.array(initial) @ np.linalg.matrix_power(matrix, row["step"]))
        if row["step"]:
            previous = np.array(initial) @ np.linalg.matrix_power(matrix, row["step"]-1)
            close(row["flows"], previous[:, None]*matrix)
        close(sum(row["distribution"]), 1)
    if state["stationary"] is not None:
        close(np.array(state["stationary"]) @ matrix, state["stationary"])
        close(state["stationary"], [b/(a+b), a/(a+b)])
    else:
        assert a == b == 0
    for n in range(5):
        totals = [F(0), F(0)]
        for path in itertools.product(range(2), repeat=n+1):
            mass = F(str(initial[path[0]]))
            for i, j in zip(path, path[1:]):
                mass *= F(str(matrix[i, j]))
            totals[path[-1]] += mass
        close(state["history"][n]["distribution"], list(map(float, totals)))

for state in data["absorption"]:
    boundary, p, start = state["boundary"], F(str(state["upward"])), state["start"]
    if p == F(1, 2):
        success = [F(i, boundary) for i in range(boundary+1)]
        duration = [F(i*(boundary-i)) for i in range(boundary+1)]
    else:
        ratio = (1-p)/p
        success = [(1-ratio**i)/(1-ratio**boundary) for i in range(boundary+1)]
        duration = [(i-boundary*success[i])/(1-2*p) for i in range(boundary+1)]
    close(state["success"], list(map(float, success)))
    close(state["meanSteps"], list(map(float, duration)))
    q = np.asarray(state["q"])
    close(state["visits"], np.linalg.inv(np.eye(boundary-1)-q))
    close(np.asarray(state["visits"]).sum(axis=1), list(map(float, duration[1:-1])))
    # Enumerate stopped paths; no copy of the model's matrix-propagation update.
    for steps in range(7):
        distribution = [F(0)]*(boundary+1)
        first = [F(0), F(0)]
        for moves in itertools.product([-1, 1], repeat=steps):
            position, weight, hit = start, F(1), 0 if start in [0, boundary] else None
            for step, move in enumerate(moves, 1):
                weight *= p if move == 1 else 1-p
                if position not in [0, boundary]:
                    position += move
                    if position in [0, boundary]:
                        hit = step
            distribution[position] += weight
            if hit == steps and position in [0, boundary]:
                first[int(position == boundary)] += weight
        row = state["history"][steps]
        close(row["distribution"], list(map(float, distribution)))
        close([row["firstLower"], row["firstUpper"]], list(map(float, first)))
        close(row["surviving"], float(sum(distribution[1:-1])))

for state in data["arrivals"]:
    r0, r1 = state["rates"]
    change, horizon = state["switchTime"], state["horizon"]
    def intensity(t):
        return quad(lambda x: r0 if x < change else r1, 0, t,
                    points=[change] if 0 < change < t else None)[0]
    close(state["totalIntensity"], intensity(horizon))
    left, right = state["interval"]
    mean = intensity(right)-intensity(left)
    close(state["intervalMean"], mean)
    close(state["zeroProbability"], poisson.pmf(0, mean))
    close(state["pmf"], poisson.pmf(np.arange(41), mean))
    close(state["pmfTail"], poisson.sf(40, mean), atol=1e-300, rtol=2e-10)
    last = 0
    for event in state["events"]:
        assert last < event["time"] <= horizon
        close(intensity(event["time"]), event["unitTime"])
        close(event["gap"], event["time"]-last)
        if state["routing"] == "alternating":
            assert event["mark"] == (event["index"]-1) % 2
        last = event["time"]
    if right <= state["observedUntil"]:
        selected = [event for event in state["events"] if left < event["time"] <= right]
        assert state["intervalCount"] == len(selected)
        assert state["markedCounts"] == [sum(event["mark"] == j for event in selected) for j in [0, 1]]
    else:
        assert state["intervalCount"] is None and state["markedCounts"] is None
    if state["complete"]:
        close(state["observedUntil"], horizon)

for row in data["split"]:
    mean, a, b, p = row["mean"], row["a"], row["b"], row["probability"]
    possible = mean > 0 or a+b == 0
    assert row["state"]["conditioningEventPossible"] == possible
    close(row["state"]["markKernel"], binom.pmf(a, a+b, p))
    if possible:
        close(row["state"]["conditional"], binom.pmf(a, a+b, p))
    else:
        assert row["state"]["conditional"] is None
    close(row["state"]["joint"], poisson.pmf(a+b, mean)*binom.pmf(a, a+b, p))
    close(row["state"]["product"], poisson.pmf(a, mean*p)*poisson.pmf(b, mean*(1-p)))

for state in data["clocks"]:
    alpha, beta = state["alpha"], state["beta"]
    generator = np.array([[-alpha, alpha], [beta, -beta]])
    close(np.asarray(state["generator"]).sum(axis=1), [0, 0])
    for row in state["history"]:
        close(row["probabilityOn"], expm(generator*row["time"])[state["initial"], 0])
    integral = quad(lambda t: expm(generator*t)[state["initial"], 0], 0, state["horizon"])[0]
    close(state["expectedOnExposure"], integral)
    close(state["stationaryOn"], beta/(alpha+beta))
    last, value = 0, state["initial"]
    exposure, departures = [0, 0], [0, 0]
    for segment in state["segments"]:
        assert segment["state"] == value
        close(segment["start"], last)
        assert segment["holdingTime"] > 0
        close(segment["end"], min(state["horizon"], last+segment["holdingTime"]))
        exposure[value] += segment["end"]-last
        departures[value] += segment["jumped"]
        value, last = 1-value, segment["end"]
    close(state["exposure"], exposure)
    assert state["departures"] == departures
    close(sum(exposure), state["observedUntil"])

for state in data["brownian"]:
    horizon, drift, scale, n = state["horizon"], state["drift"], state["scale"], state["count"]
    increments = np.diff(state["selected"])
    close(increments, state["increments"])
    close(sum(increments), state["selected"][-1])
    close(state["rawVariation"], np.dot(increments, increments))
    centered = increments-drift*horizon/n
    close(state["centeredVariation"], np.dot(centered, centered))
    mean, sd = drift*horizon/n, scale*math.sqrt(horizon/n)
    # Independently integrate Gaussian fourth/second moments.
    second = quad(lambda z: (mean+sd*z)**2*norm.pdf(z), -12, 12)[0]
    fourth = quad(lambda z: (mean+sd*z)**4*norm.pdf(z), -12, 12)[0]
    close(state["rawVariationExpectation"], n*second)
    close(state["rawVariationVariance"], n*(fourth-second**2))
    close(state["centeredVariationExpectation"], scale**2*horizon)
    close(state["centeredVariationVariance"], 2*scale**4*horizon**2/n)
    times = np.array(state["covarianceTimes"])
    close(state["covariance"], scale**2*np.minimum.outer(times, times))
    close(state["terminalMean"], drift*horizon)
    close(state["terminalVariance"], scale**2*horizon)
for first, second, third in zip(data["brownian"][::3], data["brownian"][1::3], data["brownian"][2::3]):
    close(first["paths"], np.asarray(third["paths"])[:, ::64])
    close(second["paths"], np.asarray(third["paths"])[:, ::16])
for row in data["basis"]:
    transform = np.asarray(row["paths"]).T
    times = np.linspace(0, row["horizon"], row["count"]+1)
    close(transform @ transform.T, row["scale"]**2*np.minimum.outer(times, times))

for state in data["bridges"]:
    x, y, t, s, u, b = [state[key] for key in ["left", "right", "duration", "scale", "fraction", "barrier"]]
    covariance = s*s*np.array([[u*t, u*t], [u*t, t]])
    coefficient = covariance[0, 1]/covariance[1, 1]
    close(state["mean"], x+coefficient*(y-x))
    close(state["variance"], covariance[0, 0]-coefficient*covariance[1, 0])
    log_ratio = 0 if b <= max(x, y) else norm.logpdf(2*b-y-x, scale=s*math.sqrt(t))-norm.logpdf(y-x, scale=s*math.sqrt(t))
    close(state["logCrossing"], log_ratio, atol=2e-8)
    expected = math.exp(log_ratio)
    close(state["crossingProbability"], expected, atol=1e-300, rtol=1e-8)
    assert state["probabilityBelowFloatingPointRange"] == (expected == 0)

for row in data["transitions"]:
    counts = np.zeros((3, 3), int)
    for path in row["paths"]:
        for i, j in zip(path, path[1:]):
            counts[i, j] += 1
    close(row["state"]["counts"], counts)
    for index, total in enumerate(counts.sum(axis=1)):
        expected = None if total == 0 else counts[index]/total
        if expected is None:
            assert row["state"]["estimate"][index] is None
        else:
            close(row["state"]["estimate"][index], expected)

# Changed inputs call the actual displayed program helpers, not copied implementations.
native_changed = 0
for boundary in range(3, 9):
    for p in [F(1, 4), F(1, 2), F(3, 4)]:
        result = spaces["absorption"]["reserve"](boundary, p)
        assert len(result) == 2
        success, duration = result
        for i in range(1, boundary):
            assert success[i] == (1-p)*success[i-1]+p*success[i+1]
            assert duration[i] == 1+(1-p)*duration[i-1]+p*duration[i+1]
        native_changed += 1
for n in [2, 4, 8, 16]:
    normals = [(-1)**i*(i+1)/10 for i in range(n)]
    path = spaces["brownianGrid"]["brownian_path"](normals, 2, 0.3, 0.7)
    close(path, np.r_[0, np.cumsum(0.3*2/n+0.7*math.sqrt(2/n)*np.array(normals))])
    native_changed += 1
for args in [(0, 0, 1, 1, .5, 1), (.2, -.1, .5, .8, .25, .7), (-1, 2, 3, 2, .8, 1)]:
    mean, variance, logp = spaces["bridgeVariation"]["bridge"](*args)
    x, y, t, s, u, b = args
    close(mean, (1-u)*x+u*y)
    close(variance, s*s*t*u*(1-u))
    close(logp, 0 if b <= max(x, y) else -2*(b-x)*(b-y)/(s*s*t))
    native_changed += 1
close(spaces["diagnosis"]["fitted_rate"]([.2,.4,1.1,1.5,2.2,2.8], 3), 2)
native_changed += 1
native_invalid = [
    lambda: spaces["markov"]["propagate"]([[F(1)]], [F(1)], -1),
    lambda: spaces["markov"]["propagate"]([[F(1)]], [F(1)], 1.5),
    lambda: spaces["markov"]["propagate"]([[F(1, 2)]], [F(1)], 1),
    lambda: spaces["markov"]["fit_transitions"]([[0]], 0),
    lambda: spaces["markov"]["fit_transitions"]([[3]], 3),
    lambda: spaces["markov"]["fit_transitions"]([], 3),
    lambda: spaces["absorption"]["reserve"](1, F(1, 2)),
    lambda: spaces["absorption"]["reserve"](4, F(0)),
    lambda: spaces["arrivals"]["event_times"](-1, 3, 0),
    lambda: spaces["arrivals"]["event_times"](1, 0, 0),
    lambda: spaces["arrivals"]["event_times"](math.nan, 3, 0),
    lambda: spaces["intensity"]["event_clock"]([math.nan], 1, 4, 2, 3),
    lambda: spaces["intensity"]["event_clock"]([2, 1], 1, 4, 2, 3),
    lambda: spaces["intensity"]["event_clock"]([1], -1, 4, 2, 3),
    lambda: spaces["jumpClock"]["simulate"](0, 1, 3, 0),
    lambda: spaces["jumpClock"]["simulate"](1, 1, math.inf, 0),
    lambda: spaces["brownianGrid"]["brownian_path"]([], 1),
    lambda: spaces["brownianGrid"]["brownian_path"]([math.nan], 1),
    lambda: spaces["brownianGrid"]["brownian_path"]([1], 1, math.nan),
    lambda: spaces["bridgeVariation"]["bridge"](0, 0, 0, 1, .5, 1),
    lambda: spaces["bridgeVariation"]["bridge"](0, 0, 1, 1, 1.1, 1),
    lambda: spaces["bridgeVariation"]["bridge"](math.inf, 0, 1, 1, .5, 1),
    lambda: spaces["diagnosis"]["fitted_rate"]([1, 1], 3),
    lambda: spaces["diagnosis"]["fitted_rate"]([4], 3),
]
for check in native_invalid:
    try:
        check()
        raise AssertionError("An invalid native input was accepted.")
    except ValueError:
        pass
for check in [
    lambda: spaces["arrivals"]["event_times"](6, 3, 11, cap=1),
    lambda: spaces["jumpClock"]["simulate"](6, 6, 12, 11, cap=1),
]:
    try:
        check()
        raise AssertionError("An incomplete simulated window was returned as complete.")
    except RuntimeError:
        pass
# Exact original program bodies remain intact, including their preserved comments.
import re
archive = (ROOT / "scratch/stochastic-processes-design/original-lesson.jsx").read_text(encoding="utf-8")
tick = chr(96)
original = re.findall('<CodeBlock language="python">\\{' + tick + '(.*?)' + tick + '\\}</CodeBlock>', archive, re.S)
for key, code in zip(["weather", "arrivalProbability", "brownianOriginal"], original):
    assert data["examples"][key]["code"].strip() == code.strip()
tiny_mean = F(str(data["shortInterval"]["rates"][1])) * F(2)**-51
close(data["shortInterval"]["intervalMean"], float(tiny_mean), atol=0, rtol=1e-14)

result = {
    "checkedAt": datetime.now(timezone.utc).isoformat(),
    "passed": True, "programs": len(data["examples"]), "finitePathLaws": len(data["laws"]),
    "markovStates": len(data["markov"]), "absorptionStates": len(data["absorption"]),
    "arrivalStates": len(data["arrivals"]), "splittingLaws": len(data["split"]),
    "continuousClockStates": len(data["clocks"]), "brownianStates": len(data["brownian"]),
    "basisCovarianceMaps": len(data["basis"]), "conditionalBridges": len(data["bridges"]),
    "nativeChangedInputs": native_changed, "invalidBrowserModelInputs": data["invalidInputs"],
    "nativeInvalidInputs": len(native_invalid), "nativeCapRejections": 2, "originalProgramsPreserved": 3,
    "strictInteriorRepeatableUniformDraws": data["uniformDrawChecks"],
    "numericComparisons": len(errors), "maximumAbsoluteError": max(errors),
    "method": "Actual Python stdout; Fraction path enumeration/closed forms; NumPy matrix powers/inverses and covariance transforms; SciPy Poisson/binomial, expm, Gaussian quadrature and conditioning/reflected densities. Finite contracts only, no simulation claimed to prove distributional theorems.",
}
(DIRECTORY / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
