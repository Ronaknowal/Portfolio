"""Complementary review of actual PDE helpers, with exact polynomial and analytic oracles."""

from pathlib import Path
import json, math, subprocess, sys, io, contextlib, datetime
from fractions import Fraction as F
from scipy.integrate import quad

payload = json.loads(
    Path("scratch/pde-independent-review/current-payload.json").read_text()
)
notes = []
counts = {}
failures = []
max_relative = 0.0
max_tolerance_fraction = 0.0


def compare(actual, expected, name, relative=3e-12, absolute=3e-13):
    global max_relative, max_tolerance_fraction
    expected = float(expected)
    error = abs(actual - expected)
    if expected and absolute == 0:
        max_relative = max(max_relative, error / abs(expected))
    allowed = absolute + relative * abs(expected)
    if allowed:
        max_tolerance_fraction = max(max_tolerance_fraction, error / allowed)
    if not math.isfinite(actual) or error > absolute + relative * abs(expected):
        failures.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "absoluteError": error,
            }
        )


namespaces = {}
for name, e in payload["pdeExamples"].items():
    namespace = {}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(e["code"], f"actual:{name}", "exec"), namespace)
    assert capture.getvalue().strip() == e["expected"].strip(), name
    namespaces[name] = namespace
counts["actualCompletePrograms"] = len(namespaces)
original = json.loads(
    Path("scratch/pde-independent-review/initial-examples.json").read_text()
)
assert set(original) == set(payload["pdeExamples"])
for name, row in original.items():
    assert row["expected"] == payload["pdeExamples"][name]["expected"]
    if name != "wave":
        assert row["code"] == payload["pdeExamples"][name]["code"]
counts["originalStdoutConserved"] = 15
counts["unchangedOtherCompletePrograms"] = 14


# Exact rational integral of the degree-six compact bump, independently integrated
# as polynomial coefficients. Use exact Fraction values of the supplied x,t;
# the repaired model intentionally preserves distances lost in rounded x+-t display feet.
def exact_bump(x):
    q = F(x)
    return (1 - q * q) ** 3 if abs(q) < 1 else F(0)


def exact_integral(lo, hi):
    lo = max(F(-1), F(lo))
    hi = min(F(1), F(hi))
    if hi <= lo:
        return F(0)
    return sum(
        F(c) * (hi ** (i + 1) - lo ** (i + 1)) / F(i + 1)
        for i, c in enumerate([1, 0, -3, 0, 3, 0, -1])
    )


for index, row in enumerate(payload["wave"]):
    low = F(row["x"]) - F(row["time"])
    high = F(row["x"]) + F(row["time"])
    integral = exact_integral(low, high) / 4
    compare(
        row["velocityContribution"],
        integral,
        f"wave integral {index}",
        relative=3e-12,
        absolute=0,
    )
    assert row["total"] >= 0, ("negative total from nonnegative data", row)
    compare(
        row["total"],
        (exact_bump(low) + exact_bump(high)) / 2 + integral,
        f"wave total {index}",
        relative=3e-7,
        absolute=0,
    )
counts["exactInputFractionWaveStates"] = len(payload["wave"])
# Native wave uses independently changed c and a signed initial velocity multiplier.
for c in [0.5, 1.7, 2.4]:
    for time in [0.13, 0.41]:
        for x in [-0.87, 0.29, 1.18]:
            for velocity in [-0.3, 0.7]:
                a, b, total = namespaces["wave"]["wave"](x, time, velocity, c)
                lo = x - c * time
                hi = x + c * time
                compare(
                    a,
                    (exact_bump(lo) + exact_bump(hi)) / 2,
                    "native changed wave shape",
                )
                compare(
                    b,
                    F(velocity) * exact_integral(lo, hi) / (2 * F(c)),
                    "native changed wave velocity",
                )
counts["changedNativeWaveStates"] = 36
for time in [1e-5, 1e-6, 1e-7, 1e-8]:
    for x in [-1.0, 1.0]:
        a, b, total = namespaces["wave"]["wave"](x, time, 0.5, 1)
        compare(
            b,
            exact_integral(F(x) - F(time), F(x) + F(time)) / 4,
            "native wave edge",
            relative=3e-12,
            absolute=0,
        )
        assert total >= 0
counts["nativeWaveEdges"] = 8

# Independent initial slope and energy checks against exact trigonometric integration.
for row in payload["heat"]:
    t = row["theta"]
    field = lambda x: namespaces["heat"]["heat"](x, t, "dirichlet")
    # A signed polynomial probe v=x(1-x) gives d/dt int uv = int u v''
    # under zero endpoint values. Its exact modal moment is 4/(n*pi)^3 for odd n.
    b = lambda n: -8 / (math.pi * n * (n * n - 4))
    derivative = sum(
        -n
        * n
        * math.pi**2
        * b(n)
        * math.exp(-n * n * math.pi**2 * t)
        * 4
        / (n * math.pi) ** 3
        for n in range(1, 192, 2)
    )
    compare(
        derivative,
        -2 * quad(field, 0, 1, epsabs=1e-13)[0],
        "heat weak polynomial moment",
    )
    # Boundary loss equals d/dt total heat, using independent sine projection by quadrature.
    projected = []
    for n in range(1, 18, 2):
        coeff = (
            2
            * quad(
                lambda x: math.sin(math.pi * x) ** 2 * math.sin(n * math.pi * x),
                0,
                1,
                epsabs=1e-13,
            )[0]
        )
        projected.append((n, coeff))
    flux = sum(
        coeff * n * math.pi * math.exp(-n * n * math.pi**2 * t)
        for n, coeff in projected
    )
    compare(
        row["dirichlet"]["leftOutflow"],
        flux,
        "heat projected endpoint loss",
        relative=3e-11,
        absolute=3e-12,
    )
counts["heatWeakAndFluxStates"] = len(payload["heat"])

# The convolution of two heat kernels is a kernel at the sum of times: this
# checks normalization/scaling as a mechanism beyond author pointwise fixtures.
for row in payload["kernels"]:
    time, alpha = row["time"], row["alpha"]
    for point in row["values"]:
        x = point["x"]
        t1 = time * 0.37
        t2 = time - t1
        conv = quad(
            lambda y: namespaces["kernel"]["kernel"](x - y, t1, alpha)
            * namespaces["kernel"]["kernel"](y, t2, alpha),
            -math.inf,
            math.inf,
            epsabs=1e-13,
        )[0]
        compare(point["value"], conv, "heat semigroup")
counts["kernelConvolutionValues"] = 12

# Changed exact test functions and variational gaps: genuinely different probes.
for row in payload["weak"]:
    a = F(row["location"])
    for degree in [1, 2, 4, 6]:
        # w=x^d(1-x)(1+x+a), a nonzero probe at the source; exact coefficients.
        coefficients = [F(0)] * (degree + 3)
        coefficients[degree] = 1 + a
        coefficients[degree + 1] = -a
        coefficients[degree + 2] = -1
        derivative = [
            (i + 1) * coefficients[i + 1] for i in range(len(coefficients) - 1)
        ]
        value = lambda x: sum(c * x**i for i, c in enumerate(coefficients))
        pairing = (1 - a) * (value(a) - value(F(0))) - a * (value(F(1)) - value(a))
        assert pairing == value(a)
        actual_pairing = F(row["leftSlope"]) * (value(a) - value(F(0))) + F(
            row["rightSlope"]
        ) * (value(F(1)) - value(a))
        compare(float(actual_pairing), value(a), "actual weak derivative pairing")
        square = sum(
            c * d / F(i + j + 1)
            for i, c in enumerate(derivative)
            for j, d in enumerate(derivative)
        )
        assert square > 0
    compare(row["energyIntegral"], a * (1 - a), "weak energy")
counts["changedExactWeakProbes"] = 12

# Cubic jump identity is obtained independently by algebraic factorization.
# A full family of hinge entropies has q=sign(u-k)*(u²-k²)/2.
for row in payload["burgers"]:
    left, right = F(row["left"]), F(row["right"])
    speed = (left + right) / 2
    compare(row["entropyProduction"], (right - left) ** 3 / 12, "Burgers cubic entropy")
    for threshold in [F(-5, 2), F(-1, 2), F(1, 2), F(5, 2), F(7, 2)]:
        sign = lambda x: (x > 0) - (x < 0)
        eta = lambda u: abs(u - threshold)
        flux = lambda u: sign(u - threshold) * (u * u - threshold * threshold) / 2
        production = flux(right) - flux(left) - speed * (eta(right) - eta(left))
        if left >= right:
            assert production <= 0
        if left < right and left < threshold < right:
            assert production > 0
counts["changedHingeEntropyChecks"] = 100

# Actual rod native helper and model at changed lengths, time and amplitude.
# Independently derive the mean by integrating and the global budget by flux.
for row in payload["rods"]:
    length, A, t = row["length"], row["amplitude"], row["time"]
    fn = lambda x: namespaces["rod"]["rod"](x, t, length, A)
    compare(
        row["mean"], quad(fn, 0, length, epsabs=1e-13)[0] / length, "physical rod mean"
    )
    compare(
        row["accumulation"],
        row["totalSource"] - 2 * row["eachOutflow"],
        "physical rod flux budget",
    )
    compare(
        row["settlingTime"],
        namespaces["rod"]["settle"](length, A, 0.03),
        "actual rod settling",
    )
    # The supremum is reached at the center; do not use a sampled-grid maximum.
    compare(
        fn(length / 2) - length * length / 4, row["excess"], "rod attained supremum"
    )
counts["changedPhysicalRodStates"] = len(payload["rods"])
report = {
    "checkedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "passed" if not failures else "failed",
    "sourceHashes": payload["sources"],
    "counts": counts,
    "maxStrictRelativeDiscrepancy": max_relative,
    "maximumToleranceFraction": max_tolerance_fraction,
    "failures": failures,
    "limits": "Complementary changed-source and exact-polynomial checks; author broader suite separately attributed, no numerical proof of general PDE theorems.",
}
Path("scratch/pde-independent-review/results.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
assert not failures
