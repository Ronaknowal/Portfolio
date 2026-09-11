from pathlib import Path
import contextlib, datetime, hashlib, io, json, math, subprocess, warnings
from fractions import Fraction as F
import mpmath as mp
import numpy as np
from scipy.integrate import quad

root = Path.cwd()
js = """import {functionalAnalysisExamples as examples} from './src/learn/data/functional-analysis-examples.js';
import * as m from './src/learn/data/functional-analysis-models.js';
const ridge=[];for(const fixture of ['original','curve','duplicates'])for(const [gamma,lambda] of [[.05,.0001],[.33,.007],[7.25,.21],[32,2]])ridge.push(m.kernelRidgeState(gamma,lambda,.37,fixture));
const means=[.05,.1,.37,1,2.5,8].map(g=>m.distributionEmbeddingState(g,false));
const quads=[];for(const x of [0,.03125,.25,2/3,.9375,1])for(const w of [null,-1,.125,1.875])quads.push(m.kernelQuadratureState(x,w));
console.log(JSON.stringify({examples,ridge,means,quads}));"""
payload = json.loads(
    subprocess.run(
        ["node", "--input-type=module", "-e", js],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
)
counts = {}
namespaces = {}
for e in payload["examples"]:
    ns = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(e["code"], e["id"] + ".py", "exec"), ns)
    assert out.getvalue().rstrip("\n") == e["expected"]
    namespaces[e["id"]] = ns
counts["actualDisplayedPrograms"] = len(namespaces)
original = json.loads(
    (
        root / "docs/teaching/evidence/functional-analysis-rkhs-original-content.json"
    ).read_text(encoding="utf-8-sig")
)["programs"][0]
e = next(e for e in payload["examples"] if e["id"] == "original")
assert e["code"] == original["code"] and e["expected"] == original[
    "expectedOutput"
].replace("\r\n", "\n")

mp.mp.dps = 70
maximum_ridge_error = 0
for state in payload["ridge"]:
    xs = list(map(mp.mpf, map(str, state["xs"])))
    ys = mp.matrix(list(map(mp.mpf, map(str, state["ys"]))))
    g = mp.mpf(str(state["gamma"]))
    lam = mp.mpf(str(state["lambda"]))
    n = len(xs)
    kernel = lambda a, b: mp.exp(-g * (a - b) ** 2)
    k = mp.matrix([[kernel(a, b) for b in xs] for a in xs])
    alpha = mp.lu_solve(k + n * lam * mp.eye(n), ys)
    norm = float((alpha.T * k * alpha)[0])
    prediction = float(
        sum(alpha[i] * kernel(x, mp.mpf(".37")) for i, x in enumerate(xs))
    )
    maximum_ridge_error = max(maximum_ridge_error, abs(norm - state["squaredNorm"]))
    assert np.allclose(list(map(float, alpha)), state["alpha"], rtol=3e-9, atol=3e-9)
    assert math.isclose(norm, state["squaredNorm"], rel_tol=2e-8, abs_tol=2e-10)
    assert math.isclose(prediction, state["prediction"], rel_tol=2e-8, abs_tol=3e-10)
    # Strict convex functional objective difference, with h an extra section at .37.
    q = mp.matrix([kernel(x, mp.mpf(".37")) for x in xs])
    amp = mp.mpf("-.41")
    residual = k * alpha - ys
    before = (residual.T * residual)[0] / n + lam * (alpha.T * k * alpha)[0]
    after = ((residual + amp * q).T * (residual + amp * q))[0] / n + lam * (
        (alpha.T * k * alpha)[0] + 2 * amp * (alpha.T * q)[0] + amp**2
    )
    exact = amp**2 * ((q.T * q)[0] / n + lam)
    assert abs(after - before - exact) < mp.mpf("1e-58")
counts["seventyDigitRidgeAndObjectiveIdentities"] = len(payload["ridge"])

# Fourier-domain discrepancy is a different oracle from pair/Gram enumeration.
maximum_fourier_error = 0
for state in payload["means"]:
    gamma = state["gamma"]
    # P: cos(w), Q:3/4+cos(2w)/4. Rescale to a standard Normal integration variable.
    difference = (
        lambda z: math.cos(math.sqrt(2 * gamma) * z)
        - 0.75
        - 0.25 * math.cos(2 * math.sqrt(2 * gamma) * z)
    )
    oracle = (
        2
        * quad(
            lambda z: difference(z) ** 2
            * math.exp(-z * z / 2)
            / math.sqrt(2 * math.pi),
            0,
            12,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=400,
        )[0]
    )
    maximum_fourier_error = max(
        maximum_fourier_error, abs(oracle - state["squaredMmd"])
    )
    assert math.isclose(oracle, state["squaredMmd"], rel_tol=2e-10, abs_tol=2e-12)
counts["IndependentGaussianFourierDiscrepancies"] = len(payload["means"])

native_quad = namespaces["quadrature"]["quadrature_rule"]
sets = [
    [F(0)],
    [F(1, 4)],
    [F(1, 7), F(3, 7), F(6, 7)],
    [F(3, 4), F(1, 4), F(1, 4), F(0)],
    [F(1, 9), F(2, 9), F(4, 9), F(8, 9)],
    [F(1), F(1), F(0)],
    [F(1, 5), F(4, 5), F(4, 5)],
]
for nodes in sets:
    positive = sorted(set(nodes) - {F(0)})
    breaks = [F(0)] + positive
    slopes = [1 - (a + b) / 2 for a, b in zip(breaks, breaks[1:])] + [F(0)]
    total = {node: slopes[i] - slopes[i + 1] for i, node in enumerate(positive)}
    expected = [total.get(node, F(0)) / nodes.count(node) for node in nodes]
    weights, error = native_quad(list(map(float, nodes)))
    assert np.allclose(weights, list(map(float, expected)), rtol=1e-11, atol=1e-12)
    # Exact energy with residual slope (1-t) minus all active section weights.
    points = sorted(set([F(0), F(1)] + nodes))
    energy = F(0)
    integral_error = F(0)
    for a, b in zip(points, points[1:]):
        active = sum(w for x, w in zip(nodes, expected) if x > (a + b) / 2)
        intercept = 1 - active
        energy += (
            intercept**2 * (b - a) - intercept * (b * b - a * a) + (b**3 - a**3) / 3
        )
    assert abs(error - float(energy)) < 3e-13
    # Residual representer attains the normalized bound, using exact integrals.
    mean_at = lambda t: t - t * t / 2
    residual_at = lambda t: mean_at(t) - sum(
        w * min(t, x) for x, w in zip(nodes, expected)
    )
    actual_gap = (
        F(1, 3)
        - sum(w * mean_at(x) for x, w in zip(nodes, expected))
        - sum(w * residual_at(x) for x, w in zip(nodes, expected))
    )
    assert actual_gap == energy
counts["ExactMultipleNodeProjectionAndAttainment"] = len(sets)
for s in payload["quads"]:
    x, w = F(str(s["node"])), F(str(s["weight"]))
    energy = F(1, 3) - 2 * w * (x - x * x / 2) + w * w * x
    assert abs(s["squaredError"] - float(energy)) < 8e-15
    assert s["linearError"] <= s["error"] + 1e-15
counts["ExactSingleNodeResidualPolynomials"] = len(payload["quads"])

# Changed intervals and arbitrary local amplitudes, without copying the fixture's constants.
wiggles = 0
for a, b, rise in [
    (F(1, 7), F(5, 7), F(3, 2)),
    (F(0), F(1, 3), F(-2)),
    (F(2, 5), F(1), F(0)),
]:
    for fraction in [F(1, 3), F(1, 2), F(4, 5)]:
        peak = a + (b - a) * fraction
        for amplitude in [F(-3, 2), F(1, 4), F(1)]:
            slope = rise / (b - a)
            left = amplitude / (peak - a)
            right = -amplitude / (b - peak)
            cross = slope * (left * (peak - a) + right * (b - peak))
            assert cross == 0
            base = slope * slope * (b - a)
            extra = left * left * (peak - a) + right * right * (b - peak)
            changed = (slope + left) ** 2 * (peak - a) + (slope + right) ** 2 * (
                b - peak
            )
            assert changed == base + extra
            wiggles += 1
counts["ExactChangedOrthogonalWiggles"] = wiggles

# Redundant finite features: minimum coefficient norm in the quotient function space.
rng = np.random.default_rng(420)
for trial in range(8):
    x = rng.uniform(-1, 1, 5)
    phi = np.column_stack([np.ones(5), np.ones(5), x, 2 * x])
    y = rng.normal(size=5)
    lam = 0.013 + 0.01 * trial
    w = np.linalg.solve(phi.T @ phi + 5 * lam * np.eye(4), phi.T @ y)
    alpha = np.linalg.solve(phi @ phi.T + 5 * lam * np.eye(5), y)
    assert np.allclose(w, phi.T @ alpha, rtol=1e-10, atol=1e-12)
    assert np.allclose(w, np.linalg.pinv(phi) @ (phi @ w), rtol=1e-10, atol=1e-12)
    assert abs(w[0] - w[1]) < 1e-12 and abs(w[3] - 2 * w[2]) < 1e-12
counts["ChangedRedundantFeatureQuotients"] = 8

# Explicit thresholded-inverse distinction, no arbitrary eigenvalue tolerance assertion.
small = 1e-14
w = np.diag([1.0, small])
f = np.array([[1.0], [0.0]])
exact = w @ np.diag([1.0, 1 / small]) @ w
truncated = f @ f.T
assert exact[1, 1] == small and truncated[1, 1] == 0

# Execute actual changed capstone tasks, preserving each whole selection protocol.
validation = next(e["code"] for e in payload["examples"] if e["id"] == "validation")
capstones = []
for name, code in [
    (
        "noise035",
        validation.replace("rng.normal(0, 0.15, 90)", "rng.normal(0, 0.35, 90)"),
    ),
    (
        "linear035",
        validation.replace("np.sin(2 * xs)", "0.4 + 1.2 * xs").replace(
            "rng.normal(0, 0.15, 90)", "rng.normal(0, 0.35, 90)"
        ),
    ),
]:
    ns = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, ns)
    assert math.isclose(ns["mean"], float(np.mean(ns["xs"][:40])))
    assert (
        np.isfinite(ns["prediction"]).all()
        and np.isfinite(ns["linear_prediction"]).all()
    )
    assert ns["validation_error"] == min(c[0] for c in ns["candidates"])
    assert ns["linear_error"] == min(c[0] for c in ns["linear_candidates"])
    capstones.append(
        {
            "case": name,
            "codeSha256": hashlib.sha256(code.encode()).hexdigest(),
            "actualStdout": out.getvalue(),
        }
    )
counts["ActualChangedCapstoneProtocols"] = len(capstones)

# Admitted-input precision finding: final run requires rejection rather than Infinity.
interpolate = namespaces["interpolation"]["anchored_interpolate"]
finite = interpolate([1e-308], [1.0])
assert np.isfinite(finite[0]).all() and math.isfinite(finite[1])
with warnings.catch_warnings(record=True) as captured:
    try:
        answer = interpolate([1e-310], [1.0])
        guard = {
            "rejected": False,
            "returned": repr(answer),
            "warnings": [str(w.message) for w in captured],
        }
    except (ValueError, FloatingPointError, OverflowError) as error:
        guard = {
            "rejected": True,
            "exception": type(error).__name__,
            "message": str(error),
        }
production = [
    "src/learn/data/topics/functional-analysis-rkhs.jsx",
    "src/learn/data/functional-analysis-models.js",
    "src/learn/data/functional-analysis-examples.js",
    "src/learn/components/lesson-labs/FunctionalAnalysisLabs.jsx",
    "src/learn/components/lesson-labs/functional-analysis-labs.css",
    "src/learn/data/curriculum/blueprints/functional-analysis-rkhs.js",
]
result = {
    "checkedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": (
        "passed complementary mathematics; native guard pending"
        if not guard["rejected"]
        else "passed"
    ),
    "counts": counts,
    "originalConserved": True,
    "maximumSeventyDigitRidgeNormDifference": maximum_ridge_error,
    "maximumFourierDiscrepancyDifference": maximum_fourier_error,
    "nativeArithmeticRegression": guard,
    "thresholdedInverseCounterexample": {
        "smallPositiveEigenvalue": small,
        "exactPseudoinverseRecoveredEntry": float(exact[1, 1]),
        "thresholdedRecoveredEntry": float(truncated[1, 1]),
    },
    "changedCapstones": capstones,
    "production": [
        {"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()}
        for p in production
    ],
    "limits": "Finite changed-input independent checks and source proof read; no browser or infinite-theorem proof from simulations.",
}
out = root / "scratch/functional-analysis-independent-review"
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
if not guard["rejected"]:
    (out / "prefixed-native-arithmetic-finding.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
print(
    json.dumps(
        {
            k: v
            for k, v in result.items()
            if k not in ["production", "changedCapstones"]
        },
        indent=2,
    )
)
