"""Bounded independent VI review: log-domain edges and changed Gaussian models.

Run with scratch/lesson-tools/Scripts/python.exe -X utf8 -I.
This complements the author's broad browser/native checks; it does not replace them.
"""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "scratch/variational-inference-independent-review"
OUTPUT.mkdir(parents=True, exist_ok=True)

# These are the exact observations made before the author repaired intermediate ratios.
BEFORE = {
    "model_sha256": "e17c198b9b2f0416aaf7e78044af6931ccc539aa93151e57785ad3e000678257",
    "finite": {
        "q": [1, 0], "joint": [1e-310, 1],
        "observed_elbo": -713.8013788281542, "observed_kl": "Infinity",
    },
    "gaussian": {
        "q_covariance": [[1e-100, 0], [0, 1e-100]],
        "target_covariance": [[1e100, 0], [0, 1e100]],
        "means": [0, 0], "observed_kl": "Infinity",
    },
    "interpretation": "Accepted model inputs outside the current fixed UI fixtures; no claim that ordinary controls broke.",
}

finite_cases = [
    {"q": [1, 0], "joint": [1e-310, 1]},
    {"q": [1, 0], "joint": [5e-324, 1e100]},
    {"q": [0.25, 0.75], "joint": [1e-310, 1e100]},
    {"q": [0.25, 0.75], "joint": [0.02, 0.9]},
    {"q": [1, 0], "joint": [0, 1]},
    {"q": [0, 1], "joint": [0, 1]},
]
gaussian_cases = []
for q_covariance, target_covariance, mean, target_mean in [
    ([[1e-100, 0], [0, 1e-100]], [[1e100, 0], [0, 1e100]], [0, 0], [0, 0]),
    ([[2, .4], [.4, .7]], [[.8, -.2], [-.2, 1.6]], [-1, 2], [.2, -.5]),
    ([[.3, -.15], [-.15, 2]], [[4, 1], [1, .5]], [1.3, -1.2], [-.7, .9]),
]:
    gaussian_cases.append(dict(covariance=q_covariance, targetCovariance=target_covariance,
                               mean=mean, targetMean=target_mean))

# An invertible, nonorthogonal change of coordinates must preserve KL.
transform = np.array([[2., 1.], [.5, -1.]])
shift = np.array([2., -.3])
original = gaussian_cases[1]
transformed_q = transform @ np.array(original["covariance"]) @ transform.T
transformed_p = transform @ np.array(original["targetCovariance"]) @ transform.T
# Preserve the helper's exact symmetry input contract after NumPy matrix rounding.
transformed_q = (transformed_q + transformed_q.T) / 2
transformed_p = (transformed_p + transformed_p.T) / 2
gaussian_cases.append({
    "covariance": transformed_q.tolist(),
    "targetCovariance": transformed_p.tolist(),
    "mean": (transform @ np.array(original["mean"]) + shift).tolist(),
    "targetMean": (transform @ np.array(original["targetMean"]) + shift).tolist(),
})
gradient_cases = [
    {"mean": -.7, "sd": .3, "targetMean": 2.2, "targetSd": 1.4, "seed": 41, "count": 13},
    {"mean": 1.8, "sd": 1.7, "targetMean": -.4, "targetSd": .4, "seed": 92, "count": 17},
    {"mean": -.3, "sd": 1.1, "targetMean": -.3, "targetSd": 1.1, "seed": 7, "count": 19},
]
payload = {"finite": finite_cases, "gaussian": gaussian_cases, "gradients": gradient_cases}
javascript = """
import { finiteElbo, gaussianKl, variationalGradients } from './src/learn/data/variational-inference-models.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const cases = JSON.parse(input);
const result = {
  finite: cases.finite.map(({q, joint}) => finiteElbo(q, joint)),
  gaussian: cases.gaussian.map(c => gaussianKl(c.mean, c.covariance, c.targetMean, c.targetCovariance)),
  gradients: cases.gradients.map(c => variationalGradients(c))
};
console.log(JSON.stringify(result, (_, value) =>
  typeof value === 'number' && !Number.isFinite(value) ? String(value) : value));
"""
completed = subprocess.run(["node", "--input-type=module", "-e", javascript],
                           input=json.dumps(payload), text=True, cwd=ROOT, capture_output=True)
if completed.returncode:
    raise RuntimeError(completed.stderr)
actual = json.loads(completed.stdout)
checks = []


def close(value, reference, label, tolerance=2e-10):
    if math.isinf(reference):
        assert value == ("Infinity" if reference > 0 else "-Infinity"), (label, value, reference)
        return
    assert isinstance(value, (float, int)) and math.isfinite(value), (label, value, reference)
    assert math.isclose(value, reference, rel_tol=tolerance, abs_tol=tolerance), (label, value, reference)


for case, result in zip(finite_cases, actual["finite"]):
    # Decimal.from_float preserves the represented input, including the minimum subnormal.
    with localcontext() as context:
        context.prec = 80
        q = [Decimal.from_float(float(x)) for x in case["q"]]
        joint = [Decimal.from_float(float(x)) for x in case["joint"]]
        total = sum(joint)
        support_failure = any(p > 0 and mass == 0 for p, mass in zip(q, joint))
        if support_failure:
            reference_elbo, reference_kl = -math.inf, math.inf
        else:
            reference_elbo = float(sum(p * (mass.ln() - p.ln())
                                       for p, mass in zip(q, joint) if p > 0))
            reference_kl = float(sum(p * (p / (mass / total)).ln()
                                     for p, mass in zip(q, joint) if p > 0))
        close(result["elbo"], reference_elbo, "finite ELBO")
        close(result["kl"], reference_kl, "finite KL")
        if not support_failure:
            close(result["elbo"] + result["kl"], float(total.ln()), "evidence identity")
        checks.append({"kind": "finite", **case, "reference_elbo": str(reference_elbo),
                       "reference_kl": str(reference_kl), "actual": result})

for case, value in zip(gaussian_cases, actual["gaussian"]):
    q_cov = np.array(case["covariance"])
    p_cov = np.array(case["targetCovariance"])
    delta = np.array(case["mean"]) - np.array(case["targetMean"])
    reference = .5 * (np.trace(np.linalg.solve(p_cov, q_cov))
                      + delta @ np.linalg.solve(p_cov, delta) - 2
                      + np.linalg.slogdet(p_cov)[1] - np.linalg.slogdet(q_cov)[1])
    close(value, float(reference), "Gaussian KL")
    checks.append({"kind": "gaussian", **case, "reference": float(reference), "actual": value})
close(actual["gaussian"][1], actual["gaussian"][3], "affine invariance")

# Fixed-node Gaussian quadrature is independent of the author's adaptive-integral oracle.
# These contributions are polynomial in noise, so 12 Hermite nodes integrate them exactly
# in real arithmetic; floating-point residuals are checked against a finite difference.
nodes, weights = np.polynomial.hermite.hermgauss(12)
noise = np.sqrt(2) * nodes
weights = weights / np.sqrt(np.pi)
for case, result in zip(gradient_cases, actual["gradients"]):
    m, s, t, u = [case[k] for k in ("mean", "sd", "targetMean", "targetSd")]

    def objective(current_mean, log_sd):
        current_sd = math.exp(log_sd)
        draws = current_mean + current_sd * noise
        expected_log_target = np.sum(weights * (-math.log(u * math.sqrt(2 * math.pi))
                                                 - .5 * ((draws - t) / u) ** 2))
        return float(expected_log_target + log_sd + .5 * math.log(2 * math.pi * math.e))

    h = 1e-5
    reference = [(objective(m + h, math.log(s)) - objective(m - h, math.log(s))) / (2 * h),
                 (objective(m, math.log(s) + h) - objective(m, math.log(s) - h)) / (2 * h)]
    draws = m + s * noise
    slope = (t - draws) / u ** 2
    log_ratio = math.log(s / u) - .5 * ((draws - t) / u) ** 2 + .5 * noise ** 2
    integrands = [slope, 1 + s * noise * slope, log_ratio * noise / s,
                  log_ratio * (noise ** 2 - 1)]
    integrated = [float(np.sum(weights * values)) for values in integrands]
    for value, expected in zip(integrated, reference * 2):
        close(value, expected, "gradient expectation", tolerance=2e-7)
    close(result["exactMean"], reference[0], "model exact mean", tolerance=2e-7)
    close(result["exactLogSd"], reference[1], "model exact log scale", tolerance=2e-7)
    close(result["elbo"], objective(m, math.log(s)), "Gaussian ELBO")
    for row in result["rows"]:
        z = row["noise"]
        draw = m + s * z
        g = (t - draw) / u ** 2
        ratio = math.log(s / u) - .5 * ((draw - t) / u) ** 2 + .5 * z ** 2
        for key, expected in zip(("pathMean", "pathLogSd", "scoreMean", "scoreLogSd"),
                                 (g, 1 + s * z * g, ratio * z / s, ratio * (z ** 2 - 1))):
            close(row[key], expected, "changed target per-draw contribution")
    checks.append({"kind": "gradient", **case, "integrated": integrated,
                   "finite_difference": reference, "checked_draws": len(result["rows"])})

record = {
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "before": BEFORE,
    "final_model_sha256": hashlib.sha256((ROOT / "src/learn/data/variational-inference-models.js").read_bytes()).hexdigest(),
    "checks": checks,
    "summary": {"finite_support_log_domain_cases": 6, "nonunit_gaussian_cases": 4,
                "nonorthogonal_affine_invariance": True, "changed_target_gradient_cases": 3,
                "changed_target_draws": 49, "all_passed": True},
}
(OUTPUT / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record["summary"], indent=2))
