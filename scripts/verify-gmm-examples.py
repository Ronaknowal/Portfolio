"""Execute the Gaussian mixture lesson's displayed Python programs.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/gmm-examples.js from the executed programs; without it, the
recorded output must match a fresh execution.

The Iris program reads the CSV the lesson offers for download, from
public/learn-assets/gmm, which is the same bytes a learner receives. The
Bayesian block is an appendix to that program and runs in its namespace, exactly
as the lesson instructs.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib
import hashlib
import importlib.metadata
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ASSETS = Path("public/learn-assets/gmm").resolve()

PROGRAMS = {
    "emOneDimension": {
        "file": "em_1d.py",
        "title": "Constrained EM on four measurements",
        "question": "the two components start at −1 and +1 with variance 1. Will the fitted means land exactly on the group averages −1.5 and 1.5?",
        "code": r"""
import numpy as np

x = np.array([-2., -1., 1., 2.])
weights = np.array([0.5, 0.5])
means = np.array([-1., 1.])
variances = np.array([1., 1.])
variance_floor = 0.05

def expectation(x, weights, means, variances):
    log_joint = np.log(weights) - 0.5 * (
        np.log(2 * np.pi * variances)
        + (x[:, None] - means) ** 2 / variances
    )
    row_max = log_joint.max(axis=1, keepdims=True)
    log_density = row_max[:, 0] + np.log(
        np.exp(log_joint - row_max).sum(axis=1)
    )
    responsibility = np.exp(log_joint - log_density[:, None])
    return responsibility, log_density.sum()

responsibility, previous = expectation(x, weights, means, variances)
print(f"initial log-likelihood {previous:.6f}")
for iteration in range(1, 51):
    count = responsibility.sum(axis=0)
    weights = count / len(x)
    means = (responsibility.T @ x) / count
    scatter = (responsibility * (x[:, None] - means) ** 2).sum(axis=0)
    variances = np.maximum(scatter / count, variance_floor)
    responsibility, current = expectation(x, weights, means, variances)
    print(iteration, f"{current:.6f}")
    gain = (current - previous) / len(x)
    if abs(gain) < 1e-8:
        break
    previous = current

print("means", np.round(means, 6))
print("variances", np.round(variances, 6))
print("row sums", responsibility.sum(axis=1))
""",
    },
    "irisDensity": {
        "file": "iris_density.py",
        "title": "The complete offline Iris comparison",
        "question": "validation picks the candidate and the test rows stay untouched until the end. Will the selected mixture beat one Gaussian on the reserved rows?",
        "code": r"""
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

with Path("iris.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
X = np.array([[float(row["sepal_length_cm"]),
               float(row["sepal_width_cm"])] for row in rows])
species = np.array([row["species"] for row in rows])
order = np.random.default_rng(16).permutation(len(X))
train, validation, test = order[:90], order[90:120], order[120:]
scaler = StandardScaler().fit(X[train])
Z = scaler.transform(X)

fits = []
print("covariance K validation_log_density training_BIC")
for kind in ["full", "tied", "diag", "spherical"]:
    for k in range(1, 5):
        model = GaussianMixture(
            n_components=k, covariance_type=kind,
            n_init=5, random_state=16, reg_covar=1e-4,
            tol=1e-6, max_iter=500,
        ).fit(Z[train])
        if not model.converged_:
            raise RuntimeError(f"Unfinished fit: {kind}, K={k}")
        score = model.score(Z[validation])
        fits.append((score, kind, k, model))
        print(kind, k, f"{score:.6f}", f"{model.bic(Z[train]):.3f}")

_, kind, k, selected = max(fits, key=lambda fit: fit[0])
baseline = fits[0][3]  # full covariance, K=1
print("selected", kind, k)
print("test log-density", f"{selected.score(Z[test]):.6f}")
print("baseline test log-density", f"{baseline.score(Z[test]):.6f}")
print("test ARI", f"{adjusted_rand_score(species[test], selected.predict(Z[test])):.6f}")
first = test[:1]
print("first test ID", int(first[0] + 1))
print("responsibilities", np.round(selected.predict_proba(Z[first])[0], 6))
print("log-density", np.round(selected.score_samples(Z[first])[0], 6))
""",
    },
    "bayesianWeights": {
        "file": "iris_density.py (appendix)",
        "title": "How many components stay active under a weight prior?",
        "question": "six components are available to every fit. Does a larger concentration leave more of them with meaningful weight?",
        "prelude": "irisDensity",
        "code": r"""
from sklearn.mixture import BayesianGaussianMixture

for concentration in [0.01, 1., 10.]:
    bayesian = BayesianGaussianMixture(
        n_components=6, covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=concentration,
        n_init=3, random_state=16, reg_covar=1e-4,
        tol=1e-6, max_iter=1000,
    ).fit(Z[train])
    print(concentration, np.round(bayesian.weights_, 4),
          int((bayesian.weights_ > 0.01).sum()),
          int((bayesian.weights_ > 0.05).sum()),
          bayesian.converged_)
""",
    },
}

module_path = Path("src/learn/data/gmm-examples.js")
evidence_path = Path("docs/teaching/evidence/gmm-native.json")
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


records, namespaces = {}, {}
for key, item in PROGRAMS.items():
    code = item["code"].strip() + "\n"
    if "prelude" in item:
        # The lesson says to append this block to that program, so it runs in
        # that program's namespace and only its own new output is recorded.
        namespace = namespaces[item["prelude"]]
    else:
        namespace = {"__file__": str(ASSETS / item["file"])}
    with contextlib.chdir(ASSETS), threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"gmm-example:{key}", "exec"), namespace)
    namespaces[key] = namespace
    records[key] = {
        "title": item["title"],
        "question": item["question"],
        "file": item["file"],
        "code": code.rstrip("\n"),
        "expected": output.getvalue().strip(),
        "language": "python",
    }
    print(f"Executed {key}: {len(code.splitlines())} lines, {len(records[key]['expected'].splitlines())} output lines")

if not write:
    existing = module_path.read_text(encoding="utf-8")
    prefix = "export const gmmExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], (
            f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}")

# Oracles: the manuscript's stated values must come out of these runs.
em = namespaces["emOneDimension"]
assert abs(float(em["previous"]) + 5.675741839) < 1e-6, "the run ends at the plateau the lesson prints"
assert np.allclose(em["means"], [-1.499994, 1.499994], atol=1e-6), "fitted means"
assert np.allclose(em["variances"], [0.250018, 0.250018], atol=1e-6), "fitted variances"
assert np.allclose(em["responsibility"].sum(axis=1), 1), "responsibility rows sum to one"
assert (em["variances"] > em["variance_floor"]).all(), "the floor never binds in this run"
start = em["expectation"](np.array([-2., -1., 1., 2.]), np.array([0.5, 0.5]), np.array([-1., 1.]), np.array([1., 1.]))
assert abs(start[1] + 7.158186977) < 1e-9, "the printed starting log-likelihood"
assert abs(start[0][0][0] - 0.982013790) < 1e-9, "A's first Left responsibility"
underflow = em["expectation"](np.array([0.]), np.array([0.5, 0.5]), np.array([-1000., 1000.]), np.array([1., 1.]))
assert np.isfinite(underflow[1]), "log-space evaluation survives a distant observation"

iris = namespaces["irisDensity"]
assert len(iris["fits"]) == 16, "sixteen candidates"
assert all(fit[3].converged_ for fit in iris["fits"]), "every candidate converged"
assert (iris["kind"], iris["k"]) == ("full", 2), "validation selects full K=2"
assert abs(iris["selected"].score(iris["Z"][iris["test"]]) + 3.011223) < 1e-6, "selected test score"
assert abs(iris["baseline"].score(iris["Z"][iris["test"]]) + 2.887249) < 1e-6, "baseline test score"
assert iris["baseline"].score(iris["Z"][iris["test"]]) > iris["selected"].score(iris["Z"][iris["test"]]), (
    "the simple baseline scores higher on the reserved rows")
assert abs(float(np.round(iris["scaler"].mean_[0], 9)) - 5.772222222) < 1e-9, "training mean sepal length"
assert int(iris["test"][0] + 1) == 34, "the first reserved row"
assert np.allclose(iris["selected"].predict_proba(iris["Z"][iris["test"][:1]]).sum(), 1), "a responsibility row sums to one"
lowest_bic = min(iris["fits"], key=lambda fit: fit[3].bic(iris["Z"][iris["train"]]))
assert (lowest_bic[1], lowest_bic[2]) == ("full", 4), "training BIC prefers a sharper fit"
assert abs(float(np.min(lowest_bic[3].covariances_[:, 1, 1])) - 1e-4) < 1e-12, "its narrow width sits at the regularization level"

bayes = namespaces["bayesianWeights"]
assert bayes["bayesian"].weights_.shape == (6,), "six available components"
assert int((bayes["bayesian"].weights_ > 0.05).sum()) == 2, "two components above five percent at concentration 10"
oracle_count = 18

if write:
    module_path.write_text(
        "// Complete displayed programs for the Gaussian mixture lesson, executed by\n"
        "// scripts/verify-gmm-examples.py. `file` is the filename the lesson asks the\n"
        "// learner to save the block as; the appendix runs inside that program.\n"
        "export const gmmExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-gmm-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn"]},
    "programs": {key: {"file": record["file"], "codeHash": digest(record["code"]),
                       "stdoutHash": digest(record["expected"]), "stdout": record["expected"]}
                 for key, record in records.items()},
    "oracles": oracle_count,
    "limits": [
        "The Iris program ran against the served copy of the supplied CSV; no network access was used.",
        "One fixed split of one curated balanced collection; the test comparison is a single demonstration.",
        "Species labels enter only the reported ARI diagnostic, never the fitting or the selection.",
        "Numerical fitting can differ on other library versions.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions.")
