"""Execute the PCA lesson's displayed Python programs and record their output.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/pca-examples.js from the executed programs; without it, the
recorded expected output must match a fresh execution. Continuation programs
declare the earlier program whose namespace they extend, exactly as the lesson
tells the learner to run them.
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

PROGRAMS = {
    "svd": {
        "title": "Center, decompose and reconstruct the four points with NumPy",
        "question": "Which shape does `directions` have, and why does the forward multiplication need `.T` while the reconstruction does not?",
        "code": r"""
import numpy as np

X = np.array([[1., 1.], [2., 0.], [4., 4.], [5., 3.]])
mean = X.mean(axis=0)
centered = X - mean
_, singular_values, directions = np.linalg.svd(centered, full_matrices=False)

k = 1
kept_directions = directions[:k]       # shape: (k, d)
scores = centered @ kept_directions.T # shape: (n, k)
reconstructed = scores @ kept_directions + mean
variances = singular_values**2 / (len(X) - 1)

print(np.round(variances, 4))
print(np.round(variances / variances.sum(), 4))
print(np.round(reconstructed, 4))
print(round(np.mean((X - reconstructed)**2), 4))
""",
    },
    "library": {
        "title": "The library version exposes the same operations",
        "question": "The new observation (6, 4) was not part of the fit. Which fitted quantities does `transform` reuse, and what would change if you refitted with it included?",
        "continues": "svd",
        "code": r"""
# Continue the NumPy program above: X, mean and directions are already defined.
from sklearn.decomposition import PCA

pca = PCA(n_components=1, svd_solver="full")
scores = pca.fit_transform(X)
reconstructed = pca.inverse_transform(scores)

new_observation = np.array([[6., 4.]])
new_scores = pca.transform(new_observation)
new_reconstruction = pca.inverse_transform(new_scores)

print(pca.components_.shape, scores.shape)
print(np.round(new_reconstruction, 4))
""",
    },
    "wineScaling": {
        "title": "Raw and standardized PCA on all 178 wines",
        "question": "Will the raw first component retain more or less than half of the variance, and which single measurement do you expect to dominate it?",
        "code": r"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

wine = load_wine()  # included with scikit-learn; no dataset download
X, cultivar = wine.data, wine.target

raw_pca = PCA(svd_solver="full").fit(X)
scaler = StandardScaler().fit(X)
standardized = scaler.transform(X)
scaled_pca = PCA(svd_solver="full").fit(standardized)
scores = scaled_pca.transform(standardized)

print(X.shape, np.bincount(cultivar))
print(np.round(raw_pca.explained_variance_ratio_[:2], 4))
print(np.round(scaled_pca.explained_variance_ratio_[:3], 4))
print(round(scaled_pca.explained_variance_ratio_[:2].sum(), 4))
""",
    },
    "budget": {
        "title": "Choose the smallest component count that meets a validation error budget",
        "question": "Which rows fit the scaler and the PCA, which rows are scored, and what does a ratio of 0.10 mean relative to always returning the training mean?",
        "code": r"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
train, validation = train_test_split(
    np.arange(len(wine.data)), test_size=0.25,
    random_state=42, stratify=wine.target
)
scaler = StandardScaler().fit(wine.data[train])
A = scaler.transform(wine.data[train])
B = scaler.transform(wine.data[validation])
pca = PCA(svd_solver="full").fit(A)

baseline_mse = np.mean((B - pca.mean_)**2)
ratios = []
for k in range(14):
    directions = pca.components_[:k]
    scores = (B - pca.mean_) @ directions.T
    reconstruction = scores @ directions + pca.mean_
    ratios.append(np.mean((B - reconstruction)**2) / baseline_mse)

print(len(train), len(validation))
print(round(baseline_mse, 4))
for k in [0, 2, 7, 8, 10, 13]:
    print(k, round(ratios[k], 4))
print(next(k for k, ratio in enumerate(ratios) if ratio <= 0.10))
""",
    },
    "originalUnits": {
        "title": "Recover approximate measurements in their original units",
        "question": "The output has thirteen columns. How many independently stored numbers per wine produced them?",
        "continues": "budget",
        "code": r"""
# Continue the training/validation program above.
k = 8
directions = pca.components_[:k]
scores = (B - pca.mean_) @ directions.T
reconstruction = scores @ directions + pca.mean_
original_units = scaler.inverse_transform(reconstruction)

print(original_units.shape)
""",
    },
    "eigen": {
        "title": "Check that covariance eigenvectors and SVD give the same PCA",
        "question": "Why does the comparison use reconstructions rather than the direction vectors themselves?",
        "continues": "svd",
        "code": r"""
# Continue the four-point NumPy example in section 4.
covariance = centered.T @ centered / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
leading = eigenvectors[:, -1:]
eigen_reconstruction = (centered @ leading) @ leading.T + mean

print(np.round(eigenvalues[::-1], 4))
print(np.allclose(eigen_reconstruction, reconstructed))
print(round(float(np.sum(singular_values[1:]**2)), 4))
""",
    },
    "gaussian": {
        "title": "A noise-only sample still has a leading component",
        "question": "The population covariance is the identity, so every two directions hold exactly 10% of population variance. Will the first two sample components hold exactly 10%?",
        "code": r"""
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(23)
noise = rng.normal(size=(40, 20))
pca = PCA(svd_solver="full").fit(noise)
print(round(pca.explained_variance_ratio_[:2].sum(), 4))
""",
    },
    "pipeline": {
        "title": "Evaluate PCA inside a prediction pipeline with a no-PCA baseline",
        "question": "Where are the scaler and PCA fitted in each fold, and what would be wrong with fitting them once on all 178 rows first?",
        "code": r"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

wine = load_wine()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for k in [None, 2, 8]:
    reduction = "passthrough" if k is None else PCA(
        n_components=k, svd_solver="full"
    )
    pipeline = make_pipeline(
        StandardScaler(), reduction, LogisticRegression(max_iter=2000)
    )
    accuracy = cross_val_score(
        pipeline, wine.data, wine.target,
        cv=folds, scoring="accuracy", n_jobs=1
    )
    print(k, np.round(accuracy, 4), round(accuracy.mean(), 4))
""",
    },
}

module_path = Path("src/learn/data/pca-examples.js")
evidence_path = Path("docs/teaching/evidence/pca-native.json")
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def execute(key, namespaces):
    item = PROGRAMS[key]
    namespace = {}
    if item.get("continues"):
        namespace.update(namespaces[item["continues"]])
    code = item["code"].strip() + "\n"
    with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"pca-example:{key}", "exec"), namespace)
    namespaces[key] = namespace
    return code, output.getvalue().strip()


records = {}
namespaces = {}
for key in PROGRAMS:
    code, stdout = execute(key, namespaces)
    records[key] = {
        "title": PROGRAMS[key]["title"],
        "question": PROGRAMS[key]["question"],
        **({"continues": PROGRAMS[key]["continues"]} if PROGRAMS[key].get("continues") else {}),
        "code": code.rstrip("\n"),
        "expected": stdout,
        "language": "python",
    }
    print(f"Executed {key}: {len(code.splitlines())} lines, {len(stdout.splitlines())} output lines")

if not write:
    existing = module_path.read_text(encoding="utf-8")
    prefix = "export const pcaExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}"

# Independent oracle checks tied to the manuscript's stated values.
svd_namespace = namespaces["svd"]
assert np.allclose(svd_namespace["variances"], [6, 2 / 3])
assert np.allclose(svd_namespace["reconstructed"], [[1.5, 0.5], [1.5, 0.5], [4.5, 3.5], [4.5, 3.5]])
assert np.allclose(namespaces["library"]["new_reconstruction"], [[5.5, 4.5]])
budget = namespaces["budget"]
assert budget["ratios"][0] == 1.0 and budget["ratios"][13] < 1e-12
assert next(k for k, r in enumerate(budget["ratios"]) if r <= 0.06) == 10
assert next(k for k, r in enumerate(budget["ratios"]) if r <= 0.11) == 8
assert abs(budget["ratios"][9] - 0.072354) < 1e-6 and abs(budget["ratios"][10] - 0.052066) < 1e-6
assert namespaces["originalUnits"]["original_units"].shape == (45, 13)
recovered = namespaces["originalUnits"]["scaler"].inverse_transform(budget["B"])
assert np.allclose(recovered, namespaces["budget"]["wine"].data[budget["validation"]])
scaled = namespaces["wineScaling"]
assert abs(scaled["scaled_pca"].explained_variance_ratio_[:8].sum() - 0.9202) < 5e-4
assert abs(scaled["scaled_pca"].explained_variance_ratio_[:10].sum() - 0.9617) < 5e-4
assert abs(scaled["raw_pca"].components_[0][-1]) > 0.999  # proline dominates raw PC1
weights = scaled["scaled_pca"].components_[0]
sign = 1 if weights[6] > 0 else -1
assert np.allclose(sign * weights[[5, 6, 7, 12]], [0.3947, 0.4229, -0.2985, 0.2868], atol=5e-4)
gaussian = namespaces["gaussian"]["pca"]
assert abs(gaussian.explained_variance_ratio_[:2].sum() - 0.2459) < 5e-4
whitened = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(n_components=2, whiten=True, svd_solver="full").fit_transform(svd_namespace["X"])
assert np.allclose(whitened.var(axis=0, ddof=1), [1, 1]) and np.allclose(whitened.var(axis=0, ddof=0), [0.75, 0.75])
oracle_count = 12

if write:
    module_path.write_text(
        "// Complete displayed programs for the PCA lesson, executed by scripts/verify-pca-examples.py.\n"
        "// Programs with `continues` extend the named program's namespace, as the lesson instructs.\n"
        "export const pcaExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-pca-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn"]},
    "programs": {key: {"codeHash": digest(record["code"]), "stdoutHash": digest(record["expected"]), "stdout": record["expected"], **({"continues": record["continues"]} if "continues" in record else {})} for key, record in records.items()},
    "oracles": oracle_count,
    "limits": [
        "Wine values come from the scikit-learn bundled mirror of UCI Wine; no UCI download was made.",
        "Fold accuracies are development evidence with n_jobs=1 and single-threaded BLAS; they are not a final test result or a timing benchmark.",
        "The whitening probe checks ddof conventions on the four-point fixture only.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions.")
