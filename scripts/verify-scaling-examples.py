"""Execute the feature-preparation lesson's displayed Python programs.

Run with the isolated lesson Python. Read-only by default, so an independent
reviewer can re-run it inside a no-write boundary: it executes both programs and
asserts that the recorded code and output match a fresh run. `--write`
regenerates src/learn/data/scaling-examples.js and the evidence file.

The penguin program reads the CSV the lesson offers for download, from
public/learn-assets/feature-scaling, which is the same bytes a learner receives,
so `Path(__file__).with_name("penguins.csv")` resolves exactly as the displayed
line says it does.
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

ASSETS = Path("public/learn-assets/feature-scaling").resolve()

PROGRAMS = {
    "penguinExperiment": {
        "file": "penguins_prepare.py",
        "title": "The complete offline penguin experiment",
        "question": "the same split, the same neighbour count and the same categorical preparation throughout; only the numerical ruler changes. Will any ruler leave the majority baseline behind, and will the three scaled rulers separate from one another?",
        "code": r"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler,
)

data = pd.read_csv(Path(__file__).with_name("penguins.csv"))
numeric = [
    "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
]
categorical = ["sex"]
X = data[numeric + categorical]
y = data["species"]
train, test = train_test_split(
    np.arange(len(data)), test_size=0.25, random_state=20, stratify=y,
)

def make_model(scaler):
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", scaler),
    ])
    category_steps = Pipeline([
        ("impute", SimpleImputer(
            strategy="constant", fill_value="not_recorded",
            keep_empty_features=True,
        )),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prepare = ColumnTransformer([
        ("numeric", numeric_steps, numeric),
        ("category", category_steps, categorical),
    ])
    return Pipeline([
        ("prepare", prepare),
        ("classify", KNeighborsClassifier(n_neighbors=5)),
    ])

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(np.zeros((len(train), 1)), y.iloc[train])
baseline_prediction = baseline.predict(np.zeros((len(test), 1)))
print("majority", int(np.sum(baseline_prediction == y.iloc[test])), "/", len(test))

models = {}
for name, scaler in [
    ("raw", "passthrough"),
    ("standard", StandardScaler()),
    ("minmax", MinMaxScaler()),
    ("robust", RobustScaler()),
]:
    model = make_model(scaler)
    model.fit(X.iloc[train], y.iloc[train])
    prediction = model.predict(X.iloc[test])
    correct = int(np.sum(prediction == y.iloc[test]))
    print(name, correct, "/", len(test), f"{correct / len(test):.4f}")
    models[name] = model

prepare = models["standard"].named_steps["prepare"]
print(prepare.get_feature_names_out())
print(np.round(prepare.transform(X.iloc[test[:1]]), 4))
""",
    },
    "crossFitEncoding": {
        "file": "cross_fit.py",
        "title": "Cross-fitted target encoding, and the target that changed everything else",
        "question": "row 0's own target is about to change. Which of the six encoded values can move, and which cannot?",
        "code": r"""
import numpy as np

def cross_fit(categories, target, folds, smoothing=2.0):
    encoded = np.empty(len(target), dtype=float)
    for held_fold in np.unique(folds):
        donors = folds != held_fold
        prior = target[donors].mean()
        for row in np.flatnonzero(~donors):
            matching = donors & (categories == categories[row])
            encoded[row] = (
                target[matching].sum() + smoothing * prior
            ) / (matching.sum() + smoothing)
    return encoded

categories = np.array(["A", "A", "B", "B", "C", "C"])
target = np.array([1., 1., 0., 0., 1., 0.])
folds = np.array([0, 1, 0, 1, 0, 1])
print(np.round(cross_fit(categories, target, folds), 6))
changed = target.copy()
changed[0] = 0.
print(np.round(cross_fit(categories, changed, folds), 6))
""",
    },
}

module_path = Path("src/learn/data/scaling-examples.js")
evidence_path = Path("docs/teaching/evidence/scaling-native.json")
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


records, namespaces = {}, {}
for key, item in PROGRAMS.items():
    code = item["code"].strip() + "\n"
    namespace = {"__file__": str(ASSETS / item["file"])}
    with contextlib.chdir(ASSETS), threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"scaling-example:{key}", "exec"), namespace)
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
    prefix = "export const scalingExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], (
            f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}")

# Oracles: the manuscript's stated values must come out of these runs, not out
# of a recorded table.
penguins = namespaces["penguinExperiment"]
printed = records["penguinExperiment"]["expected"].splitlines()
assert printed[0] == "majority 38 / 86", printed[0]
assert printed[1] == "raw 67 / 86 0.7791", printed[1]
assert printed[2] == "standard 84 / 86 0.9767", printed[2]
assert printed[3] == "minmax 85 / 86 0.9884", printed[3]
assert printed[4] == "robust 85 / 86 0.9884", printed[4]
assert len(penguins["data"]) == 344, "344 observations"
assert len(penguins["train"]) == 258 and len(penguins["test"]) == 86
assert int(penguins["test"][0]) == 309, "the first held-out row is zero-based source row 309"
prepare = penguins["prepare"]
names = prepare.get_feature_names_out().tolist()
assert names[4:] == ["category__sex_female", "category__sex_male", "category__sex_not_recorded"]
numeric_branch = prepare.named_transformers_["numeric"]
medians = numeric_branch.named_steps["impute"].statistics_.tolist()
assert medians == [45.0, 17.3, 197.0, 4000.0], medians
scaler = numeric_branch.named_steps["scale"]
assert np.allclose(scaler.mean_, [43.9306, 17.0992, 200.6589, 4190.9884], atol=1e-4), "fitted training means"
assert np.allclose(scaler.scale_, [5.4000, 1.9387, 14.2307, 806.6876], atol=1e-4), "fitted training scales"
first_row = prepare.transform(penguins["X"].iloc[penguins["test"][:1]])[0]
assert np.allclose(
    np.round(first_row, 4), [1.3091, 0.8773, 0.1645, -0.1128, 0, 1, 0], atol=0), "the transformed source row 309"
assert first_row[3] < 0, "a mass below the fitted mean gives a negative coordinate, not a negative mass"
# The fit used training rows only: recomputing from those rows alone reproduces
# every statistic, and the held-out block is never consulted.
training_block = penguins["X"].iloc[penguins["train"]][penguins["numeric"]].to_numpy(dtype=float)
filled = np.where(np.isnan(training_block), np.array(medians), training_block)
assert np.allclose(np.nanmedian(training_block, axis=0), medians, atol=1e-12), "medians are training medians"
assert np.allclose(filled.mean(axis=0), scaler.mean_, atol=1e-9), "means are training means"
assert np.allclose(filled.std(axis=0, ddof=0), scaler.scale_, atol=1e-9), "scales use the population divisor n"
# Practice 6: the same record with its mass missing takes the fitted median.
edited = penguins["X"].iloc[penguins["test"][:1]].copy()
edited.loc[edited.index[0], "body_mass_g"] = np.nan
edited_row = prepare.transform(edited)[0]
assert abs(edited_row[3] - (4000.0 - scaler.mean_[3]) / scaler.scale_[3]) < 1e-12
assert abs(edited_row[3] + 0.2368) < 5e-5, "practice 6's approximately -0.2368"
assert np.allclose(np.delete(edited_row, 3), np.delete(first_row, 3), atol=0), "no other coordinate moves"

encoding = namespaces["crossFitEncoding"]
assert np.allclose(
    encoding["cross_fit"](encoding["categories"], encoding["target"], encoding["folds"]),
    [5 / 9, 7 / 9, 2 / 9, 4 / 9, 2 / 9, 7 / 9], atol=1e-12), "the base cross-fitted array"
assert np.allclose(
    encoding["cross_fit"](encoding["categories"], encoding["changed"], encoding["folds"]),
    [5 / 9, 2 / 9, 2 / 9, 2 / 9, 2 / 9, 5 / 9], atol=1e-12), "the changed-target array"
assert records["crossFitEncoding"]["expected"].splitlines()[0] == (
    "[0.555556 0.777778 0.222222 0.444444 0.222222 0.777778]")
assert records["crossFitEncoding"]["expected"].splitlines()[1] == (
    "[0.555556 0.222222 0.222222 0.222222 0.222222 0.555556]")
# Practice 7: changing row 3's target moves row 0 through the fold prior alone,
# while row 3's own encoding is unchanged.
practice = encoding["target"].copy()
practice[3] = 1.0
practice_encoded = encoding["cross_fit"](encoding["categories"], practice, encoding["folds"])
assert abs(practice_encoded[0] - 7 / 9) < 1e-12 and abs(practice_encoded[3] - 4 / 9) < 1e-12
oracle_count = 24

if write:
    module_path.write_text(
        "// Complete displayed programs for the feature-preparation lesson, executed by\n"
        "// scripts/verify-scaling-examples.py. `file` is the filename the lesson asks the\n"
        "// learner to save the block as; the penguin program reads penguins.csv beside it.\n"
        "export const scalingExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-scaling-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
    "programs": {key: {"file": record["file"], "codeHash": digest(record["code"]),
                       "stdoutHash": digest(record["expected"]), "stdout": record["expected"]}
                 for key, record in records.items()},
    "oracles": oracle_count,
    "limits": [
        "The penguin program ran against the served copy of the supplied CSV; no network access was used.",
        "One fixed stratified split of one curated collection; the comparison is a single controlled demonstration.",
        "One extra correct row does not establish that min-max or robust scaling is generally better than standard scaling.",
        "Numerical results can differ on other library versions.",
    ],
}
if write:
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
elif evidence_path.exists():
    recorded_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert recorded_evidence["sourceHash"] == evidence["sourceHash"], "recorded evidence describes a different module; re-run with --write"
    for key, record in evidence["programs"].items():
        assert recorded_evidence["programs"][key]["stdoutHash"] == record["stdoutHash"], f"{key}: recorded stdout hash differs; re-run with --write"
print(
    f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions"
    f"{', module and evidence rewritten' if write else ' (read-only)'}."
)
