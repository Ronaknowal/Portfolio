"""Execute the feature-preparation lesson's displayed Python programs.

Run with the isolated lesson Python. Read-only by default, so an independent
reviewer can re-run it inside a no-write boundary: it executes all programs and
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
    "fittedStateParity": {
        "file": "fitted_preparation.py",
        "title": "Build the fitted state, then match the library transformation",
        "question": "Follow the saved medians, means, scales and category order into two new rows, including a changed constant feature and an unseen category.",
        "code": r"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def inputs(numeric, categories):
    numeric = np.asarray(numeric, dtype=float)
    if numeric.ndim != 2 or 0 in numeric.shape or np.isinf(numeric).any():
        raise ValueError("Use a nonempty numeric matrix: finite values or NaN.")
    if len(categories) != len(numeric) or any(
        value is not None and not isinstance(value, str) for value in categories
    ):
        raise ValueError("Use one string or None category per row.")
    if "not_recorded" in categories:
        raise ValueError("not_recorded is reserved; use None for a missing category.")
    category = np.array(["not_recorded" if value is None else value
                         for value in categories])
    return numeric, category

def fit_preparation(numeric, categories):
    numeric, category = inputs(numeric, categories)
    if np.isnan(numeric).all(axis=0).any():
        raise ValueError("This teaching fit requires an observation in every column.")
    with np.errstate(over="raise", invalid="raise"):
        median = np.nanmedian(numeric, axis=0)
        filled = np.where(np.isnan(numeric), median, numeric)
        scale = filled.std(axis=0, ddof=0)
        mean = filled.mean(axis=0)
    return {"median": median, "mean": mean,
            "scale": np.where(scale == 0, 1.0, scale),
            "categories": np.unique(category)}

def transform_preparation(state, numeric, categories):
    numeric, category = inputs(numeric, categories)
    if numeric.shape[1] != len(state["median"]):
        raise ValueError("Keep the fitted numeric feature schema.")
    filled = np.where(np.isnan(numeric), state["median"], numeric)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scaled = (filled - state["mean"]) / state["scale"]
    indicators = (category[:, None] == state["categories"]).astype(float)
    return np.column_stack([scaled, indicators])

train = np.array([[10., 100., 7.], [20., np.nan, 7.],
                  [30., 300., 7.], [np.nan, 500., 7.]])
train_categories = ["b", "a", None, "a"]
query = np.array([[25., np.nan, 9.], [np.nan, 700., 7.]])
query_categories = ["new", None]
state = fit_preparation(train, train_categories)
snapshot = {key: value.copy() for key, value in state.items()}
manual = transform_preparation(state, query, query_categories)

imputer = SimpleImputer(strategy="median").fit(train)
scaler = StandardScaler().fit(imputer.transform(train))
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(inputs(train, train_categories)[1][:, None])
library = np.column_stack([
    scaler.transform(imputer.transform(query)),
    encoder.transform(inputs(query, query_categories)[1][:, None]),
])
np.testing.assert_allclose(manual, library, rtol=0, atol=1e-12)
np.testing.assert_allclose(state["median"], imputer.statistics_)
np.testing.assert_allclose(state["mean"], scaler.mean_)
np.testing.assert_allclose(state["scale"], scaler.scale_)
np.testing.assert_array_equal(state["categories"], encoder.categories_[0])

changed = query.copy()
changed[1, 1] = 1000.
edited = transform_preparation(state, changed, query_categories)
assert all(np.array_equal(state[key], before) for key, before in snapshot.items())
assert np.array_equal(edited[0], manual[0])
print("category columns:", state["categories"].tolist())
print("medians:", state["median"].tolist())
print("new rows:", np.round(manual, 6).tolist())
print("library parity:", np.allclose(manual, library, rtol=0, atol=1e-12))
print("edited second coordinate:", round(edited[1, 1], 6))
""",
    },
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
paired = namespaces["fittedStateParity"]
assert np.allclose(paired["manual"][:, :3], [[2**-.5, 0, 2], [0, 2**1.5, 0]])
assert paired["manual"][0, 3:].tolist() == [0, 0, 0], "unknown category has no invented coordinate"
assert paired["manual"][1, 3:].tolist() == [0, 0, 1], "missing is a fitted category"
assert np.isclose(paired["edited"][1, 1], 7/2**.5), "only the edited value changes its coordinate"
for label, call in [
    ("all-missing training column", lambda: paired["fit_preparation"]([[np.nan]], ["a"])),
    ("changed feature width", lambda: paired["transform_preparation"](paired["state"], [[2., 3.]], ["a"])),
    ("infinite measurement", lambda: paired["transform_preparation"](paired["state"], [[np.inf, 1., 2.]], ["a"])),
    ("reserved missing marker", lambda: paired["fit_preparation"]([[1.]], ["not_recorded"])),
]:
    try:
        call()
    except ValueError:
        passed = True
    else:
        passed = False
    assert passed, label
try:
    paired["fit_preparation"]([[1e300], [-1e300]], ["a", "b"])
except FloatingPointError:
    range_rejected = True
else:
    range_rejected = False
assert range_rejected, "an unrepresentable variance must not silently produce a false zero coordinate"
# Execute the practice modification itself, with an independent library indicator.
from sklearn.impute import MissingIndicator
practice_code = PROGRAMS["fittedStateParity"]["code"].replace(
    '    filled = np.where(np.isnan(numeric), state["median"], numeric)',
    '    missing = np.isnan(numeric).astype(float)\n    filled = np.where(np.isnan(numeric), state["median"], numeric)',
).replace('return np.column_stack([scaled, indicators])',
          'return np.column_stack([scaled, indicators, missing])')
# Only function definitions, so existing six-column demonstration assertions stay intact.
practice_namespace = {}
exec(practice_code.split("train = np.array(")[0], practice_namespace)
practice_input = np.array([[15., 500., np.nan]])
before_practice = {key: value.copy() for key, value in paired["state"].items()}
practice_result = practice_namespace["transform_preparation"](paired["state"], practice_input, ["b"])
assert np.allclose(practice_result, [[-2**-.5, 2**.5, 0, 0, 1, 0, 0, 0, 1]])
assert np.array_equal(practice_result[:, :6], paired["transform_preparation"](paired["state"], practice_input, ["b"]))
assert all(np.array_equal(paired["state"][key], before) for key, before in before_practice.items())
library_indicator = MissingIndicator(features="all", error_on_new=False).fit(paired["train"])
assert np.array_equal(practice_result[:, -3:], library_indicator.transform(practice_input))
oracle_count = 37

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
