"""Execute the cross-validation lesson's displayed Python programs.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/validation-examples.js from the executed programs; without it the
recorded output must match a fresh execution, so a stale block cannot ship.

The penguin programs run inside public/learn-assets/cross-validation, beside the
same CSV bytes a learner downloads, so `Path(__file__).with_name("penguins.csv")`
resolves exactly as the lesson instructs. The supplementary Optuna study imports
the main program by name, which is what its first line says it does.

The oracle block at the end asserts the manuscript's stated values against these
runs. A program whose printed output drifts from the prose fails here rather
than on the page.
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

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public/learn-assets/cross-validation"
MODULE = ROOT / "src/learn/data/validation-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/validation-native.json"
PACKET = ROOT / "docs/teaching/drafts/cross-validation-hyperparameter-tuning"

PROGRAMS = {
    "tinySplitter": {
        "file": "cv_tiny.py",
        "title": "A splitter that keeps every row",
        "question": "seven rows split into three consecutive folds. Will the unweighted fold mean equal the pooled row accuracy?",
        "code": r'''
import numpy as np

def kfold_indices(n, k, shuffle=False, seed=0):
    if not 2 <= k <= n:
        raise ValueError("Require 2 <= k <= n.")
    indices = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, k)
    for held in range(k):
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != held]))
        yield train, folds[held]

x = np.arange(7)
y = np.array([0, 0, 0, 1, 1, 1, 1])
fold_scores = []
total_correct = 0
for train, validation in kfold_indices(len(x), 3):
    distances = np.abs(x[validation, None] - x[train])
    nearest = train[np.argmin(distances, axis=1)]
    prediction = y[nearest]
    correct = int(np.sum(prediction == y[validation]))
    fold_scores.append(correct / len(validation))
    total_correct += correct
    print(validation.tolist(), prediction.tolist(), correct, "/", len(validation))
print("fold mean", np.mean(fold_scores))
print("pooled", total_correct / len(x))
''',
    },
    "nestedPenguins": {
        "file": "cv_penguins.py",
        "title": "A complete nested experiment on 344 real penguins",
        "question": "each outer fold runs its own six-candidate search. Will the three outer folds select the same neighbour count and scaler?",
        "code": r'''
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

def load_data():
    data = pd.read_csv(Path(__file__).with_name("penguins.csv"))
    columns = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex",
    ]
    return data[columns], data["species"]

def make_model():
    numeric = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
    ]
    prepare = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("category", Pipeline([
            ("impute", SimpleImputer(
                strategy="constant", fill_value="not_recorded",
                keep_empty_features=True,
            )),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["sex"]),
    ])
    return Pipeline([
        ("prepare", prepare),
        ("classify", KNeighborsClassifier()),
    ])

def main():
    X, y = load_data()
    grid = {
        "prepare__numeric__scale": [StandardScaler(), RobustScaler()],
        "classify__n_neighbors": [3, 5, 11],
    }
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=41)
    held_out_predictions = np.empty(len(y), dtype=object)
    fold_scores = []
    for fold, (train, test) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=73)
        search = GridSearchCV(
            make_model(), grid, cv=inner, scoring="accuracy",
            n_jobs=1, error_score="raise",
        )
        search.fit(X.iloc[train], y.iloc[train])
        prediction = search.predict(X.iloc[test])
        held_out_predictions[test] = prediction
        correct = int(np.sum(prediction == y.iloc[test]))
        fold_scores.append(correct / len(test))
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(np.zeros((len(train), 1)), y.iloc[train])
        base_prediction = baseline.predict(np.zeros((len(test), 1)))
        base_correct = int(np.sum(base_prediction == y.iloc[test]))
        chosen_k = search.best_params_["classify__n_neighbors"]
        chosen_scaler = type(search.best_params_["prepare__numeric__scale"]).__name__
        print(fold, chosen_k, chosen_scaler, correct, "/", len(test),
              "baseline", base_correct)

    print("outer fold mean", f"{np.mean(fold_scores):.7f}")
    print("outer pooled", f"{np.mean(held_out_predictions == y):.7f}")
    final = GridSearchCV(
        make_model(), grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=73),
        scoring="accuracy", n_jobs=1, error_score="raise",
    )
    final.fit(X, y)
    print("final settings", final.best_params_)
    print("final selection score", f"{final.best_score_:.7f}")

if __name__ == "__main__":
    main()
''',
    },
    "optunaStudy": {
        "file": "cv_optuna.py",
        "title": "An optional adaptive search over a different space",
        "question": "a TPE sampler replaces the six-candidate grid with a larger space. Is its selection score comparable evidence to the grid's?",
        "interpreter": "optuna",
        "code": r'''
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from cv_penguins import load_data, make_model

X, y = load_data()
splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=73)
splits = list(splitter.split(X, y))

def objective(trial):
    model = make_model()
    model.set_params(
        classify__n_neighbors=trial.suggest_int("neighbors", 1, 21, step=2),
        classify__p=trial.suggest_int("distance_power", 1, 2),
        classify__weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
    )
    scores = cross_val_score(
        model, X, y, cv=splits, scoring="accuracy", n_jobs=1,
        error_score="raise",
    )
    return float(scores.mean())

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=29, n_startup_trials=10),
)
study.optimize(objective, n_trials=20, n_jobs=1)
print("selection score", study.best_value)
print("selected settings", study.best_params)
''',
    },
}

write = "--write" in sys.argv
digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(code, namespace, quiet=False):
    """Execute a displayed program beside the served CSV and capture stdout."""
    with contextlib.chdir(ASSETS), contextlib.redirect_stdout(io.StringIO()) as stream:
        if quiet:
            import logging
            logging.disable(logging.CRITICAL)
        exec(compile(code, "displayed-program", "exec"), namespace)
        if quiet:
            import logging
            logging.disable(logging.NOTSET)
    return stream.getvalue().strip()


records, namespaces = {}, {}
for key, item in PROGRAMS.items():
    code = item["code"].strip() + "\n"
    namespace = {"__file__": str(ASSETS / item["file"]), "__name__": "__main__"}
    if key == "optunaStudy":
        # The lesson says to save this beside cv_penguins.py and import from it.
        (ASSETS / "cv_penguins.py").write_text(
            PROGRAMS["nestedPenguins"]["code"].strip() + "\n", encoding="utf-8", newline="\n")
        sys.path.insert(0, str(ASSETS))
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            output = run(code, namespace)
        finally:
            sys.path.remove(str(ASSETS))
            (ASSETS / "cv_penguins.py").unlink()
            sys.modules.pop("cv_penguins", None)
            # The served asset folder must contain only the files a learner gets.
            cache = ASSETS / "__pycache__"
            if cache.exists():
                for cached in cache.iterdir():
                    cached.unlink()
                cache.rmdir()
    else:
        output = run(code, namespace)
    namespaces[key] = namespace
    records[key] = {
        "title": item["title"],
        "question": item["question"],
        "file": item["file"],
        "code": code.rstrip("\n"),
        "expected": output,
        "language": "python",
    }
    print(f"Executed {key}: {len(code.splitlines())} lines, {len(output.splitlines())} output lines")

if not write:
    existing = MODULE.read_text(encoding="utf-8")
    prefix = "export const validationExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    assert set(recorded) == set(records), "the recorded program set changed"
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], (
            f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}")

# ------------------------------------------------------------------ #
# Oracles: the manuscript's stated values must come out of these runs.
# ------------------------------------------------------------------ #
oracles = 0


def oracle(condition, message):
    global oracles
    assert condition, message
    oracles += 1


saved = json.loads((PACKET / "calculated-inputs.json").read_text(encoding="utf-8"))

tiny = records["tinySplitter"]["expected"].splitlines()
oracle(tiny[0] == "[0, 1, 2] [1, 1, 1] 0 / 3", "first fold predicts all ones and gets 0 of 3")
oracle(tiny[1] == "[3, 4] [0, 1] 1 / 2", "second fold predicts 0 then 1")
oracle(tiny[2] == "[5, 6] [1, 1] 2 / 2", "third fold is correct twice")
oracle(tiny[3] == "fold mean 0.5", "the unweighted fold mean is 0.5")
oracle(tiny[4] == "pooled 0.42857142857142855", "pooled accuracy is 3/7 as printed")
tiny_namespace = namespaces["tinySplitter"]
oracle([len(fold) for fold in np.array_split(np.arange(7), 3)] == [3, 2, 2],
       "array_split keeps the remainder: folds of 3, 2, 2")
oracle(sum(len(v) for _, v in tiny_namespace["kfold_indices"](11, 3)) == 11,
       "n=11, k=3 still gives every row a validation turn")
oracle(sorted(int(i) for _, v in tiny_namespace["kfold_indices"](11, 3) for i in v) == list(range(11)),
       "no row is dropped and none is assessed twice at n=11")
try:
    list(tiny_namespace["kfold_indices"](5, 6))
    raise AssertionError("k > n must be refused")
except ValueError:
    oracles += 1
for saved_fold, line in zip(saved["tiny"], tiny[:3]):
    shown = line.split("] [")
    oracle(shown[0] + "]" == str(saved_fold["validation"]), "the printed held-out IDs match the packet trace")
    oracle("[" + shown[1].split("]")[0] + "]" == str(saved_fold["prediction"]), "the printed predictions match the packet trace")
oracle([fold["correct"] for fold in saved["tiny"]] == [0, 1, 2], "packet tiny trace records 0, 1 and 2 correct")
oracle(sum(fold["correct"] for fold in saved["tiny"]) == 3, "three of seven held-out predictions are correct")

penguins = records["nestedPenguins"]["expected"].splitlines()
oracle(penguins[0] == "1 11 StandardScaler 114 / 115 baseline 51", "outer fold 1 selects 11/Standard and gets 114 of 115")
oracle(penguins[1] == "2 3 StandardScaler 114 / 115 baseline 51", "outer fold 2 selects 3/Standard and gets 114 of 115")
oracle(penguins[2] == "3 3 RobustScaler 112 / 114 baseline 50", "outer fold 3 selects 3/Robust and gets 112 of 114")
oracle(penguins[3] == "outer fold mean 0.9883549", "the unweighted outer-fold mean is 0.9883549")
oracle(penguins[4] == "outer pooled 0.9883721", "pooled accuracy is 340/344 = 0.9883721")
oracle(penguins[6] == "final selection score 0.9913043", "the final inner selection score is 0.9913043")
oracle("'classify__n_neighbors': 3" in penguins[5] and "StandardScaler()" in penguins[5],
       "the final all-data search selects 3 neighbours with standard scaling")
oracle(abs(saved["fold_mean"] - 0.9883549453343504) < 1e-15, "packet fold mean")
oracle(abs(saved["pooled_accuracy"] - 340 / 344) < 1e-15, "packet pooled accuracy is exactly 340/344")
oracle(saved["final_selected"] == {"k": 3, "scaler": "StandardScaler", "inner_score": 0.991304347826087},
       "packet final selection")
oracle([record["correct"] for record in saved["outer"]] == [114, 114, 112], "packet outer correct counts")
oracle([record["baseline_correct"] for record in saved["outer"]] == [51, 51, 50], "packet baseline counts")
oracle([len(record["test_rows"]) for record in saved["outer"]] == [115, 115, 114], "packet outer fold sizes")
oracle([len(record["train_rows"]) for record in saved["outer"]] == [229, 229, 230], "packet outer training sizes")
selected = [record["candidates"][record["best_index"]] for record in saved["outer"]]
oracle([(row["k"], row["scaler"]) for row in selected] == [(11, "StandardScaler"), (3, "StandardScaler"), (3, "RobustScaler")],
       "packet selections match the printed ones")
for record in saved["outer"]:
    best = max(row["mean_score"] for row in record["candidates"])
    first = next(index for index, row in enumerate(record["candidates"]) if row["mean_score"] == best)
    oracle(first == record["best_index"], f"fold {record['fold']} breaks its tie on the first enumerated candidate")
oracle(sum(1 for row in saved["outer"][2]["candidates"] if row["mean_score"] == max(r["mean_score"] for r in saved["outer"][2]["candidates"])) == 5,
       "outer fold 3 really has a five-way tie that the rule resolves")
oracle(abs(sum(record["accuracy"] for record in saved["outer"]) / 3 - saved["fold_mean"]) < 1e-15,
       "the fold mean is the unweighted average of the three fold accuracies")
oracle(abs(sum(record["correct"] for record in saved["outer"]) / 344 - saved["pooled_accuracy"]) < 1e-15,
       "pooled accuracy weights each row equally")
oracle(saved["fold_mean"] != saved["pooled_accuracy"], "the two summaries genuinely differ here")

optuna_output = records["optunaStudy"]["expected"].splitlines()
oracle(optuna_output[0].startswith("selection score "), "the study prints its selection score")
oracle(optuna_output[1].startswith("selected settings {"), "the study prints its selected settings")
study = namespaces["optunaStudy"]["study"]
oracle(len(study.trials) == 20, "twenty trials were run")
oracle(all(trial.value is not None for trial in study.trials), "every trial returned a score")
oracle(study.best_value == max(trial.value for trial in study.trials), "the reported best is the best observed")
oracle(set(study.best_params) == {"neighbors", "distance_power", "weights"}, "the search space is the one the program declares")
oracle(study.best_params["neighbors"] % 2 == 1 and 1 <= study.best_params["neighbors"] <= 21,
       "the neighbour count respects the declared odd grid")
oracle(len(namespaces["optunaStudy"]["splits"]) == 3, "the study reuses one fixed three-fold split")
oracle(study.best_value <= 1.0, "an accuracy cannot exceed 1")

# The sentence the lesson prints about this run must be a checked fact, not a
# recollection: the study's space contains the grid's selected setting, the
# folds are the same ones the final grid search uses, and that point scores
# strictly higher than twenty TPE trials reached.
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

optuna_namespace = namespaces["optunaStudy"]
X_study, y_study = optuna_namespace["X"], optuna_namespace["y"]
reference_splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=73).split(X_study, y_study))
oracle(all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
           for a, b in zip(optuna_namespace["splits"], reference_splits)),
       "the study scores candidates on the same three folds the final grid search uses")
grid_point = optuna_namespace["make_model"]()
grid_point.set_params(classify__n_neighbors=3, classify__p=2, classify__weights="uniform")
grid_point_score = float(cross_val_score(grid_point, X_study, y_study, cv=optuna_namespace["splits"],
                                         scoring="accuracy", n_jobs=1, error_score="raise").mean())
oracle(abs(grid_point_score - 0.991304347826087) < 1e-12,
       "the grid's selected setting scores 0.9913043 on those same folds")
oracle(grid_point_score > study.best_value,
       "that point is strictly better than the best of the study's twenty trials")
oracle(3 in range(1, 22, 2) and 2 in (1, 2),
       "and it lies inside the study's declared space: 3 is an odd count in 1..21, power 2 and uniform weights are both offered")
oracle(all((trial.params["neighbors"], trial.params["distance_power"], trial.params["weights"])
           != (3, 2, "uniform") for trial in study.trials),
       "no trial in this run actually evaluated that point")

versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "optuna"]}

if write:
    MODULE.write_text(
        "// Complete displayed programs for the cross-validation lesson, executed by\n"
        "// scripts/verify-validation-examples.py. `file` is the filename the lesson asks\n"
        "// the learner to save the block as. Do not edit by hand: the verifier re-runs\n"
        "// every program and refuses to pass when a recorded output drifts.\n"
        "export const validationExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": "src/learn/data/validation-examples.js",
    "sourceHash": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-validation-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python": sys.version.split()[0],
    "versions": versions,
    "programs": {
        key: {
            "file": record["file"],
            "codeHash": digest(record["code"]),
            "stdoutHash": digest(record["expected"]),
            "stdout": record["expected"],
        } for key, record in records.items()
    },
    "oracles": oracles,
    "limits": [
        "The penguin programs ran beside the served copy of the supplied CSV; no network request was made.",
        "One dataset, one stratified seed pair and one candidate grid: a demonstration of a protocol, not new evidence about penguins or about any scaler.",
        "The Optuna study is a development search only. Its printed score selected its own settings and is not an assessment of them.",
        "The Optuna program was explicitly unexecuted in the content phase; this run used Optuna 5.0.0 in an isolated environment built for phase two. A different Optuna or scikit-learn version can move its trial sequence.",
        "Fit counts are exact; no timing or speed claim is recorded.",
    ],
}
EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(records)} displayed programs executed, {oracles} oracle assertions.")
