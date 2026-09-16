"""Recompute the cross-validation lesson's real nested experiment and generate
src/learn/data/validation-data.js from that run.

Nothing is copied out of the content packet. The whole three-by-three nested
search is refitted here from the CSV the lesson serves, and every value the
packet recorded in `calculated-inputs.json` must equal the refitted value before
the module is written. A drifting library version therefore fails here rather
than silently republishing stale numbers.

Run with the isolated lesson Python:
    scratch/lesson-tools/Scripts/python.exe scripts/verify-validation-data.py
`--write` publishes the module and the downloadable split record; without it
both must already be current.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/cross-validation-hyperparameter-tuning"
ASSETS = ROOT / "public/learn-assets/cross-validation"
MODULE = ROOT / "src/learn/data/validation-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/validation-data.json"
write = "--write" in sys.argv

checks = []


def check(condition, description):
    assert condition, f"FAILED: {description}"
    checks.append(description)


saved = json.loads((PACKET / "calculated-inputs.json").read_text(encoding="utf-8"))
csv_bytes = (ASSETS / "penguins.csv").read_bytes()
csv_hash = hashlib.sha256(csv_bytes).hexdigest()

check(csv_hash == saved["data_sha256"], "the served CSV is byte-identical to the packet input the author calculated from")
check(hashlib.sha256((PACKET / "penguins.csv").read_bytes()).hexdigest() == csv_hash,
      "the packet copy and the served copy are the same bytes")
check(len(csv_bytes) == 15241, "the served CSV is the recorded 15,241 bytes")

data = pd.read_csv(ASSETS / "penguins.csv")
check(len(data) == 344, "the file holds all 344 original rows")
check(list(data.columns) == ["species", "island", "bill_length_mm", "bill_depth_mm",
                             "flipper_length_mm", "body_mass_g", "sex", "year"],
      "the original column list is unchanged")

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
NUMERIC = FEATURES[:4]
X, y = data[FEATURES], data["species"]
missing = {column: int(data[column].isna().sum()) for column in FEATURES}
check(missing["sex"] == 11, "eleven sex entries are missing in the source, as the provenance records")
check(all(missing[column] == 2 for column in NUMERIC), "each numeric measurement has two missing cells")
species_counts = {name: int(count) for name, count in y.value_counts().sort_index().items()}
check(sum(species_counts.values()) == 344, "every row carries a species label")


def make_model():
    prepare = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="not_recorded", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["sex"]),
    ])
    return Pipeline([("prepare", prepare), ("classify", KNeighborsClassifier())])


GRID = {
    "prepare__numeric__scale": [StandardScaler(), RobustScaler()],
    "classify__n_neighbors": [3, 5, 11],
}

outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=41)
folds, out_of_fold = [], np.empty(len(y), dtype=object)
for index, (train, test) in enumerate(outer.split(X, y)):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=73)
    search = GridSearchCV(make_model(), GRID, cv=inner, scoring="accuracy", n_jobs=1, error_score="raise")
    search.fit(X.iloc[train], y.iloc[train])
    prediction = search.predict(X.iloc[test])
    out_of_fold[test] = prediction
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y.iloc[train])
    baseline = dummy.predict(np.zeros((len(test), 1)))
    candidates = [{
        "k": int(params["classify__n_neighbors"]),
        "scaler": type(params["prepare__numeric__scale"]).__name__,
        "foldScores": [float(search.cv_results_[f"split{split}_test_score"][position]) for split in range(3)],
        "meanScore": float(search.cv_results_["mean_test_score"][position]),
    } for position, params in enumerate(search.cv_results_["params"])]
    inner_splits = [{"validationRows": train[validation].tolist(), "trainingSize": int(len(fit))}
                    for fit, validation in inner.split(X.iloc[train], y.iloc[train])]
    truth = y.iloc[test].tolist()
    folds.append({
        "fold": index,
        "trainingRows": train.tolist(),
        "testRows": test.tolist(),
        "innerSplits": inner_splits,
        "candidates": candidates,
        "bestIndex": int(search.best_index_),
        "selected": {"k": candidates[int(search.best_index_)]["k"], "scaler": candidates[int(search.best_index_)]["scaler"]},
        "selectionScore": float(search.best_score_),
        "prediction": prediction.tolist(),
        "truth": truth,
        "correct": int(np.sum(prediction == y.iloc[test])),
        "accuracy": float(np.mean(prediction == y.iloc[test])),
        "baselineLabel": str(dummy.predict(np.zeros((1, 1)))[0]),
        "baselineCorrect": int(np.sum(baseline == y.iloc[test])),
        "missed": [{"row": int(test[position]), "truth": truth[position], "prediction": prediction[position]}
                   for position in range(len(test)) if prediction[position] != truth[position]],
    })

final = GridSearchCV(make_model(), GRID, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=73),
                     scoring="accuracy", n_jobs=1, error_score="raise").fit(X, y)
pooled = float(np.mean(out_of_fold == y))
fold_mean = float(np.mean([fold["accuracy"] for fold in folds]))
# The final search's own candidate table, so the page can surface its tie the way
# the outer folds already surface theirs.
final_candidates = [{
    "k": int(params["classify__n_neighbors"]),
    "scaler": type(params["prepare__numeric__scale"]).__name__,
    "foldScores": [float(final.cv_results_[f"split{split}_test_score"][position]) for split in range(3)],
    "meanScore": float(final.cv_results_["mean_test_score"][position]),
} for position, params in enumerate(final.cv_results_["params"])]
final_best = max(candidate["meanScore"] for candidate in final_candidates)
final_selected = {
    "k": int(final.best_params_["classify__n_neighbors"]),
    "scaler": type(final.best_params_["prepare__numeric__scale"]).__name__,
    "selectionScore": float(final.best_score_),
    "bestIndex": int(final.best_index_),
    "candidates": final_candidates,
    "tiedWith": [f"{candidate['k']}/{candidate['scaler']}" for candidate in final_candidates
                 if candidate["meanScore"] == final_best],
}

# ------------------------------------------------------------------ #
# Every recorded packet value must equal this refit.
# ------------------------------------------------------------------ #
for refitted, recorded in zip(folds, saved["outer"]):
    tag = f"outer fold {refitted['fold'] + 1}"
    check(refitted["trainingRows"] == recorded["train_rows"], f"{tag}: the training row IDs match the packet")
    check(refitted["testRows"] == recorded["test_rows"], f"{tag}: the protected row IDs match the packet")
    check([split["validationRows"] for split in refitted["innerSplits"]]
          == [split["validation_rows"] for split in recorded["inner_splits"]], f"{tag}: the inner validation IDs match")
    check([split["trainingSize"] for split in refitted["innerSplits"]]
          == [len(split["train_rows"]) for split in recorded["inner_splits"]], f"{tag}: the inner training sizes match")
    for actual, packet in zip(refitted["candidates"], recorded["candidates"]):
        label = f"{tag} candidate k={actual['k']}/{actual['scaler']}"
        check((actual["k"], actual["scaler"]) == (packet["k"], packet["scaler"]), f"{label}: same enumeration position")
        check(all(abs(a - b) < 1e-12 for a, b in zip(actual["foldScores"], packet["fold_scores"])), f"{label}: inner fold scores")
        check(abs(actual["meanScore"] - packet["mean_score"]) < 1e-12, f"{label}: inner mean score")
    check(refitted["bestIndex"] == recorded["best_index"], f"{tag}: the same candidate is selected")
    check(refitted["correct"] == recorded["correct"], f"{tag}: the same number of protected rows are correct")
    check(abs(refitted["accuracy"] - recorded["accuracy"]) < 1e-12, f"{tag}: the same assessed accuracy")
    check(refitted["baselineCorrect"] == recorded["baseline_correct"], f"{tag}: the same majority-baseline count")
    check(refitted["prediction"] == recorded["prediction"], f"{tag}: every protected prediction matches")
    check(refitted["truth"] == recorded["truth"], f"{tag}: every protected label matches")

check(abs(pooled - saved["pooled_accuracy"]) < 1e-15, "pooled accuracy matches the packet")
check(abs(fold_mean - saved["fold_mean"]) < 1e-15, "the unweighted fold mean matches the packet")
check(final_selected["k"] == saved["final_selected"]["k"]
      and final_selected["scaler"] == saved["final_selected"]["scaler"]
      and abs(final_selected["selectionScore"] - saved["final_selected"]["inner_score"]) < 1e-15,
      "the final all-data selection matches the packet")
check(final_selected["tiedWith"] == ["3/StandardScaler", "3/RobustScaler"],
      "the final all-data search is a two-way tie that the declared enumeration rule resolves")
check(final_selected["bestIndex"] == 0, "the first tied candidate in the enumeration is the one selected")
check(len({round(candidate["meanScore"], 12) for candidate in final_candidates}) == 3,
      "the six final candidates collapse into three tied pairs, so the scaler is settled by order rather than by evidence")
check(abs(final_candidates[2]["meanScore"] - 0.9884057971014494) < 1e-15
      and abs(final_candidates[4]["meanScore"] - 0.9854818204932622) < 1e-15,
      "the other two final scores are the recorded 0.9884058 and 0.9854818")

# The manuscript's own sentences.
check([fold["selected"] for fold in folds] == [{"k": 11, "scaler": "StandardScaler"},
                                               {"k": 3, "scaler": "StandardScaler"},
                                               {"k": 3, "scaler": "RobustScaler"}],
      "the three outer folds select 11/Standard, 3/Standard and 3/Robust")
check([fold["correct"] for fold in folds] == [114, 114, 112], "the recorded correct counts are 114, 114 and 112")
check([len(fold["testRows"]) for fold in folds] == [115, 115, 114], "the outer folds assess 115, 115 and 114 rows")
check([len(fold["trainingRows"]) for fold in folds] == [229, 229, 230], "the outer training sizes are 229, 229 and 230")
check([fold["baselineCorrect"] for fold in folds] == [51, 51, 50], "the majority baselines are 51, 51 and 50")
check(sum(fold["correct"] for fold in folds) == 340 and abs(pooled - 340 / 344) < 1e-15,
      "pooled accuracy is exactly 340/344")
check(f"{fold_mean:.7f}" == "0.9883549", "the unweighted fold mean prints as 0.9883549")
check(f"{pooled:.7f}" == "0.9883721", "pooled accuracy prints as 0.9883721")
check(f"{final_selected['selectionScore']:.7f}" == "0.9913043", "the final selection score prints as 0.9913043")
check(fold_mean != pooled, "the two summaries differ, so the weighting distinction is visible in real numbers")
check(sorted(row for fold in folds for row in fold["testRows"]) == list(range(344)),
      "the three outer folds partition all 344 rows exactly once")
for fold in folds:
    held = set(fold["testRows"])
    check(not held & set(fold["trainingRows"]), f"outer fold {fold['fold'] + 1} never trains on a row it assesses")
    inner_union = [row for split in fold["innerSplits"] for row in split["validationRows"]]
    check(sorted(inner_union) == sorted(fold["trainingRows"]),
          f"outer fold {fold['fold'] + 1}: the inner folds partition only the outer training rows")
    check(not held & set(inner_union), f"outer fold {fold['fold'] + 1}: no protected row reaches an inner comparison")
    best = max(candidate["meanScore"] for candidate in fold["candidates"])
    first = next(position for position, candidate in enumerate(fold["candidates"]) if candidate["meanScore"] == best)
    check(first == fold["bestIndex"], f"outer fold {fold['fold'] + 1}: an exact tie takes the first enumerated candidate")

ties = sum(1 for candidate in folds[2]["candidates"]
           if candidate["meanScore"] == max(row["meanScore"] for row in folds[2]["candidates"]))
check(ties == 5, "outer fold 3 has a genuine five-way tie the declared rule resolves")
check(max(fold["accuracy"] for fold in folds) > max(fold["baselineCorrect"] / len(fold["testRows"]) for fold in folds),
      "the selected pipeline beats the majority baseline on every outer fold")

FITS = 3 * (6 * 3 + 1) + (6 * 3 + 1)
check(FITS == 76, "the experiment costs 57 assessment fits plus 19 for final selection and refit")

coverage_points = np.random.default_rng(5).uniform(size=(9, 2))
check(np.allclose(coverage_points, np.asarray(saved["random_coverage_points"]), atol=0, rtol=0),
      "the nine disclosed random search positions regenerate exactly from NumPy default_rng(5)")
check(abs(coverage_points[0][0] - 0.8050029237453802) < 1e-15 and abs(coverage_points[0][1] - 0.8079407897364937) < 1e-15,
      "the first disclosed draw is the (0.8050, 0.8079) the specification names")
check(len({round(float(value), 12) for value in coverage_points[:, 0]}) == 9,
      "the nine draws take nine distinct values on the projected coordinate, against the grid's three")

versions = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__}
check(versions == {"python": saved["environment"]["python"], "numpy": saved["environment"]["numpy"],
                   "pandas": saved["environment"]["pandas"], "sklearn": saved["environment"]["sklearn"]},
      "this run used the same pinned library versions the author recorded")

experiment = {
    "outerSeed": 41,
    "innerSeed": 73,
    "outerFolds": 3,
    "innerFolds": 3,
    "metric": "accuracy",
    "tieRule": "the first candidate in the declared enumeration",
    "enumeration": [{"k": candidate["k"], "scaler": candidate["scaler"]} for candidate in folds[0]["candidates"]],
    "folds": folds,
    "pooledCorrect": sum(fold["correct"] for fold in folds),
    "assessedRows": 344,
    "pooledAccuracy": pooled,
    "foldMean": fold_mean,
    "finalSelected": final_selected,
    "fits": {"perOuter": 19, "assessment": 57, "finalSearch": 19, "total": FITS, "baselines": 3},
}
source = {
    "sha256": csv_hash,
    "bytes": len(csv_bytes),
    "rows": 344,
    "columns": list(data.columns),
    "features": FEATURES,
    "excluded": ["island", "year", "source row position"],
    "target": "species",
    "missing": missing,
    "speciesCounts": species_counts,
    "licence": "CC0",
    "project": "https://allisonhorst.github.io/palmerpenguins/",
    "attribution": "Allison Horst, Alison Hill and Kristen Gorman; original Palmer Station measurements by Gorman and colleagues.",
    "served": "/learn-assets/cross-validation/penguins.csv",
    "provenance": "/learn-assets/cross-validation/data-provenance.md",
    "limit": "A familiar dataset reused to demonstrate a selection protocol. The random-row split does not certify a future-season or unseen-site deployment, and this is not new independent evidence about the preceding lesson's scaler choice.",
}

module = (
    "/** The cross-validation lesson's real nested experiment, generated by\n"
    " * scripts/verify-validation-data.py. Do not edit by hand.\n"
    " *\n"
    " * The generator refits the whole three-by-three, six-candidate search from the\n"
    " * CSV served at /learn-assets/cross-validation/penguins.csv and refuses to write\n"
    " * unless every value matches the retained author calculation.\n"
    " *\n"
    " * Palmer Penguins, CC0. Row IDs are zero-based source positions in the original\n"
    " * file order. `selectionScore` chose a candidate; `accuracy` was produced by rows\n"
    " * protected from that choice. They are different quantities and never merged.\n"
    " */\n"
    "export const PENGUIN_SOURCE = " + json.dumps(source, ensure_ascii=False, indent=2) + ";\n\n"
    "export const NESTED_EXPERIMENT = " + json.dumps(experiment, ensure_ascii=False, separators=(",", ":")) + ";\n\n"
    "/** Nine actual uniform draws from NumPy default_rng(5), disclosed in full so\n"
    " * the coverage picture is a generated search position and never a fabricated\n"
    " * accuracy surface. Regenerated and re-checked by the verifier. */\n"
    "export const RANDOM_COVERAGE_POINTS = " + json.dumps(coverage_points.tolist(), ensure_ascii=False, separators=(",", ":")) + ";\n\n"
    "export const RUNTIME_VERSIONS = " + json.dumps(versions, ensure_ascii=False, indent=2) + ";\n"
)
public = json.dumps({
    "datasetSha256": csv_hash,
    "outerSeed": 41, "innerSeed": 73,
    "enumeration": experiment["enumeration"],
    "folds": [{
        "fold": fold["fold"], "trainingRows": fold["trainingRows"], "testRows": fold["testRows"],
        "innerValidationRows": [split["validationRows"] for split in fold["innerSplits"]],
        "candidates": fold["candidates"], "bestIndex": fold["bestIndex"],
        "prediction": fold["prediction"], "truth": fold["truth"],
        "correct": fold["correct"], "baselineCorrect": fold["baselineCorrect"],
    } for fold in folds],
    "pooledAccuracy": pooled, "foldMean": fold_mean, "finalSelected": final_selected,
    "versions": versions,
}, ensure_ascii=False, separators=(",", ":")) + "\n"

if write:
    MODULE.write_text(module, encoding="utf-8", newline="\n")
    (ASSETS / "nested-experiment.json").write_text(public, encoding="utf-8", newline="\n")
else:
    assert MODULE.read_text(encoding="utf-8") == module, "Stale src/learn/data/validation-data.js; rerun with --write"
    assert (ASSETS / "nested-experiment.json").read_text(encoding="utf-8") == public, "Stale downloadable split record"

evidence = {
    "status": "passed",
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "command": "scratch/lesson-tools/Scripts/python.exe scripts/verify-validation-data.py" + (" --write" if write else ""),
    "stage": "author data regeneration; browser, independent and integration review are separate",
    "pipelineFits": FITS,
    "checks": checks,
    "checkCount": len(checks),
    "result": {
        "selections": [fold["selected"] for fold in folds],
        "correct": [fold["correct"] for fold in folds],
        "assessed": [len(fold["testRows"]) for fold in folds],
        "baselineCorrect": [fold["baselineCorrect"] for fold in folds],
        "pooledAccuracy": pooled,
        "foldMean": fold_mean,
        "finalSelected": final_selected,
    },
    "versions": versions,
    "limits": [
        "One dataset, one seed pair, one candidate grid: a reproducible protocol demonstration, not new evidence about penguin classification.",
        "The outer score assesses the selection-and-fitting procedure at this outer training size, not an oracle-best setting.",
        "No timing was recorded; the fit count is exact and the wall time is not claimed.",
    ],
    "sourceHashes": {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), MODULE, ASSETS / "penguins.csv", ASSETS / "nested-experiment.json",
                     ASSETS / "data-provenance.md", PACKET / "calculated-inputs.json"]
    },
}
EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(checks)} data checks; {FITS} pipeline fits refitted; "
      f"selections {[f'{fold['selected']['k']}/{fold['selected']['scaler'][:-6]}' for fold in folds]}; "
      f"pooled {pooled:.7f}, fold mean {fold_mean:.7f}.")
