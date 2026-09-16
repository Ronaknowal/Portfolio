"""Bounded author calculations for CV, selection and resource allocation."""
from pathlib import Path
from itertools import product
import hashlib
import json
import platform
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

ROOT = Path(__file__).resolve().parent

def kfold_indices(n, k, shuffle=False, seed=0):
    if not 2 <= k <= n:
        raise ValueError("Require 2 <= k <= n.")
    indices = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, k)
    for held in range(k):
        yield np.sort(np.concatenate([folds[j] for j in range(k) if j != held])), folds[held]

def make_model():
    numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    return Pipeline([
        ("prepare", ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("category", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="not_recorded", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ["sex"]),
        ])),
        ("classify", KNeighborsClassifier()),
    ])

def tiny_nested(x, y):
    rows = np.arange(len(y))
    def predict(train, test, k):
        distance = abs(x[test, None] - x[train])
        neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
        return (y[train[neighbors]].mean(axis=1) > .5).astype(int)
    trace = []
    for held in [0, 1]:
        test = rows[rows % 2 == held]
        train = rows[rows % 2 != held]
        inner = [(train[1::2], train[::2]), (train[::2], train[1::2])]
        candidates = []
        for k in [1, 3]:
            scores = [float(np.mean(predict(a,b,k) == y[b])) for a,b in inner]
            candidates.append({"k":k,"scores":scores,"mean":float(np.mean(scores))})
        best = max(candidates, key=lambda value: value["mean"])
        predicted = predict(train,test,best["k"])
        trace.append({"train":train.tolist(),"test":test.tolist(),"inner":[{"train":a.tolist(),"validation":b.tolist()} for a,b in inner],"candidates":candidates,"selected_k":best["k"],"prediction":predicted.tolist(),"correct":int(np.sum(predicted==y[test]))})
    return trace

def main():
    source = ROOT.parent / "feature-scaling-encoding-imputation" / "penguins.csv"
    if not (ROOT / "penguins.csv").exists():
        (ROOT / "penguins.csv").write_bytes(source.read_bytes())
    data = pd.read_csv(ROOT / "penguins.csv")
    columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    X, y = data[columns], data.species
    grid = {
        "prepare__numeric__scale": [StandardScaler(), RobustScaler()],
        "classify__n_neighbors": [3, 5, 11],
    }
    outer = StratifiedKFold(3, shuffle=True, random_state=41)
    records = []
    out_of_fold = np.empty(len(y), dtype=object)
    for fold, (train, test) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(3, shuffle=True, random_state=73)
        search = GridSearchCV(make_model(), grid, cv=inner, scoring="accuracy", n_jobs=1, error_score="raise")
        search.fit(X.iloc[train], y.iloc[train])
        prediction = search.predict(X.iloc[test])
        out_of_fold[test] = prediction
        dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y.iloc[train])
        baseline = dummy.predict(np.zeros((len(test), 1)))
        candidate_rows = []
        for j, params in enumerate(search.cv_results_["params"]):
            candidate_rows.append({
                "k":int(params["classify__n_neighbors"]),
                "scaler":type(params["prepare__numeric__scale"]).__name__,
                "fold_scores":[float(search.cv_results_[f"split{v}_test_score"][j]) for v in range(3)],
                "mean_score":float(search.cv_results_["mean_test_score"][j]),
            })
        records.append({
            "fold":fold, "train_rows":train.tolist(), "test_rows":test.tolist(),
            "inner_splits":[{"train_rows":train[a].tolist(),"validation_rows":train[b].tolist()} for a,b in inner.split(X.iloc[train], y.iloc[train])],
            "candidates":candidate_rows, "best_index":int(search.best_index_),
            "correct":int(np.sum(prediction==y.iloc[test])),
            "accuracy":float(np.mean(prediction==y.iloc[test])),
            "baseline_correct":int(np.sum(baseline==y.iloc[test])),
            "prediction":prediction.tolist(),"truth":y.iloc[test].tolist(),
        })
    final = GridSearchCV(make_model(),grid,cv=StratifiedKFold(3,shuffle=True,random_state=73),scoring="accuracy",n_jobs=1,error_score="raise")
    final.fit(X,y)
    tiny_x=np.arange(7);tiny_y=np.array([0,0,0,1,1,1,1])
    tiny=[]
    for train,test in kfold_indices(7,3):
        predictions=tiny_y[train[np.argmin(abs(tiny_x[test,None]-tiny_x[train]),axis=1)]]
        tiny.append({"train":train.tolist(),"validation":test.tolist(),"prediction":predictions.tolist(),"truth":tiny_y[test].tolist(),"correct":int(np.sum(predictions==tiny_y[test]))})
    label_sets=np.array(list(product([0,1],repeat=4)))
    fixed=np.array([[0,0,0,0],[1,1,1,1]])
    selected_score=np.mean(np.max(np.mean(label_sets[:,None,:]==fixed[None,:,:],axis=2),axis=1))
    hit={str(p):{str(t):1-(1-p)**t for t in [20,59,60,299]} for p in [.05,.01]}
    nested_x = np.arange(16)
    nested_y = (nested_x >= 8).astype(int)
    nested_changed_y = nested_y.copy()
    nested_changed_y[7] = 1
    nested_selection_y = nested_y.copy()
    nested_selection_y[3] = 1
    output={
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__},
        "data_sha256":hashlib.sha256((ROOT/"penguins.csv").read_bytes()).hexdigest(),
        "outer":records,"pooled_accuracy":float(np.mean(out_of_fold==y)),
        "fold_mean":float(np.mean([r["accuracy"] for r in records])),
        "final_selected":{"k":int(final.best_params_["classify__n_neighbors"]),"scaler":type(final.best_params_["prepare__numeric__scale"]).__name__,"inner_score":float(final.best_score_)},
        "tiny":tiny,"selection_enumeration":{"two_constant_candidates_mean_selected_accuracy":float(selected_score),"all16_patterns_mean_selected_accuracy":1.,"true_fresh_accuracy":.5},
        "random_hit":hit,
        "random_coverage_points":np.random.default_rng(5).uniform(size=(9,2)).tolist(),
        "tiny_nested":{"x":nested_x.tolist(),"y":nested_y.tolist(),"trace":tiny_nested(nested_x,nested_y),"changed_row7_y":nested_changed_y.tolist(),"changed_trace":tiny_nested(nested_x,nested_changed_y),"changed_row3_y":nested_selection_y.tolist(),"selection_changed_trace":tiny_nested(nested_x,nested_selection_y)},
    }
    (ROOT/"calculated-inputs.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"folds":[{"fold":r["fold"],"test_size":len(r["test_rows"]),"best":r["candidates"][r["best_index"]],"correct":r["correct"],"baseline_correct":r["baseline_correct"]} for r in records],"pooled_accuracy":output["pooled_accuracy"],"fold_mean":output["fold_mean"],"final_selected":output["final_selected"],"tiny":tiny,"selection_enumeration":output["selection_enumeration"],"random_hit":hit},indent=2))

if __name__=="__main__":
    main()
