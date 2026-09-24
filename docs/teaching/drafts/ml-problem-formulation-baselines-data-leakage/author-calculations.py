"""Bounded content calculations; no web/runtime implementation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, log_loss, confusion_matrix

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "bank-additional.csv", sep=";")
y = (data["y"] == "yes").to_numpy(dtype=int)
development, reserved = train_test_split(
    np.arange(len(y)), train_size=3295, stratify=y, random_state=53
)
train, validation = train_test_split(
    development, train_size=2471, stratify=y[development], random_state=54
)
features = data.copy()
features["contacted_before"] = (features["pdays"] != 999).astype(int)
features["days_since_previous"] = features["pdays"].replace(999, np.nan)
numeric = ["age", "previous", "contacted_before", "days_since_previous"]
categorical = ["job", "marital", "education", "default", "housing", "loan",
               "poutcome"]

def pipeline(include_duration):
    columns = numeric + (["duration"] if include_duration else [])
    preparation = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median",
             keep_empty_features=True), StandardScaler()), columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return make_pipeline(preparation, LogisticRegression(C=1., max_iter=600))

def score(probability):
    prediction = (probability >= .5).astype(int)
    # Deterministic ties: smaller original source row index first.
    ranked = np.lexsort((validation, -probability))
    found = int(y[validation][ranked[:50]].sum())
    return {"averagePrecision": float(average_precision_score(y[validation], probability)),
            "logLoss": float(log_loss(y[validation], probability)),
            "confusion": confusion_matrix(y[validation], prediction, labels=[0,1]).tolist(),
            "correct": int((prediction == y[validation]).sum()),
            "top50Positives": found, "precisionAt50": found/50,
            "recallAt50": found/int(y[validation].sum()),
            "probabilities": probability.tolist(), "rankedSourceRows": validation[ranked].tolist()}

results = {}
prior = y[train].mean()
results["training_prior"] = score(np.full(len(validation), prior))
for name, duration in [("candidate_pre_call", False), ("unavailable_duration", True)]:
    model = pipeline(duration).fit(features.iloc[train], y[train])
    results[name] = score(model.predict_proba(features.iloc[validation])[:, 1])
    results[name]["iterations"] = model[-1].n_iter_.tolist()
    results[name]["featureNames"] = model[0].get_feature_names_out().tolist()

# Original event/knowledge-time fixture; arrival versions never overwrite history.
records = [
    {"entity": "sensor_A", "event": 1, "available": 1, "version": 1, "value": 10},
    {"entity": "sensor_A", "event": 4, "available": 8, "version": 1, "value": 20},
    {"entity": "sensor_A", "event": 1, "available": 6, "version": 2, "value": 12},
    {"entity": "sensor_B", "event": 4, "available": 4, "version": 1, "value": 99},
]
def as_known(rows, entity, cutoff, maximum_age):
    eligible = [r for r in rows if r["entity"] == entity
                and cutoff-maximum_age <= r["event"] <= cutoff
                and r["available"] <= cutoff]
    return max(eligible, key=lambda r: (r["event"], r["available"], r["version"]),
               default=None)
timeline = []
for name, rows, cutoff, age in [
    ("default", records, 5, 5),
    ("arrive_earlier", [{**r, "available": 4} if r["event"] == 4 and r["entity"] == "sensor_A" else r for r in records], 5, 5),
    ("latest_known_revision", records, 7, 7),
    ("new_event_now_known", records, 9, 9),
    ("too_old", records, 5, 2),
    ("unrelated_entity_null", [{**r, "value": 88} if r["entity"] == "sensor_B" else r for r in records], 5, 5),
]:
    timeline.append({"name":name, "selected":as_known(rows, "sensor_A", cutoff, age)})

target_by_id = dict(zip(validation.tolist(), y[validation].tolist()))
ranked_ids = results["candidate_pre_call"]["rankedSourceRows"]
selected = ranked_ids[:25]
remove_id = next(i for i in selected if target_by_id[i] == 1)
add_id = next(i for i in ranked_ids[25:] if target_by_id[i] == 0)
changed_set = [add_id if i == remove_id else i for i in selected]
policy = {
    "top25": {"ids": selected, "positives": sum(target_by_id[i] for i in selected)},
    "positiveToNegativeSwap": {
        "removeId": remove_id, "addId": add_id,
        "positives": sum(target_by_id[i] for i in changed_set)},
    "reorderNullPositives": sum(target_by_id[i] for i in reversed(selected)),
}

output = {"software":{"numpy":np.__version__,"pandas":pd.__version__,
                      "scikitLearn":sklearn.__version__},
          "inputRows":len(y), "positiveRows":int(y.sum()),
          "trainRows":train.tolist(),"validationRows":validation.tolist(),
          "reservedRows":reserved.tolist(),"trainPositives":int(y[train].sum()),
          "validationPositives":int(y[validation].sum()),
          "trainPrior":float(prior),"validationTargets":y[validation].tolist(),
          "numericFeatures":numeric,"categoricalFeatures":categorical,
          "results":results,"timelineInput":records,"timelineResults":timeline,
          "policyFixtures":policy}
(ROOT/"calculated-inputs.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
print({k:{a:b for a,b in v.items() if a not in ["probabilities","rankedSourceRows","featureNames"]}
       for k,v in results.items()})
print("counts",len(train),len(validation),len(reserved),y.sum(),y[train].sum(),y[validation].sum())
print("timeline",timeline)
print("policy",policy)
