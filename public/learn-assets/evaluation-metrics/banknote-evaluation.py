"""Fixed measured-data evaluation protocol with a development-selected threshold."""
from pathlib import Path
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss, precision_recall_curve, roc_curve)

HERE = Path(__file__).resolve().parent


def run():
    data = np.genfromtxt(HERE / "banknote-subset.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    # Two predeclared features keep the experiment an inspectable imperfect baseline.
    x = np.column_stack([data["variance"], data["entropy"]])
    y = data["class"]
    train, development, test = np.arange(320), np.arange(320,400), np.arange(400,480)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=500))
    model.fit(x[train], y[train])
    assert model.classes_.tolist() == [0, 1]
    dev_score = model.predict_proba(x[development])[:, 1]
    thresholds = np.r_[np.inf, np.unique(dev_score)[::-1]]
    records = []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y[development], dev_score >= threshold, labels=[0,1]).ravel()
        records.append({"threshold": "above_maximum" if np.isinf(threshold) else float(threshold),
                        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                        "cost_fp1_fn5": int(fp+5*fn)})
    best = min(range(len(records)), key=lambda i: records[i]["cost_fp1_fn5"])
    threshold = thresholds[best]  # First minimum = highest threshold among equal-cost candidates.
    test_score = model.predict_proba(x[test])[:, 1]
    prior = float(y[train].mean())

    def report(score, threshold):
        tn, fp, fn, tp = confusion_matrix(y[test], score >= threshold, labels=[0,1]).ravel()
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "accuracy": float((tp+tn)/len(test)), "precision": None if tp+fp == 0 else float(tp/(tp+fp)),
                "recall": float(tp/(tp+fn)), "f1": float(2*tp/(2*tp+fp+fn)),
                "cost_fp1_fn5": int(fp+5*fn), "roc_auc": float(roc_auc_score(y[test], score)),
                "ap": float(average_precision_score(y[test], score)), "log_loss": float(log_loss(y[test], score)),
                "brier": float(brier_score_loss(y[test], score))}

    roc_fpr, roc_tpr, roc_threshold = roc_curve(y[test], test_score, drop_intermediate=False)
    precision, recall, pr_threshold = precision_recall_curve(y[test], test_score, drop_intermediate=False)
    return {"features": ["variance", "entropy"], "split_sizes": [320,80,80],
            "source_rows": {name: data["source_row"][rows].tolist() for name, rows in [("train",train),("development",development),("test",test)]},
            "training_class1_prior": prior, "development_scores": dev_score.tolist(),
            "development_labels": y[development].tolist(), "development_thresholds": records,
            "selected_development_record": records[best], "test_labels": y[test].tolist(), "test_scores": test_score.tolist(),
            "reports": {"fixed_0.5": report(test_score, .5), "development_selected": report(test_score, threshold),
                        "training_prior_0.5": report(np.full(len(test), prior), .5),
                        "training_prior_cost_rule": report(np.full(len(test), prior), 1/6)},
            "test_curves": {"roc_fpr": roc_fpr.tolist(), "roc_tpr": roc_tpr.tolist(),
                            "roc_thresholds": ["above_maximum" if np.isinf(t) else float(t) for t in roc_threshold],
                            "pr_precision": precision.tolist(), "pr_recall": recall.tolist(), "pr_thresholds": pr_threshold.tolist()}}


if __name__ == "__main__":
    result = run()
    (HERE / "banknote-evaluation-results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("selected development threshold", result["selected_development_record"])
    print(json.dumps(result["reports"], indent=2))
