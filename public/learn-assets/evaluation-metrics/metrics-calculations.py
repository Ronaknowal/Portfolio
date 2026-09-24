"""Complete instructional metrics program. Constructed fixtures, never model benchmarks."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             auc, ndcg_score, precision_recall_fscore_support,
                             matthews_corrcoef, log_loss, brier_score_loss)

HERE = Path(__file__).resolve().parent


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def binary_counts(y, score, threshold):
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or len(y) == 0:
        raise ValueError("Use nonempty matching one-dimensional label and score arrays.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all() or np.isnan(threshold):
        raise ValueError("Use binary labels and finite scores.")
    predicted = score >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def rates(counts, beta=1.):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive.")
    tp, fp, fn, tn = (counts[key] for key in ["tp", "fp", "fn", "tn"])
    return {"precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
            "specificity": ratio(tn, tn+fp), "fpr": ratio(fp, tn+fp),
            "accuracy": ratio(tp+tn, tp+fp+fn+tn),
            "f1": ratio(2*tp, 2*tp+fp+fn),
            "fbeta": ratio((1+beta**2)*tp, (1+beta**2)*tp+fp+beta**2*fn)}


class StreamingBinaryCounts:
    def __init__(self, threshold):
        self.threshold = threshold
        self.counts = dict.fromkeys(["tp", "fp", "fn", "tn"], 0)

    def update(self, labels, scores):
        chunk = binary_counts(labels, scores, self.threshold)
        for key in self.counts:
            self.counts[key] += chunk[key]

    def summary(self):
        return {"counts": self.counts.copy(), "rates": rates(self.counts)}


def threshold_curve(y, score):
    """Group every equal-score block; increasing recall as threshold decreases."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    binary_counts(y, score, np.inf)  # Validate arrays once.
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    order = np.argsort(-score, kind="stable")
    ordered_y, ordered_score = y[order], score[order]
    end_indices = np.r_[np.flatnonzero(np.diff(ordered_score) != 0), len(y)-1]
    tp = np.r_[0, np.cumsum(ordered_y == 1)[end_indices]]
    fp = np.r_[0, np.cumsum(ordered_y == 0)[end_indices]]
    recall = tp/positives if positives else None
    fpr = fp/negatives if negatives else None
    precision = np.divide(tp, tp+fp, out=np.ones(len(tp), dtype=float), where=(tp+fp) != 0)
    # precision[0]=1 is a plotting sentinel, not an estimated no-alert precision.
    result = {"thresholds": ["above_maximum"]+ordered_score[end_indices].tolist(),
              "tp": tp.tolist(), "fp": fp.tolist(), "precision_for_plot": precision.tolist(),
              "recall": None if recall is None else recall.tolist(),
              "fpr": None if fpr is None else fpr.tolist()}
    result["roc_auc"] = None if positives*negatives == 0 else float(np.trapezoid(recall, fpr))
    result["ap"] = None if positives == 0 else float(np.sum(np.diff(recall)*precision[1:]))
    return result


def pair_auc(y, score):
    positive = np.asarray(score)[np.asarray(y) == 1]
    negative = np.asarray(score)[np.asarray(y) == 0]
    if not len(positive) or not len(negative):
        return None, None
    contributions = (positive[:, None] > negative).astype(float) + .5*(positive[:, None] == negative)
    return float(contributions.mean()), contributions.tolist()


def regression_metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != predicted.shape or not len(y) or not np.isfinite([y, predicted]).all():
        raise ValueError("Use matching nonempty finite target and prediction vectors.")
    residual = y-predicted
    sse = float(residual @ residual)
    centered = y-y.mean()
    sst = float(centered @ centered)
    return {"residual": residual.tolist(), "absolute": np.abs(residual).tolist(),
            "squared": (residual**2).tolist(), "mae": float(np.abs(residual).mean()),
            "mse": float(np.mean(residual**2)), "rmse": float(np.sqrt(np.mean(residual**2))),
            "median_absolute_error": float(np.median(np.abs(residual))),
            "sse": sse, "sst": sst, "evaluation_mean": float(y.mean()),
            "r2": None if sst == 0 or len(y) < 2 else 1-sse/sst}


def retrieval_metrics(relevance, k):
    """Input is a specific document order; exponential gain, binary relevance >0."""
    relevance = np.asarray(relevance, dtype=float)
    if not 1 <= k <= len(relevance) or np.any(relevance < 0):
        raise ValueError("Use nonnegative grades and a valid cutoff.")
    discount = 1/np.log2(np.arange(2, len(relevance)+2))
    gain = 2**relevance-1
    dcg = float(gain[:k] @ discount[:k])
    ideal = float(np.sort(gain)[::-1][:k] @ discount[:k])
    binary = relevance > 0
    cumulative = np.cumsum(binary)/np.arange(1, len(relevance)+1)
    total_relevant = int(binary.sum())
    rr = 0 if not total_relevant else 1/(np.flatnonzero(binary)[0]+1)
    return {"relevance": relevance.tolist(), "gain": gain.tolist(), "discount": discount.tolist(),
            "dcg": dcg, "ideal_dcg": ideal, "ndcg": ratio(dcg, ideal),
            "precision_at_k": float(binary[:k].mean()), "recall_at_k": ratio(int(binary[:k].sum()), total_relevant),
            "reciprocal_rank": float(rr), "ap": ratio(float(cumulative[binary].sum()), total_relevant)}


def examples():
    y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    score = np.array([.95, .8, .8, .6, .5, .3, .2, .1])
    core = {"ids": list("ABCDEFGH"), "y": y.tolist(), "score": score.tolist(), "thresholds": {}}
    for threshold in [.8, .5, .3, 1.1]:
        counts = binary_counts(y, score, threshold)
        core["thresholds"][str(threshold)] = {"counts": counts, "rates": rates(counts), "cost_fp1_fn3": counts["fp"]+3*counts["fn"]}
    core["curve"] = threshold_curve(y, score)
    core["pair_auc"], core["pair_contributions"] = pair_auc(y, score)
    assert np.isclose(core["curve"]["roc_auc"], roc_auc_score(y, score))
    assert np.isclose(core["curve"]["ap"], average_precision_score(y, score))
    precision, recall, _ = precision_recall_curve(y, score)
    core["trapezoidal_pr_area"] = float(auc(recall, precision))
    core["probability_scores"] = {}
    for name, probability in {"original": score, "squared": score**2, "constant": np.full(8, .5)}.items():
        core["probability_scores"][name] = {"probabilities": probability.tolist(), "log_loss": float(log_loss(y, probability)),
                                          "brier": float(brier_score_loss(y, probability)),
                                          "roc_auc": float(roc_auc_score(y, probability)),
                                          "ap": float(average_precision_score(y, probability))}
    reordered = np.array([0, 2, 1, 3, 4, 5, 6, 7])
    assert threshold_curve(y[reordered], score[reordered]) == core["curve"]
    stream = StreamingBinaryCounts(.5)
    for rows in [slice(0,3),slice(3,6),slice(6,8)]:
        stream.update(y[rows], score[rows])
    assert stream.counts == binary_counts(y, score, .5)
    core["streamed"] = stream.summary()
    ranking = {}
    for name, positions in {"A": [1, 2, 4], "B": [0, 1, 7]}.items():
        label = np.zeros(8, dtype=int); label[positions] = 1
        ranking[name] = {"labels_in_descending_order": label.tolist(),
                         "auc": float(roc_auc_score(label, -np.arange(8))),
                         "ap": float(average_precision_score(label, -np.arange(8)))}
    nulls = {"constant_scores": threshold_curve(y, np.full(8, .5)),
             "no_positives": threshold_curve(np.zeros(8, dtype=int), score),
             "no_negatives": threshold_curve(np.ones(8, dtype=int), score)}
    matrix = np.array([[8, 1, 1], [2, 2, 0], [1, 1, 0]])
    true, predicted = [], []
    for actual in range(3):
        for pred in range(3):
            true.extend([actual]*int(matrix[actual, pred])); predicted.extend([pred]*int(matrix[actual, pred]))
    averages = {}
    for average in [None, "macro", "micro", "weighted"]:
        result = precision_recall_fscore_support(true, predicted, labels=[0,1,2], average=average, zero_division=0)
        averages[str(average)] = [v.tolist() if hasattr(v, "tolist") else v for v in result]
    regression = {name: regression_metrics([1, 2, 3, 4, 10], prediction)
                  for name, prediction in {"A": [1,2,3,4,4], "B": [-1,0,1,2,8],
                                           "evaluation_mean": [4]*5, "training_mean3": [3]*5,
                                           "perfect": [1,2,3,4,10]}.items()}
    regression["constant_target"] = regression_metrics([4]*5, [4]*5)
    retrieval = {"original": retrieval_metrics([0,3,1,2,0], 3),
                 "changed_order": retrieval_metrics([3,0,1,2,0], 3),
                 "all_zero": retrieval_metrics([0,0,0,0,0], 3)}
    relevance = np.array([0,3,1,2,0]); scores = -np.arange(5)
    retrieval["sklearn_linear_gain"] = float(ndcg_score([relevance], [scores], k=3))
    retrieval["sklearn_exponential_gain"] = float(ndcg_score([2**relevance-1], [scores], k=3))
    assert np.isclose(retrieval["original"]["ndcg"], retrieval["sklearn_exponential_gain"])
    prevalence = {str(pi): {"tpr": .8, "fpr": .1, "precision": (.8*pi)/(.8*pi+.1*(1-pi)),
                            "tp_per10000": .8*pi*10000, "fp_per10000": .1*(1-pi)*10000}
                  for pi in [.5,.01]}
    edited_score = score.copy(); edited_score[2] = .45
    contrast = {"edited_C_score045": threshold_curve(y, edited_score),
                "A_last_prediction8": regression_metrics([1,2,3,4,10], [1,2,3,4,8]),
                "A_seconds": regression_metrics(np.array([1,2,3,4,10])*60, np.array([1,2,3,4,4])*60)}
    assert binary_counts(y, score, .7) == binary_counts(y, score, .75)
    assert retrieval_metrics([0,3,1,2,0], 3) == retrieval_metrics(np.array([0,3,1,2,0])[[4,1,2,3,0]], 3)
    practice_y = [1,0,1,0,1,0]; practice_score = [.9,.2,.7,.8,.1,.05]
    practice_batches = [binary_counts(practice_y[:2],practice_score[:2],.5), binary_counts(practice_y[2:],practice_score[2:],.5)]
    practice = {"tied_ranking": threshold_curve([1,1,0,0],[.7,.4,.7,.1]),
                "probability_losses": {str(p): {"log_loss": float(log_loss([0,1],[p,1-p])), "brier":float(brier_score_loss([0,1],[p,1-p]))} for p in [.49,.1]},
                "r2": regression_metrics([2,4,6],[3,3,3]),
                "retrieval": [retrieval_metrics(r,3) for r in [[0,2,1],[2,0,1]]],
                "streaming": {"labels":practice_y,"scores":practice_score,"threshold":.5,"batches":practice_batches,
                              "pooled":binary_counts(practice_y,practice_score,.5),
                              "pooled_f1":rates(binary_counts(practice_y,practice_score,.5))["f1"],
                              "mean_batch_f1":float(np.mean([rates(c)["f1"] for c in practice_batches]))}}
    return {"core": core, "auc_ap_counterexample": ranking, "nulls": nulls, "contrasts":contrast,"practice":practice,
            "multiclass": {"matrix": matrix.tolist(), "averages": averages},
            "regression": regression, "retrieval": retrieval, "prevalence": prevalence}


if __name__ == "__main__":
    result = examples()
    (HERE / "metric-fixtures.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("core ROC-AUC", result["core"]["pair_auc"], "AP", result["core"]["curve"]["ap"])
    print("trapezoidal PR area", result["core"]["trapezoidal_pr_area"])
    print("AUC/AP counterexample", result["auc_ap_counterexample"])
    print("regression", {k: {m:v[m] for m in ["mae", "rmse", "r2"]} for k,v in result["regression"].items()})
    print("retrieval", result["retrieval"])
