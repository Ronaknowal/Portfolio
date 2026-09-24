"""Small executable calibration/conformal mechanisms. Run with Python 3.12+.

Dependencies: numpy, scipy, scikit-learn. All fixtures are constructed; no
numbers are represented as model benchmarks. Output: checked-results.json.
"""
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import json
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


def reliability(probabilities, outcomes, edges):
    p, y, edges = np.asarray(probabilities, float), np.asarray(outcomes, float), np.asarray(edges, float)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or not len(p):
        raise ValueError("Need equally sized nonempty prediction and outcome vectors")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Probabilities must be finite in [0,1] and outcomes binary")
    if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("Increasing bin boundaries must span [0,1]")
    # Internal boundary belongs to the bin on its right; p=1 belongs to last.
    indices = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        selected = indices == b
        count = int(selected.sum())
        if not count:
            rows.append({"bin": b, "count": 0, "mean_p": None, "fraction_positive": None})
            continue
        positive = int(y[selected].sum())
        rows.append({"bin": b, "count": count, "positive": positive,
                     "mean_p": float(p[selected].mean()), "fraction_positive": positive / count,
                     "source_indices": np.flatnonzero(selected).tolist()})
    ece = sum(r["count"] * abs(r["mean_p"] - r["fraction_positive"]) for r in rows if r["count"]) / len(p)
    return {"bins": rows, "ece": ece, "count": len(p)}


def fit_sigmoid(scores, labels, smoothing=True):
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    if s.ndim != 1 or y.ndim != 1 or len(s) != len(y) or set(y) != {0, 1}:
        raise ValueError("Need finite scores with both binary classes")
    if np.any(~np.isfinite(s)):
        raise ValueError("Scores must be finite")
    positives, negatives = int(y.sum()), int((1-y).sum())
    target = np.where(y == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2)) if smoothing else y
    def objective(parameters):
        a, b = parameters
        z = a*s+b
        loss = np.mean(np.logaddexp(0, z) - target*z)
        residual = expit(z) - target
        return loss, np.array([np.mean(residual*s), residual.mean()])
    result = minimize(objective, [1.0, 0.0], jac=True, method="BFGS", options={"gtol": 1e-9})
    if not result.success and np.linalg.norm(result.jac) > 1e-6:
        raise RuntimeError(result.message)
    return {"a": float(result.x[0]), "b": float(result.x[1]),
            "objective": float(result.fun), "smoothed_targets": target.tolist(),
            "probabilities": expit(result.x[0]*s+result.x[1]).tolist()}


def fit_pav(scores, labels):
    """Group equal scores first; stack PAV is linear after sorting."""
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    if s.ndim != 1 or y.ndim != 1 or not len(s) or len(s) != len(y):
        raise ValueError("Need equally sized nonempty score and label vectors")
    if np.any(~np.isfinite(s)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Scores must be finite and labels binary")
    order = np.argsort(s, kind="stable")
    knots, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    sums = np.add.reduceat(y[order], first)
    blocks, trace = [], []
    for j, (total, count) in enumerate(zip(sums, counts)):
        blocks.append([j, j, float(total), int(count)])
        while len(blocks) >= 2 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            merged = [left[0], right[1], left[2]+right[2], left[3]+right[3]]
            blocks.append(merged)
            trace.append({"left": left, "right": right, "merged": merged})
    fitted = np.empty(len(knots))
    for start, end, total, count in blocks:
        fitted[start:end+1] = total/count
    expected = IsotonicRegression(out_of_bounds="clip").fit(s, y).predict(knots)
    np.testing.assert_allclose(fitted, expected)
    return {"knots": knots.tolist(), "fitted": fitted.tolist(), "blocks": blocks, "merges": trace}


def temperature_probabilities(logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    z = np.asarray(logits, float)/temperature
    return np.exp(z-logsumexp(z, axis=-1, keepdims=True))


def conformal_rank(n, alpha):
    if n < 1:
        raise ValueError("At least one calibration score is required")
    a = Fraction(str(alpha))
    if not 0 < a < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return math.ceil((n+1)*(1-a))


def fit_temperature(logits, labels):
    z, y = np.asarray(logits,float), np.asarray(labels,int)
    if z.ndim != 2 or len(z) != len(y) or np.any(y<0) or np.any(y>=z.shape[1]):
        raise ValueError("Need an observation-by-class logit matrix and matching class indices")
    def loss(inverse_temperature):
        scaled=z*inverse_temperature
        return float(np.mean(logsumexp(scaled,axis=1)-scaled[np.arange(len(y)),y]))
    result=minimize_scalar(loss,bounds=(.05,20),method="bounded",options={"xatol":1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    return {"temperature":1/float(result.x),"nll_before":loss(1),"nll_after":float(result.fun),
            "inverse_temperature_bounds":[.05,20],"near_search_boundary":bool(min(result.x-.05,20-result.x)<1e-5),
            "probabilities":temperature_probabilities(z,1/result.x).tolist()}


def conformal_threshold(scores, alpha):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not len(scores) or np.any(~np.isfinite(scores)):
        raise ValueError("Need a nonempty vector of finite scores")
    k = conformal_rank(len(scores), alpha)
    return float(np.partition(scores, k-1)[k-1]) if k <= len(scores) else math.inf


def rank_rotation(scores, alpha):
    """Hold each exchangeable position out once; exact conditional rank check."""
    rows = []
    for i, test in enumerate(scores):
        calibration = scores[:i] + scores[i+1:]
        q = conformal_threshold(calibration, alpha)
        rows.append({"test": test, "q": q if math.isfinite(q) else "infinity", "covered": test <= q})
    return {"rows": rows, "covered": sum(r["covered"] for r in rows), "total": len(rows)}


def class_sets(probabilities, q):
    return (1 - np.asarray(probabilities, float) <= q).tolist()


def aps_scores(probabilities):
    p = np.asarray(probabilities, float)
    order = np.argsort(-p, kind="stable")
    cumulative = np.cumsum(p[order])
    result = np.empty(len(p))
    result[order] = cumulative
    return result.tolist()


def main():
    p = [.2]*5 + [.8]*5
    y = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    repaired = [.4]*5 + [.6]*5
    score = [-3, -2, -1, 0, 1, 2, 3, 4]
    labels = [0, 1, 0, 0, 1, 0, 1, 1]
    cal = [.05, .1, .15, .2, .25, .3, .4, .6, .9]
    q = conformal_threshold(cal, .2)
    assert conformal_rank(9, .2) == 8 and q == .6
    assert math.isinf(conformal_threshold(cal, .05))
    assert conformal_threshold(list(reversed(cal)), .2) == q
    assert conformal_threshold([.01]+cal[1:], .2) == q
    assert np.isclose(conformal_threshold(cal[:-2]+[.75,.9], .2), .75)
    # Scaling every score and future candidate score preserves memberships.
    np.testing.assert_array_equal(np.array(cal) <= q, np.array(cal)*3 <= q*3)
    result = {
        "reliability": {"predictions": p, "outcomes": y,
            "two_bins": reliability(p,y,[0,.5,1]), "one_bin": reliability(p,y,[0,1]),
            "repaired": reliability(repaired,y,[0,.5,1]),
            "brier_before": float(np.mean((np.array(p)-y)**2)),
            "brier_repaired": float(np.mean((np.array(repaired)-y)**2)),
            "auc_before": float(roc_auc_score(y,p)), "auc_repaired": float(roc_auc_score(y,repaired)),
            "boundary_fixture": reliability([0,.5,1],[0,1,1],[0,.5,1])},
        "sigmoid": fit_sigmoid(score, labels), "pav": fit_pav(score, labels),
        "pav_ties": fit_pav([-2,-2,0,1],[1,0,0,1]),
        "pav_ties_changed": fit_pav([-2,-2,0,1],[1,0,1,1]),
        "temperature": {str(t): temperature_probabilities([3,1,0],t).tolist() for t in [.5,1,2,4]},
        "temperature_fit":fit_temperature([[3,1,0],[3,1,0],[3,1,0],[0,2,1]],[0,0,1,2]),
        "rank": {"calibration_scores": cal, "k": 8, "q": q,
            "linear_quantile_at_k_over_n": float(np.quantile(cal,8/9)),
            "higher_quantile_at_k_over_n": float(np.quantile(cal,8/9,method="higher")),
            "tiny_alpha_rank": conformal_rank(9,.05),
            "rotation": rank_rotation(cal+[.95],.2),
            "tie_rotation": rank_rotation([.2]*10,.2),
            "probabilities": [[.8,.15,.05],[.45,.4,.15],[.34,.33,.33]],
            "prediction_sets": class_sets([[.8,.15,.05],[.45,.4,.15],[.34,.33,.33]],q)},
        "aps": {"probabilities": [.5,.3,.2], "scores": aps_scores([.5,.3,.2])},
        "group_example": {"weights": [.8,.2],"coverage": [1,.5],"marginal": .8*1+.2*.5},
        "resolution": {"rates": [.1,.3], "coarse": .2,
            "coarse_brier": .16,"full_brier": .15,"coarse_cost":16,"full_cost":14},
        "label_shift": {"sensitivity": .8,"false_positive_rate":.2,
            "ppv_at_prior_half": .8,"ppv_at_prior_tenth": (.8*.1)/(.8*.1+.2*.9)},
        "practice": {"rank_n14_alpha_point2": conformal_rank(14,.2),
            "pav": fit_pav([1,2,3,4,5,6],[0,1,0,1,0,1]),
            "cqr_scores": [-2,-1,0,1,3],"cqr_q": conformal_threshold([-2,-1,0,1,3],.4),
            "cqr_negative_q": conformal_threshold([-5,-4,-3,-2,-1],.4)}
    }
    residuals=np.array([.5,1,1.5,2,3,4,5,6,8])
    scales=np.array([1,1,1,1,2,2,2,3,4])
    normalized_q=conformal_threshold(residuals/scales,.2)
    changed=residuals.copy();changed[-2:]=[12,16]
    result["adaptive"]={"residuals":residuals.tolist(),"scales":scales.tolist(),
        "absolute_q":conformal_threshold(residuals,.2),"normalized_q":normalized_q,
        "changed_residuals":changed.tolist(),"changed_q":conformal_threshold(changed/scales,.2),
        "all_scales_doubled_q":conformal_threshold(residuals/(2*scales),.2),
        "query_predictions":[10,20],"query_scales":[1,3]}
    assert normalized_q==2 and result["adaptive"]["changed_q"]==4
    # Equality must not disappear after replacing s<=q by a rounded 1-q.
    boundary_p=103/240
    boundary_q=1-boundary_p
    result["floating_boundary"]={"p":boundary_p,"q":boundary_q,
        "score_comparison":1-boundary_p<=boundary_q,"rearranged_comparison":boundary_p>=1-boundary_q}
    # Every PAV permutation preserves equal-score grouping, even with tied labels.
    for perm in permutations(range(4)):
        fit = fit_pav(np.array([-2,-2,0,1])[list(perm)],np.array([1,0,0,1])[list(perm)])
        np.testing.assert_allclose(fit["fitted"],[1/3,1/3,1])
    path = Path(__file__).with_name("checked-results.json")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"ece":result["reliability"]["two_bins"]["ece"],"q":q,
                      "rank_covered":result["rank"]["rotation"]["covered"],"saved":path.name},indent=2))


if __name__ == "__main__":
    main()
