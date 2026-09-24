"""Small content-author probes; these are not phase-two runtime verification.

Run from this directory. Data are supplied offline. No downloads or plot output.
"""

from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits


BASE = Path(__file__).resolve().parent


def c_exact(n):
    if n <= 1:
        return Fraction(0)
    return 2 * sum((Fraction(1, j) for j in range(1, n)), Fraction(0)) - Fraction(2 * (n - 1), n)


def expected_paths(values, limit):
    """Integrate every 1D cut interval exactly, using harmonic leaf correction."""
    @lru_cache(None)
    def visit(indices, remaining):
        if len(indices) == 1 or remaining == 0 or values[indices[0]] == values[indices[-1]]:
            return tuple(c_exact(len(indices)) for _ in indices)
        result = [Fraction(0) for _ in indices]
        span = values[indices[-1]] - values[indices[0]]
        for cut in range(1, len(indices)):
            gap = values[indices[cut]] - values[indices[cut - 1]]
            if gap == 0:
                continue
            weight = Fraction(gap, span)
            left = visit(indices[:cut], remaining - 1)
            right = visit(indices[cut:], remaining - 1)
            for i, length in enumerate(left + right):
                result[i] += weight * (1 + length)
        return tuple(result)
    return visit(tuple(range(len(values))), limit)


def tiny_lof(values, k):
    """Exactly k others; stable index tie-break; distinct finite 1D inputs."""
    x = np.asarray(values, dtype=float)
    distances = abs(x[:, None] - x)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]
    radii = distances[np.arange(len(x)), neighbors[:, -1]]
    reach = np.maximum(distances[np.arange(len(x))[:, None], neighbors], radii[neighbors])
    lrd = 1 / reach.mean(axis=1)
    lof = (lrd[neighbors] / lrd[:, None]).mean(axis=1)
    return {"neighbors": neighbors, "radii": radii, "reach": reach, "lrd": lrd, "lof": lof}


def score_new(values, k, queries):
    state = tiny_lof(values, k)
    distances = abs(np.asarray(queries)[:, None] - np.asarray(values))
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]
    reach = np.maximum(distances[np.arange(len(queries))[:, None], neighbors], state["radii"][neighbors])
    lrd = 1 / reach.mean(axis=1)
    return (state["lrd"][neighbors] / lrd[:, None]).mean(axis=1)


def real_data_probe():
    raw = pd.read_csv(BASE / "machine_temperature_system_failure.csv", parse_dates=["timestamp"])
    # Duplicate timestamp measurements become one explicitly averaged observation.
    series = raw.groupby("timestamp", sort=True)["value"].mean()
    delayed = series.reindex(series.index - pd.Timedelta(hours=1))
    features = pd.DataFrame({"level": series.to_numpy(), "one_hour_change": series.to_numpy() - delayed.to_numpy()}, index=series.index).dropna()
    fit = features.index < "2013-12-06"
    calibration = (features.index >= "2013-12-06") & (features.index < "2013-12-10")
    test = features.index >= "2013-12-10"
    scaler = StandardScaler().fit(features.loc[fit])
    train_x, calibration_x, test_x = [scaler.transform(features.loc[mask]) for mask in [fit, calibration, test]]
    windows = json.loads((BASE / "nab-event-windows.json").read_text())["realKnownCause/machine_temperature_system_failure.csv"]
    times = features.index[test]
    masks = [(times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end)) for start, end in windows]
    in_window = np.logical_or.reduce(masks)
    baseline_center = float(features.loc[fit, "level"].median())
    baseline_scale = float((features.loc[fit, "level"] - baseline_center).abs().median())
    pairs = {"median absolute level deviation": (
        (features.loc[calibration, "level"] - baseline_center).abs().to_numpy() / baseline_scale,
        (features.loc[test, "level"] - baseline_center).abs().to_numpy() / baseline_scale)}
    models = {
        "Isolation Forest": IsolationForest(n_estimators=100, max_samples=256, contamination="auto", random_state=17, n_jobs=1),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma=0.5, nu=0.05),
        "LOF novelty": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto"),
    }
    for name, model in models.items():
        model.fit(train_x)
        pairs[name] = (-model.score_samples(calibration_x), -model.score_samples(test_x))
    results = []
    for name, (cal_scores, test_scores) in pairs.items():
        for quantile in [0.95, 0.99]:
            threshold = float(np.quantile(cal_scores, quantile, method="higher"))
            alert = test_scores > threshold
            results.append({"method": name, "calibrationQuantile": quantile, "threshold": threshold,
                "calibrationAlerts": int((cal_scores > threshold).sum()), "testAlerts": int(alert.sum()),
                "alertsInsideWindows": int((alert & in_window).sum()), "alertsOutsideWindows": int((alert & ~in_window).sum()),
                "windowHits": [bool((alert & mask).any()) for mask in masks],
                "firstAlertInEachWindow": [str(times[alert & mask][0]) if (alert & mask).any() else None for mask in masks]})
    return {"rawRows": len(raw), "uniqueTimes": len(series), "averagedDuplicateExcess": len(raw) - len(series),
        "removedMissingLagRows": len(series) - len(features), "fitRows": int(fit.sum()), "calibrationRows": int(calibration.sum()),
        "testRows": int(test.sum()), "testRowsInsideWindows": int(in_window.sum()), "testRowsOutsideWindows": int((~in_window).sum()),
        "windows": windows, "baselineMedian": baseline_center, "baselineMAD": baseline_scale,
        "fitMeans": scaler.mean_.tolist(), "fitScales": scaler.scale_.tolist(), "results": results,
        "scope": "One fixed retrospective chronology and two prespecified threshold choices; not NAB official scoring, fault ground truth, timing benchmark, early-warning study or a model-selection campaign."}


def main():
    output = {"scope": "Content-phase elementary reasoning and fixture checks only; implementation and formal independent/browser verification deferred.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikitLearn": sklearn.__version__}}
    output["isolation"] = []
    for values in [(0, 1, 2, 3, 12), (0, 1, 2, 3, 4), (0, 0, 0, 0, 0)]:
        lengths = expected_paths(values, 3)
        output["isolation"].append({"values": values, "maxDepth": 3, "normalizer": str(c_exact(len(values))),
            "meanLengths": [str(h) for h in lengths], "scores": [2 ** (-float(h / c_exact(len(values)))) for h in lengths]})
    values = [0, 1, 2, 20, 24, 28]
    output["lof"] = []
    for k in [2, 3, 5]:
        state = tiny_lof(values, k)
        fit = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(np.array(values)[:, None])
        output["lof"].append({"k": k, "values": values, "state": {name: val.tolist() for name, val in state.items()},
            "queries": [3, 6, 17, 25], "queryLOF": score_new(values, k, [3, 6, 17, 25]).tolist(),
            "libraryQueryLOF": (-fit.score_samples(np.array([3, 6, 17, 25])[:, None])).tolist(),
            "trainingLOF": (-fit.negative_outlier_factor_).tolist(),
            "incorrectTrainingAsQuery": (-fit.score_samples(np.array(values)[:, None])).tolist()})
    output["kernel"] = []
    for gamma in [0.1, 0.5, 1, 2]:
        rho = (1 + np.exp(-4 * gamma)) / 2
        def decision(x):
            return (np.exp(-gamma * (x + 1) ** 2) + np.exp(-gamma * (x - 1) ** 2)) / 2 - rho
        output["kernel"].append({"gamma": gamma, "rho": rho, "atCenter": decision(0), "atReference": decision(1), "atFarQuery": decision(3)})
    output["alarm"] = [{"prevalence": p, "sensitivity": t, "falsePositiveRate": f, "trueAlerts": 100000*p*t,
         "falseAlerts": 100000*(1-p)*f, "precision": p*t/(p*t+(1-p)*f) if p*t+(1-p)*f else None}
         for p,t,f in [(0.001,.8,.01),(.01,.8,.01),(.001,.8,0),(.001,.8,.001)]]
    with threadpool_limits(limits=1):
        output["realData"] = real_data_probe()
    output["sourceHash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (BASE / "author-calculations.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"realData":output["realData"],"isolation":output["isolation"],"kernel":output["kernel"],
        "lof": [{"k": v["k"],"queryLOF":v["queryLOF"],"trainingLOF":v["trainingLOF"],"trainingAsQuery":v["incorrectTrainingAsQuery"]} for v in output["lof"]]},indent=2))


if __name__ == "__main__":
    main()
