"""Content-author calculations. No network, website runtime or model persistence."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def baseline_forecasts(history, horizon=7, period=7):
    history = np.asarray(history, dtype=float)
    lead = np.arange(1, horizon + 1)
    return {
        "mean": np.repeat(history.mean(), horizon),
        "naive": np.repeat(history[-1], horizon),
        "seasonal": history[-period + (lead - 1) % period],
        "drift": np.maximum(
            0, history[-1] + lead * (history[-1] - history[0]) / (len(history) - 1)
        ),
    }


def origin_features(counts, dates, origins, horizon):
    """Each feature row uses counts through its own origin, plus a known date."""
    rows = []
    for origin in origins:
        target_date = dates[origin + horizon]
        annual_phase = 2 * np.pi * (target_date.dayofyear - 1) / 365.25
        weekday = [int(target_date.dayofweek == day) for day in range(7)]
        rows.append([
            counts[origin], counts[origin - 1], counts[origin - 6],
            counts[origin - 6:origin + 1].mean(), *weekday,
            np.sin(annual_phase), np.cos(annual_phase),
            (target_date - dates[0]).days / 365.25,
        ])
    return np.asarray(rows, dtype=float)


def direct_ridge_forecast(counts, dates, origin, window=None):
    result = []
    for horizon in range(1, 8):
        eligible = np.arange(6, origin - horizon + 1)
        if window is not None:
            eligible = eligible[-window:]
        features = origin_features(counts, dates, eligible, horizon)
        labels = counts[eligible + horizon]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="svd"))
        model.fit(features, labels)
        prediction = model.predict(origin_features(counts, dates, [origin], horizon))[0]
        result.append(max(0.0, float(prediction)))
    return np.asarray(result)


def summarize(records):
    errors = np.asarray([r["actual"] for r in records]) - np.asarray(
        [r["predicted"] for r in records]
    )
    return {
        "origins": len(records),
        "forecasts": int(errors.size),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "bias_actual_minus_forecast": float(errors.mean()),
        "mae_by_horizon": np.abs(errors).mean(axis=0).tolist(),
        "rmse_by_horizon": np.sqrt(np.square(errors).mean(axis=0)).tolist(),
    }


def evaluate(counts, dates, origins, methods):
    records = {name: [] for name in methods}
    for origin in origins:
        predictions = baseline_forecasts(counts[:origin + 1])
        for name in methods:
            if name == "ridge_expanding":
                predicted = direct_ridge_forecast(counts, dates, origin)
            elif name == "ridge_90":
                predicted = direct_ridge_forecast(counts, dates, origin, 90)
            else:
                predicted = predictions[name]
            records[name].append({
                "origin_index": int(origin),
                "origin_date": str(dates[origin].date()),
                "target_dates": [str(d.date()) for d in dates[origin + 1:origin + 8]],
                "actual": counts[origin + 1:origin + 8].tolist(),
                "predicted": predicted.tolist(),
            })
    return {name: {"summary": summarize(rows), "records": rows}
            for name, rows in records.items()}


def toy_fixtures():
    history = np.array([10, 20, 10, 20, 12, 22], dtype=float)
    future = np.array([12, 22, 12, 22], dtype=float)
    base = baseline_forecasts(history, 4, 2)
    changed = history.copy()
    changed[-2] = 18
    changed_base = baseline_forecasts(changed, 4, 2)
    # Original recursion example: one-step rule (last + 2), not a fitted model.
    recursive = [24, 26, 28, 30]
    teacher_forced = [24, 14, 24, 14]
    changed_future = np.array([22, 24, 26, 28], dtype=float)
    changed_teacher_forced = [24, 24, 26, 28]
    # A label y[s+h] is available on date s+h+delay in this constructed clock.
    eligibility = []
    for cutoff, horizon, delay in [(12, 3, 2), (12, 1, 0), (12, 3, 0), (10, 3, 2)]:
        origins = list(range(6, 13))
        allowed = [s for s in origins if s + horizon + delay <= cutoff]
        eligibility.append({"cutoff": cutoff, "horizon": horizon, "delay": delay,
                            "eligible_origins": allowed})
    return {
        "history": history.tolist(), "future": future.tolist(),
        "baseline_forecasts": {key: value.tolist() for key, value in base.items()},
        "all_period_forecasts": {
            str(period): baseline_forecasts(history, 8, period)["seasonal"].tolist()
            for period in range(1, 7)
        },
        "baseline_mae": {key: float(np.abs(value - future).mean())
                         for key, value in base.items()},
        "changed_history": changed.tolist(),
        "changed_baseline_forecasts": {
            key: value.tolist() for key, value in changed_base.items()
        },
        "recursive": recursive, "teacher_forced": teacher_forced,
        "recursive_mae": float(np.abs(np.array(recursive) - future).mean()),
        "teacher_forced_mae": float(np.abs(np.array(teacher_forced) - future).mean()),
        "changed_future": changed_future.tolist(),
        "changed_teacher_forced": changed_teacher_forced,
        "changed_recursive_mae": float(np.abs(np.array(recursive) - changed_future).mean()),
        "changed_teacher_forced_mae": float(
            np.abs(np.array(changed_teacher_forced) - changed_future).mean()),
        "eligibility": eligibility,
    }


def main():
    source = ROOT / "bike-sharing-daily.csv"
    frame = pd.read_csv(source, parse_dates=["dteday"])
    dates = pd.DatetimeIndex(frame["dteday"])
    counts = frame["cnt"].to_numpy(dtype=float)
    assert len(frame) == 731 and dates.is_unique
    assert (np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1).all()
    assert np.array_equal(frame["casual"] + frame["registered"], frame["cnt"])
    origins = np.arange(364, len(counts) - 7, 7)
    assert len(origins) == 52
    development_origins, final_origins = origins[:36], origins[36:]
    methods = ["mean", "naive", "seasonal", "drift", "ridge_expanding", "ridge_90"]
    development = evaluate(counts, dates, development_origins, methods)
    selected = min(methods, key=lambda name: development[name]["summary"]["mae"])
    final_methods = list(dict.fromkeys([selected, "naive", "seasonal"]))
    final = evaluate(counts, dates, final_origins, final_methods)
    # Actual-data investigation is restricted to development origins.
    investigations = []
    for origin in [364, 371, 476]:
        actual = counts[origin + 1:origin + 8]
        for period in [1, 7, 14]:
            predicted = baseline_forecasts(counts[:origin + 1], 7, period)["seasonal"]
            investigations.append({
                "origin_index": origin, "origin_date": str(dates[origin].date()),
                "period": period, "actual": actual.tolist(),
                "predicted": predicted.tolist(),
                "mae": float(np.abs(predicted - actual).mean()),
            })
    # Changing an observation after origin cannot change this origin's forecast.
    changed_counts = counts.copy()
    changed_counts[365] += 1000
    null_before = baseline_forecasts(counts[:365])["seasonal"]
    null_after = baseline_forecasts(changed_counts[:365])["seasonal"]
    assert np.array_equal(null_before, null_after)
    output = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "protocol": {
            "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
            "development_origin_indices": development_origins.tolist(),
            "final_origin_indices": final_origins.tolist(),
            "unused_tail_dates": [str(d.date()) for d in dates[729:]],
            "horizons": list(range(1, 8)), "ridge_alpha": 1.0,
            "ridge_solver": "svd", "ridge_windows": [None, 90],
            "development_ridge_fits": 36 * 7 * 2,
            "final_ridge_fits": 16 * 7 if selected.startswith("ridge") else 0,
            "selection": "Development mean absolute error across all origins/horizons",
            "final_policy": "Predeclared rolling refit; no candidate reselection",
            "count_availability_assumption": "At the end of each recorded day",
            "prediction_postprocessing": "Clip drift and ridge below zero",
        },
        "development": development, "selected_method": selected, "final": final,
        "toy": toy_fixtures(), "real_investigation": investigations,
        "future_edit_null": {
            "origin": 364, "edited_row": 365, "increment": 1000,
            "prediction_before": null_before.tolist(),
            "prediction_after": null_after.tolist(),
            "original_actual": counts[365:372].tolist(),
            "changed_actual": changed_counts[365:372].tolist(),
        },
    }
    (ROOT / "calculated-inputs.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "development": {k: v["summary"] for k, v in development.items()},
        "selected": selected,
        "final": {k: v["summary"] for k, v in final.items()},
        "toy": output["toy"], "real_investigation": investigations,
    }, indent=2))


if __name__ == "__main__":
    main()
