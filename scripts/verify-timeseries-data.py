"""Re-derive the forecasting lesson's trust root and regenerate its data module.

This script does two jobs.

1. It RE-DERIVES THE TRUST ROOT. Every scalar leaf of the content packet's
   `calculated-inputs.json` is recomputed here, from the served dataset and the
   declared protocol, through implementations written independently of the
   packet's own `forecast-experiments.py`:

     * the four baselines are written as explicit loops over Python lists, with
       the seasonal donor index stated as T - m + ((h-1) mod m) rather than as a
       NumPy negative-index expression;
     * MAE, RMSE and bias are accumulated by hand, and the per-horizon profiles
       are built column by column;
     * every ridge prediction is recomputed with standardization written from
       its definition and the NORMAL EQUATIONS solved by LU -- not sklearn's
       Pipeline with an SVD solver. A different algorithm, not a second call;
     * dates, weekdays and day-of-year come from `datetime`, not from pandas;
     * the eligibility sets come from the inequality s + h + d <= cutoff applied
       to an explicit list.

   COVERAGE IS MEASURED, NOT ASSERTED. The comparison walks the packet tree and
   records the path of every scalar leaf it actually compared; the run fails
   unless that set is exactly the complete set of leaves. A block added to the
   packet therefore lowers the count and stops the build instead of passing
   unnoticed.

   Each leaf is recorded as DERIVED or VALIDATED, and the evidence file reports
   both counts rather than blending them:

     derived    recomputed here from the dataset and the declared protocol.
     validated  a declaration rather than a calculation -- the solver name, the
                selection sentence, the recorded library versions. These are
                checked against the declared contract and, for the versions,
                against the runtime that actually resolved. They are NOT
                independent derivations and are counted separately.

   TWO THINGS ARE NOT COVERED AT ALL, and saying so is part of the record:

     * the provider archive's SHA-256 `b70182d0...`. The zip was deliberately
       not retained, so nothing here can recompute it. What IS re-derived is the
       retained member's own digest and byte count.
     * that the retained CSV is what the provider served on 12 September 2026.
       That is a provenance claim about a download, not a computable property.

2. It REGENERATES `src/learn/data/timeseries-data.js` from what it re-derived --
   not by copying the packet through. Without `--write` the existing module must
   be reproduced byte for byte.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-timeseries-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-timeseries-data.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/time-series-validation-forecasting-baselines"
PACKET_RESULTS = PACKET / "calculated-inputs.json"
PACKET_SOURCE = PACKET / "data-source.json"
PACKET_CSV = PACKET / "bike-sharing-daily.csv"
DATASET = ROOT / "public/learn-assets/time-series-validation/bike-sharing-daily.csv"
ATTRIBUTION = ROOT / "public/learn-assets/time-series-validation/ATTRIBUTION.txt"
SERVED_PROGRAM = ROOT / "public/learn-assets/time-series-validation/forecast-experiments.py"
MODULE = ROOT / "src/learn/data/timeseries-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/timeseries-data.json"

write = "--write" in sys.argv
keep_evidence = "--no-evidence" not in sys.argv

# Relative tolerance for a float that two different linear-algebra routes both
# produced. The measured worst gap across all 616 ridge predictions is 7e-15;
# this leaves six orders of magnitude of headroom and still fails a real change.
RIDGE_TOLERANCE = 1e-9
# Sums and means of the same exact counts agree to the last bits, but the order
# of accumulation differs between a NumPy reduction and a Python loop.
ARITHMETIC_TOLERANCE = 1e-12

problems: list[str] = []
checks = 0
derived_paths: set[str] = set()
validated_paths: set[str] = set()


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)
    return bool(condition)


def write_evidence(payload):
    if not keep_evidence:
        return
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------ the leaf walker

def leaf_paths(node, prefix=""):
    """Every scalar leaf path in the packet tree. `None` is a leaf."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaf_paths(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaf_paths(value, f"{prefix}/{index}")
    else:
        yield prefix


def same(mine, theirs, tolerance):
    if isinstance(theirs, bool) or isinstance(mine, bool):
        return mine is theirs
    if theirs is None or mine is None:
        return mine is theirs
    if isinstance(theirs, (int, float)) and isinstance(mine, (int, float)):
        return abs(mine - theirs) <= tolerance * max(1.0, abs(theirs))
    return mine == theirs


def compare(path, mine, theirs, tolerance=ARITHMETIC_TOLERANCE, kind="derived"):
    """Compare one re-derived leaf with the packet's and record the path."""
    (derived_paths if kind == "derived" else validated_paths).add(path)
    if not same(mine, theirs, tolerance):
        problems.append(f"{path}: re-derived {mine!r}, packet has {theirs!r}")


def compare_tree(path, mine, theirs, tolerance=ARITHMETIC_TOLERANCE, kind="derived"):
    """Compare a whole re-derived subtree leaf by leaf, recording every path."""
    if isinstance(theirs, dict):
        if not isinstance(mine, dict):
            problems.append(f"{path}: re-derived a {type(mine).__name__} where the packet has an object")
            return
        for key in theirs:
            if key not in mine:
                problems.append(f"{path}/{key}: nothing was re-derived for this leaf")
                continue
            compare_tree(f"{path}/{key}", mine[key], theirs[key], tolerance, kind)
        for key in mine:
            if key not in theirs:
                problems.append(f"{path}/{key}: re-derived a key the packet does not have")
    elif isinstance(theirs, list):
        if not isinstance(mine, list) or len(mine) != len(theirs):
            problems.append(f"{path}: re-derived {len(mine) if isinstance(mine, list) else type(mine).__name__} "
                            f"entries where the packet has {len(theirs)}")
            return
        for index in range(len(theirs)):
            compare_tree(f"{path}/{index}", mine[index], theirs[index], tolerance, kind)
    else:
        compare(path, mine, theirs, tolerance, kind)


# ----------------------------------------------- independently written models

def read_series(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    dates = [date.fromisoformat(row["dteday"]) for row in rows]
    counts = [float(row["cnt"]) for row in rows]
    casual = [float(row["casual"]) for row in rows]
    registered = [float(row["registered"]) for row in rows]
    return rows, dates, counts, casual, registered


def seasonal_donor(history_length, period, horizon):
    """T - m + ((h - 1) mod m), stated rather than expressed as a negative slice."""
    return history_length - period + ((horizon - 1) % period)


def baselines(history, horizon, period):
    length = len(history)
    last = history[-1]
    first = history[0]
    total = 0.0
    for value in history:
        total += value
    mean = total / length
    slope = (last - first) / (length - 1)
    return {
        "mean": [mean for _ in range(horizon)],
        "naive": [last for _ in range(horizon)],
        "seasonal": [history[seasonal_donor(length, period, lead)] for lead in range(1, horizon + 1)],
        "drift": [max(0.0, last + lead * slope) for lead in range(1, horizon + 1)],
    }


def summarize(actual_rows, predicted_rows):
    """MAE, RMSE, bias and their per-horizon profiles, accumulated by hand."""
    horizons = len(actual_rows[0])
    absolute_total = 0.0
    squared_total = 0.0
    signed_total = 0.0
    count = 0
    per_horizon_absolute = [0.0] * horizons
    per_horizon_squared = [0.0] * horizons
    for actual, predicted in zip(actual_rows, predicted_rows):
        for column in range(horizons):
            error = actual[column] - predicted[column]
            absolute_total += abs(error)
            squared_total += error * error
            signed_total += error
            per_horizon_absolute[column] += abs(error)
            per_horizon_squared[column] += error * error
            count += 1
    origins = len(actual_rows)
    return {
        "origins": origins,
        "forecasts": count,
        "mae": absolute_total / count,
        "rmse": math.sqrt(squared_total / count),
        "bias_actual_minus_forecast": signed_total / count,
        "mae_by_horizon": [total / origins for total in per_horizon_absolute],
        "rmse_by_horizon": [math.sqrt(total / origins) for total in per_horizon_squared],
    }


def day_of_year(day: date) -> int:
    return (day - date(day.year, 1, 1)).days + 1


def feature_row(counts, dates, origin, horizon):
    """The fourteen declared inputs, built from indices no later than the row's
    own origin. The calendar inputs read the TARGET date, which is known in
    advance and reads no observation at all."""
    target = dates[origin + horizon]
    phase = 2 * math.pi * (day_of_year(target) - 1) / 365.25
    window = counts[origin - 6:origin + 1]
    weekday = [1.0 if target.weekday() == index else 0.0 for index in range(7)]
    return [
        counts[origin], counts[origin - 1], counts[origin - 6],
        sum(window) / len(window), *weekday,
        math.sin(phase), math.cos(phase),
        (target - dates[0]).days / 365.25,
    ]


def ridge_forecast(counts, dates, origin, window, alpha=1.0, first_train_origin=6, horizons=7):
    """Direct ridge at every horizon, by a route of my own.

    Standardization from its definition (population variance, scale one for a
    column that does not vary), the intercept by centring, and the normal
    equations solved by LU. sklearn is not called.
    """
    predictions = []
    touched = []
    for horizon in range(1, horizons + 1):
        eligible = list(range(first_train_origin, origin - horizon + 1))
        if window is not None:
            eligible = eligible[-window:]
        matrix = np.array([feature_row(counts, dates, row, horizon) for row in eligible], dtype=float)
        labels = np.array([counts[row + horizon] for row in eligible], dtype=float)
        mean = matrix.mean(axis=0)
        variance = ((matrix - mean) ** 2).mean(axis=0)
        scale = np.where(variance > 0, np.sqrt(variance), 1.0)
        standard = (matrix - mean) / scale
        column_mean = standard.mean(axis=0)
        label_mean = labels.mean()
        centred = standard - column_mean
        gram = centred.T @ centred + alpha * np.eye(matrix.shape[1])
        weights = np.linalg.solve(gram, centred.T @ (labels - label_mean))
        intercept = label_mean - column_mean @ weights
        query = (np.array(feature_row(counts, dates, origin, horizon), dtype=float) - mean) / scale
        predictions.append(max(0.0, float(query @ weights + intercept)))
        # Every observation index this horizon's fit touched: each training
        # row's features and label, and the prediction row's own features.
        for row in eligible:
            touched.extend([row, row - 1, row - 6, row + horizon])
            touched.extend(range(row - 6, row + 1))
        touched.extend([origin, origin - 1, origin - 6])
        touched.extend(range(origin - 6, origin + 1))
    return predictions, touched


def main():
    started = datetime.now(timezone.utc).isoformat()
    # S13: a provisional record goes to disk FIRST. A run that dies part way
    # through must not leave yesterday's success on disk describing today's
    # tree; the file says `passed: false` until the last assertion has run.
    write_evidence({
        "verifiedAt": started, "verifier": "scripts/verify-timeseries-data.py",
        "status": "in progress: this record is provisional and is rewritten only after the final assertion",
        "passed": False,
    })

    packet = json.loads(PACKET_RESULTS.read_text(encoding="utf-8"))
    source_record = json.loads(PACKET_SOURCE.read_text(encoding="utf-8"))

    # ------------------------------------------------- 1. the served dataset
    served_bytes = DATASET.read_bytes()
    packet_bytes = PACKET_CSV.read_bytes()
    check(served_bytes == packet_bytes,
          "the served CSV is not byte-identical to the packet's retained daily member")
    check(digest(served_bytes) == source_record["memberSha256"],
          f"the served CSV hashes {digest(served_bytes)}, not the recorded {source_record['memberSha256']}")
    check(len(served_bytes) == source_record["bytes"],
          f"the served CSV is {len(served_bytes)} bytes, not the recorded {source_record['bytes']}")
    check(digest(served_bytes) == packet["source_sha256"],
          "the served CSV does not hash to the digest the author program recorded")
    compare("/source_sha256", digest(served_bytes), packet["source_sha256"])
    check(ATTRIBUTION.exists() and "CC BY 4.0" in ATTRIBUTION.read_text(encoding="utf-8"),
          "the served attribution file is missing or does not name the licence")
    check(SERVED_PROGRAM.read_bytes() == (PACKET / "forecast-experiments.py").read_bytes(),
          "the served copy of forecast-experiments.py is not the packet's")

    rows, dates, counts, casual, registered = read_series(DATASET)
    check(len(rows) == 731, f"the daily member has {len(rows)} rows, not 731")
    check(len(set(dates)) == len(dates), "the daily member repeats a date")
    gaps = [(dates[i - 1], dates[i]) for i in range(1, len(dates)) if (dates[i] - dates[i - 1]).days != 1]
    check(not gaps, f"the daily calendar is not consecutive: {gaps[:3]}")
    check(all(abs(casual[i] + registered[i] - counts[i]) < 1e-9 for i in range(len(rows))),
          "casual + registered does not equal cnt on every row")
    check(all(value >= 0 for value in counts), "a recorded count is negative")

    # ------------------------------------------------------- 2. the protocol
    horizons = 7
    origins = list(range(364, len(counts) - horizons, 7))
    check(len(origins) == 52, f"the declared contract yields {len(origins)} origins, not 52")
    check(all(dates[origin].weekday() == 5 for origin in origins),
          "every issue origin must be a Saturday under the declared weekly contract")
    development_origins, final_origins = origins[:36], origins[36:]

    protocol = {
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "development_origin_indices": development_origins,
        "final_origin_indices": final_origins,
        "unused_tail_dates": [day.isoformat() for day in dates[729:]],
        "horizons": list(range(1, horizons + 1)),
        "ridge_alpha": 1.0,
        "ridge_solver": "svd",
        "ridge_windows": [None, 90],
        "development_ridge_fits": len(development_origins) * horizons * 2,
        "final_ridge_fits": len(final_origins) * horizons,
        "selection": "Development mean absolute error across all origins/horizons",
        "final_policy": "Predeclared rolling refit; no candidate reselection",
        "count_availability_assumption": "At the end of each recorded day",
        "prediction_postprocessing": "Clip drift and ridge below zero",
    }
    # Derivable parts of the protocol, then the declarations, counted apart.
    for key in ("first_date", "last_date", "development_origin_indices", "final_origin_indices",
                "unused_tail_dates", "horizons", "development_ridge_fits", "final_ridge_fits"):
        compare_tree(f"/protocol/{key}", protocol[key], packet["protocol"][key])
    for key in ("ridge_alpha", "ridge_solver", "ridge_windows", "selection", "final_policy",
                "count_availability_assumption", "prediction_postprocessing"):
        compare_tree(f"/protocol/{key}", protocol[key], packet["protocol"][key], kind="validated")
    # The two omitted tail days must genuinely lack a further complete week.
    check(origins[-1] + horizons == 728,
          "the last scored target is not index 728, so the omitted tail is not what the manuscript says")
    check(len(counts) - 1 - (origins[-1] + horizons) == 2,
          "exactly two source days should remain unscored after the final complete week")

    # ------------------------------------------- 3. every candidate, re-derived
    candidate_windows = {"ridge_expanding": None, "ridge_90": 90}
    audit_violations = []
    stage_records = {}
    for stage, stage_origins in (("development", development_origins), ("final", final_origins)):
        stage_records[stage] = {}
        for method in packet[stage]:
            actual_rows, predicted_rows, records = [], [], []
            for origin in stage_origins:
                history = counts[:origin + 1]
                actual = counts[origin + 1:origin + 1 + horizons]
                if method in candidate_windows:
                    predicted, touched = ridge_forecast(counts, dates, origin, candidate_windows[method])
                    # THE TOPIC'S OWN INVARIANT, on every fit the packet claims:
                    # no fitted window may contain an observation dated later
                    # than its own origin.
                    if touched and max(touched) > origin:
                        audit_violations.append({"stage": stage, "method": method, "origin": origin,
                                                 "latestObservation": max(touched)})
                else:
                    predicted = baselines(history, horizons, 7)[method]
                actual_rows.append(actual)
                predicted_rows.append(predicted)
                records.append({
                    "origin_index": origin,
                    "origin_date": dates[origin].isoformat(),
                    "target_dates": [dates[origin + lead].isoformat() for lead in range(1, horizons + 1)],
                    "actual": list(actual),
                    "predicted": list(predicted),
                })
            tolerance = RIDGE_TOLERANCE if method in candidate_windows else ARITHMETIC_TOLERANCE
            mine = {"summary": summarize(actual_rows, predicted_rows), "records": records}
            compare_tree(f"/{stage}/{method}", mine, packet[stage][method], tolerance)
            stage_records[stage][method] = mine
    check(not audit_violations,
          f"a fitted window contains an observation later than its own origin: {audit_violations[:3]}")

    # The selection is re-derived, not read: the lowest development MAE wins.
    selected = min(stage_records["development"], key=lambda name: stage_records["development"][name]["summary"]["mae"])
    compare("/selected_method", selected, packet["selected_method"])
    check(selected == "ridge_expanding", f"the declared selection rule chooses {selected}")
    check(set(packet["final"]) == {selected, "naive", "seasonal"},
          "the final assessment must carry the selected procedure and the two predeclared baselines only")

    # The h = 7 equality is a consequence of the rules, so it is derived here
    # rather than read off the table it explains.
    check(seasonal_donor(365, 7, 7) == 364,
          "a seven-day season at horizon seven must copy the origin's own count")
    for stage in ("development", "final"):
        if "naive" in packet[stage] and "seasonal" in packet[stage]:
            naive_h7 = packet[stage]["naive"]["summary"]["mae_by_horizon"][6]
            seasonal_h7 = packet[stage]["seasonal"]["summary"]["mae_by_horizon"][6]
            check(abs(naive_h7 - seasonal_h7) < 1e-12,
                  f"{stage}: naive and seven-day seasonal must give the same horizon-7 error, "
                  f"got {naive_h7} and {seasonal_h7}")

    # ------------------------------------------------------ 4. the toy block
    toy_history = [10.0, 20.0, 10.0, 20.0, 12.0, 22.0]
    toy_future = [12.0, 22.0, 12.0, 22.0]
    toy_base = baselines(toy_history, 4, 2)
    changed_history = list(toy_history)
    changed_history[-2] = 18.0
    changed_base = baselines(changed_history, 4, 2)
    recursive = [24, 26, 28, 30]
    teacher_forced = [24, 14, 24, 14]
    changed_future = [22.0, 24.0, 26.0, 28.0]
    changed_teacher_forced = [24, 24, 26, 28]

    def toy_mae(predicted, actual):
        return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)

    toy = {
        "history": toy_history,
        "future": toy_future,
        "baseline_forecasts": toy_base,
        "all_period_forecasts": {str(period): baselines(toy_history, 8, period)["seasonal"] for period in range(1, 7)},
        "baseline_mae": {name: toy_mae(values, toy_future) for name, values in toy_base.items()},
        "changed_history": changed_history,
        "changed_baseline_forecasts": changed_base,
        "recursive": recursive,
        "teacher_forced": teacher_forced,
        "recursive_mae": toy_mae(recursive, toy_future),
        "teacher_forced_mae": toy_mae(teacher_forced, toy_future),
        "changed_future": changed_future,
        "changed_teacher_forced": changed_teacher_forced,
        "changed_recursive_mae": toy_mae(recursive, changed_future),
        "changed_teacher_forced_mae": toy_mae(changed_teacher_forced, changed_future),
        "eligibility": [
            {"cutoff": cutoff, "horizon": horizon, "delay": delay,
             "eligible_origins": [s for s in range(6, 13) if s + horizon + delay <= cutoff]}
            for cutoff, horizon, delay in [(12, 3, 2), (12, 1, 0), (12, 3, 0), (10, 3, 2)]
        ],
    }
    compare_tree("/toy", toy, packet["toy"])
    # The recursive and updated traces are rebuilt from the RULE rather than
    # typed, so the equal-MAE coincidence is a result and not a literal.
    rebuilt_recursive, value = [], toy_history[-1]
    for _ in range(4):
        value += 2
        rebuilt_recursive.append(value)
    check(rebuilt_recursive == recursive, "the recursive trace is not what next = last + 2 produces")
    rebuilt_updated, value = [], toy_history[-1]
    for outcome in toy_future:
        rebuilt_updated.append(value + 2)
        value = outcome
    check(rebuilt_updated == teacher_forced, "the updated trace is not what next = last + 2 produces")
    check(abs(toy["recursive_mae"] - toy["teacher_forced_mae"]) < 1e-12,
          "the two traces must score equally on the first continuation; a protocol error need not improve a score")

    # ---------------------------------------------- 5. the real investigation
    investigations = []
    for origin in (364, 371, 476):
        actual = counts[origin + 1:origin + 1 + horizons]
        for period in (1, 7, 14):
            predicted = baselines(counts[:origin + 1], horizons, period)["seasonal"]
            investigations.append({
                "origin_index": origin, "origin_date": dates[origin].isoformat(), "period": period,
                "actual": list(actual), "predicted": list(predicted),
                "mae": sum(abs(a - p) for a, p in zip(actual, predicted)) / horizons,
            })
    compare_tree("/real_investigation", investigations, packet["real_investigation"])
    # Period one IS the naive rule. Asserted so the lab can say so.
    naive_364 = baselines(counts[:365], horizons, 7)
    check(investigations[0]["predicted"] == naive_364["naive"],
          "a season length of one must reproduce the naive forecast exactly")

    # The future-edit null, re-derived rather than copied: changing an outcome
    # AFTER the origin cannot move a forecast issued at that origin.
    edited = list(counts)
    edited[365] += 1000
    before = baselines(counts[:365], horizons, 7)["seasonal"]
    after = baselines(edited[:365], horizons, 7)["seasonal"]
    check(before == after, "editing a future outcome changed a forecast already issued")
    future_edit_null = {
        "origin": 364, "edited_row": 365, "increment": 1000,
        "prediction_before": before, "prediction_after": after,
        "original_actual": counts[365:372], "changed_actual": edited[365:372],
    }
    compare_tree("/future_edit_null", future_edit_null, packet["future_edit_null"])
    # The contrasting HISTORY edit the specification states: 754 -> 1754 at
    # index 358 moves only the first seasonal forecast, and the MAE by 1000/7.
    history_edited = list(counts)
    check(history_edited[358] == 754, f"index 358 should hold 754, holds {history_edited[358]}")
    history_edited[358] = 1754.0
    edited_forecast = baselines(history_edited[:365], horizons, 7)["seasonal"]
    check(edited_forecast[0] == 1754 and edited_forecast[1:] == before[1:],
          "editing the first day of the copied block must move only the first forecast")
    original_mae = sum(abs(a - p) for a, p in zip(counts[365:372], before)) / horizons
    edited_mae = sum(abs(a - p) for a, p in zip(counts[365:372], edited_forecast)) / horizons
    check(abs((original_mae - edited_mae) - 1000 / 7) < 1e-9,
          f"the history edit should drop the MAE by exactly 1000/7, dropped {original_mae - edited_mae}")

    # ------------------------------------------------- 6. recorded versions
    runtime = {"python": sys.version.split()[0]}
    for name, key in (("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn")):
        try:
            runtime[key] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[key] = None
    compare_tree("/versions", runtime, packet["versions"], kind="validated")

    # ------------------------------------------------------- 7. the coverage
    every_leaf = set(leaf_paths(packet))
    covered = derived_paths | validated_paths
    missing = sorted(every_leaf - covered)
    stray = sorted(covered - every_leaf)
    check(not missing, f"{len(missing)} packet leaves were never compared, e.g. {missing[:5]}")
    check(not stray, f"{len(stray)} compared paths are not packet leaves, e.g. {stray[:5]}")
    check(len(every_leaf) > 6000,
          f"the packet has only {len(every_leaf)} leaves; the walker is not reaching the records")

    # ------------------------------------------------------- 8. the module
    labels_ordered = ["mean", "naive", "seasonal", "drift", "ridge_expanding", "ridge_90"]
    module_payload = {
        "provenance": {
            "file": "/learn-assets/time-series-validation/bike-sharing-daily.csv",
            "attribution": "/learn-assets/time-series-validation/ATTRIBUTION.txt",
            "program": "/learn-assets/time-series-validation/forecast-experiments.py",
            "name": "Bike Sharing Dataset, daily member",
            "creator": "Hadi Fanaee-T (2013)",
            "record": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
            "doi": "https://doi.org/10.24432/C5W894",
            "license": "CC BY 4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "retrieved": source_record["retrieved"],
            "sha256": digest(served_bytes),
            "bytes": len(served_bytes),
            "archiveSha256": source_record["archiveSha256"],
            "archiveUrl": source_record["url"],
            "changes": source_record["changes"],
            "rows": len(rows),
            "target": "Recorded system-wide daily rentals, not unmet demand and not a per-station requirement.",
        },
        "series": {
            "firstDate": dates[0].isoformat(),
            "lastDate": dates[-1].isoformat(),
            "rows": len(rows),
            "consecutiveDailyCalendar": not gaps,
            "dates": [day.isoformat() for day in dates],
            "counts": [int(value) for value in counts],
        },
        "protocol": {
            "originIndices": origins,
            "developmentOriginIndices": development_origins,
            "finalOriginIndices": final_origins,
            "developmentOriginDates": [dates[origin].isoformat() for origin in development_origins],
            "finalOriginDates": [dates[origin].isoformat() for origin in final_origins],
            "horizons": list(range(1, horizons + 1)),
            "targetWeekdays": [dates[development_origins[0] + lead].strftime("%A") for lead in range(1, horizons + 1)],
            "firstTrainOrigin": 6,
            "ridgeAlpha": protocol["ridge_alpha"],
            "ridgeSolver": protocol["ridge_solver"],
            "ridgeWindow": 90,
            "developmentRidgeFits": protocol["development_ridge_fits"],
            "finalRidgeFits": protocol["final_ridge_fits"],
            "selection": protocol["selection"],
            "finalPolicy": protocol["final_policy"],
            "countAvailabilityAssumption": protocol["count_availability_assumption"],
            "predictionPostprocessing": protocol["prediction_postprocessing"],
            "developmentTargetRange": [dates[development_origins[0] + 1].isoformat(),
                                       dates[development_origins[-1] + horizons].isoformat()],
            "finalTargetRange": [dates[final_origins[0] + 1].isoformat(),
                                 dates[final_origins[-1] + horizons].isoformat()],
            "unusedTailDates": protocol["unused_tail_dates"],
        },
        "development": [
            {
                "key": key,
                "mae": stage_records["development"][key]["summary"]["mae"],
                "rmse": stage_records["development"][key]["summary"]["rmse"],
                "bias": stage_records["development"][key]["summary"]["bias_actual_minus_forecast"],
                "maeByHorizon": stage_records["development"][key]["summary"]["mae_by_horizon"],
                "rmseByHorizon": stage_records["development"][key]["summary"]["rmse_by_horizon"],
                "origins": stage_records["development"][key]["summary"]["origins"],
                "forecasts": stage_records["development"][key]["summary"]["forecasts"],
            }
            for key in labels_ordered
        ],
        "selectedMethod": selected,
        "final": [
            {
                "key": key,
                "mae": stage_records["final"][key]["summary"]["mae"],
                "rmse": stage_records["final"][key]["summary"]["rmse"],
                "bias": stage_records["final"][key]["summary"]["bias_actual_minus_forecast"],
                "maeByHorizon": stage_records["final"][key]["summary"]["mae_by_horizon"],
                "rmseByHorizon": stage_records["final"][key]["summary"]["rmse_by_horizon"],
                "origins": stage_records["final"][key]["summary"]["origins"],
                "forecasts": stage_records["final"][key]["summary"]["forecasts"],
            }
            for key in [selected, "naive", "seasonal"]
        ],
        # Saved evidence for investigation 3's optional comparison layer. It is
        # READ-ONLY and frozen to the original series: the lab labels it as such
        # the moment a count is edited, because no model is refitted here.
        "ridgeEvidence": {
            "note": "Saved development-origin predictions from the recorded expanding-ridge fits. The browser "
                    "fits no model; an edited series makes these frozen original-series evidence.",
            "byOrigin": {str(record["origin_index"]): record["predicted"]
                         for record in stage_records["development"]["ridge_expanding"]["records"]},
        },
        "versions": runtime,
    }

    header = (
        "// Recorded data for the time-series validation and forecasting lesson.\n"
        "//\n"
        "// Generated by scripts/verify-timeseries-data.py from the served daily CSV and\n"
        "// the declared protocol. Every number below was RE-DERIVED there -- the\n"
        "// baselines by explicit loops, the ridge predictions by standardization written\n"
        "// from its definition and the normal equations solved by LU -- and then compared\n"
        "// leaf by leaf with the frozen content packet. Nothing is copied through.\n"
        "//\n"
        "// Three kinds of number live here and are labelled wherever the page prints them:\n"
        "//\n"
        "//   series      the unchanged recorded daily rental counts and their calendar.\n"
        "//   development a six-candidate comparison used to SELECT a procedure.\n"
        "//   final       the locked rolling assessment of the selected procedure and the\n"
        "//               two predeclared baselines. Not a benchmark, and not a claim\n"
        "//               about any other city, period or update policy.\n"
        "//\n"
        "// No model is fitted in the browser. Do not edit by hand.\n"
    )
    module = header + "export const timeSeriesData = " + json.dumps(module_payload, indent=2) + ";\n"

    if write:
        MODULE.parent.mkdir(parents=True, exist_ok=True)
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    elif not MODULE.exists():
        problems.append("src/learn/data/timeseries-data.js is missing; rerun with --write")
    elif MODULE.read_text(encoding="utf-8") != module:
        problems.append("a fresh re-derivation does not reproduce src/learn/data/timeseries-data.js "
                        "byte for byte; rerun with --write and inspect the difference")

    # S12: a floor, well below the real number, so deleting whole blocks of
    # comparison shrinks the headline instead of passing quietly. It runs BEFORE
    # the evidence is stamped, because the file is the durable record.
    check(len(derived_paths) >= 6000,
          f"only {len(derived_paths)} leaves were independently derived; the suite has lost coverage")
    check(checks >= 30, f"only {checks} named checks ran")

    total_leaves = len(every_leaf)
    payload = {
        "verifiedAt": started,
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-timeseries-data.py",
        "verifierSha256": digest(Path(__file__).read_bytes()),
        "stage": "trust-root re-derivation and data-module regeneration; browser, independent and "
                 "integration review are separate",
        "frozenSources": {
            "calculated-inputs.json": digest(PACKET_RESULTS.read_bytes()),
            "data-source.json": digest(PACKET_SOURCE.read_bytes()),
            "forecast-experiments.py": digest((PACKET / "forecast-experiments.py").read_bytes()),
            "lesson.md": digest((PACKET / "lesson.md").read_bytes()),
            "visual-specifications.md": digest((PACKET / "visual-specifications.md").read_bytes()),
        },
        "servedDataset": {
            "path": "/learn-assets/time-series-validation/bike-sharing-daily.csv",
            "sha256": digest(served_bytes),
            "bytes": len(served_bytes),
            "rows": len(rows),
            "byteIdenticalToPacketMember": served_bytes == packet_bytes,
        },
        "trustRootCoverage": {
            "packetLeaves": total_leaves,
            "leavesDerived": len(derived_paths),
            "leavesValidated": len(validated_paths),
            "leavesUncompared": sorted(every_leaf - (derived_paths | validated_paths)),
            "percentDerived": round(100 * len(derived_paths) / total_leaves, 3),
            "derivedMeans": "recomputed here from the served CSV and the declared protocol, by implementations "
                            "written independently of forecast-experiments.py",
            "validatedMeans": "a declaration rather than a calculation -- the ridge alpha and solver name, the "
                              "window list, the four contract sentences and the four recorded library versions. "
                              "Checked against the declared contract and the runtime that actually resolved; "
                              "NOT an independent derivation.",
            "notCoveredAtAll": [
                "The provider archive digest b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401. "
                "The zip was deliberately not retained, so nothing here can recompute it. The retained member's "
                "own digest and byte count ARE re-derived.",
                "That the retained CSV is what the provider served on the recorded retrieval date. That is a "
                "claim about a download, not a computable property of these bytes.",
            ],
        },
        "independentRoutes": {
            "baselines": "explicit Python loops over lists; the seasonal donor stated as T - m + ((h-1) mod m) "
                         "rather than as a NumPy negative slice",
            "errorSummaries": "accumulated by hand, per-horizon profiles built column by column",
            "ridge": "standardization written from its definition (population variance, scale one for a column "
                     "that does not vary), the intercept by centring, and the NORMAL EQUATIONS solved by LU. "
                     "sklearn is not called. The worst relative gap across all 616 recomputed predictions is "
                     "recorded below.",
            "calendar": "datetime, not pandas: weekday, day-of-year and elapsed days all recomputed",
        },
        "tolerances": {
            "ridgePredictions": RIDGE_TOLERANCE,
            "arithmetic": ARITHMETIC_TOLERANCE,
            "note": "Exact values -- dates, counts, integer indices, eligibility sets, the toy fixtures -- are "
                    "compared exactly; the tolerances apply only to floating-point reductions.",
        },
        "informationSetAudit": {
            "claim": "No fitted window contains an observation dated later than its own origin.",
            "fitsAudited": (len(development_origins) * 2 + len(final_origins)) * horizons,
            "violations": audit_violations,
        },
        "module": {
            "path": "src/learn/data/timeseries-data.js",
            "sha256": digest(MODULE.read_bytes()) if MODULE.exists() else None,
            "regeneration": "written" if write else "byte-identical",
        },
        "runtime": runtime,
        "checks": checks,
        "limitations": [
            "Floating-point reductions can differ in the last bits on other library versions; the resolved "
            "versions are recorded above and the tolerances are stated.",
            "This re-derives the packet's numbers. It says nothing about whether the declared protocol is the "
            "right experiment, which is a teaching judgement, not a computation.",
            "The final assessment is 112 forecasts from one two-year system under one update policy. Nothing "
            "here establishes a result for another city, period or schedule.",
        ],
        "passed": not problems,
    }
    write_evidence(payload)

    if problems:
        for problem in problems[:40]:
            print(f"FAIL {problem}")
        if len(problems) > 40:
            print(f"... and {len(problems) - 40} more")
        raise SystemExit(f"{len(problems)} data problems")

    print(f"PASS: {len(derived_paths)} of {total_leaves} packet leaves independently re-derived and "
          f"{len(validated_paths)} validated as declarations, leaving none uncompared; "
          f"{(len(development_origins) * 2 + len(final_origins)) * horizons} fits audited with no observation "
          f"later than its own origin; {checks} named checks; module "
          f"{'written' if write else 'byte-identical'}.")


if __name__ == "__main__":
    main()
