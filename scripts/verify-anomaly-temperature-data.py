"""Derive the anomaly lesson's temperature runtime data from the pinned NAB bytes.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/anomaly-temperature-data.js; without it, every emitted value is
recomputed and compared with the file on disk.

The browser never recomputes detector scores. This script fits the detectors
once from the raw CSV and emits exact counts for every threshold the
investigation can reach, so the lesson displays native results rather than
browser arithmetic on truncated values. It also re-derives the packet's score
table independently and checks the author's published comparison rows.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits

PACKET = Path("docs/teaching/drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof")
ASSETS = Path("public/learn-assets/anomaly-detection")
MODULE = Path("src/learn/data/anomaly-temperature-data.js")
EVIDENCE = Path("docs/teaching/evidence/anomaly-temperature-data.json")
SERIES_KEY = "realKnownCause/machine_temperature_system_failure.csv"
METHODS = ["baseline", "isolation", "oneClassSvm", "lof"]
LABELS = {
    "baseline": "Median absolute level deviation",
    "isolation": "Isolation Forest",
    "oneClassSvm": "One-Class SVM",
    "lof": "LOF novelty",
}
OVERVIEW_BINS = 480
SWEEP_POINTS = 240
CONTEXT_ROWS = 72  # six hours of five-minute rows on each side of a window


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    """Fit once on the reference period and score every later row, natively."""
    raw = pd.read_csv(PACKET / "machine_temperature_system_failure.csv", parse_dates=["timestamp"])
    level = raw.groupby("timestamp", sort=True)["value"].mean()
    previous = level.reindex(level.index - pd.Timedelta(hours=1))
    features = pd.DataFrame(
        {"level": level.to_numpy(), "change": level.to_numpy() - previous.to_numpy()},
        index=level.index,
    ).dropna()

    fit = features.index < "2013-12-06"
    cal = (features.index >= "2013-12-06") & (features.index < "2013-12-10")
    test = features.index >= "2013-12-10"
    scaler = StandardScaler().fit(features.loc[fit])
    x_fit, x_cal, x_test = (scaler.transform(features.loc[mask]) for mask in (fit, cal, test))

    median = features.loc[fit, "level"].median()
    mad = (features.loc[fit, "level"] - median).abs().median()
    scores = {
        "baseline": (
            abs(features.loc[cal, "level"].to_numpy() - median) / mad,
            abs(features.loc[test, "level"].to_numpy() - median) / mad,
        )
    }
    models = {
        "isolation": IsolationForest(
            n_estimators=100, max_samples=256, contamination="auto", random_state=17, n_jobs=1
        ),
        "oneClassSvm": OneClassSVM(kernel="rbf", gamma=0.5, nu=0.05),
        "lof": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto"),
    }
    with threadpool_limits(limits=1):
        for name, model in models.items():
            model.fit(x_fit)
            scores[name] = (-model.score_samples(x_cal), -model.score_samples(x_test))

    windows = json.loads((PACKET / "nab-event-windows.json").read_text())[SERIES_KEY]
    times = features.index[test]
    masks = [
        (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end)) for start, end in windows
    ]
    return {
        "features": features,
        "masks": {"fit": fit, "cal": cal, "test": test},
        "scores": scores,
        "windows": windows,
        "windowMasks": masks,
        "median": float(median),
        "mad": float(mad),
        "scaler": scaler,
        "times": times,
    }


def outcome(test_score, cal_score, threshold, window_masks, inside):
    """Exact native counts for one threshold, using strict greater-than.

    Returns a positional row: threshold, calibration alerts, test alerts, alerts
    inside and outside the annotated windows, and the four window hits packed
    into one integer whose bit i is window i + 1.
    """
    alert = test_score > threshold
    hits = sum(int(bool((alert & mask).any())) << index for index, mask in enumerate(window_masks))
    return [float(threshold), int((cal_score > threshold).sum()), int(alert.sum()),
            int((alert & inside).sum()), int((alert & ~inside).sum()), hits]


def number(value, decimals):
    rounded = round(float(value), decimals)
    return int(rounded) if rounded == int(rounded) else rounded


def emit_array(values, decimals=None, per_line=16):
    items = [repr(v if decimals is None else number(v, decimals)) for v in values]
    lines, current = [], []
    for item in items:
        if len(current) == per_line:
            lines.append(", ".join(current))
            current = []
        current.append(item)
    lines.append(", ".join(current))
    return "[\n  " + ",\n  ".join(lines) + "\n]"


def main():
    write = "--write" in sys.argv
    data = build()
    features, scores, masks = data["features"], data["scores"], data["masks"]
    times = data["times"]
    inside = np.logical_or.reduce(data["windowMasks"])
    counts = {name: int(np.asarray(mask).sum()) for name, mask in masks.items()}
    assert counts == {"fit": 885, "cal": 1152, "test": 20634}, counts
    assert int(inside.sum()) == 2268 and int((~inside).sum()) == 18366

    # The retained feature rows happen to be exactly five minutes apart, which the
    # module relies on to derive every timestamp from the first one.
    step = np.diff(features.index.to_numpy()).astype("timedelta64[m]").astype(int)
    assert set(step.tolist()) == {5}, sorted(set(step.tolist()))

    # Independent check against the packet's derived score table.
    packet_rows = list(csv.DictReader(open(PACKET / "nab-derived-scores.csv")))
    packet_columns = {"baseline": "baseline", "isolation": "isolation",
                      "oneClassSvm": "one_class_svm", "lof": "lof"}
    for period, slot in (("calibration", 0), ("test", 1)):
        for name, column in packet_columns.items():
            recorded = np.array([float(r[column]) for r in packet_rows if r["period"] == period])
            np.testing.assert_allclose(scores[name][slot], recorded, rtol=0, atol=1e-9,
                                       err_msg=f"{name} {period} differs from the packet table")

    cal_index = features.index[masks["cal"]]
    reachable = sorted({int(np.ceil(q * (counts["cal"] - 1))) for q in np.arange(0.9, 1.00001, 0.0001)})
    quantile_table, sweep_table, published = {}, {}, {}
    for name in METHODS:
        cal_score, test_score = scores[name]
        ordered = np.sort(cal_score)
        quantile_table[name] = [
            [i, *outcome(test_score, cal_score, ordered[i], data["windowMasks"], inside)]
            for i in reachable
        ]
        # Thresholds the learner can type directly. Each sits midway between two
        # consecutive observed test scores, so the resolution follows the data and
        # no threshold coincides with a score: strict greater-than is then
        # unambiguous and a rounded copy of a score cannot cross it.
        distinct = np.unique(test_score)
        ranks = np.unique(np.linspace(0, len(distinct) - 2, SWEEP_POINTS).round().astype(int))
        levels = (distinct[ranks] + distinct[ranks + 1]) / 2
        span = float(distinct[-1] - distinct[0])
        levels = np.concatenate([[float(distinct[0]) - max(span * 0.05, 1e-3)], levels,
                                 [float(distinct[-1]) + max(span * 0.05, 1e-3)]])
        sweep_table[name] = [
            outcome(test_score, cal_score, t, data["windowMasks"], inside) for t in levels
        ]
        published[name] = {}
        for q in (0.95, 0.975, 0.99):
            threshold = float(np.quantile(cal_score, q, method="higher"))
            published[name][f"{q}"] = outcome(test_score, cal_score, threshold,
                                              data["windowMasks"], inside)

    # The manuscript's published comparison rows must come out of this run.
    expected = {
        ("baseline", "0.95"): (4416, 1379, 3037), ("baseline", "0.99"): (1461, 1016, 445),
        ("isolation", "0.95"): (10530, 1722, 8808), ("isolation", "0.99"): (1548, 971, 577),
        ("oneClassSvm", "0.95"): (9719, 1347, 8372), ("oneClassSvm", "0.99"): (9232, 1319, 7913),
        ("lof", "0.95"): (8771, 1275, 7496), ("lof", "0.99"): (7837, 1138, 6699),
    }
    for (name, q), (total, hit, miss) in expected.items():
        row = published[name][q]
        assert (row[2], row[3], row[4]) == (total, hit, miss), (name, q, row)
        assert row[5] == 0b1111, (name, q, row)
    # The specification's unsolved q = 0.975 cases.
    for name, threshold, alerts, outside in (
        ("baseline", 4.566579723695049, 2170, 1036),
        ("isolation", 0.5555652392489324, 5632, 4263),
        ("oneClassSvm", -6.415545934270259, 9661, 8318),
        ("lof", 1.4936764075602469, 8191, 7002),
    ):
        row = published[name]["0.975"]
        assert abs(row[0] - threshold) < 1e-9, (name, row[0])
        assert (row[2], row[4], row[1]) == (alerts, outside, 28), (name, row)

    # Overview: extrema-preserving bins over the whole feature series. A bin's
    # maximum score decides exactly whether it contains an alert at any threshold.
    order = np.arange(len(features))
    period_of = np.where(np.asarray(masks["fit"]), 0, np.where(np.asarray(masks["cal"]), 1, 2))
    full = {name: np.concatenate([np.full(counts["fit"], np.nan), scores[name][0], scores[name][1]])
            for name in METHODS}
    edges = np.linspace(0, len(features), OVERVIEW_BINS + 1).round().astype(int)
    bins = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        chunk = features.iloc[start:end]
        bins.append([
            int(start), int(end - start),
            number(chunk["level"].min(), 2), number(chunk["level"].max(), 2),
            int(np.bincount(period_of[start:end], minlength=3).argmax()),
            *[None if np.isnan(full[name][start:end]).all()
              else number(np.nanmax(full[name][start:end]), 6) for name in METHODS],
        ])

    # Detail rows: each annotated window with six hours of context on both sides.
    test_offset = counts["fit"] + counts["cal"]
    # Every threshold the investigation can offer, per method, in ascending order.
    available = {name: sorted([row[1] for row in quantile_table[name]]
                              + [row[0] for row in sweep_table[name]])
                 for name in METHODS}
    blocks = []
    for window_number, mask in enumerate(data["windowMasks"], start=1):
        positions = np.flatnonzero(mask)
        low = max(0, int(positions[0]) - CONTEXT_ROWS)
        high = min(counts["test"] - 1, int(positions[-1]) + CONTEXT_ROWS)
        blocks.append((window_number, low, high, int(positions[0]), int(positions[-1])))

    # Rounded scores cannot decide alerts faithfully: a threshold can lie within
    # 1e-10 of a score. Store instead how many available thresholds each row's
    # native score exceeds. A row alerts at threshold i exactly when that count
    # is greater than i, so marking stays exact while the shown score is rounded.
    detail_decimals = 4
    detail_windows = []
    for window_number, low, high, inside_from, inside_to in blocks:
        rows = slice(low, high + 1)
        entry = {
            "window": window_number,
            "firstRow": low + test_offset,
            "insideFrom": inside_from - low,
            "insideTo": inside_to - low,
            "level": [number(v, 2) for v in features.iloc[test_offset + low:test_offset + high + 1]["level"]],
            "change": [number(v, 2) for v in features.iloc[test_offset + low:test_offset + high + 1]["change"]],
            "score": {name: [number(v, detail_decimals) for v in scores[name][1][rows]] for name in METHODS},
            "exceeds": {name: [int(np.searchsorted(available[name], v, side="left"))
                               for v in scores[name][1][rows]] for name in METHODS},
        }
        for name in METHODS:
            native = scores[name][1][rows]
            for index, threshold in enumerate(available[name]):
                marked = np.array(entry["exceeds"][name]) > index
                assert np.array_equal(native > threshold, marked), (name, window_number, threshold)
        detail_windows.append(entry)

    first = features.index[0]
    hits_note = "bit i of the packed hits is window i + 1"

    def rows_text(rows, indent="    "):
        return ",\n".join(indent + json.dumps(row) for row in rows)

    module = f'''/** Machine-temperature monitoring data for the anomaly-detection lesson.
 *
 * Source: Numenta Anomaly Benchmark, pinned commit
 * ea702d75cc2258d9d7dd35ca8e5e2539d71f3140, MIT licensed. The raw series, the
 * annotation windows and the licence are served under their upstream names from
 * /learn-assets/anomaly-detection/.
 * Temperature units and timezone are unspecified upstream; values and timestamps
 * are used exactly as recorded.
 *
 * Every number here was produced by scripts/verify-anomaly-temperature-data.py,
 * which fits the four detectors once on the reference period with scikit-learn
 * {importlib.metadata.version("scikit-learn")} and scores every later row. The browser refits nothing and
 * recomputes no count: the threshold tables carry exact native outcomes for
 * every threshold the investigation can reach, derived from all {counts["test"]:,} test rows.
 */

/** The {len(features):,} feature rows are exactly five minutes apart, so every timestamp is
 * derived from the first one. That spacing was checked, not assumed. */
export const seriesStart = '{first.isoformat(sep=" ")}';
export const stepMinutes = 5;
export const rowCounts = Object.freeze({{ total: {len(features)}, reference: {counts["fit"]}, calibration: {counts["cal"]}, test: {counts["test"]}, insideWindows: 2268, outsideWindows: 18366 }});

/** Facts about the supplied raw file, before features were formed. */
export const sourceFacts = Object.freeze({{ rawRows: 22695, uniqueTimestamps: 22683, duplicateExcess: 12, droppedMissingLag: 12 }});

/** Inclusive annotation windows: published event windows, not per-row fault
 * labels and not verified onset times. */
export const eventWindows = Object.freeze({json.dumps(data["windows"])}.map(pair => Object.freeze(pair)));

/** Reference-period constants: the baseline's median and MAD of level, and the
 * scaler the learned detectors share. */
export const referenceFit = Object.freeze({{
  median: {data["median"]!r},
  mad: {data["mad"]!r},
  means: {json.dumps([float(v) for v in data["scaler"].mean_])},
  scales: {json.dumps([float(v) for v in data["scaler"].scale_])}
}});

export const methodOrder = Object.freeze({json.dumps(METHODS)});
export const methodLabels = Object.freeze({json.dumps(LABELS)});

/** {len(bins)} extrema-preserving bins of the whole series for the overview trace:
 * [firstRow, rows, lowestLevel, highestLevel, period, highest score per method].
 * Period codes are 0 reference, 1 calibration, 2 test; reference rows are unscored,
 * so their score entries are null. The highest score in a bin decides exactly
 * whether that bin contains an alert, because an alert is a score strictly above
 * the threshold. */
export const overviewBins = Object.freeze([
{rows_text(bins, "  ")}
]);

/** Exact outcomes for every calibration quantile the investigation accepts:
 * [sortedCalibrationIndex, threshold, calibrationAlerts, testAlerts,
 *  alertsInsideWindows, alertsOutsideWindows, packedWindowHits], where
 * {hits_note}. NumPy's higher-interpolation quantile maps q to
 * index ceil(q x {counts["cal"] - 1}) among the {counts["cal"]} sorted calibration scores. */
export const quantileOutcomes = Object.freeze({{
{",".join(chr(10) + f"  {name}: [" + chr(10) + rows_text(quantile_table[name]) + chr(10) + "  ]" for name in METHODS)}
}});

/** Exact outcomes for directly typed thresholds:
 * [threshold, calibrationAlerts, testAlerts, inside, outside, packedWindowHits].
 * Levels sit at actual test-score ranks, so the resolution follows the data; a
 * typed value snaps to the nearest level and the lab names the level it used.
 * The first level lies below every test score, so it alerts on all {counts["test"]:,} rows. */
export const sweepOutcomes = Object.freeze({{
{",".join(chr(10) + f"  {name}: [" + chr(10) + rows_text(sweep_table[name]) + chr(10) + "  ]" for name in METHODS)}
}});

/** The manuscript's worked 0.95 and 0.99 comparison, and the 0.975 case the
 * investigation starts from unsolved. Same positional layout as a sweep row. */
export const publishedOutcomes = Object.freeze({{
{",".join(chr(10) + "  " + name + ": {" + ",".join(chr(10) + "    '" + q + "': " + json.dumps(published[name][q]) for q in ("0.95", "0.975", "0.99")) + chr(10) + "  }" for name in METHODS)}
}});

/** Actual rows around each annotated window: {CONTEXT_ROWS} rows of context on either
 * side, as parallel arrays. Levels and changes are rounded to two decimals for
 * display, and so are scores. Alert marking does not use those rounded scores:
 * `exceeds` counts how many of this method's available thresholds the row's
 * native score is above, in the ascending order of the quantile and sweep
 * tables combined, so a row alerts at threshold index i exactly when its count
 * exceeds i. That equivalence was checked against the native scores. */
export const windowDetail = Object.freeze([
{",".join(chr(10) + "  " + json.dumps(entry) for entry in detail_windows)}
].map(entry => Object.freeze(entry)));
'''

    if write:
        MODULE.write_text(module, encoding="utf-8")
        ASSETS.mkdir(parents=True, exist_ok=True)
        # Keep the upstream filenames so the lesson's program runs against the
        # files a learner downloads without renaming anything.
        for name in ("machine_temperature_system_failure.csv", "nab-event-windows.json", "NAB-LICENSE.txt"):
            (ASSETS / name).write_bytes((PACKET / name).read_bytes())
    else:
        current = MODULE.read_text(encoding="utf-8")
        assert current == module, "The emitted data module differs from the file on disk."

    evidence = {
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native data derivation for the browser lesson; independent and browser review are separate",
        "generator": "scripts/verify-anomaly-temperature-data.py",
        "generatorHash": digest("scripts/verify-anomaly-temperature-data.py"),
        "module": str(MODULE).replace("\\", "/"),
        "moduleHash": digest(MODULE),
        "moduleBytes": MODULE.stat().st_size,
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        "sourceHashes": {
            f"{PACKET}/{name}".replace("\\", "/"): digest(PACKET / name)
            for name in ("machine_temperature_system_failure.csv", "nab-event-windows.json",
                         "nab-derived-scores.csv", "NAB-LICENSE.txt")
        },
        "checks": [
            "Refitted every detector from the raw CSV and matched the packet's 21,786 recorded scores within 1e-9.",
            "Row counts 885/1152/20634 and window membership 2268/18366 reproduced.",
            "All retained feature rows are exactly five minutes apart, so derived timestamps are exact.",
            "The manuscript's eight published 0.95/0.99 rows and the specification's four 0.975 rows reproduced exactly.",
            f"{len(reachable)} reachable calibration-quantile thresholds and {SWEEP_POINTS + 2} direct threshold levels per method "
            f"({SWEEP_POINTS} midpoints between consecutive distinct scores plus the two bounds) carry exact native counts.",
        ],
        "limits": [
            "Outside-window alerts are unmatched workload, not verified false positives.",
            "One fixed retrospective chronology; no NAB official scoring, onset timing or early-warning claim.",
            "Overview bins are a drawing summary; every published count comes from all 20,634 test rows.",
        ],
        "passed": True,
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(features)} feature rows, {len(reachable)} quantile thresholds and "
          f"{SWEEP_POINTS + 2} direct threshold levels per method, module {MODULE.stat().st_size / 1024:.0f} KB.")


main()
