"""Recompute the ICA lesson's real-recording module from the supplied extract.

Run with scratch/lesson-tools/Scripts/python.exe. With --write the module is
regenerated; without it the script only checks, so it can be run read-only
against src/. Every value recorded in the content packet - the extract hash, the calibration, the fixed split, the twelve
signed development correlations, the three selected coordinates, the held-out
diagnostics, the iteration count and the reconstruction residual - is recomputed
here and asserted before src/learn/data/ica-data.js is written. Nothing is
hand-copied; an unflattering result is published exactly as it came out.

The generated module carries only what the page displays: the exact numeric
table, a deterministic peak-preserving envelope of the 16-18 second traces and
a 20-row exact window. The 20,000-row diagnostic itself is always computed from
every sample here, never from the plotted envelope.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/independent-component-analysis-ica"
ASSETS = ROOT / "public/learn-assets/ica"
MODULE = ROOT / "src/learn/data/ica-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/ica-data.json"

CSV_SHA256 = "7c95ef45ceaf96254950ce633b2ab0b5089b15a4fdbd617e1b843846beef23cc"
SOURCE_SHA256 = "7549bbd378ea23851c20c0b7924f0a1f9fd909a3a3683c2334144a4c156dcb62"
HEADER = "direct_adc,abdomen1_adc,abdomen2_adc,abdomen3_adc,abdomen4_adc"

# Exactly the values the content packet recorded, asserted before publication.
PACKET_DEVELOPMENT = {
    "channel": [0.043935, 0.101807, 0.201203, 0.079348],
    "PCA": [-0.034740, 0.127795, 0.010770, 0.184580],
    "ICA": [0.079635, 0.169804, -0.043955, -0.116516],
}
PACKET_SELECTION = {"channel": (3, 0.201203, 0.119806), "PCA": (4, 0.184580, 0.450178), "ICA": (2, 0.169804, 0.343966)}

checks = []


def check(description, condition):
    assert condition, f"check failed: {description}"
    checks.append(description)


assert (ASSETS / "r01-first20s.csv").exists(), "run scripts/verify-ica-examples.py --write first"
check("the served extract has the provenance SHA-256", hashlib.sha256((ASSETS / "r01-first20s.csv").read_bytes()).hexdigest() == CSV_SHA256)
check("the served extract is byte-identical to the retained packet CSV",
      (ASSETS / "r01-first20s.csv").read_bytes() == (PACKET / "r01-first20s.csv").read_bytes())
text = (ASSETS / "r01-first20s.csv").read_text(encoding="utf-8")
check("the header names the direct reference and four abdominal channels", text.split("\n")[0].strip() == HEADER)
check("the provenance file records the original EDF hash", SOURCE_SHA256 in (ASSETS / "data-provenance.md").read_text(encoding="utf-8"))

digital = np.loadtxt(ASSETS / "r01-first20s.csv", delimiter=",", skiprows=1)
check("20,000 instants and five channels", digital.shape == (20000, 5))
check("every stored value is a signed 16-bit integer count",
      np.array_equal(digital, np.round(digital)) and digital.min() >= -32768 and digital.max() <= 32767)

microvolts = (digital + 32768) * (6553.6 / 65535) - 3276.8
check("the calibration is the EDF header's endpoint mapping, not an assumed 0.1 uV per count",
      abs(float((microvolts[1, 0] - microvolts[0, 0]) / (digital[1, 0] - digital[0, 0])) - 6553.6 / 65535) < 1e-12
      and abs(6553.6 / 65535 - 0.1) > 1e-7)
reference, X = microvolts[:, 0], microvolts[:, 1:]
train, development, test = slice(0, 12000), slice(12000, 16000), slice(16000, 20000)
check("the chronological split is 12,000 / 4,000 / 4,000 half-open samples at 1,000 Hz",
      (train.stop - train.start, development.stop - development.start, test.stop - test.start) == (12000, 4000, 4000))


def correlation_with_reference(values, target):
    joined = np.column_stack([values, target])
    return np.corrcoef(joined, rowvar=False)[-1, :-1]


pca = PCA(n_components=4, svd_solver="full").fit(X[train])
ica = FastICA(n_components=4, whiten="unit-variance", whiten_solver="svd", algorithm="parallel",
              fun="logcosh", random_state=7, max_iter=1000, tol=1e-5).fit(X[train])
check("ICA converged in the 14 iterations the packet recorded", int(ica.n_iter_) == 14)
check("unit-variance whitening gives fitted sources variance one",
      np.allclose(ica.transform(X[train]).var(axis=0), np.ones(4), atol=1e-6))
reconstruction_mse = float(np.mean((ica.inverse_transform(ica.transform(X[test])) - X[test]) ** 2))
check("a full-rank round trip reconstructs held-out sensors to numerical precision", reconstruction_mse < 1e-20)
check("the reference never enters either fit", pca.n_features_in_ == 4 and ica.n_features_in_ == 4)

representations = [("channel", X, "Original abdominal channel"), ("PCA", pca.transform(X), "PCA coordinate"),
                   ("ICA", ica.transform(X), "ICA coordinate")]
results, traces = [], {}
INTERVAL = slice(16000, 18000)
COLUMNS = 500
GROUP = (INTERVAL.stop - INTERVAL.start) // COLUMNS


def envelope(series):
    """Deterministic peak-preserving envelope: the min and max of each group."""
    blocks = np.asarray(series, dtype=float).reshape(COLUMNS, GROUP)
    return [[round(float(low), 4), round(float(high), 4)] for low, high in zip(blocks.min(axis=1), blocks.max(axis=1))]


def display_scale(series):
    values = np.asarray(series, dtype=float)
    spread = values.std()
    assert spread > 0, "a constant displayed trace has no display z-score"
    return (values - values.mean()) / spread


for name, values, description in representations:
    dev = correlation_with_reference(values[development], reference[development])
    chosen = int(np.argmax(np.abs(dev)))
    signed_test = correlation_with_reference(values[test], reference[test])[chosen]
    index, dev_expected, test_expected = PACKET_SELECTION[name]
    check(f"{name} development correlations match the packet to six decimals",
          np.allclose(np.round(dev, 6), PACKET_DEVELOPMENT[name], atol=1e-9))
    check(f"{name} selects coordinate {index} on development data only", chosen + 1 == index)
    check(f"{name} development |r| is {dev_expected:.6f}", abs(round(abs(float(dev[chosen])), 6) - dev_expected) < 1e-9)
    check(f"{name} held-out |r| is {test_expected:.6f}", abs(round(abs(float(signed_test)), 6) - test_expected) < 1e-9)
    results.append({
        "method": name,
        "description": description,
        "development": [round(float(value), 6) for value in dev],
        "chosen": chosen + 1,
        "developmentAbs": round(abs(float(dev[chosen])), 6),
        "testSigned": round(float(signed_test), 6),
        "testAbs": round(abs(float(signed_test)), 6),
        # Frozen before the test interval is displayed; never a test-driven flip.
        "displaySign": 1 if dev[chosen] >= 0 else -1,
        "unit": "microvolts" if name == "channel" else "arbitrary units",
    })
    scaled = display_scale(values[INTERVAL, chosen]) * (1 if dev[chosen] >= 0 else -1)
    traces[name] = envelope(scaled)

check("PCA keeps its held-out advantage over ICA and the raw channel in this fixed comparison",
      results[1]["testAbs"] > results[2]["testAbs"] > results[0]["testAbs"])
check("every method offered four candidate coordinates", all(len(item["development"]) == 4 for item in results))
traces["reference"] = envelope(display_scale(reference[INTERVAL]))
check("each envelope has 500 peak-preserving columns over 2,000 displayed samples",
      all(len(item) == COLUMNS for item in traces.values()) and COLUMNS * GROUP == 2000)
for name, values, _ in representations:
    chosen = next(item for item in results if item["method"] == name)["chosen"] - 1
    raw = display_scale(values[INTERVAL, chosen]) * next(item for item in results if item["method"] == name)["displaySign"]
    check(f"the {name} envelope retains the extreme displayed values",
          abs(max(high for low, high in traces[name]) - round(float(raw.max()), 4)) < 1e-9
          and abs(min(low for low, high in traces[name]) - round(float(raw.min()), 4)) < 1e-9)

window = []
for offset in range(20):
    row = {"sample": 16000 + offset, "second": round((16000 + offset) / 1000, 3),
           "reference": round(float(reference[16000 + offset]), 3)}
    for item in results:
        values = dict((name, array) for name, array, _ in representations)[item["method"]]
        row[item["method"]] = round(float(values[16000 + offset, item["chosen"] - 1]), 3)
    window.append(row)
check("the exact window exposes 20 consecutive rows from the start of the test interval",
      len(window) == 20 and window[0]["second"] == 16.0 and window[-1]["sample"] == 16019)

versions = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn"]}
versions["python"] = sys.version.split()[0]

module_data = {
    "record": "r01",
    "dataset": "Abdominal and Direct Fetal ECG Database v1.0.0",
    "datasetUrl": "https://physionet.org/content/adfecgdb/1.0.0/",
    "license": "ODC-By 1.0",
    "licenseUrl": "https://opendatacommons.org/licenses/by/1-0/",
    "attribution": "Jezewski J, Matonia A, Kupka T, Roj D, Czabanski R (2012). Extract retained under ODC-By 1.0; retrieved 12 September 2026.",
    "csv": "/learn-assets/ica/r01-first20s.csv",
    "provenance": "/learn-assets/ica/data-provenance.md",
    "sha256": CSV_SHA256,
    "sourceSha256": SOURCE_SHA256,
    "samples": 20000,
    "samplingHz": 1000,
    "calibration": "(count + 32768) x 6553.6 / 65535 - 3276.8 microvolts, from the EDF header endpoints",
    "channels": HEADER.split(","),
    "split": {"train": [0, 12000], "development": [12000, 16000], "test": [16000, 20000]},
    "settings": "FastICA: 4 components, parallel, logcosh, unit-variance whitening, SVD whiten solver, seed 7, tol 1e-5, max_iter 1000. PCA: 4 components, full SVD.",
    "iterations": int(ica.n_iter_),
    "reconstructionMse": reconstruction_mse,
    "results": results,
    "trace": {"start": 16.0, "end": 18.0, "columns": COLUMNS, "samplesPerColumn": GROUP,
              "note": "Display z-score within the shown interval, sign-aligned by the development correlation; not fitted model normalization.",
              "series": traces},
    "window": window,
    "versions": versions,
    "limits": [
        "Absolute Pearson correlation to a directly recorded waveform, over one 4-second held-out interval in one recording.",
        "Not a fetal-beat detector, a clinical accuracy measure or a count of recovered physiological sources.",
        "The provider filtered the signal offline; this extract cannot establish a real-time pipeline.",
        "Temporal autocorrelation means 20,000 instants do not supply the information of 20,000 independent draws.",
    ],
}

module = (
    "/** Real-recording results for the ICA lesson, generated by scripts/verify-ica-data.py.\n"
    " * Source: Abdominal and Direct Fetal ECG Database v1.0.0 (PhysioNet), record r01,\n"
    " * first 20 seconds, ODC-By 1.0. Jezewski J, Matonia A, Kupka T, Roj D, Czabanski R (2012).\n"
    " * Every number here is recomputed from the served CSV; the plotted envelope is a\n"
    " * peak-preserving summary, while all correlations use all 20,000 instants.\n"
    " */\n"
    "export const ICA_RECORDING = " + json.dumps(module_data, ensure_ascii=False, separators=(",", ":")) + ";\n"
)
write = "--write" in sys.argv
if write:
    MODULE.write_text(module, encoding="utf-8", newline="\n")
else:
    # Without --write this script only checks, so it can be run read-only
    # against src/. A drifted module is a failure, never a silent republish.
    assert MODULE.exists(), "src/learn/data/ica-data.js is missing; run this script with --write first"
    assert MODULE.read_text(encoding="utf-8") == module, "stale src/learn/data/ica-data.js; rerun with --write"
written = write

evidence = {
    "status": "passed",
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "command": "scratch/lesson-tools/Scripts/python.exe scripts/verify-ica-data.py" + (" --write" if write else ""),
    "versions": versions,
    "moduleRewritten": written,
    "checkCount": len(checks),
    "checks": checks,
    "results": results,
    "iterations": int(ica.n_iter_),
    "reconstructionMse": reconstruction_mse,
    "sourceHashes": {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), MODULE, ASSETS / "r01-first20s.csv", ASSETS / "data-provenance.md"]
    },
}
EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(checks)} recomputed checks; raw {results[0]['testAbs']}, PCA {results[1]['testAbs']}, ICA {results[2]['testAbs']} held out.")
