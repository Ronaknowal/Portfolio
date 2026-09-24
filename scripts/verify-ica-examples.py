"""Execute the ICA lesson's displayed Python programs in the isolated runtime.

Run with scratch/lesson-tools/Scripts/python.exe. With `--write` the two
```python blocks are taken from the retained manuscript, executed, and written
to src/learn/data/ica-examples.js together with their actual output; the two
programs are also published under public/learn-assets/ica so a learner receives
exactly the bytes that ran. Without `--write` the recorded code and output must
match a fresh execution of the published module.

Both programs run with the asset directory as the working directory, so the real
one reads the same CSV the lesson offers for download. No network access is used.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib
import hashlib
import importlib.metadata
import io
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/independent-component-analysis-ica"
ASSETS = ROOT / "public/learn-assets/ica"
MODULE = ROOT / "src/learn/data/ica-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/ica-native.json"
WRITE = "--write" in sys.argv
PREFIX = "export const icaExamples = "

METADATA = [
    (
        "handSeparation",
        "ica_by_hand.py",
        "Separate the exact four-state mixture",
        "the four rows enumerate a designed distribution, not a sample. Will the whitened covariance be the identity, and will each source match exactly one recovered component?",
    ),
    (
        "realRecording",
        "ica_recording.py",
        "Fit blindly, select on development data, evaluate later",
        "predict which of raw channels, PCA coordinates or ICA coordinates will give the largest held-out absolute correlation.",
    ),
]

ASSETS.mkdir(parents=True, exist_ok=True)
shutil.copyfile(PACKET / "r01-first20s.csv", ASSETS / "r01-first20s.csv")


def serve_provenance():
    """Publish the attribution document with two facts corrected in place.

    The retained packet copy says the CSV has LF lines; every one of its 20,001
    lines is in fact CRLF, which changes the byte count and the SHA-256 anyone
    regenerating it would get. The packet also carries two repository-relative
    links that 404 when served. Both are corrected here and the change is
    declared in the served file, which is what ODC-By asks of a derivative; the
    retained packet copy is left untouched because the ledger binds its hash.
    """
    text = (PACKET / "data-provenance.md").read_text(encoding="utf-8")
    csv_bytes = (ASSETS / "r01-first20s.csv").read_bytes()
    newline = chr(13) + chr(10)
    assert csv_bytes.count(newline.encode()) == 20001, "the CRLF claim must match the served bytes"
    assert csv_bytes.count(chr(10).encode()) == 20001, "every line ends CRLF, none bare LF"
    replacements = [
        ("comma-delimited, LF lines.",
         "comma-delimited, CRLF lines. Every one of the 20,001 lines ends CRLF; the byte count and SHA-256 above are of exactly those bytes."),
        ("The retained [author-calculations.py](author-calculations.py) reconstructs",
         "The retained `docs/teaching/drafts/independent-component-analysis-ica/author-calculations.py` in the lesson repository reconstructs"),
        ("Exact details and phase boundaries are in [the design record](../../ICA-LESSON-DESIGN.md).",
         "Exact details and phase boundaries are in the lesson repository's `docs/teaching/ICA-LESSON-DESIGN.md`."),
    ]
    for old_text, new_text in replacements:
        assert old_text in text, f"provenance text changed: {old_text[:40]!r}"
        text = text.replace(old_text, new_text)
    notice = [
        "> **Served copy.** This is the packet's provenance document with two corrections applied for publication:",
        "> the retained line-ending claim (LF) was corrected to the file's actual CRLF endings, and two",
        "> repository-relative links were replaced by repository paths, because they do not resolve when this",
        "> document is served. Nothing about the source, licence, attribution, calibration or hashes was changed.",
        "> The uncorrected original is retained at",
        "> `docs/teaching/drafts/independent-component-analysis-ica/data-provenance.md`.",
        "",
    ]
    lines = text.split(chr(10))
    # Placed after the licence and attribution block, before "What was collected".
    marker = next(index for index, line in enumerate(lines) if line.startswith("## What was collected"))
    served = chr(10).join(lines[:marker] + notice + lines[marker:])
    (ASSETS / "data-provenance.md").write_text(served, encoding="utf-8", newline=chr(10))


serve_provenance()

digest_bytes = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
digest_text = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()

if WRITE:
    blocks = re.findall(r"```python\n(.*?)\n```", (PACKET / "lesson.md").read_text(encoding="utf-8"), flags=re.S)
    assert len(blocks) == 2, f"expected two displayed Python programs, found {len(blocks)}"
    programs = {
        key: {"title": title, "question": question, "file": file, "code": code.strip(), "language": "python"}
        for (key, file, title, question), code in zip(METADATA, blocks)
    }
else:
    existing = MODULE.read_text(encoding="utf-8")
    programs = json.loads(existing[existing.index(PREFIX) + len(PREFIX):].rstrip().rstrip(";"))

records, namespaces = {}, {}
for key, program in programs.items():
    code = program["code"].strip() + "\n"
    namespace = {"__file__": str(ASSETS / program["file"]), "__name__": "__main__"}
    with contextlib.chdir(ASSETS), threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as stream:
        exec(compile(code, program["file"], "exec"), namespace)
    output = stream.getvalue().strip()
    if not WRITE:
        assert output == program["expected"], f"{key}: output changed\n{output}\n---expected---\n{program['expected']}"
    namespaces[key] = namespace
    records[key] = {**program, "code": code.rstrip("\n"), "expected": output}
    print(f"{key} ({program['file']}):\n{output}\n", flush=True)

# ---------------------------------------------------------------------------
# Oracles: every value the manuscript states must come out of these actual runs.
# ---------------------------------------------------------------------------
oracles = []


def oracle(description, condition):
    assert condition, f"oracle failed: {description}"
    oracles.append(description)


hand = namespaces["handSeparation"]
S, A, X, Z, W, K, B, C = (hand[name] for name in ["S", "A", "X", "Z", "W", "K", "B", "C"])
oracle("four equiprobable source states in the packet's order", np.array_equal(S, [[-1, -1], [-1, 1], [1, -1], [1, 1]]))
oracle("mixing matrix [[2,1],[1,2]]", np.array_equal(A, [[2.0, 1.0], [1.0, 2.0]]))
oracle("observed states (-3,-3), (-1,1), (1,-1), (3,3)", np.array_equal(X, [[-3, -3], [-1, 1], [1, -1], [3, 3]]))
oracle("mixed covariance is [[5,4],[4,5]] with denominator four", np.allclose(X.T @ X / 4, [[5, 4], [4, 5]], atol=1e-12))
oracle("covariance eigenvalues are 9 and 1", np.allclose(sorted(np.linalg.eigvalsh(X.T @ X / 4), reverse=True), [9, 1], atol=1e-12))
oracle("whitened covariance is the identity", np.allclose(Z.T @ Z / 4, np.eye(2), atol=1e-12))
oracle("each whitened coordinate is +/- sqrt(2) or 0", np.allclose(np.sort(np.abs(Z).ravel()), [0, 0, 0, 0, np.sqrt(2)] + [np.sqrt(2)] * 3, atol=1e-12))
oracle("no whitened point sits at the joint origin", np.all(np.abs(Z).sum(axis=1) > 1e-9))
oracle("the recovered unmixing W is orthogonal", np.allclose(W @ W.T, np.eye(2), atol=1e-12))
oracle("B = W K reproduces the estimated sources from centred observations", np.allclose((X - X.mean(axis=0)) @ B.T, Z @ W.T, atol=1e-12))
oracle("both maximum absolute correlations are 1", np.allclose(np.max(np.abs(C), axis=1), [1, 1], atol=1e-12))
oracle("the two sources match different recovered components", int(np.argmax(np.abs(C), axis=1)[0]) != int(np.argmax(np.abs(C), axis=1)[1]))
oracle("the full absolute correlation matrix is a permutation of the identity", np.allclose(np.sort(np.abs(C).ravel()), [0, 0, 1, 1], atol=1e-9))
oracle("reconstruction returns the observed values to numerical precision", np.max(np.abs(hand["reconstructed"] - X)) < 1e-10)
oracle("estimated sources have unit variance in this fixture", np.allclose(hand["estimated"].var(axis=0), [1, 1], atol=1e-12))

# The section-5 hand trace is an independent calculation on the same fixture.
whitener = np.array([[1 / (3 * np.sqrt(2)), 1 / (3 * np.sqrt(2))], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
white = X @ whitener.T
w = np.array([0.8, 0.6])
projection = white @ w
weighted_mean = white.T @ projection ** 3 / 4
raw_update = weighted_mean - (3 * projection ** 2).mean() * w
aligned = -raw_update / np.linalg.norm(raw_update)
oracle("the hand whitener produces the diamond (-sqrt2,0), (0,-sqrt2), (0,sqrt2), (sqrt2,0)",
       np.allclose(white, [[-np.sqrt(2), 0], [0, -np.sqrt(2)], [0, np.sqrt(2)], [np.sqrt(2), 0]], atol=1e-12))
oracle("the four projections are -0.8sqrt2, -0.6sqrt2, 0.6sqrt2, 0.8sqrt2",
       np.allclose(projection, np.sqrt(2) * np.array([-0.8, -0.6, 0.6, 0.8]), atol=1e-12))
oracle("the averaged weighted vector is (1.024, 0.432)", np.allclose(weighted_mean, [1.024, 0.432], atol=1e-12))
oracle("the average derivative is 3", abs((3 * projection ** 2).mean() - 3) < 1e-12)
oracle("the raw update is (-1.376, -1.368)", np.allclose(raw_update, [-1.376, -1.368], atol=1e-12))
oracle("the sign-aligned next direction is (0.7091653, 0.70504225)", np.allclose(np.round(aligned, 8), [0.7091653, 0.70504225], atol=1e-8))

# Section 4 and practice 2: the exact projection kurtoses in the packet.
for name, kappa, expected in [("binary", -2.0, [-2, -1.25, -1, -2]), ("Laplace", 3.0, [3, 1.875, 1.5, 3]), ("Gaussian", 0.0, [0, 0, 0, 0])]:
    values = [kappa * (np.cos(np.deg2rad(a)) ** 4 + np.sin(np.deg2rad(a)) ** 4) for a in [0, 30, 45, 90]]
    oracle(f"{name} projection kurtoses at 0, 30, 45 and 90 degrees", np.allclose(values, expected, atol=1e-12))
oracle("the binary source has excess kurtosis -2", abs(np.mean(np.array([-1.0, 1.0]) ** 4) / np.mean(np.array([-1.0, 1.0]) ** 2) ** 2 - 3 + 2) < 1e-12)
zero_kurtosis = np.array([0.0] * 4 + [np.sqrt(3), -np.sqrt(3)])
weights = np.array([1 / 6] * 4 + [1 / 6, 1 / 6])
oracle("the zero-kurtosis non-Gaussian counterexample has variance 1 and fourth moment 3",
       abs((weights * zero_kurtosis ** 2).sum() - 1) < 1e-12 and abs((weights * zero_kurtosis ** 4).sum() - 3) < 1e-12)
oracle("its sixth moment is 9, against a Gaussian's 15", abs((weights * zero_kurtosis ** 6).sum() - 9) < 1e-12)

real = namespaces["realRecording"]
digital, microvolts, reference = real["digital"], real["microvolts"], real["reference"]
oracle("the CSV supplies 20000 instants and five channels", digital.shape == (20000, 5))
oracle("stored values are integer ADC counts", np.array_equal(digital, np.round(digital)))
oracle("the header calibration converts counts to microvolts", np.allclose(microvolts, (digital + 32768) * (6553.6 / 65535) - 3276.8, atol=1e-12))
oracle("the direct channel is the reference and never an input", np.array_equal(reference, microvolts[:, 0]) and real["X"].shape == (20000, 4))
oracle("the chronological split is 12000 / 4000 / 4000",
       (real["train"], real["development"], real["test"]) == (slice(0, 12000), slice(12000, 16000), slice(16000, 20000)))
oracle("PCA and ICA are fitted on training rows only", real["pca"].n_features_in_ == 4 and real["ica"].n_features_in_ == 4)
oracle("ICA converged in 14 iterations", int(real["ica"].n_iter_) == 14)
oracle("components_ is the 4x4 unmixing operator and mixing_ its pseudoinverse",
       real["ica"].components_.shape == (4, 4) and real["ica"].mixing_.shape == (4, 4)
       and np.allclose(real["ica"].mixing_ @ real["ica"].components_, np.eye(4), atol=1e-8))
oracle("fitted source variances are one under unit-variance whitening",
       np.allclose(real["ica"].transform(real["X"][real["train"]]).var(axis=0), np.ones(4), atol=1e-6))
oracle("a full-rank round trip reconstructs the held-out sensors",
       float(np.mean((real["ica"].inverse_transform(real["ica"].transform(real["X"][real["test"]])) - real["X"][real["test"]]) ** 2)) < 1e-20)

expected_results = {"channel": (3, 0.201203, 0.119806), "PCA": (4, 0.184580, 0.450178), "ICA": (2, 0.169804, 0.343966)}
for name, values in real["representations"]:
    dev = real["correlation_with_reference"](values[real["development"]], reference[real["development"]])
    chosen = int(np.argmax(np.abs(dev)))
    test = real["correlation_with_reference"](values[real["test"]], reference[real["test"]])[chosen]
    index, dev_expected, test_expected = expected_results[name]
    oracle(f"{name} selects coordinate {index} on development data", chosen + 1 == index)
    oracle(f"{name} development absolute correlation is {dev_expected:.6f}", abs(abs(dev[chosen]) - dev_expected) < 5e-7)
    oracle(f"{name} held-out absolute correlation is {test_expected:.6f}", abs(abs(test) - test_expected) < 5e-7)
oracle("PCA coordinate 4 has the largest held-out value in this fixed comparison", 0.450178 > 0.343966 > 0.119806)
oracle("the supplied extract matches the recorded provenance hash",
       digest_bytes(ASSETS / "r01-first20s.csv") == "7c95ef45ceaf96254950ce633b2ab0b5089b15a4fdbd617e1b843846beef23cc")

versions = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn"]}
versions["python"] = sys.version.split()[0]

if WRITE:
    MODULE.write_text(
        "// Complete displayed programs for the ICA lesson with their actual executed\n"
        "// output. Regenerate with scripts/verify-ica-examples.py --write; running the\n"
        "// script without --write proves the recorded output matches a fresh run.\n"
        + PREFIX + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )
    for record in records.values():
        (ASSETS / record["file"]).write_text(record["code"] + "\n", encoding="utf-8", newline="\n")
    (ASSETS / "runtime-versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8", newline="\n")
else:
    for record in records.values():
        assert (ASSETS / record["file"]).read_text(encoding="utf-8") == record["code"] + "\n", f"published {record['file']} differs from the displayed program"

evidence = {
    "status": "passed",
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native execution of the displayed programs; browser, model and independent review are separate",
    "command": "scratch/lesson-tools/Scripts/python.exe scripts/verify-ica-examples.py" + (" --write" if WRITE else ""),
    "versions": versions,
    "module": "src/learn/data/ica-examples.js",
    "moduleSha256": digest_bytes(MODULE),
    "verifierSha256": digest_bytes(Path(__file__)),
    "datasetSha256": digest_bytes(ASSETS / "r01-first20s.csv"),
    "programs": {
        key: {"file": record["file"], "codeSha256": digest_text(record["code"]), "stdoutSha256": digest_text(record["expected"]), "stdout": record["expected"]}
        for key, record in records.items()
    },
    "oracleCount": len(oracles),
    "oracles": oracles,
    "limits": [
        "One record, one fixed 20-second extract and one declared split; a within-recording comparison, not a clinical or cross-person result.",
        "Absolute Pearson correlation to a directly recorded waveform is the whole diagnostic; it is not beat detection or physiological source recovery.",
        "The provider filtered the signal offline, so this cannot demonstrate a real-time pipeline.",
        "Numerical results can differ on other library versions; the recorded snapshot is in versions above.",
    ],
}
EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(records)} displayed programs executed, {len(oracles)} oracle assertions.")
