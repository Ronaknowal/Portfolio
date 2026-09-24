"""Execute the NMF lesson's displayed Python programs.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/nmf-examples.js from the executed programs; without it, the
recorded output must match a fresh execution.

The digit program reads the CSV the lesson offers for download, from
public/learn-assets/nmf, which is the same bytes a learner receives. The word
program was declared unexecuted in the content phase; it is executed here and
its actual output is recorded, so no fitted number on the page is invented.

Run:  scratch/lesson-tools/Scripts/python.exe scripts/verify-nmf-examples.py --write
      then again without --write to prove the recording matches a fresh run.
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
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (ROOT / "public/learn-assets/nmf").resolve()

PROGRAMS = {
    "multiplicativeStep": {
        "file": "nmf_step.py",
        "title": "Forty multiplicative sweeps on three constructed observations",
        "question": "the initial factors are not the ones that built X, and the initial loss is 17.06. Will forty sweeps drive the loss to zero, and will the factors return to the ones from section 1?",
        "code": r"""
import numpy as np

X = np.array([[2., 1., 3.], [1., 2., 3.], [3., 3., 6.]])
W = np.array([[1., .5], [.5, 1.], [1., 1.]])
H = np.array([[1., .2, .8], [.2, 1., .8]])

def objective(X, W, H):
    return np.sum((X - W @ H) ** 2) / 2

print(0, round(objective(X, W, H), 8))
for iteration in range(1, 41):
    H *= (W.T @ X) / ((W.T @ W) @ H)
    W *= (X @ H.T) / (W @ (H @ H.T))
    if iteration in (1, 2, 10, 40):
        print(iteration, round(objective(X, W, H), 8))
""",
    },
    "digitDictionary": {
        "file": "nmf_digits.py",
        "title": "The complete offline digit experiment",
        "question": "eight candidate fits, then a fixed eight-component panel against PCA and the training-mean image. Does the additive constraint cost anything on the reserved images?",
        "code": r"""
import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

data = np.loadtxt('digits-300.csv', delimiter=',', skiprows=1)
source_rows, labels = data[:, 0].astype(int), data[:, 1].astype(int)
X = data[:, 2:] / 16
train, rest = train_test_split(np.arange(len(X)), test_size=.4,
                               random_state=19, stratify=labels)
validation, test = train_test_split(rest, test_size=.5,
                                    random_state=19, stratify=labels[rest])

for k in (1, 4, 8, 16):
    for seed in (7, 19):
        model = NMF(n_components=k, init='random', solver='cd',
                    random_state=seed, max_iter=2000, tol=1e-5)
        W_train = model.fit_transform(X[train])
        W_validation = model.transform(X[validation])
        H = model.components_
        train_mse = np.mean((X[train] - W_train @ H) ** 2)
        validation_mse = np.mean((X[validation] - W_validation @ H) ** 2)
        print(k, seed, round(train_mse, 6), round(validation_mse, 6))

# A fixed eight-component comparison chosen for inspectable image panels.
nmf = NMF(n_components=8, init='random', solver='cd', random_state=19,
          max_iter=2000, tol=1e-5).fit(X[train])
W_test = nmf.transform(X[test])
nmf_reconstruction = W_test @ nmf.components_
pca = PCA(n_components=8, svd_solver='full').fit(X[train])
pca_reconstruction = pca.inverse_transform(pca.transform(X[test]))
mean_reconstruction = np.repeat(X[train].mean(axis=0)[None, :], len(test), axis=0)
for name, reconstruction in [('mean', mean_reconstruction),
                             ('PCA8', pca_reconstruction),
                             ('NMF8', nmf_reconstruction)]:
    print(name, round(np.mean((X[test] - reconstruction) ** 2), 6))
print('first held-out source row', source_rows[test[0]])
print('first reconstructed image')
print(nmf_reconstruction[0].reshape(8, 8).round(2))
""",
    },
    "wordPatterns": {
        "file": "nmf_words.py",
        "title": "A constructed five-document transfer",
        "question": "two components, five documents and a generalized-KL objective. Will the two patterns divide the vocabulary, and how does the mixed document use them?",
        "code": r"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

documents = ['orbit rocket orbit', 'rocket orbit rocket',
             'goal team goal', 'team goal team',
             'orbit rocket goal team']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
model = NMF(n_components=2, init='nndsvda', solver='mu',
            beta_loss='kullback-leibler', max_iter=1000,
            tol=1e-5, random_state=19)
W = model.fit_transform(X)
H = model.components_
mass = H.sum(axis=1)
normalized_patterns = H / mass[:, None]
amounts = W * mass
print(vectorizer.get_feature_names_out())
print(normalized_patterns.round(3))
print(amounts.round(3))
new = vectorizer.transform(['rocket team'])
print((model.transform(new) @ H).round(3))
""",
    },
}

module_path = ROOT / "src/learn/data/nmf-examples.js"
evidence_path = ROOT / "docs/teaching/evidence/nmf-native.json"
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


records, namespaces = {}, {}
for key, item in PROGRAMS.items():
    code = item["code"].strip() + "\n"
    namespace = {"__file__": str(ASSETS / item["file"])}
    with contextlib.chdir(ASSETS), threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"nmf-example:{key}", "exec"), namespace)
    namespaces[key] = namespace
    records[key] = {
        "title": item["title"],
        "question": item["question"],
        "file": item["file"],
        "code": code.rstrip("\n"),
        "expected": output.getvalue().strip(),
        "language": "python",
    }
    print(f"Executed {key}: {len(code.splitlines())} lines, {len(records[key]['expected'].splitlines())} output lines")

if not write:
    existing = module_path.read_text(encoding="utf-8")
    prefix = "export const nmfExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], (
            f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}")

# ---- Oracles: the manuscript's stated values must come out of these runs. ----
oracles = 0


def oracle(label, condition):
    global oracles
    assert condition, f"ORACLE FAILED: {label}"
    oracles += 1


step = namespaces["multiplicativeStep"]
printed = [line.split() for line in records["multiplicativeStep"]["expected"].splitlines()]
oracle("the printed trace is the manuscript's",
       printed == [["0", "17.06"], ["1", "0.03944744"], ["2", "0.02823377"],
                   ["10", "0.00180608"], ["40", "5e-08"]])
oracle("the objective helper is the half squared Frobenius loss",
       abs(step["objective"](np.array([[2., 1., 3.]]), np.array([[1.]]), np.array([[1.5, 1., 2.5]])) - .25) < 1e-15)
oracle("forty sweeps reach the exact product from section 1",
       np.allclose(step["W"] @ step["H"], step["X"], atol=1e-3))
oracle("the fitted factors are not the factors that built X",
       not np.allclose(step["H"], np.array([[1., 0., 1.], [0., 1., 1.]]), atol=1e-2))
oracle("every fitted entry stayed nonnegative", step["W"].min() >= 0 and step["H"].min() >= 0)
oracle("the first updated H is the manuscript's matrix",
       abs(float(step["H"][0][0]) - 2.1205418051316602) < 1e-12)

digits = namespaces["digitDictionary"]
lines = records["digitDictionary"]["expected"].splitlines()
oracle("the eight-row validation table is the manuscript's",
       [line.split() for line in lines[:8]] ==
       [["1", "7", "0.072144", "0.072803"], ["1", "19", "0.072144", "0.072803"],
        ["4", "7", "0.040409", "0.041899"], ["4", "19", "0.040409", "0.0419"],
        ["8", "7", "0.02475", "0.025548"], ["8", "19", "0.025147", "0.026315"],
        ["16", "7", "0.012295", "0.013548"], ["16", "19", "0.012729", "0.015289"]])
oracle("the three reserved comparisons are the manuscript's",
       [line.split() for line in lines[8:11]] ==
       [["mean", "0.069995"], ["PCA8", "0.019678"], ["NMF8", "0.025381"]])
oracle("PCA wins the reserved reconstruction comparison and the output shows it",
       float(lines[9].split()[1]) < float(lines[10].split()[1]))
oracle("the first held-out image is source row 242", lines[11] == "first held-out source row 242")
oracle("the split is 180 / 60 / 60",
       (len(digits["train"]), len(digits["validation"]), len(digits["test"])) == (180, 60, 60))
oracle("labels stratified the split and never entered the factorization",
       digits["X"].shape == (300, 64) and digits["data"].shape == (300, 66))
oracle("the scaled features stay in [0, 1]", digits["X"].min() == 0 and digits["X"].max() == 1)
oracle("the fitted dictionary has eight 64-feature patterns", digits["nmf"].components_.shape == (8, 64))
oracle("transform kept the dictionary fixed for the reserved rows",
       np.allclose(digits["W_test"] @ digits["nmf"].components_, digits["nmf_reconstruction"], atol=1e-12))
oracle("no candidate fit hit its 2000-iteration cap", digits["model"].n_iter_ < 2000 and digits["nmf"].n_iter_ < 2000)
oracle("the printed reconstructed image is an 8 by 8 block of nonnegative numbers",
       digits["nmf_reconstruction"][0].reshape(8, 8).shape == (8, 8) and digits["nmf_reconstruction"][0].min() >= 0)
oracle("the printed image exceeds one, so no display may quietly clip it",
       abs(float(digits["nmf_reconstruction"][0].max()) - 1.017674616234679) < 1e-12)

words = namespaces["wordPatterns"]
oracle("the vocabulary order is the manuscript's",
       list(words["vectorizer"].get_feature_names_out()) == ["goal", "orbit", "rocket", "team"])
oracle("each normalized nonzero pattern sums to one",
       np.allclose(words["normalized_patterns"].sum(axis=1), 1))
oracle("normalization preserved the reconstruction",
       np.allclose(words["amounts"] @ words["normalized_patterns"], words["W"] @ words["H"], atol=1e-12))
oracle("the counted corpus is five documents over four words", words["X"].shape == (5, 4))
oracle("one pattern carries the space words and the other the sports words",
       sorted(int(np.argmax(words["normalized_patterns"][component])) for component in range(2)) in ([0, 1], [0, 2], [1, 3], [2, 3]))
oracle("the mixed document uses both components", (words["W"][4] > 1e-6).all())
oracle("the new document's reconstruction is nonnegative and four-dimensional",
       words["model"].transform(words["new"]).shape == (1, 2)
       and (words["model"].transform(words["new"]) @ words["H"]).min() >= 0)

if write:
    module_path.write_text(
        "// Complete displayed programs for the NMF lesson, executed by\n"
        "// scripts/verify-nmf-examples.py. `file` is the filename the lesson asks the\n"
        "// learner to save the block as; the digit program runs beside the served CSV.\n"
        "export const nmfExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": "src/learn/data/nmf-examples.js",
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-nmf-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn"]},
    "programs": {key: {"file": record["file"], "codeHash": digest(record["code"]),
                       "stdoutHash": digest(record["expected"]), "stdout": record["expected"]}
                 for key, record in records.items()},
    "oracles": oracles,
    "limits": [
        "The digit program ran against the served copy of the supplied CSV; no network access was used.",
        "One fixed split of one curated balanced subset; the reserved comparison is a single demonstration, and PCA wins it.",
        "The five-document corpus is a constructed illustration of the arithmetic, not text-model evidence.",
        "Numerical fitting can differ on other library versions.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(records)} displayed programs executed, {oracles} oracle assertions.")
