"""Regenerate and verify the NMF lesson's offline data module.

Everything the lesson states about real digits is recomputed here from the CSV
the page serves, with the isolated lesson Python, and compared against the
content packet's recorded `calculated-inputs.json` before anything is written.
The constructed fixtures (the multiplicative trace, the altered step, the loss
comparison, the two exact factorizations, the practice values and the
nonnegative-rank matrix) are recomputed as well, because the browser model has
to reproduce them and a stale recording would hide a drift.

Run:  scratch/lesson-tools/Scripts/python.exe scripts/verify-nmf-data.py
      add --write to publish src/learn/data/nmf-data.js
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/non-negative-matrix-factorization-nmf"
ASSETS = ROOT / "public/learn-assets/nmf"
MODULE = ROOT / "src/learn/data/nmf-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/nmf-data.json"

checks = []


def check(label, condition):
    assert condition, f"FAILED: {label}"
    checks.append(label)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label, tolerance=1e-12):
    check(label, abs(float(actual) - float(expected)) <= tolerance * max(1.0, abs(float(expected))))


saved = json.loads((PACKET / "calculated-inputs.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- the served CSV
csv_hash = digest(ASSETS / "digits-300.csv")
check("served CSV is the byte-identical licensed subset",
      csv_hash == "d93f963c4b2610eb07122a312eec3ddceac18a031477370d71e33835eced728e")
check("packet CSV and served CSV are the same bytes", csv_hash == digest(PACKET / "digits-300.csv"))
raw = np.loadtxt(ASSETS / "digits-300.csv", delimiter=",", skiprows=1)
source_rows, labels, pixels = raw[:, 0].astype(int), raw[:, 1].astype(int), raw[:, 2:]
check("300 rows with an ID, a label and 64 block counts", raw.shape == (300, 66))
check("source rows are unique and ascending", len(set(source_rows)) == 300 and np.all(source_rows[:-1] < source_rows[1:]))
check("thirty examples of every digit", np.array_equal(np.bincount(labels), [30] * 10))
check("block counts are integers from 0 to 16",
      np.array_equal(pixels, pixels.astype(int)) and pixels.min() == 0 and pixels.max() == 16)
native = load_digits()
check("features and labels match the bundled original collection",
      np.array_equal(native.data[source_rows], pixels) and np.array_equal(native.target[source_rows], labels))
selection = np.sort(np.concatenate([np.flatnonzero(native.target == label)[:30] for label in range(10)]))
check("the subset is the first thirty loader rows of each label", np.array_equal(source_rows, selection))

X = pixels / 16
check("scaling by the known maximum keeps every value in [0, 1] and preserves zero",
      X.min() == 0 and X.max() == 1 and np.count_nonzero(X == 0) == np.count_nonzero(pixels == 0))

# ------------------------------------------------------------------- the splits
train, rest = train_test_split(np.arange(len(X)), test_size=.4, random_state=19, stratify=labels)
validation, test = train_test_split(rest, test_size=.5, random_state=19, stratify=labels[rest])
check("the recorded training split is reproduced", train.tolist() == saved["splits"]["train"])
check("the recorded validation split is reproduced", validation.tolist() == saved["splits"]["validation"])
check("the recorded test split is reproduced", test.tolist() == saved["splits"]["test"])
check("180 / 60 / 60 disjoint rows covering the collection once",
      len(train) == 180 and len(validation) == 60 and len(test) == 60
      and len(set(train) | set(validation) | set(test)) == 300)
check("the split is label-stratified, six of each digit in each held-out part",
      np.array_equal(np.bincount(labels[validation]), [6] * 10) and np.array_equal(np.bincount(labels[test]), [6] * 10))
check("the first reserved image is source row 242", int(source_rows[test[0]]) == 242)

# ------------------------------------------------------------- candidate fits
runs = []
dictionary = None
activations = None
for k in (1, 4, 8, 16):
    for seed in (7, 19):
        model = NMF(n_components=k, init="random", solver="cd", random_state=seed, max_iter=2000, tol=1e-5)
        W_train = model.fit_transform(X[train])
        H = model.components_
        W_validation = model.transform(X[validation])
        train_mse = float(np.mean((X[train] - W_train @ H) ** 2))
        validation_mse = float(np.mean((X[validation] - W_validation @ H) ** 2))
        runs.append({"k": k, "seed": seed, "iterations": int(model.n_iter_),
                     "trainMse": train_mse, "validationMse": validation_mse})
        recorded = next(row for row in saved["runs"] if row["k"] == k and row["seed"] == seed)
        close(train_mse, recorded["train_mse"], f"training MSE for k={k}, seed={seed}", 1e-12)
        close(validation_mse, recorded["validation_mse"], f"validation MSE for k={k}, seed={seed}", 1e-12)
        check(f"the k={k}, seed={seed} fit stopped before the 2000-iteration cap", model.n_iter_ < 2000)
        check(f"nonnegative factors for k={k}, seed={seed}", W_train.min() >= 0 and H.min() >= 0)
        if k == 8 and seed == 19:
            dictionary = H
            activations = model.transform(X[test])

check("the displayed validation table rounds to the manuscript's six decimals",
      [(row["k"], row["seed"], round(row["trainMse"], 6), round(row["validationMse"], 6)) for row in runs]
      == [(1, 7, .072144, .072803), (1, 19, .072144, .072803), (4, 7, .040409, .041899), (4, 19, .040409, .0419),
          (8, 7, .02475, .025548), (8, 19, .025147, .026315), (16, 7, .012295, .013548), (16, 19, .012729, .015289)])
check("more components never hurt validation over the inspected range for seed 7",
      [row["validationMse"] for row in runs if row["seed"] == 7]
      == sorted([row["validationMse"] for row in runs if row["seed"] == 7], reverse=True))
check("the two initializations agree at one component and disagree more at sixteen",
      abs(runs[0]["validationMse"] - runs[1]["validationMse"]) < 1e-6
      and abs(runs[6]["validationMse"] - runs[7]["validationMse"]) > 1e-3)
check("the best inspected validation candidate is k=16 with seed 7",
      min(runs, key=lambda row: row["validationMse"])["k"] == 16
      and min(runs, key=lambda row: row["validationMse"])["seed"] == 7)

check("the inspection dictionary is the recorded one", np.allclose(dictionary, np.asarray(saved["visual"]["H"]), atol=1e-12, rtol=0))
check("the reserved activations are the recorded ones", np.allclose(activations, np.asarray(saved["visual"]["W_test"]), atol=1e-12, rtol=0))
reconstruction = activations @ dictionary
check("the recorded reserved reconstruction is exactly the product of the two recorded factors",
      np.allclose(reconstruction, np.asarray(saved["visual"]["reconstructed_test"]), atol=1e-12, rtol=0))
nmf_test_mse = float(np.mean((X[test] - reconstruction) ** 2))
close(nmf_test_mse, saved["visual"]["test_mse"], "NMF8 reserved MSE", 1e-12)

pca = PCA(n_components=8, svd_solver="full").fit(X[train])
pca_test_mse = float(np.mean((X[test] - pca.inverse_transform(pca.transform(X[test]))) ** 2))
mean_image = X[train].mean(axis=0)
mean_test_mse = float(np.mean((X[test] - mean_image) ** 2))
close(pca_test_mse, saved["pca_test_mse"], "PCA8 reserved MSE", 1e-12)
close(mean_test_mse, saved["mean_test_mse"], "training-mean reserved MSE", 1e-12)
check("the manuscript's three reserved comparisons round as printed",
      (round(mean_test_mse, 6), round(pca_test_mse, 6), round(nmf_test_mse, 6)) == (.069995, .019678, .025381))
check("PCA wins this reconstruction comparison and the lesson says so", pca_test_mse < nmf_test_mse)

# ---------------------------------------- the first reserved image, component by component
row = 0
observed = X[test[row]]
coefficients = activations[row]
contributions = coefficients[:, None] * dictionary
check("the contributions add to the reconstruction", np.allclose(contributions.sum(axis=0), reconstruction[row], atol=1e-12, rtol=0))
check("every contribution is nonnegative", contributions.min() >= 0)
base_mse = float(np.mean((observed - reconstruction[row]) ** 2))
close(base_mse, .021503642603, "row 242 reconstruction MSE", 1e-9)
removals = []
recorded_removals = [.024686350768, .068409087403, .037522476594, .022283346218,
                     .030620852640, .067307402976, .023776212335, .021503642603]
for component in range(8):
    without = reconstruction[row] - contributions[component]
    value = float(np.mean((observed - without) ** 2))
    removals.append(value)
    close(value, recorded_removals[component], f"row 242 without component {component + 1}", 1e-9)
check("component 8 has exactly zero activation on this image, so removing it changes nothing",
      coefficients[7] == 0 and removals[7] == base_mse)
close(coefficients[1], .740160683895, "the largest recorded coefficient", 1e-9)
check("no single removal improves the total row error at this row optimum",
      all(value >= base_mse - 1e-8 for value in removals))
check("the all-zero mask leaves the mean squared observation",
      abs(float(np.mean(observed ** 2)) - float(np.mean((observed - np.zeros(64)) ** 2))) < 1e-15)
check("the reserved reconstruction exceeds one, so a display must not clip at one",
      reconstruction.max() > 1 and abs(reconstruction[row].max() - 1.017674616234679) < 1e-12
      and int((reconstruction.max(axis=1) > 1).sum()) == 38)
contribution_totals = contributions.sum(axis=1)
pattern_mass = dictionary.sum(axis=1)
check("the eight patterns carry different total mass, so a raw coefficient is not a contribution",
      pattern_mass.max() / pattern_mass.min() > 1.4)
check("ranking by contribution total agrees with the raw coefficients on source row 242",
      list(np.argsort(-contribution_totals)) == list(np.argsort(-coefficients)))
disagreeing = [index for index in range(60)
               if list(np.argsort(-(activations[index] * pattern_mass))) != list(np.argsort(-activations[index]))]
check("but the two rankings disagree on most reserved images, including the second one",
      1 in disagreeing and len(disagreeing) > 30)

# ---------------------------------------------- constructed fixtures: exact products
X3 = np.array([[2., 1., 3.], [1., 2., 3.], [3., 3., 6.]])
W1 = np.array([[2., 1.], [1., 2.], [3., 3.]])
H1 = np.array([[1., 0., 1.], [0., 1., 1.]])
W2 = np.array([[1.5, .5], [.5, 1.5], [2., 2.]])
H2 = np.array([[1.25, .25, 1.5], [.25, 1.25, 1.5]])
check("the first factorization is exact", np.array_equal(W1 @ H1, X3))
check("the second factorization is exact", np.array_equal(W2 @ H2, X3))
check("the recorded ambiguity products are both X",
      saved["ambiguity_products"] == [X3.tolist(), X3.tolist()])
check("the second dictionary is not the first reordered or rescaled",
      not any(np.allclose(H2[r] / H2[r].sum(), H1[s] / H1[s].sum()) for r in range(2) for s in range(2)))
check("feature three is the sum of the first two throughout the fixture",
      np.allclose(X3[:, 2], X3[:, 0] + X3[:, 1]))
sums = H1.sum(axis=1)
normalized = H1 / sums[:, None]
rescaled = W1 * sums
check("normalization preserves the product", np.allclose(rescaled @ normalized, X3))
check("the normalized patterns and activations are the stated ones",
      np.array_equal(normalized[0], [.5, 0, .5]) and np.array_equal(normalized[1], [0, .5, .5])
      and np.array_equal(rescaled[0], [4, 2]) and rescaled[0].sum() == 6)

# ------------------------------------------------ constructed fixtures: the update trace
def loss(data, W, H):
    return float(np.sum((data - W @ H) ** 2) / 2)


def update(data, W, H):
    H = H * (W.T @ data) / ((W.T @ W) @ H)
    W = W * (data @ H.T) / (W @ (H @ H.T))
    return W, H


W0 = np.array([[1., .5], [.5, 1.], [1., 1.]])
H0 = np.array([[1., .2, .8], [.2, 1., .8]])
W, H = W0.copy(), H0.copy()
trace = [{"iteration": 0, "W": W.tolist(), "H": H.tolist(), "loss": loss(X3, W, H)}]
for step in range(1, 41):
    W, H = update(X3, W, H)
    trace.append({"iteration": step, "W": W.tolist(), "H": H.tolist(), "loss": loss(X3, W, H)})
for step in range(41):
    close(trace[step]["loss"], saved["trace"][step]["loss"], f"recorded loss at sweep {step}", 1e-12)
    check(f"recorded factors at sweep {step}",
          np.allclose(trace[step]["W"], saved["trace"][step]["W"], atol=1e-12, rtol=0)
          and np.allclose(trace[step]["H"], saved["trace"][step]["H"], atol=1e-12, rtol=0))
check("the trace never increases the objective",
      all(trace[step + 1]["loss"] <= trace[step]["loss"] + 1e-15 for step in range(40)))
check("the printed trace values match the manuscript",
      [round(trace[step]["loss"], 8) for step in (0, 1, 2, 10, 40)] == [17.06, 0.03944744, 0.02823377, 0.00180608, 5e-08])
numerator = float((W0.T @ X3)[0, 0])
denominator = float(((W0.T @ W0) @ H0)[0, 0])
close(numerator, 5.5, "the H11 numerator")
close(denominator, 2.65, "the H11 denominator")
close(H0[0, 0] * numerator / denominator, 2.075471698113208, "the updated H11")
close(trace[1]["H"][0][0], 2.075471698113208, "the first sweep's H11", 1e-12)

altered = X3.copy()
altered[0, 1] = 2
alteredW, alteredH = update(altered, W0.copy(), H0.copy())
check("the altered-input step matches the recording",
      np.allclose(alteredW, saved["altered_one_step"]["W"], atol=1e-12, rtol=0)
      and np.allclose(alteredH, saved["altered_one_step"]["H"], atol=1e-12, rtol=0))
close(loss(altered, alteredW, alteredH), saved["altered_one_step"]["loss"], "the altered-input sweep loss", 1e-12)
close(alteredH[0, 1], .489795918367347, "H[0,1] after the altered step", 1e-12)
close(alteredH[1, 1], 2.2641509433962264, "H[1,1] after the altered step", 1e-12)
close(loss(altered, alteredW, alteredH), .224942552, "the altered sweep loss to the recorded digits", 1e-8)

bigger = H0.copy()
bigger[0, 0] = 3.
big_numerator = float((W0.T @ X3)[0, 0])
big_denominator = float(((W0.T @ W0) @ bigger)[0, 0])
close(big_denominator, 7.15, "the denominator when H11 starts at 3")
close(bigger[0, 0] * big_numerator / big_denominator, 30 / 13, "H11 shrinks from 3 to 30/13", 1e-12)
check("the contrast the investigation needs: the default H11 grows and the edited one shrinks",
      H0[0, 0] * numerator / denominator > H0[0, 0] and bigger[0, 0] * big_numerator / big_denominator < bigger[0, 0])

exactW, exactH = update(X3, W1.copy(), H1.copy())
check("the exact-fit null leaves both factors where they were",
      np.allclose(exactW, W1, atol=1e-12, rtol=0) and np.allclose(exactH, H1, atol=1e-12, rtol=0))
close(loss(X3, exactW, exactH), 0, "the exact-fit null keeps zero loss", 1e-24)
check("the zero pattern entries stay at zero in the exact-fit null", exactH[0, 1] == 0 and exactH[1, 0] == 0)

# ------------------------------------------------- constructed fixtures: losses and gradients
comparison = []
for x, y in ((2., 4.), (20., 22.)):
    frobenius = .5 * (x - y) ** 2
    kl = x * np.log(x / y) - x + y
    itakura = x / y - np.log(x / y) - 1
    comparison.append({"observed": x, "reconstructed": y, "frobeniusHalf": float(frobenius),
                       "kl": float(kl), "itakuraSaito": float(itakura)})
    recorded = saved["loss_comparison"][str(int(x))]
    close(frobenius, recorded["frobenius_half"], f"half squared error at {x}")
    close(kl, recorded["kl"], f"generalized KL at {x}")
    close(itakura, recorded["is"], f"Itakura-Saito at {x}")
check("the loss table rounds to the manuscript's printed digits",
      [(round(row["kl"], 6), round(row["itakuraSaito"], 6)) for row in comparison]
      == [(.613706, .193147), (.093796, .004401)])
for c in (2., 3., 7.5):
    close(.5 * (c * 2 - c * 4) ** 2, c * c * 2., f"squared loss scales by c squared at c={c}", 1e-12)
    close(c * 2 * np.log(c * 2 / (c * 4)) - c * 2 + c * 4, c * comparison[0]["kl"], f"KL scales by c at c={c}", 1e-12)
    close(c * 2 / (c * 4) - np.log(c * 2 / (c * 4)) - 1, comparison[0]["itakuraSaito"], f"IS is unchanged at c={c}", 1e-12)

zeroW = np.array([[1.]])
zeroX = np.array([[2., 1.]])
zeroH = np.array([[0., 1.]])
zero_gradient = (zeroW.T @ zeroW) @ zeroH - zeroW.T @ zeroX
check("the zero-locked coordinate has gradient minus two", np.array_equal(zero_gradient, [[-2., 0.]]))
guarded = np.where(zeroH > 0, zeroH * np.divide(zeroW.T @ zeroX, (zeroW.T @ zeroW) @ zeroH,
                                                out=np.zeros_like(zeroH), where=((zeroW.T @ zeroW) @ zeroH) > 0), 0.)
check("a guarded multiplicative rule leaves the zero at zero", guarded[0, 0] == 0 and guarded[0, 1] == 1)
check("nonnegative least squares with the same fixed W recovers [2, 1]",
      np.allclose(np.linalg.lstsq(zeroW, zeroX, rcond=None)[0], [[2., 1.]]))

practice_gradient = (np.array([[1.]]).T @ np.array([[1.]])) @ np.array([[0., 2.]]) - np.array([[1.]]).T @ np.array([[3., 1.]])
check("practice 8's gradient is [-3, 1]", np.array_equal(practice_gradient, [[-3., 1.]]))
close(.5 * (1 - 1.5 * .75) ** 2, .0078125, "the joint nonconvexity midpoint loss", 1e-15)
close(.5 * (2 - 2.5 * 1.25) ** 2, .6328125, "practice 6's midpoint loss", 1e-15)
practice_one = np.array([1., 3.]) @ np.array([[2., 0., 1.], [0., 1., 2.]])
check("practice 1 reconstructs [2, 3, 7]", np.array_equal(practice_one, [2., 3., 7.]))
practice_two_numerator = np.array([[1.], [2.]]).T @ np.array([[2., 1.], [4., 3.]])
practice_two_denominator = (np.array([[1.], [2.]]).T @ np.array([[1.], [2.]])) @ np.array([[1., 1.]])
practice_two = np.array([[1., 1.]]) * practice_two_numerator / practice_two_denominator
check("practice 2 gives H = [2, 1.4]", np.allclose(practice_two, [[2., 1.4]]))
close(loss(np.array([[2., 1.], [4., 3.]]), np.array([[1.], [2.]]), practice_two), .1, "practice 2's half-squared loss", 1e-12)

# ------------------------------------- constructed fixtures: nonnegative rank
S = np.array([[0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]])
check("the support matrix has ordinary rank three", np.linalg.matrix_rank(S) == 3)
check("rows one and three sum to rows two and four", np.array_equal(S[0] + S[2], S[1] + S[3]))
check("the first three rows are independent", np.linalg.matrix_rank(S[:3]) == 3)
marked = [(0, 2), (1, 3), (2, 0), (3, 1)]
pairs = []
for a in range(4):
    for b in range(a + 1, 4):
        (r1, c1), (r2, c2) = marked[a], marked[b]
        crossed = [(r, c) for r, c in ((r1, c2), (r2, c1)) if S[r, c] == 0]
        check(f"the rectangle spanning {'ABCD'[a]} and {'ABCD'[b]} crosses a zero", len(crossed) > 0)
        pairs.append({"pair": ["ABCD"[a], "ABCD"[b]], "cells": [[r1, c1], [r2, c2]],
                      "crossed": [[r + 1, c + 1] for r, c in crossed]})
check("all six pairs were inspected", len(pairs) == 6)
check("the identity times S is an exact four-component nonnegative factorization",
      np.array_equal(np.eye(4) @ S, S))

# ------------------------------------------- constructed fixtures: words and spectra
spectrum = .3 * np.array([.2, .6, .4]) + .7 * np.array([.8, .3, .1])
check("the three-band mixture is [.62, .39, .19]", np.allclose(spectrum, [.62, .39, .19]))
words = np.array([2., 1.]) @ np.array([[3., 2., 0., 0.], [0., 0., 1., 4.]])
check("the word example reconstructs [6, 4, 1, 4]", np.array_equal(words, [6., 4., 1., 4.]))
check("exact rational check of the normalized mixture proportions",
      [Fraction(2, 3), Fraction(1, 3)] == [Fraction(4, 6), Fraction(2, 6)])

# --------------------------------------------------------------------- the module
module_data = {
    "datasetSha256": csv_hash,
    "license": "CC BY 4.0",
    "attribution": "E. Alpaydin and C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository",
    "provenance": "/learn-assets/nmf/data-provenance.md",
    "csv": "/learn-assets/nmf/digits-300.csv",
    "scale": 16,
    "side": 8,
    "splitSizes": {"train": 180, "validation": 60, "test": 60},
    "fit": {"k": 8, "seed": 19, "init": "random", "solver": "cd", "maxIter": 2000, "tol": 1e-5},
    "dictionary": [[float(value) for value in pattern] for pattern in dictionary],
    "activations": [[float(value) for value in coefficient] for coefficient in activations],
    "test": [{"sourceRow": int(source_rows[index]), "digit": int(labels[index]),
              "pixels": [int(value) for value in pixels[index]]} for index in test],
    "runs": runs,
    "baselines": {"meanTestMse": mean_test_mse, "pcaTestMse": pca_test_mse, "nmfTestMse": nmf_test_mse},
    "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
}
module_fixtures = {
    "lossComparison": comparison,
    "supportPairs": pairs,
    "recordedTraceLosses": [saved["trace"][step]["loss"] for step in range(41)],
    "alteredStep": {"H": alteredH.tolist(), "W": alteredW.tolist(), "loss": loss(altered, alteredW, alteredH)},
    "removals": {"sourceRow": 242, "baseMse": base_mse, "byComponent": removals,
                 "coefficients": [float(value) for value in coefficients],
                 "contributionTotals": [float(value) for value in contribution_totals]},
}
header = (
    "/** Real digit data and recorded native probes for the NMF lesson.\n"
    " * Generated by scripts/verify-nmf-data.py; do not edit by hand.\n"
    " * Original data: E. Alpaydin and C. Kaynak (1998), UCI Optical Recognition of Handwritten\n"
    " * Digits, CC BY 4.0. Selection, split, licence and stated limits:\n"
    " * /learn-assets/nmf/data-provenance.md. Only the 60 reserved images, the fitted eight-row\n"
    " * dictionary and their activations are published here; the full 300-row CSV is served for\n"
    " * download. Labels never enter the factorization.\n"
    " */\n"
)
module_text = header
module_text += "export const NMF_DIGITS = " + json.dumps(module_data, ensure_ascii=False, separators=(",", ":")) + ";\n"
module_text += "export const NMF_RECORDED = " + json.dumps(module_fixtures, ensure_ascii=False, separators=(",", ":")) + ";\n"

if "--write" in sys.argv:
    MODULE.write_text(module_text, encoding="utf-8", newline="\n")
else:
    check("the published data module is current", MODULE.exists() and MODULE.read_text(encoding="utf-8") == module_text)

evidence = {
    "status": "passed",
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "command": "scratch/lesson-tools/Scripts/python.exe scripts/verify-nmf-data.py" + (" --write" if "--write" in sys.argv else ""),
    "stage": "author regeneration of the real-data module and every recorded fixture; browser, independent and integration review are separate",
    "versions": module_data["versions"],
    "observations": 300,
    "features": 64,
    "candidateFits": len(runs),
    "checks": checks,
    "checkCount": len(checks),
    "reservedComparison": {"mean": mean_test_mse, "pca8": pca_test_mse, "nmf8": nmf_test_mse,
                           "interpretation": "PCA reconstructs the reserved images better; the example is not tuned until NMF wins."},
    "sourceHashes": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in
                     [Path(__file__), MODULE, PACKET / "calculated-inputs.json", PACKET / "digits-300.csv",
                      ASSETS / "digits-300.csv", ASSETS / "data-provenance.md"]},
    "limits": [
        "One fixed split of one curated balanced subset; writer identities are unavailable, so this is a withheld-image evaluation.",
        "Random-seed repeatability is tied to the recorded library versions.",
        "No elapsed time, classification accuracy or optimal component count is claimed.",
    ],
}
EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(checks)} checks; {len(runs)} candidate fits recomputed; "
      f"mean {mean_test_mse:.6f}, PCA8 {pca_test_mse:.6f}, NMF8 {nmf_test_mse:.6f}.")
