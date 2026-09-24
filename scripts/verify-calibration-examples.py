"""Assemble, execute and independently check every program the calibration lesson shows.

Nothing here is transcribed. Each displayed program is built from functions
copied VERBATIM out of the frozen content packet's `calibration_calculations.py`
by walking its abstract syntax tree and slicing the exact source lines, plus a
short driver that lives in this file. The packet file is pinned by SHA-256 and
each extracted function is pinned by the SHA-256 of its own source segment, so a
copy-and-paste slip, an edited packet or a hand-edited asset all fail here rather
than drifting quietly away from the page.

Five programs:

  reliability_bins  the binning and ECE mechanism of section 2.
  monotone_map      pool-adjacent-violators and its weighted tie grouping, section 3.
  conformal_rank    the exact finite rank, its threshold and the rotation, section 5.
  calculations      the complete `calibration_calculations.py`, served as a download.
  experiments       the complete `uncertainty_experiments.py`, served as a download.

The two complete programs are run in a scratch workspace beside copies of the
two served CSV files. What is checked for them is the strongest available
statement: they must regenerate the packet's frozen `checked-results.json` and
`experiment-results.json` BYTE FOR BYTE. Nothing inside the packet directory is
written at any point.

Every printed number is then checked against an oracle computed here by a
different route. The conformal rank in particular is re-derived by scanning the
integers with exact rational arithmetic and asking for the smallest j whose
j/(n+1) reaches the requested coverage — it never calls the page's or the
packet's quantile helper. The monotone fit is recomputed by the max-min formula
over lower and upper sets, which shares no code with the stack algorithm.

`--write` regenerates the served .py assets and src/learn/data/calibration-examples.js
from what actually ran. Without it, both must already match a fresh execution.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-calibration-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-calibration-examples.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/calibration-conformal-prediction"
CALCULATIONS = PACKET / "calibration_calculations.py"
EXPERIMENTS = PACKET / "uncertainty_experiments.py"
CHECKED = PACKET / "checked-results.json"
RECORDED = PACKET / "experiment-results.json"
ASSETS = ROOT / "public/learn-assets/calibration"
MODULE = ROOT / "src/learn/data/calibration-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/calibration-native.json"
WORKSPACE = ROOT / "scratch/calibration-programs"
PYTHON = sys.executable

PACKET_CALCULATIONS_SHA = "c2884d2a47fa601b5f1b8aeee35afe4ad6074490cf9b62c71903d6d081ea21a2"
PACKET_EXPERIMENTS_SHA = "92463e8a10ed7e338fb3c8af411a4ca5d76fabb020b84fb7b0c941ab6d7c0e19"

write = "--write" in sys.argv
oracle_count = 0
failures: list[str] = []


def oracle(condition, label):
    global oracle_count
    oracle_count += 1
    if not condition:
        failures.append(label)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------- extraction

def function_sources(path: Path, names: list[str]) -> dict[str, str]:
    """Exact source lines of each named top-level function, docstring included."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = "".join(lines[node.lineno - 1:node.end_lineno])
    missing = [name for name in names if name not in found]
    if missing:
        raise SystemExit(f"{path.name} no longer defines {', '.join(missing)}")
    return found


RELIABILITY_DRIVER = '''
if __name__ == "__main__":
    forecasts = [.2] * 5 + [.8] * 5
    outcomes = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    repaired = [.4] * 5 + [.6] * 5
    settings = [
        ("two bins", forecasts, [0, .5, 1]),
        ("one bin", forecasts, [0, 1]),
        ("two bins, forecasts replaced by the observed fractions", repaired, [0, .5, 1]),
    ]
    for name, probabilities, edges in settings:
        report = reliability(probabilities, outcomes, edges)
        print(f"{name}: ECE {report['ece']:.6f} over {report['count']} cards")
        for row in report["bins"]:
            if row["count"]:
                print(f"    bin {row['bin']}: {row['count']} cards, {row['positive']} positive,"
                      f" mean forecast {row['mean_p']:.3f}, observed fraction {row['fraction_positive']:.3f}")
            else:
                print(f"    bin {row['bin']}: empty, so it has no observed fraction")
    boundary = reliability([0, .5, 1], [0, 1, 1], [0, .5, 1])
    counts = [row["count"] for row in boundary["bins"]]
    print(f"boundary fixture: counts {counts}, ECE {boundary['ece']:.6f}")
'''

MONOTONE_DRIVER = '''
if __name__ == "__main__":
    scores = [-3, -2, -1, 0, 1, 2, 3, 4]
    labels = [0, 1, 0, 0, 1, 0, 1, 1]
    fit = fit_pav(scores, labels)
    print("scores", fit["knots"])
    print("labels", [float(label) for label in labels])
    print("fitted", [round(value, 6) for value in fit["fitted"]])
    for step, merge in enumerate(fit["merges"], start=1):
        left, right, merged = merge["left"], merge["right"], merge["merged"]
        print(f"merge {step}: blocks {left[0]}-{left[1]} at {left[2] / left[3]:.6f}"
              f" and {right[0]}-{right[1]} at {right[2] / right[3]:.6f}"
              f" pool to {merged[2] / merged[3]:.6f} over {merged[3]} observations")
    tied = fit_pav([-2, -2, 0, 1], [1, 0, 0, 1])
    print("tied scores", tied["knots"])
    print("tied fitted", [round(value, 6) for value in tied["fitted"]])
    print(f"count weighted {tied['fitted'][0]:.6f}, not the average of block means {(0.5 + 0.0) / 2:.6f}")
'''

RANK_DRIVER = '''
if __name__ == "__main__":
    calibration = [.05, .10, .15, .20, .25, .30, .40, .60, .90]
    n = len(calibration)
    for alpha in (.2, .05):
        k = conformal_rank(n, alpha)
        q = conformal_threshold(calibration, alpha)
        print(f"alpha {alpha}: n {n}, k = ceil({n + 1} * {1 - alpha:.2f}) = {k}, q = {q}")
    q = conformal_threshold(calibration, .2)
    print(f"linear quantile at 8/9 {np.quantile(calibration, 8 / 9):.6f}")
    print(f"higher quantile at 8/9 {np.quantile(calibration, 8 / 9, method='higher'):.6f}")
    print(f"eighth order statistic {q:.6f}")
    for probabilities in ([.80, .15, .05], [.45, .40, .15], [.34, .33, .33]):
        included = class_sets(probabilities, q)
        names = [name for name, keep in zip("ABC", included) if keep]
        shown = "{" + ", ".join(names) + "}" if names else "the empty set"
        print("probabilities", probabilities, "scores",
              [round(1 - value, 2) for value in probabilities], "set", shown)
    rotation = rank_rotation(calibration + [.95], .2)
    print(f"rotation: {rotation['covered']} of {rotation['total']} held-out scores covered")
    tied = rank_rotation([.2] * 10, .2)
    print(f"every score tied at .2: {tied['covered']} of {tied['total']} covered")
'''

PROGRAMS = [
    {
        "key": "reliability",
        "file": "reliability_bins.py",
        "title": "Bin the forecasts, then make the aggregate error disappear",
        "question": "the same ten cards under three binnings. Which of the three reports zero error, and has the "
                    "model changed between them?",
        "docstring": "Reliability bins and the binned ECE summary, from section 2.\n\n"
                     "The `reliability` function below is copied verbatim from the lesson's complete program,\n"
                     "calibration_calculations.py. Its guard block is that program's own: an internal boundary\n"
                     "belongs to the bin on its right and a forecast of exactly 1 belongs to the last bin, and\n"
                     "an edge list that does not span [0, 1] would silently drop observations.\n\n"
                     "Needs numpy. Run: python reliability_bins.py",
        "imports": "import numpy as np\n",
        "functions": ["reliability"],
        "driver": RELIABILITY_DRIVER,
    },
    {
        "key": "monotone",
        "file": "monotone_map.py",
        "title": "Pool the adjacent violators, with the counts attached",
        "question": "eight scores whose labels are not monotone, and four scores of which two are equal. Where "
                    "does the first required merge happen, and what does it pool to?",
        "docstring": "Isotonic calibration by pool-adjacent-violators, from section 3.\n\n"
                     "`fit_pav` is copied verbatim from calibration_calculations.py. Equal scores are grouped\n"
                     "before any merging, and a merge takes the count-weighted mean of two blocks rather than\n"
                     "the average of their two means. The final line prints both so the difference is visible.\n"
                     "The function checks its own result against scikit-learn's independent implementation.\n\n"
                     "Needs numpy and scikit-learn. Run: python monotone_map.py",
        "imports": "import numpy as np\nfrom sklearn.isotonic import IsotonicRegression\n",
        "functions": ["fit_pav"],
        "driver": MONOTONE_DRIVER,
    },
    {
        "key": "rank",
        "file": "conformal_rank.py",
        "title": "The exact finite rank, and the two quantiles it is not",
        "question": "nine calibration scores and a target of 80%. Which score becomes the threshold, and what "
                    "happens when the target is 95% instead?",
        "docstring": "The split-conformal rank, threshold, prediction sets and rotation, from section 5.\n\n"
                     "All four functions are copied verbatim from calibration_calculations.py. The rank uses\n"
                     "exact rational arithmetic on alpha and selects an indexed order statistic; it is not a\n"
                     "percentile call, and the two NumPy quantile conventions printed below return two other\n"
                     "numbers on the same data. When the requested rank exceeds the calibration set, the\n"
                     "threshold is infinity, which means every candidate answer is included.\n\n"
                     "Needs numpy. Run: python conformal_rank.py",
        "imports": "import math\nfrom fractions import Fraction\n\nimport numpy as np\n",
        "functions": ["conformal_rank", "conformal_threshold", "class_sets", "rank_rotation"],
        "driver": RANK_DRIVER,
    },
]

DOWNLOADS = [
    {
        "key": "calculations",
        "file": "calibration_calculations.py",
        "source": CALCULATIONS,
        "sha256": PACKET_CALCULATIONS_SHA,
        "title": "Every constructed mechanism in one file",
        "question": "the reliability bins, the sigmoid and monotone fits, the temperature search, the exact "
                    "rank and every invariance check. Does it still produce exactly the recorded results?",
        "writes": "checked-results.json",
    },
    {
        "key": "experiments",
        "file": "uncertainty_experiments.py",
        "source": EXPERIMENTS,
        "sha256": PACKET_EXPERIMENTS_SHA,
        "title": "The two offline experiments, with their four label roles",
        "question": "a banknote classifier and an airfoil regressor, each with its data roles fixed before any "
                    "comparison. Which method wins on coverage, and does it also win on usefulness?",
        "writes": "experiment-results.json",
    },
]


# -------------------------------------------------------------------- oracles

def independent_rank(n: int, alpha) -> int:
    """The smallest rank whose share of n+1 places reaches the requested coverage.

    This is the rank argument itself, scanned in exact rational arithmetic: it
    never forms (n+1)*(1-alpha), never calls ceil and never calls a quantile.
    """
    target = 1 - Fraction(str(alpha))
    for j in range(1, n + 2):
        if Fraction(j, n + 1) >= target:
            return j
    raise AssertionError("no rank reaches the requested coverage")


def independent_isotonic(scores, labels):
    """Isotonic regression by the max-min formula over lower and upper sets.

    ghat(i) = max over u <= i of min over v >= i of the mean of y[u..v], applied
    to the score-grouped, count-weighted blocks. This shares no code with the
    stack algorithm the packet uses.
    """
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    knots: list[float] = []
    totals: list[float] = []
    weights: list[int] = []
    for index in order:
        if knots and knots[-1] == scores[index]:
            totals[-1] += labels[index]
            weights[-1] += 1
        else:
            knots.append(scores[index])
            totals.append(float(labels[index]))
            weights.append(1)
    size = len(knots)
    fitted = []
    for i in range(size):
        best = None
        for u in range(0, i + 1):
            smallest = None
            for v in range(i, size):
                total = sum(totals[u:v + 1])
                weight = sum(weights[u:v + 1])
                mean = total / weight
                smallest = mean if smallest is None else min(smallest, mean)
            best = smallest if best is None else max(best, smallest)
        fitted.append(best)
    return knots, fitted


def independent_reliability(probabilities, outcomes, edges):
    """Bin, count and average with plain Python loops and no array library."""
    rows = []
    for index in range(len(edges) - 1):
        members = []
        for position, value in enumerate(probabilities):
            above = next((slot for slot, edge in enumerate(edges) if edge > value), len(edges))
            chosen = min(above - 1, len(edges) - 2)
            if chosen == index:
                members.append(position)
        if not members:
            rows.append({"count": 0, "mean_p": None, "fraction_positive": None})
            continue
        rows.append({
            "count": len(members),
            "positive": sum(outcomes[position] for position in members),
            "mean_p": sum(probabilities[position] for position in members) / len(members),
            "fraction_positive": sum(outcomes[position] for position in members) / len(members),
        })
    ece = sum(row["count"] * abs(row["mean_p"] - row["fraction_positive"])
              for row in rows if row["count"]) / len(probabilities)
    return rows, ece


def independent_rotation(scores, alpha):
    """Cover the held-out score exactly when its combined rank is at most k.

    The proof's own statement, counted directly, rather than by rebuilding a
    threshold from the other n scores.
    """
    k = independent_rank(len(scores) - 1, alpha)
    covered = 0
    for index, held in enumerate(scores):
        others = [value for slot, value in enumerate(scores) if slot != index]
        strictly_below = sum(1 for value in others if value < held)
        if strictly_below + 1 <= k:
            covered += 1
    return covered, k


# -------------------------------------------------------------------- assembly

def assemble(program, sources):
    parts = ['"""' + program["docstring"] + '\n"""\n', "from __future__ import annotations\n\n",
             program["imports"]]
    for name in program["functions"]:
        parts.append("\n\n")
        parts.append(sources[name])
    parts.append("\n")
    parts.append(program["driver"])
    return "".join(parts)


def run(path: Path, cwd: Path) -> str:
    finished = subprocess.run([PYTHON, str(path)], cwd=str(cwd), capture_output=True, text=True)
    if finished.returncode != 0:
        raise SystemExit(f"{path.name} exited {finished.returncode}:\n{finished.stdout}\n{finished.stderr}")
    return finished.stdout.rstrip("\n")


def main():
    calculations_bytes = CALCULATIONS.read_bytes()
    experiments_bytes = EXPERIMENTS.read_bytes()
    oracle(sha(calculations_bytes) == PACKET_CALCULATIONS_SHA,
           "the packet's calibration_calculations.py is not the pinned file")
    oracle(sha(experiments_bytes) == PACKET_EXPERIMENTS_SHA,
           "the packet's uncertainty_experiments.py is not the pinned file")

    wanted = sorted({name for program in PROGRAMS for name in program["functions"]})
    sources = function_sources(CALCULATIONS, wanted)
    segment_hashes = {name: sha(text.encode("utf-8")) for name, text in sources.items()}

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for name in ("banknote-subset.csv", "airfoil-subset.csv"):
        oracle((ASSETS / name).read_bytes() == (PACKET / name).read_bytes(),
               f"the served {name} is not byte-identical to the packet's")
        shutil.copyfile(ASSETS / name, WORKSPACE / name)

    records: dict[str, dict] = {}

    # --- the three displayed programs -------------------------------------
    for program in PROGRAMS:
        text = assemble(program, sources)
        served = ASSETS / program["file"]
        if write:
            served.write_text(text, encoding="utf-8", newline="\n")
        oracle(served.exists(), f"{program['file']} is not served")
        if served.exists():
            oracle(served.read_text(encoding="utf-8") == text,
                   f"{program['file']} differs from a fresh assembly out of the packet")
        for name in program["functions"]:
            oracle(sources[name] in text, f"{program['file']} lost the verbatim body of {name}")
        shutil.copyfile(served, WORKSPACE / program["file"])
        output = run(WORKSPACE / program["file"], WORKSPACE)
        records[program["key"]] = {
            "title": program["title"], "question": program["question"], "code": text,
            "language": "python", "file": program["file"], "executed": True,
            "expected": output,
            "verbatimFunctions": program["functions"],
            "functionSha256": {name: segment_hashes[name] for name in program["functions"]},
        }

    # --- the two complete programs ----------------------------------------
    for download in DOWNLOADS:
        served = ASSETS / download["file"]
        oracle(served.exists(), f"{download['file']} is not served")
        oracle(sha(served.read_bytes()) == download["sha256"],
               f"the served {download['file']} is not the pinned packet file")
        shutil.copyfile(served, WORKSPACE / download["file"])
    calculations_output = run(WORKSPACE / "calibration_calculations.py", WORKSPACE)
    experiments_output = run(WORKSPACE / "uncertainty_experiments.py", WORKSPACE)
    regenerated_checked = (WORKSPACE / "checked-results.json").read_bytes()
    regenerated_recorded = (WORKSPACE / "experiment-results.json").read_bytes()
    oracle(regenerated_checked == CHECKED.read_bytes(),
           "calibration_calculations.py no longer regenerates checked-results.json byte for byte")
    oracle(regenerated_recorded == RECORDED.read_bytes(),
           "uncertainty_experiments.py no longer regenerates experiment-results.json byte for byte")
    records["calculations"] = {
        "title": DOWNLOADS[0]["title"], "question": DOWNLOADS[0]["question"],
        "code": CALCULATIONS.read_text(encoding="utf-8"), "language": "python",
        "file": DOWNLOADS[0]["file"], "executed": True, "downloadOnly": True,
        "expected": calculations_output,
    }
    records["experiments"] = {
        "title": DOWNLOADS[1]["title"], "question": DOWNLOADS[1]["question"],
        "code": EXPERIMENTS.read_text(encoding="utf-8"), "language": "python",
        "file": DOWNLOADS[1]["file"], "executed": True, "downloadOnly": True,
        "expected": experiments_output,
    }

    # --- oracles over what was printed ------------------------------------
    checked = json.loads(CHECKED.read_text(encoding="utf-8"))
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))

    reliability_out = records["reliability"]["expected"]
    forecasts = [.2] * 5 + [.8] * 5
    outcomes = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    repaired = [.4] * 5 + [.6] * 5
    _, two_ece = independent_reliability(forecasts, outcomes, [0, .5, 1])
    _, one_ece = independent_reliability(forecasts, outcomes, [0, 1])
    _, repaired_ece = independent_reliability(repaired, outcomes, [0, .5, 1])
    oracle(abs(two_ece - checked["reliability"]["two_bins"]["ece"]) < 1e-12,
           "the independent two-bin ECE disagrees with the packet")
    oracle(f"ECE {two_ece:.6f}" in reliability_out, "the two-bin ECE is not printed as the oracle computes it")
    oracle(f"ECE {one_ece:.6f}" in reliability_out, "the one-bin ECE is not printed as the oracle computes it")
    oracle(f"ECE {repaired_ece:.6f}" in reliability_out, "the repaired ECE is not printed as the oracle computes it")
    oracle(abs(one_ece) < 1e-15 and abs(repaired_ece) < 1e-15,
           "merging the bins and repairing the forecasts must both reach exactly zero binned ECE")
    oracle(two_ece > 0.19, "the two-bin ECE oracle has lost the discrepancy it is meant to show")
    oracle("counts [1, 2]" in reliability_out,
           "the boundary fixture must keep its p=1 observation in the last bin")
    boundary_rows, boundary_ece = independent_reliability([0, .5, 1], [0, 1, 1], [0, .5, 1])
    oracle([row["count"] for row in boundary_rows] == [1, 2],
           "the oracle disagrees that p=.5 goes right and p=1 stays in the last bin")
    oracle(abs(boundary_ece - checked["reliability"]["boundary_fixture"]["ece"]) < 1e-12,
           "the independent boundary ECE disagrees with the packet")

    monotone_out = records["monotone"]["expected"]
    knots, fitted = independent_isotonic([-3, -2, -1, 0, 1, 2, 3, 4], [0, 1, 0, 0, 1, 0, 1, 1])
    oracle(knots == checked["pav"]["knots"], "the independent isotonic knots disagree with the packet")
    oracle(all(abs(a - b) < 1e-12 for a, b in zip(fitted, checked["pav"]["fitted"])),
           "the max-min isotonic solution disagrees with the packet's stack algorithm")
    oracle(str([round(value, 6) for value in fitted]) in monotone_out,
           "the printed fitted values are not the ones the max-min oracle computes")
    tied_knots, tied_fitted = independent_isotonic([-2, -2, 0, 1], [1, 0, 0, 1])
    oracle(all(abs(a - b) < 1e-12 for a, b in zip(tied_fitted, checked["pav_ties"]["fitted"])),
           "the tied-score isotonic oracle disagrees with the packet")
    oracle(abs(tied_fitted[0] - 1 / 3) < 1e-12,
           "the tied fixture must pool to one third, not to one quarter")
    oracle("count weighted 0.333333, not the average of block means 0.250000" in monotone_out,
           "the program no longer contrasts the count-weighted pool with the average of block means")
    oracle(f"merge 1:" in monotone_out and monotone_out.count("merge ") == len(checked["pav"]["merges"]),
           "the printed merge trace has a different number of steps from the packet's")

    rank_out = records["rank"]["expected"]
    oracle(independent_rank(9, .2) == checked["rank"]["k"],
           "the independently scanned rank disagrees with the packet at alpha .2")
    oracle(independent_rank(9, .05) == checked["rank"]["tiny_alpha_rank"],
           "the independently scanned rank disagrees with the packet at alpha .05")
    oracle(independent_rank(9, .05) == 10 and "q = inf" in rank_out,
           "a rank beyond the calibration set must give an infinite threshold")
    oracle(f"k = ceil(10 * 0.80) = {independent_rank(9, .2)}, q = 0.6" in rank_out,
           "the printed rank and threshold are not the ones the oracle derives")
    oracle("linear quantile at 8/9 0.633333" in rank_out and "higher quantile at 8/9 0.900000" in rank_out
           and "eighth order statistic 0.600000" in rank_out,
           "the three competing conventions are no longer all printed and distinct")
    covered, rotation_k = independent_rotation([.05, .10, .15, .20, .25, .30, .40, .60, .90, .95], .2)
    oracle(covered == checked["rank"]["rotation"]["covered"],
           "the combined-rank rotation oracle disagrees with the packet")
    oracle(rotation_k == 8, "the rotation's rank should be 8 for nine calibration scores at alpha .2")
    oracle(f"rotation: {covered} of 10 held-out scores covered" in rank_out,
           "the printed rotation count is not the one the rank oracle derives")
    tied_covered, _ = independent_rotation([.2] * 10, .2)
    oracle(tied_covered == checked["rank"]["tie_rotation"]["covered"] == 10,
           "ties must enlarge coverage to all ten under the weak comparison")
    oracle(f"every score tied at .2: {tied_covered} of 10 covered" in rank_out,
           "the tied rotation is not printed as the oracle computes it")
    oracle("set {A}" in rank_out and "set {A, B}" in rank_out and "set the empty set" in rank_out,
           "the three prediction sets must include a singleton, a pair and an empty set")

    # The two complete programs' stdout, against the recorded experiment file.
    for name, row in recorded["classification"]["methods"].items():
        line = f"{name} {row['correct']} {round(row['brier'], 6)} {row['covered']} {row['mean_set_size']}"
        oracle(line in experiments_output, f"the classification line for {name} does not match the recorded run")
    for name, row in recorded["regression"]["methods"].items():
        line = f"{name} {row['covered']} {round(row['mean_width'], 6)}"
        oracle(line in experiments_output, f"the regression line for {name} does not match the recorded run")
    oracle(f'"q": {checked["rank"]["q"]}' in calculations_output
           or f'"q": {checked["rank"]["q"]}' in calculations_output.replace(" ", " "),
           "the calculations program no longer prints the threshold the packet recorded")
    printed = json.loads(calculations_output)
    oracle(printed["rank_covered"] == checked["rank"]["rotation"]["covered"],
           "the calculations program prints a rotation count the packet does not record")
    oracle(printed["q"] == checked["rank"]["q"], "the calculations program prints a different threshold")
    oracle(printed["ece"] == checked["reliability"]["two_bins"]["ece"],
           "the calculations program prints a different ECE")

    versions = json.loads((WORKSPACE / "experiment-results.json").read_text(encoding="utf-8"))["versions"]
    oracle(versions == {"numpy": "2.3.5", "scipy": "1.18.1", "sklearn": "1.9.1"},
           f"the runtime versions moved: {versions}")

    if failures:
        for problem in failures:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(failures)} of {oracle_count} oracles failed")

    if write:
        header = (
            "// Programs for the Calibration & Conformal Prediction lesson.\n"
            "//\n"
            "// Generated by scripts/verify-calibration-examples.py --write. Do not edit by hand.\n"
            "//\n"
            "// The three displayed programs are assembled from functions copied VERBATIM out of the\n"
            "// content packet's calibration_calculations.py by slicing its syntax tree, plus a short\n"
            "// driver held in the verifier. The two complete programs are the packet's own bytes,\n"
            "// served for download. `expected` is what each program actually printed on this machine.\n"
            "//\n"
            "// Both complete programs regenerate the packet's checked-results.json and\n"
            "// experiment-results.json byte for byte under NumPy 2.3.5, SciPy 1.18.1 and\n"
            "// scikit-learn 1.9.1; that equality is rechecked on every run of the verifier.\n"
            "export const calibrationExamples = "
        )
        MODULE.write_text(header + json.dumps(records, indent=2) + ";\n", encoding="utf-8", newline="\n")

    # A counter that is reported but never floored is decoration: a block that
    # stops running still prints PASS, with a smaller number nobody reads. The
    # models verifier has had these floors since phase A; the guard audit found
    # that the other four verifiers report their headline counts unfloored.
    if oracle_count < 63 or len(PROGRAMS) + len(DOWNLOADS) < 5:
        raise SystemExit(f"only {oracle_count} oracles over {len(PROGRAMS) + len(DOWNLOADS)} programs ran; "
                         "a block of checks did not execute, so this PASS covers less than it claims")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-calibration-examples.py",
        "verifierSha256": sha(Path(__file__).read_bytes()),
        "python": sys.version.split()[0],
        "runtimeVersions": versions,
        "packetSha256": {
            "calibration_calculations.py": PACKET_CALCULATIONS_SHA,
            "uncertainty_experiments.py": PACKET_EXPERIMENTS_SHA,
        },
        "verbatimFunctionSha256": segment_hashes,
        "servedSha256": {
            path.name: sha(path.read_bytes())
            for path in sorted(ASSETS.iterdir()) if path.is_file()
        },
        "programsExecuted": len(PROGRAMS) + len(DOWNLOADS),
        "oracles": oracle_count,
        "trustRootRegeneration": {
            "checked-results.json": "byte-identical",
            "experiment-results.json": "byte-identical",
        },
        "scope": "Five programs. Three are assembled from verbatim syntax-tree slices of the packet's "
                 "calibration_calculations.py and a driver held in this verifier, written to "
                 "public/learn-assets/calibration/ and executed there; the assembled text must equal the "
                 "served asset byte for byte, so a hand edit to either side fails. Two are the packet's "
                 "complete programs, served unchanged and run in a scratch workspace beside copies of the "
                 "two served CSV files, where they must regenerate both of the packet's recorded JSON files "
                 "byte for byte. Every printed number is then checked against an oracle computed here by a "
                 "different route: the conformal rank by scanning integers in exact rational arithmetic "
                 "rather than by any ceiling or quantile call, the rotation by the combined-rank argument "
                 "rather than by rebuilding thresholds, the isotonic fit by the max-min formula over lower "
                 "and upper sets rather than by the stack algorithm, and the bins and ECE by plain Python "
                 "loops with no array library.",
        "limitations": [
            "Byte-identical regeneration is claimed for these library versions only. A different NumPy, "
            "SciPy or scikit-learn can move the last digits of a fitted optimum.",
            "The drivers around the verbatim functions are authored here rather than extracted; they are "
            "pinned by the served file's own hash and executed, not transcribed onto the page.",
            "MAPIE is documentation-based guidance in the lesson and is not installed or executed anywhere.",
            "Browser rendering of these programs is checked separately by scripts/verify-calibration-browser.cjs.",
        ],
        "passed": True,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"PASS: {len(PROGRAMS) + len(DOWNLOADS)} programs executed and {oracle_count} oracles checked. "
          f"Both trust-root JSON files regenerate byte for byte; the rank, rotation, isotonic fit and ECE "
          f"are each recomputed by a second route.")


def _record_failure(evidence_path, verifier, error):
    """Write a FAILING evidence record.

    Every check in this file raises or exits, and a raise skips the evidence
    write at the end of main() — which leaves the PREVIOUS run's
    `"passed": true` on disk describing a tree that fails. A reviewer reading
    the evidence directory afterwards sees green for red. A sibling lesson
    shipped exactly that, so the failure path writes its own record.
    """
    import traceback
    try:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps({
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "verifier": verifier,
            "passed": False,
            "failure": {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "traceback": traceback.format_exc()[-4000:],
            },
            "note": "This run failed. Written from the failure path so a red tree cannot be read as green "
                    "from an earlier run's evidence file.",
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
    except Exception:  # noqa: BLE001 - the original failure must still surface
        pass


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:  # noqa: BLE001 - includes SystemExit from a FAIL path
        if isinstance(error, SystemExit) and not error.code:
            raise
        _record_failure(EVIDENCE, __file__.replace("\\", "/").split("/scripts/")[-1], error)
        raise
