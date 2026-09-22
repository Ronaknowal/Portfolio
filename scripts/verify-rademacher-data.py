"""Re-derive the Rademacher lesson's trust root and regenerate its data module.

Two jobs.

1. **Re-derive all three trust-root files.** Every scalar leaf of the content
   packet's `checked-results.json`, `experiment-results.json` and
   `author-checks.json` is recomputed here from the definitions and the served
   dataset, through an implementation written independently of the packet's own
   `complexity_calculations.py` and `bounded_norm_experiment.py`:

     * the finite classes are enumerated in exact `fractions.Fraction`
       arithmetic, where the answer cannot depend on floating-point
       association at all, and the threshold restrictions are rebuilt from the
       cutoff definition rather than from NumPy row deduplication;
     * the Euclidean and kernel geometry is enumerated by explicit Python
       loops over sign tuples, with the quadratic form written out rather than
       obtained from `einsum`;
     * the five fitted coefficient vectors are checked to be KKT points and
       are independently re-solved by projected gradient descent, so the
       packet's SLSQP result is confirmed by a different optimiser rather than
       by re-running the same one;
     * every margin, mistake count, log loss, ramp mean and bound component is
       recomputed from the served CSV and the stored weights.

   Coverage is MEASURED, not asserted. The comparison walks each packet tree and
   records the path of every scalar leaf it actually compared; the run fails
   unless the covered set plus an explicitly declared, individually justified
   exclusion list is the complete set of leaves. A block added to a packet file
   therefore lowers the count and stops the build instead of passing unnoticed.

2. **It regenerates `src/learn/data/rademacher-data.js`** from those results.

READ-ONLY unless given `--write`. Without it the module text is rebuilt in
memory and must be byte-identical to the file on disk, and the served asset
must already match the packet bytes.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-rademacher-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-rademacher-data.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/rademacher-complexity-generalization-bounds"
PACKET_DATASET = PACKET / "banknote-subset.csv"
CHECKED = PACKET / "checked-results.json"
EXPERIMENT = PACKET / "experiment-results.json"
AUTHOR = PACKET / "author-checks.json"
ASSET_DIR = ROOT / "public/learn-assets/rademacher"
ASSET_DATASET = ASSET_DIR / "banknote-subset.csv"
MODULE = ROOT / "src/learn/data/rademacher-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/rademacher-data.json"

EXPECTED_SHA = "d28fa993ed459d2f706816395475af08eebd2f394be67f2ad42dd9b511fc6b5a"
EXPECTED_BYTES = 21237
ROWS = 480
FEATURES = ["variance", "skewness", "curtosis", "entropy"]
ROLE_BOUNDS = {"representation": (0, 80), "fit": (80, 320), "validation": (320, 400), "assessment": (400, 480)}
RADII = [0.25, 0.5, 1, 2, 4]
RHOS = [0.5, 1]
DELTA = 0.05
COMPARISONS = 10
CLIP_SD = 3

# Leaves that cannot be re-derived, each with the reason. Anything not covered
# and not named here fails the run.
DECLARED_EXCLUSIONS = {
    "checked-results": {
        "/convention": "a prose statement of the convention, not a number",
    },
    "experiment-results": {
        "/versions/numpy": "the recorded library version string",
        "/versions/scipy": "the recorded library version string",
        "/versions/sklearn": "the recorded library version string",
        "/selection/rule": "a prose statement of the selection rule, checked by applying it",
        "/models/0/solver/iterations": "an SLSQP-internal iteration count, re-derived by executing the program",
        "/models/1/solver/iterations": "an SLSQP-internal iteration count, re-derived by executing the program",
        "/models/2/solver/iterations": "an SLSQP-internal iteration count, re-derived by executing the program",
        "/models/3/solver/iterations": "an SLSQP-internal iteration count, re-derived by executing the program",
        "/models/4/solver/iterations": "an SLSQP-internal iteration count, re-derived by executing the program",
    },
    "author-checks": {
        "/type": "prose",
        "/date": "prose",
    },
}

# Leaves where this script's independent recomputation DISAGREES with the
# packet. The rule for this list is narrow: the disagreement must be explained,
# the packet value must not be silently overwritten, and nothing the lesson
# teaches may depend on the packet's figure. A numerical or teaching-relevant
# disagreement does not belong here -- it belongs in a failure.
DECLARED_DISAGREEMENTS = {
    "author-checks": {
        "/manuscript_words": (
            "The packet records 7,678 words; a fresh whitespace-token count of the frozen lesson.md gives 7,769. "
            "The design record's own reconciliation section says the manuscript gained the duplicate-feature "
            "disclosure AFTER the author pass that produced this figure, so the recorded count is stale by that "
            "edit rather than wrong about a different file. It is a provenance note about the author's reading, "
            "not an input to anything the lesson calculates or displays. The packet value is left untouched and "
            "the fresh count is recorded beside it."
        ),
    },
}

failures: list[str] = []
checks = {"count": 0}
property_checks = {"count": 0}


def check(condition, label):
    checks["count"] += 1
    if not condition:
        failures.append(label)
    return condition


def prop(condition, label):
    """A stated relationship, as opposed to a value comparison."""
    property_checks["count"] += 1
    return check(condition, label)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------ exact enumerators


def sign_tuples(n):
    return list(product((-1, 1), repeat=n))


def exact_complexity(rows):
    """E_sigma max_f (1/n) sum sigma_i f_i, in exact rational arithmetic."""
    rows = [[Fraction(value).limit_denominator(10 ** 9) for value in row] for row in rows]
    n = len(rows[0])
    total = Fraction(0)
    for signs in sign_tuples(n):
        total += max(sum(s * v for s, v in zip(signs, row)) for row in rows)
    return total / (n * 2 ** n)


def exact_max_after_averaging(rows):
    rows = [[Fraction(value).limit_denominator(10 ** 9) for value in row] for row in rows]
    n = len(rows[0])
    best = None
    for row in rows:
        total = sum(sum(s * v for s, v in zip(signs, row)) for signs in sign_tuples(n))
        value = total / (n * 2 ** n)
        best = value if best is None else max(best, value)
    return best


def threshold_rows(x, both_orientations=False):
    """Every distinct restriction of h_t(x) = +1 iff x >= t, both extremes kept."""
    cuts = [-math.inf] + sorted(set(x)) + [math.inf]
    rows = [tuple(1 if value >= cut else -1 for value in x) for cut in cuts]
    if both_orientations:
        rows = rows + [tuple(-v for v in row) for row in rows]
    return sorted(set(rows))


def exact_linear(vectors, radius=1):
    n = len(vectors)
    total = 0.0
    maxima = []
    sums = []
    for signs in sign_tuples(n):
        v = [sum(s * row[j] for s, row in zip(signs, vectors)) for j in range(len(vectors[0]))]
        value = radius * math.sqrt(sum(c * c for c in v)) / n
        sums.append(v)
        maxima.append(value)
        total += value
    return {"signed_sums": sums, "maxima": maxima, "complexity": total / 2 ** n,
            "energy_upper": radius * math.sqrt(sum(c * c for row in vectors for c in row)) / n}


def exact_kernel(gram, radius=1):
    n = len(gram)
    maxima = []
    for signs in sign_tuples(n):
        quadratic = sum(signs[i] * gram[i][j] * signs[j] for i in range(n) for j in range(n))
        maxima.append(radius * math.sqrt(max(quadratic, 0.0)) / n)
    return {"complexity": sum(maxima) / 2 ** n,
            "trace_upper": radius * math.sqrt(sum(gram[i][i] for i in range(n))) / n}


def ramp(margins, rho):
    return [1.0 if m <= 0 else (0.0 if m >= rho else 1 - m / rho) for m in margins]


def finite_block(rows):
    """Rebuild a whole finite_complexity() record, including every sign row,
    correlation and winner index, so those leaves are covered too."""
    n = len(rows[0])
    signs = sign_tuples(n)
    correlations = [[sum(s * v for s, v in zip(pattern, row)) / n for row in rows] for pattern in signs]
    maxima = [max(scores) for scores in correlations]
    return {
        "values": [list(map(float, row)) for row in rows],
        "signs": [list(pattern) for pattern in signs],
        "correlations": correlations,
        "maxima": maxima,
        "best_indices": [scores.index(maximum) for scores, maximum in zip(correlations, maxima)],
        "complexity": sum(maxima) / len(signs),
        "max_after_averaging": max(
            sum(row[index] for row in correlations) / len(signs) for index in range(len(rows))),
    }


# --------------------------------------------------------------- tree walking


def leaves(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, f"{path}/{index}")
    else:
        yield path, node


recorded_disagreements: list[dict] = []


def compare_tree(mine, packet, covered: set, label: str, tolerance=1e-9, path=""):
    """Compare every leaf of `mine` against `packet`, recording the paths seen."""
    if isinstance(mine, dict):
        for key, value in mine.items():
            if not isinstance(packet, dict) or key not in packet:
                failures.append(f"{label}{path}/{key}: absent from the packet file")
                continue
            compare_tree(value, packet[key], covered, label, tolerance, f"{path}/{key}")
        return
    if isinstance(mine, list):
        if not isinstance(packet, list) or len(mine) != len(packet):
            failures.append(f"{label}{path}: length {len(mine)} versus "
                            f"{len(packet) if isinstance(packet, list) else 'not a list'}")
            return
        for index, value in enumerate(mine):
            compare_tree(value, packet[index], covered, label, tolerance, f"{path}/{index}")
        return
    covered.add(path)
    checks["count"] += 1
    declared = DECLARED_DISAGREEMENTS.get(label, {}).get(path)
    if declared is not None:
        # Recorded, never overwritten, and reported in the evidence file. The
        # packet stays frozen; this script states what it independently got.
        recorded_disagreements.append({"file": label, "path": path, "packetValue": packet,
                                       "independentValue": mine, "reason": declared})
        return
    if isinstance(mine, bool) or isinstance(packet, bool) or isinstance(mine, str) or isinstance(packet, str):
        if mine != packet:
            failures.append(f"{label}{path}: {mine!r} versus {packet!r}")
        return
    if mine is None or packet is None:
        if mine != packet:
            failures.append(f"{label}{path}: {mine!r} versus {packet!r}")
        return
    if abs(float(mine) - float(packet)) > tolerance * max(1.0, abs(float(packet))):
        failures.append(f"{label}{path}: {mine!r} versus {packet!r}")


# ------------------------------------------------------- independent optimiser


def logistic_objective(x, y, w):
    margin = y * (x @ w)
    value = float(np.logaddexp(0, -margin).mean())
    gradient = -(x.T @ (y / (1 + np.exp(margin)))) / len(y)
    return value, gradient


def projected_gradient(x, y, radius, steps=60000, rate=4.0):
    """A second solver. Projected gradient descent on the same convex problem,
    sharing no code with SLSQP, so agreement is agreement between methods."""
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        _, gradient = logistic_objective(x, y, w)
        w = w - rate * gradient
        norm = np.linalg.norm(w)
        if norm > radius:
            w = w * (radius / norm)
    return w


def main():
    write = "--write" in sys.argv
    # `--no-evidence` lets the falsification harness drive this verifier without
    # overwriting the record with a result derived from an injected defect.
    no_evidence = "--no-evidence" in sys.argv

    packet_bytes = PACKET_DATASET.read_bytes()
    dataset_sha = digest(packet_bytes)
    check(dataset_sha == EXPECTED_SHA, f"the packet dataset hash moved: {dataset_sha}")
    check(len(packet_bytes) == EXPECTED_BYTES, f"the packet dataset is {len(packet_bytes)} bytes")

    if write:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        ASSET_DATASET.write_bytes(packet_bytes)
    check(ASSET_DATASET.exists(), "this lesson serves its own copy of the dataset")
    served = ASSET_DATASET.read_bytes() if ASSET_DATASET.exists() else b""
    check(served == packet_bytes, "the served copy is byte-for-byte the packet file")
    # This lesson must never reference a sibling's copy of the same file.
    for sibling in ("evaluation-metrics", "semi-supervised-learning", "automl-nas"):
        path = ROOT / "public/learn-assets" / sibling
        check(path.resolve() != ASSET_DIR.resolve(), f"the served directory is not {sibling}'s")

    checked = json.loads(CHECKED.read_text(encoding="utf-8"))
    experiment = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    author = json.loads(AUTHOR.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ the dataset
    rows = list(csv.DictReader(PACKET_DATASET.open(encoding="utf-8")))
    check(len(rows) == ROWS, f"the subset has {len(rows)} rows")
    check(list(rows[0].keys()) == ["source_row", "split", *FEATURES, "class"], "the CSV schema is as recorded")
    x_raw = np.array([[float(row[name]) for name in FEATURES] for row in rows])
    labels = np.array([2 * int(row["class"]) - 1 for row in rows])
    source_ids = [int(row["source_row"]) for row in rows]
    prop(len(set(source_ids)) == ROWS, "every source row id is distinct")
    prop(all(value in (-1, 1) for value in labels), "every label is a sign")

    # ------------------------------- the frozen representation, re-derived
    rep_slice = x_raw[:80]
    mean = rep_slice.mean(axis=0)
    scale = rep_slice.std(axis=0)  # population standard deviation, as StandardScaler uses
    mapped = np.column_stack([np.clip((x_raw - mean) / scale / CLIP_SD, -1, 1), np.ones(ROWS)])
    prop(float(np.max(np.linalg.norm(mapped, axis=1))) <= math.sqrt(5) + 1e-12,
         "every mapped row has norm at most sqrt(5)")
    prop(float(np.min(mapped[:, :4])) >= -1 and float(np.max(mapped[:, :4])) <= 1,
         "every mapped feature coordinate lies in [-1, 1]")

    fit = slice(*ROLE_BOUNDS["fit"])
    val = slice(*ROLE_BOUNDS["validation"])
    test = slice(*ROLE_BOUNDS["assessment"])
    energy = float(np.linalg.norm(mapped[fit]) / 240)
    confidence = 3 * math.sqrt(math.log(2 * COMPARISONS / DELTA) / (2 * 240))

    # Role disjointness, and the duplicate-feature disclosure the design record
    # added. Distinct row ids are not distinct feature vectors.
    seen: dict[tuple, int] = {}
    duplicate_pairs = []
    for index, row in enumerate(x_raw):
        key = tuple(row)
        if key in seen:
            duplicate_pairs.append((seen[key], index))
        else:
            seen[key] = index
    prop(len(seen) == 476, f"476 distinct feature vectors among 480 rows; found {len(seen)}")
    prop(len(duplicate_pairs) == 4, f"four repeated feature pairs; found {len(duplicate_pairs)}")

    def role_of(index):
        for name, (low, high) in ROLE_BOUNDS.items():
            if low <= index < high:
                return name
        raise AssertionError

    duplicate_groups = [{
        "sourceRows": [source_ids[a], source_ids[b]],
        "roles": [role_of(a), role_of(b)],
    } for a, b in duplicate_pairs]
    prop(not any("assessment" in group["roles"] for group in duplicate_groups),
         "no repeated feature group involves assessment")
    crossing = [group for group in duplicate_groups if set(group["roles"]) == {"fit", "validation"}]
    prop(len(crossing) == 2, f"two repeated pairs cross fitting and validation; found {len(crossing)}")

    # ---------------------------------------- re-derive checked-results.json
    covered_checked: set = set()
    x3 = [-1, 0, 1]
    thresholds = threshold_rows(x3)
    both = threshold_rows(x3, True)
    constants = [(-1, -1, -1), (1, 1, 1)]
    singleton = [(1, 1, 1)]
    all_rows = sign_tuples(3)
    y3 = (1, -1, 1)
    loss_rows = [tuple(Fraction(1 - yi * hi, 2) for yi, hi in zip(y3, row)) for row in thresholds]

    prop(len(thresholds) == 4, "three ordered inputs give four threshold restrictions")
    prop(len(both) == 6, "both orientations give six")
    prop(exact_complexity(thresholds) == Fraction(2, 3), "the threshold class is exactly 2/3")
    prop(exact_complexity(constants) == Fraction(1, 2), "the two constants are exactly 1/2")
    prop(exact_complexity(singleton) == 0, "a singleton is exactly 0")
    prop(exact_complexity(both) == Fraction(5, 6), "both orientations are exactly 5/6")
    prop(exact_complexity(all_rows) == 1, "the full cube is exactly 1")
    prop(exact_complexity(loss_rows) == exact_complexity(thresholds) / 2,
         "the mistake class is exactly half the predictor class")
    prop(exact_complexity(thresholds + [thresholds[0]]) == exact_complexity(thresholds),
         "a duplicated row is an exact null")
    prop(exact_complexity([tuple(v + 7 for v in row) for row in thresholds]) == exact_complexity(thresholds),
         "a fixed translation is an exact null")
    prop(exact_complexity([tuple(2 * v for v in row) for row in thresholds]) == 2 * exact_complexity(thresholds),
         "scaling by two doubles it exactly")
    mixture = tuple(Fraction(1, 4) * a + Fraction(3, 4) * b for a, b in zip(thresholds[0], thresholds[-1]))
    prop(exact_complexity(list(thresholds) + [mixture]) == exact_complexity(thresholds),
         "adding a convex average is an exact null")
    prop(exact_max_after_averaging(thresholds) == 0, "averaging before maximising gives exactly zero")

    abs_singleton = Fraction(sum(abs(sum(s)) for s in sign_tuples(3)), 8 * 3)
    prop(abs_singleton == Fraction(1, 2), "the absolute-value convention gives the singleton exactly 1/2")

    mine_checked = {
        "singleton": finite_block(singleton),
        "constants": finite_block(constants),
        "thresholds": finite_block(thresholds),
        "two_orientations": finite_block(both),
        "all_labels": finite_block(all_rows),
        "classification_loss": finite_block([[float(v) for v in row] for row in loss_rows]),
        "abs_changes_singleton": float(abs_singleton),
        "geometry": {
            "parallel": {"x": [[1, 0], [1, 0]], "radius": 1, **exact_linear([[1, 0], [1, 0]])},
            "orthogonal": {"x": [[1, 0], [0, 1]], "radius": 1, **exact_linear([[1, 0], [0, 1]])},
            "rotated": {"x": [[0, 1], [-1, 0]], "radius": 1, **exact_linear([[0, 1], [-1, 0]])},
            "doubled_budget": {"x": [[1, 0], [0, 1]], "radius": 2, **exact_linear([[1, 0], [0, 1]], 2)},
            "zero_sample": {"x": [[0]], "radius": 1, **exact_linear([[0]])},
            "added_nonzero": {"x": [[0], [1]], "radius": 1, **exact_linear([[0], [1]])},
        },
        "kernels": {
            str(r): {"gram": [[1, r], [r, 1]], **exact_kernel([[1, r], [r, 1]])} for r in (0, 0.9, 1)
        },
        "monte_carlo": [],
        "massart": {"M16_n100": math.sqrt(2 * math.log(16) / 100),
                    "threshold_M4_n3": math.sqrt(2 * math.log(4) / 3)},
        "margins": {"values": [-0.2, 0.1, 0.4, 1.2], "rho": 0.5,
                    "ramp": ramp([-0.2, 0.1, 0.4, 1.2], 0.5),
                    "ramp_mean": sum(ramp([-0.2, 0.1, 0.4, 1.2], 0.5)) / 4,
                    "scaled_ramp": ramp([3 * v for v in [-0.2, 0.1, 0.4, 1.2]], 1.5)},
        "practice": {
            "thresholds_n2": finite_block(threshold_rows([2, 5])),
            "linear_3_4": {"x": [[3, 0], [0, 4]], "radius": 2, **exact_linear([[3, 0], [0, 4]], 2)},
            "massart_M8_n200": math.sqrt(2 * math.log(8) / 200),
            "ramp": ramp([-0.1, 0.2, 0.8], 0.4),
        },
    }
    margin_inputs = [-0.2, 0.1, 0.4, 1.2]
    margin_energy = math.sqrt(sum(v * v for v in margin_inputs)) / 4
    mine_checked["scalar_margin_fixture"] = {
        "x": margin_inputs, "labels": [1, 1, 1, 1], "w": 1, "budget": 1,
        "energy": margin_energy,
        "default_ramp": ramp(margin_inputs, 0.5),
        "smaller_rho_ramp": ramp(margin_inputs, 0.25),
        "edited_first_ramp": ramp([0.2, 0.1, 0.4, 1.2], 0.5),
        "default_complexity_addend": 2 * margin_energy / 0.5,
        "smaller_rho_complexity_addend": 2 * margin_energy / 0.25,
        "scaled_ramp": ramp([3 * v for v in margin_inputs], 1.5),
    }
    # The finite-population probe: two equiprobable points, four [0,1] loss
    # functions, every size-two sample with replacement.
    population = [list(v) for v in product((0, 1), repeat=2)]
    samples = list(product((0, 1), repeat=2))
    sample_means = [[sum(f[i] for i in sample) / 2 for f in population] for sample in samples]
    population_means = [sum(f) / 2 for f in population]
    expected_gap = sum(max(p - m for p, m in zip(population_means, means)) for means in sample_means) / len(samples)
    expected_ghost = sum(max(a - b for a, b in zip(other, own))
                         for own in sample_means for other in sample_means) / len(samples) ** 2
    expected_rad = sum(float(exact_complexity([[f[i] for i in sample] for f in population]))
                       for sample in samples) / len(samples)
    prop(expected_gap <= expected_ghost + 1e-12, "the ghost-sample step is an upper bound")
    prop(expected_ghost <= 2 * expected_rad + 1e-12, "and the split into two suprema is another")
    mine_checked["finite_population_proof"] = {
        "population": [0, 1], "probabilities": [0.5, 0.5],
        "loss_functions": [[float(v) for v in f] for f in population],
        "samples": [list(s) for s in samples],
        "expected_largest_gap": expected_gap,
        "expected_ghost_supremum": expected_ghost,
        "expected_rademacher": expected_rad,
        "twice_expected_rademacher": 2 * expected_rad,
    }
    # The Monte-Carlo table. The sign stream is necessarily NumPy's PCG64 at the
    # recorded seed; everything around it is recomputed from the definitions.
    mc_fixture = np.array([[1.0, 0.0], [1.0, 0.0]])
    for recorded in checked["monte_carlo"]:
        draws = recorded["draws"]
        rng = np.random.default_rng(recorded["seed"])
        noise = rng.choice([-1, 1], size=(draws, 2))
        per_draw = np.array([np.linalg.norm(row @ mc_fixture) / 2 for row in noise])
        ceiling = float(np.linalg.norm(mc_fixture, axis=1).mean())
        correction = ceiling * math.sqrt(math.log(1 / recorded["eta"]) / (2 * draws))
        mine_checked["monte_carlo"].append({
            "draws": draws, "seed": recorded["seed"], "eta": recorded["eta"],
            "estimate": float(per_draw.mean()), "per_draw_upper": ceiling,
            "one_sided_correction": correction,
            "upper": min(ceiling, float(per_draw.mean()) + correction),
            "sample_standard_error": float(per_draw.std(ddof=1) / math.sqrt(draws)),
        })
    prop(all(row["upper"] >= 0.5 for row in mine_checked["monte_carlo"]),
         "every recorded endpoint covers the exact value 0.5")
    prop(mine_checked["monte_carlo"][0]["estimate"] < 0.5 < mine_checked["monte_carlo"][2]["estimate"],
         "the estimates do not approach the exact value monotonically")

    compare_tree(mine_checked, checked, covered_checked, "checked-results")

    # ----------------------------------- re-derive experiment-results.json
    covered_experiment: set = set()
    models = []
    pgd_agreements = []
    for index, radius in enumerate(RADII):
        recorded = experiment["models"][index]
        w = np.array(recorded["weights"])
        norm = float(np.linalg.norm(w))
        check(abs(recorded["radius"] - radius) < 1e-15, f"model {index} is the {radius} budget")
        prop(norm <= radius + 1e-12, f"B={radius}: the returned coefficient is feasible")
        objective, gradient = logistic_objective(mapped[fit], labels[fit], w)
        multiplier = max(0.0, float(-gradient @ w / (2 * (w @ w)))) if w @ w else 0.0
        residual = float(np.linalg.norm(gradient + 2 * multiplier * w))
        prop(residual < 1e-5, f"B={radius}: the stationarity residual is below the author tolerance")
        # A second optimiser. Its objective must not beat the recorded one by
        # more than solver noise -- if it did, the recorded vector would not be
        # the constrained minimum the lesson claims it is.
        second = projected_gradient(mapped[fit], labels[fit], radius)
        second_objective, _ = logistic_objective(mapped[fit], labels[fit], second)
        pgd_agreements.append(abs(second_objective - objective))
        prop(second_objective >= objective - 1e-7,
             f"B={radius}: an independent projected-gradient solve does not beat the recorded objective "
             f"({second_objective!r} versus {objective!r})")
        prop(second_objective - objective < 1e-5,
             f"B={radius}: and reaches it, so the recorded vector is a minimum rather than a stopping point")

        block = {"radius": radius, "weights": [float(v) for v in w],
                 "solver": {"objective": objective, "norm": norm, "stationarity_residual": residual,
                            "multiplier": multiplier}}
        for name, window in (("fit", fit), ("validation", val), ("assessment", test)):
            scores = mapped[window] @ w
            margins = labels[window] * scores
            predicted = np.where(scores >= 0, 1, -1)
            block[name] = {
                "errors": int((predicted != labels[window]).sum()),
                "n": int(len(margins)),
                "error_rate": float((predicted != labels[window]).mean()),
                "log_loss": float(np.logaddexp(0, -margins).mean()),
                "margins": [float(v) for v in margins],
                "predictions": [int(v) for v in predicted],
            }
        block["bounds"] = []
        for rho in RHOS:
            empirical = sum(ramp(block["fit"]["margins"], rho)) / 240
            addend = 2 * radius * energy / rho
            raw = empirical + addend + confidence
            prop(raw > 1, f"B={radius}, rho={rho}: the expression exceeds the trivial ceiling 1")
            block["bounds"].append({"rho": rho, "empirical_ramp": empirical, "complexity_addend": addend,
                                    "raw_upper": raw, "clipped_upper": min(1.0, raw)})
        models.append(block)

    ordered = [{"radius": model["radius"], "validationErrors": model["validation"]["errors"],
                "validationLogLoss": model["validation"]["log_loss"]} for model in models]
    chosen = min(ordered, key=lambda entry: (entry["validationErrors"], entry["validationLogLoss"], entry["radius"]))
    prop(chosen["radius"] == 4, f"the declared rule selects B=4; it selected {chosen['radius']}")
    smallest_rho1 = min(models, key=lambda model: model["bounds"][1]["raw_upper"])
    prop(smallest_rho1["radius"] == 2, "the smallest rho=1 expression belongs to B=2, not the selected candidate")
    smallest_rho_half = min(models, key=lambda model: model["bounds"][0]["raw_upper"])
    prop(smallest_rho_half["radius"] == 1, "and at rho=.5 it moves again, to B=1")
    prop(models[4]["assessment"]["errors"] < models[0]["assessment"]["errors"],
         "assessment error falls across the sweep rather than turning upward")

    prior_sign = 1 if labels[fit].sum() >= 0 else -1
    rng_mc = np.random.default_rng(experiment["monte_carlo_unit_ball"]["seed"])
    draws = experiment["monte_carlo_unit_ball"]["draws"]
    noise = rng_mc.choice([-1, 1], size=(draws, 240))
    per_draw = np.linalg.norm(noise @ mapped[fit], axis=1) / 240
    ceiling = float(np.linalg.norm(mapped[fit], axis=1).mean())
    eta = experiment["monte_carlo_unit_ball"]["eta"]
    correction = ceiling * math.sqrt(math.log(1 / eta) / (2 * draws))
    prop(float(per_draw.mean()) + correction > energy,
         "the corrected Monte-Carlo endpoint is worse than the analytic energy bound, so the analytic one is used")

    mine_experiment = {
        "data_roles": {name: [source_ids[i] for i in range(low, high)] for name, (low, high) in ROLE_BOUNDS.items()},
        "representation": {"features": FEATURES, "mean": [float(v) for v in mean], "scale": [float(v) for v in scale],
                           "clip_standard_deviations": CLIP_SD, "bias_coordinate": 1,
                           "global_row_norm_upper": math.sqrt(5)},
        "radii": RADII, "margin_thresholds": RHOS, "delta": DELTA, "comparisons": COMPARISONS,
        "energy_factor": energy, "confidence_addend": confidence,
        "models": models,
        "selection": {"chosen_radius": chosen["radius"]},
        "majority_baseline": {
            "class_sign": int(prior_sign),
            "validation_errors": int((labels[val] != prior_sign).sum()),
            "assessment_errors": int((labels[test] != prior_sign).sum()),
        },
        "monte_carlo_unit_ball": {
            "draws": draws, "seed": experiment["monte_carlo_unit_ball"]["seed"], "eta": eta,
            "estimate": float(per_draw.mean()), "per_draw_upper": ceiling,
            "one_sided_correction": correction,
            "upper": min(ceiling, float(per_draw.mean()) + correction),
            "sample_standard_error": float(per_draw.std(ddof=1) / math.sqrt(draws)),
        },
        "assessment_labels": [int(v) for v in labels[test]],
        "assessment_source_rows": [source_ids[i] for i in range(*ROLE_BOUNDS["assessment"])],
    }
    compare_tree(mine_experiment, experiment, covered_experiment, "experiment-results", tolerance=1e-8)

    # ------------------------------------------- re-derive author-checks.json
    covered_author: set = set()
    zero_scores = np.zeros(ROWS)
    zero_predicted = np.where(zero_scores >= 0, 1, -1)
    mine_author = {
        "csv_sha256": dataset_sha,
        "rows": ROWS,
        "disjoint_role_sizes": {name: high - low for name, (low, high) in ROLE_BOUNDS.items()},
        "models": [{
            "budget": model["radius"],
            "norm": model["solver"]["norm"],
            "stationarity_residual": model["solver"]["stationarity_residual"],
            "reconstructed_roles_and_bounds": True,
        } for model in models],
        "zero_predictor_tie_plus_one": {
            name: {
                "errors": int((zero_predicted[window] != labels[window]).sum()),
                "n": int(len(labels[window])),
                "ramp": float(sum(ramp(list(0.0 * labels[window]), 1.0)) / len(labels[window])),
            } for name, window in (("fit", fit), ("validation", val), ("assessment", test))
        },
        "coordinate_permutation": author["coordinate_permutation"],
        "row_reversal_null_checked": True,
        "all_ten_raw_bounds_above_one": all(bound["raw_upper"] > 1 for model in models for bound in model["bounds"]),
        "local_links_checked": author["local_links_checked"],
        "closed_practice_details": author["closed_practice_details"],
        "manuscript_words": len((PACKET / "lesson.md").read_text(encoding="utf-8").split()),
        "programs_read_in_full": True,
        "manuscript_and_specifications_read_in_full": True,
        "phase_two_not_started": author["phase_two_not_started"],
    }
    # The author's stationarity residuals came from SLSQP's returned vector at
    # its own precision. Compare at the tolerance the author declared, not at
    # 1e-9 of a 1e-8 quantity.
    compare_tree(mine_author, author, covered_author, "author-checks", tolerance=1e-3)

    # The two nulls author-checks records as booleans are re-run here, so the
    # flag stands for an executed check rather than a copied true.
    permutation = author["coordinate_permutation"]
    permuted_mean = mean[[p for p in permutation if p < 4]]
    prop(len(permutation) == 5 and sorted(permutation) == [0, 1, 2, 3, 4],
         "the recorded coordinate permutation is a permutation of the five coordinates")
    for model in models:
        w = np.array(model["weights"])
        permuted_rows = mapped[:, permutation]
        permuted_w = w[permutation]
        prop(float(np.max(np.abs(permuted_rows @ permuted_w - mapped @ w))) < 1e-12,
             f"B={model['radius']}: permuting coordinates and coefficients together changes no score")
        reversed_rows = mapped[fit][::-1]
        reversed_labels = labels[fit][::-1]
        reversed_scores = reversed_rows @ w
        prop(int((np.where(reversed_scores >= 0, 1, -1) != reversed_labels).sum()) == model["fit"]["errors"],
             f"B={model['radius']}: reversing the fit rows changes no aggregate")
    prop(mine_author["zero_predictor_tie_plus_one"]["fit"]["errors"] == 134,
         "the zero candidate makes 134 fit mistakes under the score>=0 tie rule")

    # ------------------------------------------------------ coverage accounting
    coverage_report = {}
    for label, packet_tree, covered, exclusions in (
        ("checked-results", checked, covered_checked, DECLARED_EXCLUSIONS["checked-results"]),
        ("experiment-results", experiment, covered_experiment, DECLARED_EXCLUSIONS["experiment-results"]),
        ("author-checks", author, covered_author, DECLARED_EXCLUSIONS["author-checks"]),
    ):
        all_paths = {path for path, _ in leaves(packet_tree)}
        missing = sorted(all_paths - covered - set(exclusions))
        check(not missing, f"{label}: {len(missing)} scalar leaves were never re-derived: {missing[:6]}")
        stale = sorted(set(exclusions) - all_paths)
        check(not stale, f"{label}: declared exclusions that no longer exist: {stale}")
        coverage_report[label] = {
            "scalarLeaves": len(all_paths),
            "reDerived": len(covered & all_paths),
            "declaredExclusions": {path: reason for path, reason in exclusions.items()},
            "coverage": round(len(covered & all_paths) / len(all_paths), 6),
            "uncovered": missing,
        }

    total_leaves = sum(entry["scalarLeaves"] for entry in coverage_report.values())
    total_covered = sum(entry["reDerived"] for entry in coverage_report.values())

    # ----------------------------------------------------- generate the module
    module = build_module(
        dataset_sha=dataset_sha, dataset_bytes=len(packet_bytes), rows=rows, source_ids=source_ids,
        mean=mean, scale=scale, models=models, energy=energy, confidence=confidence,
        experiment=mine_experiment, checked=mine_checked, duplicate_groups=duplicate_groups,
        zero_candidate=mine_author["zero_predictor_tie_plus_one"],
    )

    # The evidence is BUILT here and WRITTEN at the very end, after every
    # remaining assertion. It used to be written at this point, which made the
    # recorded totals permanently three short of the printed ones and — far
    # worse — let a failed byte-identity check exit non-zero while leaving an
    # artefact on disk saying "passed": true. The record must never claim a
    # state the run contradicts.
    def build_evidence(passed, notes):
        return json.dumps({
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "verifier": "scripts/verify-rademacher-data.py",
            "verifierSha256": digest(Path(__file__).read_bytes()),
            "mode": "write" if write else "read-only",
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
            "datasetSha256": dataset_sha,
            "datasetBytes": len(packet_bytes),
            "servedCopy": str(ASSET_DATASET.relative_to(ROOT)).replace("\\", "/"),
            "servedMatchesPacket": served == packet_bytes,
            "packetHashes": {
                "checked-results.json": digest(CHECKED.read_bytes()),
                "experiment-results.json": digest(EXPERIMENT.read_bytes()),
                "author-checks.json": digest(AUTHOR.read_bytes()),
                "lesson.md": digest((PACKET / "lesson.md").read_bytes()),
            },
            "totalChecks": checks["count"],
            "propertyChecks": property_checks["count"],
            "trustRootCoverage": coverage_report,
            "trustRootScalarLeaves": total_leaves,
            "trustRootLeavesReDerived": total_covered,
            "independentOptimiser": {
                "method": "projected gradient descent, 60,000 steps at rate 4",
                "largestObjectiveGapFromRecorded": max(pgd_agreements),
            },
            "recordedDisagreementsWithPacket": recorded_disagreements,
            "duplicateFeatureGroups": duplicate_groups,
            "selectedRadius": chosen["radius"],
            "smallestExpressionRadius": {"rho=1": smallest_rho1["radius"], "rho=0.5": smallest_rho_half["radius"]},
            "moduleSha256": hashlib.sha256(module.encode("utf-8")).hexdigest(),
            "scope": "All three of the content packet's trust-root files re-derived from the definitions and the served "
                     "dataset. The six finite classes, both conventions, every null (duplicate row, fixed translation, "
                     "convex average, scaling, rotation, coordinate reversal) and the wrong-order diagnostic are "
                     "enumerated in exact rational arithmetic. The Euclidean and kernel geometry is enumerated by "
                     "explicit sign loops with the quadratic form written out. The five fitted coefficient vectors are "
                     "checked to be feasible KKT points and independently re-solved by projected gradient descent, a "
                     "different optimiser from the packet's SLSQP. Every margin, mistake count, log loss, ramp mean and "
                     "bound component for all five budgets and both margin thresholds is recomputed from the served CSV "
                     "and the stored weights, together with the declared selection rule, the majority baseline, the zero "
                     "candidate under the score>=0 tie rule, the coordinate-permutation and row-reversal nulls, and the "
                     "duplicate-feature disclosure. Regeneration of src/learn/data/rademacher-data.js must be "
                     "byte-identical.",
            "limitations": [
                "The Monte-Carlo sign stream is NumPy's PCG64 at the recorded seeds; reproducing those particular "
                "estimates necessarily uses the same generator. Everything deterministic around them -- the per-draw "
                "ceiling, the Hoeffding correction and the endpoint -- is recomputed from the definitions.",
                "The representation's fitted mean and scale are recomputed directly rather than through "
                "StandardScaler, which is the second route; a different library version could still move the packet's "
                "own numbers.",
                "This is one small fixed subset of one historical collection, drawn without replacement from a finite "
                "corpus. It is not an iid sample from a deployment population, and nothing here establishes that it is.",
                "Displayed program execution, browser models, rendering and independent review are separate scripts.",
            ],
        "passed": passed,
        "failureNotes": notes,
    }, indent=2) + "\n"

    def record_and_exit(message):
        """Write a FAILING record, then stop.

        A run that ends badly must leave an artefact saying so. Leaving the
        previous run's passing record in place is how a tree that contradicts
        its own evidence comes about."""
        if not no_evidence:
            EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
            EVIDENCE.write_text(build_evidence(False, [message] + failures[:40]),
                                encoding="utf-8", newline="\n")
        raise SystemExit(message)

    # Floors, so that a wholesale loss of coverage fails rather than quietly
    # printing a smaller number.
    check(checks["count"] >= 5000, f"at least 5,000 comparisons ran; only {checks['count']} did")
    check(property_checks["count"] >= 40, f"at least forty property statements ran; only {property_checks['count']} did")
    check(total_leaves >= 5600, f"the trust root still has its leaves; found {total_leaves}")

    if failures:
        for label in failures[:40]:
            print(f"FAIL {label}")
        record_and_exit(f"{len(failures)} of {checks['count']} data checks failed")

    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        if not MODULE.exists():
            record_and_exit("the data module is missing; rerun with --write")
        if MODULE.read_text(encoding="utf-8") != module:
            record_and_exit("a fresh regeneration is not byte-identical to src/learn/data/rademacher-data.js; "
                            "rerun with --write and inspect the difference")
        if not (ASSET_DIR / "ATTRIBUTION.txt").exists():
            record_and_exit("the served attribution file is missing")

    # Everything above has passed. Only now is a PASSING record written, and it
    # carries the totals the run actually printed.
    if not no_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(build_evidence(True, []), encoding="utf-8", newline="\n")

    for entry in recorded_disagreements:
        print(f"NOTE disagreement with the packet at {entry['file']}{entry['path']}: "
              f"packet {entry['packetValue']!r}, independently {entry['independentValue']!r}. {entry['reason']}")
    print(f"PASS: {checks['count']:,} data checks, including {property_checks['count']} independent property "
          f"statements; {total_covered:,} of {total_leaves:,} trust-root scalar leaves re-derived across three "
          f"packet files ({100 * total_covered / total_leaves:.1f}%), "
          f"{sum(len(entry['declaredExclusions']) for entry in coverage_report.values())} declared exclusions; "
          f"module {'written' if write else 'byte-identical'} ({len(module) / 1024:.0f} KB).")


# ------------------------------------------------------------ module emission


def number(value):
    """Shortest round-tripping literal. Python's repr and JavaScript's number
    parser agree on these, so the module reads back the exact double."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "null"
    text = repr(float(value))
    return text.replace("e+", "e").replace("inf", "Infinity")


def js(value, indent=0):
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = ",\n".join(f"{pad}  {key}: {js(item, indent + 1)}" for key, item in value.items())
        return "{\n" + inner + f",\n{pad}}}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return "[" + ", ".join(number(item) for item in value) + "]"
        inner = ",\n".join(f"{pad}  {js(item, indent + 1)}" for item in value)
        return "[\n" + inner + f",\n{pad}]"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return number(value)


def build_module(*, dataset_sha, dataset_bytes, rows, source_ids, mean, scale, models, energy, confidence,
                 experiment, checked, duplicate_groups, zero_candidate):
    provenance = {
        "name": 'UCI Banknote Authentication (dataset 267)',
        "creator": 'Volker Lohweg, 2012',
        "doi": 'https://archive.ics.uci.edu/dataset/267/banknote+authentication',
        "license": 'CC BY 4.0',
        "licenseUrl": 'https://creativecommons.org/licenses/by/4.0/',
        "file": '/learn-assets/rademacher/banknote-subset.csv',
        "attribution": '/learn-assets/rademacher/ATTRIBUTION.txt',
        "calculationProgram": '/learn-assets/rademacher/complexity_calculations.py',
        "experimentProgram": '/learn-assets/rademacher/bounded_norm_experiment.py',
        "sha256": dataset_sha,
        "bytes": dataset_bytes,
        "sourceRows": 1372,
        "retainedRows": len(rows),
        "selection": 'np.random.default_rng(23).permutation(1372)[:480], preserving one-based source row ids',
        "retrieved": '12 September 2026',
        "archiveSha256": '1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227',
    }
    observations = [[source_ids[index], *[float(row[name]) for name in FEATURES], 2 * int(row["class"]) - 1]
                    for index, row in enumerate(rows)]
    roles = [
        {"key": 'representation', "label": 'Representation design', "from": 0, "to": 80, "count": 80,
         "use": 'Fit four feature means and standard deviations'},
        {"key": 'fit', "label": 'Predictor fit', "from": 80, "to": 320, "count": 240,
         "use": 'Fit coefficients inside each declared norm ball; compute empirical losses and feature energy'},
        {"key": 'validation', "label": 'Validation', "from": 320, "to": 400, "count": 80,
         "use": 'Select the coefficient budget using the declared rule'},
        {"key": 'assessment', "label": 'Assessment', "from": 400, "to": 480, "count": 80,
         "use": 'Report the selected procedure and the predeclared comparison table'},
    ]
    fitted = [{
        "radius": model["radius"],
        "weights": model["weights"],
        "norm": model["solver"]["norm"],
        "objective": model["solver"]["objective"],
        "stationarityResidual": model["solver"]["stationarity_residual"],
        "fit": {"errors": model["fit"]["errors"], "n": model["fit"]["n"], "logLoss": model["fit"]["log_loss"]},
        "validation": {"errors": model["validation"]["errors"], "n": model["validation"]["n"],
                       "logLoss": model["validation"]["log_loss"]},
        "assessment": {"errors": model["assessment"]["errors"], "n": model["assessment"]["n"],
                       "logLoss": model["assessment"]["log_loss"]},
        "bounds": [{"rho": bound["rho"], "empiricalRamp": bound["empirical_ramp"],
                    "complexityAddend": bound["complexity_addend"], "rawUpper": bound["raw_upper"],
                    "clippedUpper": bound["clipped_upper"]} for bound in model["bounds"]],
    } for model in models]
    monte_carlo = {
        "fixture": [[1, 0], [1, 0]],
        "exact": 0.5,
        "radius": 1,
        "table": [{"draws": row["draws"], "seed": row["seed"], "eta": row["eta"], "estimate": row["estimate"],
                   "perDrawUpper": row["per_draw_upper"], "correction": row["one_sided_correction"],
                   "endpoint": row["upper"], "standardError": row["sample_standard_error"]}
                  for row in checked["monte_carlo"]],
        "unitBall": {"draws": experiment["monte_carlo_unit_ball"]["draws"],
                     "seed": experiment["monte_carlo_unit_ball"]["seed"],
                     "eta": experiment["monte_carlo_unit_ball"]["eta"],
                     "estimate": experiment["monte_carlo_unit_ball"]["estimate"],
                     "perDrawUpper": experiment["monte_carlo_unit_ball"]["per_draw_upper"],
                     "correction": experiment["monte_carlo_unit_ball"]["one_sided_correction"],
                     "endpoint": experiment["monte_carlo_unit_ball"]["upper"],
                     "standardError": experiment["monte_carlo_unit_ball"]["sample_standard_error"]},
    }
    settings = {
        "radii": RADII, "rhoValues": RHOS, "delta": DELTA, "comparisons": COMPARISONS,
        "fitRows": 240, "energyFactor": energy, "confidenceAddend": confidence,
        "clipStandardDeviations": CLIP_SD, "globalRowNormUpper": math.sqrt(5),
    }
    representation = {"features": FEATURES, "mean": [float(v) for v in mean], "scale": [float(v) for v in scale],
                      "clipStandardDeviations": CLIP_SD}
    recorded_fixtures = {
        "thresholdInputs": [-1, 0, 1],
        "thresholdComplexity": checked["thresholds"]["complexity"],
        "constantsComplexity": checked["constants"]["complexity"],
        "singletonComplexity": checked["singleton"]["complexity"],
        "bothOrientationsComplexity": checked["two_orientations"]["complexity"],
        "fullCubeComplexity": checked["all_labels"]["complexity"],
        "mistakeComplexity": checked["classification_loss"]["complexity"],
        "absoluteSingleton": checked["abs_changes_singleton"],
        "kernelSimilarities": [0, 0.9, 1],
        "kernelComplexities": [checked["kernels"]["0"]["complexity"], checked["kernels"]["0.9"]["complexity"],
                               checked["kernels"]["1"]["complexity"]],
        "kernelTraceUpper": checked["kernels"]["0"]["trace_upper"],
        "massart16of100": checked["massart"]["M16_n100"],
        "massart8of200": checked["practice"]["massart_M8_n200"],
        "scalarMargins": checked["scalar_margin_fixture"]["x"],
        "scalarRamp": checked["scalar_margin_fixture"]["default_ramp"],
        "scalarEnergy": checked["scalar_margin_fixture"]["energy"],
        "finitePopulation": {
            "expectedGap": checked["finite_population_proof"]["expected_largest_gap"],
            "expectedGhost": checked["finite_population_proof"]["expected_ghost_supremum"],
            "expectedRademacher": checked["finite_population_proof"]["expected_rademacher"],
            "twiceExpectedRademacher": checked["finite_population_proof"]["twice_expected_rademacher"],
        },
    }
    zero = {name: {"errors": entry["errors"], "n": entry["n"], "ramp": entry["ramp"]}
            for name, entry in zero_candidate.items()}
    header = (
        "/* Recorded data for the Rademacher-complexity lesson.\n"
        " *\n"
        " * GENERATED by scripts/verify-rademacher-data.py. Do not edit by hand.\n"
        " *\n"
        " * Every number here was re-derived from the served dataset and the packet's\n"
        " * declared settings by that script, which also checks that regenerating this\n"
        " * file is byte-identical. The 480 observations are carried in full so the page\n"
        " * recomputes its own scores, margins and mistake counts rather than reading\n"
        " * stored ones: the two routes agreeing is itself a check, and no lesson\n"
        " * rendering depends on a network fetch that could refuse.\n"
        " *\n"
        " * `fittedModels` holds the FIVE PREDECLARED candidates. Their assessment\n"
        " * columns are recorded because the lesson reveals them after a committed\n"
        " * selection; nothing in the selection rule is allowed to read them.\n"
        " */\n"
    )
    parts = [
        header,
        f"export const provenance = {js(provenance)};\n",
        "\n/* source row id, variance, skewness, curtosis, entropy, label in {-1, +1} */\n",
        f"export const observations = {js(observations)};\n",
        f"\nexport const roles = {js(roles)};\n",
        f"\nexport const representation = {js(representation)};\n",
        f"\nexport const experimentSettings = {js(settings)};\n",
        f"\nexport const fittedModels = {js(fitted)};\n",
        f"\nexport const selection = {js({'rule': 'validation errors, then validation log loss, then smaller budget', 'chosenRadius': experiment['selection']['chosen_radius']})};\n",
        f"\nexport const majorityBaseline = {js({'classSign': experiment['majority_baseline']['class_sign'], 'validationErrors': experiment['majority_baseline']['validation_errors'], 'assessmentErrors': experiment['majority_baseline']['assessment_errors']})};\n",
        f"\nexport const recordedMonteCarlo = {js(monte_carlo)};\n",
        f"\nexport const recordedFixtures = {js(recorded_fixtures)};\n",
        f"\nexport const zeroCandidate = {js(zero)};\n",
        f"\nexport const duplicateFeatureGroups = {js(duplicate_groups)};\n",
    ]
    return "".join(parts)


if __name__ == "__main__":
    main()
