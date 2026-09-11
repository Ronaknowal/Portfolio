"""Independent NumPy/Fraction oracles for actual JS and displayed programs."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
subprocess.run(["node", "scripts/export-recommender-model-cases.mjs"], check=True)
cases = json.loads((ROOT / "scratch/recommender-native/model-cases.json").read_text())
total = 0


def close(actual, expected, atol=2e-9):
    global total
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-9)
    total += 1


for case in cases["feedback"]:
    data, result = case["input"], case["result"]
    explicit = data["mode"] == "explicit"
    included = not explicit or data["rating"] is not None
    target = data["rating"] if explicit else int(data["count"] > 0)
    weight = int(included) if explicit else 1+2*data["count"]
    close(result["loss"], weight*(target-data["prediction"])**2 if included else 0)

ratings = np.array([[5, 3, np.nan, 1, np.nan, 4, np.nan], [4, np.nan, 4, 1, 2, np.nan, 3], [np.nan, 3, np.nan, np.nan, 4, 3, np.nan], [1, np.nan, np.nan, 5, 4, np.nan, 2], [np.nan, 1, 5, 4, np.nan, np.nan, 3]])
means = np.nanmean(ratings, axis=1)
for case in cases["neighbors"]:
    data, result = case["input"], case["result"]
    selected = []
    for item in np.flatnonzero(np.isfinite(ratings[0])):
        common = np.isfinite(ratings[:, item]) & np.isfinite(ratings[:, data["item"]])
        left, right = ratings[common, data["item"]], ratings[common, item]
        if data["centered"]:
            left, right = left-means[common], right-means[common]
        norm = np.linalg.norm(left)*np.linalg.norm(right)
        weight = left@right/norm*common.sum()/(common.sum()+data["shrinkage"]) if common.sum() >= data["minimumOverlap"] and norm else 0.
        if weight > 0 or data["signed"] and weight != 0:
            selected.append((int(item), weight, ratings[0, item]-(means[0] if data["centered"] else 0)))
    selected.sort(key=lambda row: (-abs(row[1]), row[0]))
    selected = selected[:3]
    expected = (means[0] if data["centered"] else 0) + sum(w*r for _, w, r in selected)/sum(abs(w) for _, w, _ in selected) if selected else means[0]
    close(result["prediction"], expected)

for key in ("updates", "pairs"):
    for case in cases[key]:
        data, result = case["input"], case["result"]
        if key == "updates":
            state = np.r_[data["userFactors"], data["itemFactors"], data["userBias"], data["itemBias"]]
            def loss(values):
                error = data["rating"]-3-values[4]-values[5]-values[:2]@values[2:4]
                return .5*(error**2+data["penalty"]*(values@values))
            after = np.r_[result["after"]["userFactors"], result["after"]["itemFactors"], result["after"]["userBias"], result["after"]["itemBias"]]
        else:
            state = np.r_[data["user"], data["positive"], data["negative"]]
            def loss(values):
                return np.logaddexp(0., -values[:2]@(values[2:4]-values[4:])) + .5*data["penalty"]*(values@values)
            after = np.r_[result["after"]["user"], result["after"]["positive"], result["after"]["negative"]]
        step = 1e-5
        # Five-point derivative; independent of the implemented closed-form gradient.
        gradient = np.array([(loss(state-2*step*e)-8*loss(state-step*e)+8*loss(state+step*e)-loss(state+2*step*e))/(12*step) for e in np.eye(6)])
        close(after, state-data["rate"]*gradient, atol=2e-8)
        close(result["loss"], loss(state))
        close(result["nextLoss"], loss(after))

items = np.array([[1., 0.], [0., 1.], [1., 1.]])
for case in cases["implicit"]:
    data, result = case["input"], case["result"]
    counts = np.array(data["counts"])
    confidence = np.where((counts > 0) | data["includeMissing"], 1+data["alpha"]*counts, 0.)
    target = counts > 0
    design = np.vstack((np.sqrt(confidence)[:, None]*items, np.sqrt(data["penalty"])*np.eye(2)))
    response = np.r_[np.sqrt(confidence)*target, [0., 0.]]
    fitted = np.linalg.lstsq(design, response, rcond=None)[0]
    close(result["user"], fitted)
    optimum = np.linalg.norm(design@fitted-response)**2
    close(result["objective"], optimum)
    for contour in result["contours"]:
        points = np.array(contour["points"])
        values = np.sum((points@design.T-response)**2, axis=1)
        close(values, optimum+contour["excess"])

for case in cases["rotations"]:
    result = case["result"]
    close(result["scores"], result["originalScores"])
    close(np.linalg.norm(result["items"], axis=1), np.linalg.norm(result["originalItems"], axis=1))

for case in cases["rankings"]:
    data, result = case["input"], case["result"]
    grades, order, cutoff = data["grades"], data["order"], data["cutoff"]
    positions = [rank for rank, item in enumerate(order[:cutoff], 1) if grades[item] > 0]
    relevant = sum(grade > 0 for grade in grades)
    close(result["precision"], len(positions)/cutoff)
    if not relevant:
        assert all(result[key] is None for key in ("recall", "ndcg", "reciprocalRank", "averagePrecision", "bestCandidateRecall"))
        continue
    close(result["recall"], len(positions)/relevant)
    close(result["averagePrecision"], sum(index/rank for index, rank in enumerate(positions, 1))/min(cutoff, relevant))
    close(result["reciprocalRank"], 1/positions[0] if positions else 0)
    dcg = sum((2**grades[item]-1)/math.log2(rank+1) for rank, item in enumerate(order[:cutoff], 1))
    ideal = sum((2**grade-1)/math.log2(rank+1) for rank, grade in enumerate(sorted(grades, reverse=True)[:cutoff], 1))
    close(result["ndcg"], dcg/ideal)
    close(result["bestCandidateRecall"], min(cutoff, sum(grades[item] > 0 for item in order))/relevant)

for case in cases["policies"]:
    data, result = case["input"], case["result"]
    logging = [F(str(data["loggingA"])), 1-F(str(data["loggingA"]))]
    target = [F(str(data["targetA"])), 1-F(str(data["targetA"]))]
    quality = [F(str(data["qualityA"])), F(4, 5)]
    supported = all(p > 0 or target[index] == 0 for index, p in enumerate(logging))
    assert result["supported"] == supported
    if not supported:
        assert result["expectation"] is result["variance"] is None
        continue
    expectation, second = F(0), F(0)
    for item in range(2):
        if not logging[item]:
            continue
        for reward in (0, 1):
            mass = logging[item]*(quality[item] if reward else 1-quality[item])
            weighted = reward*target[item]/logging[item]
            expectation += mass*weighted
            second += mass*weighted**2
    close(result["expectation"], float(expectation))
    close(result["variance"], float(second-expectation**2))

raw = subprocess.check_output(["node", "--input-type=module", "-e", "import {recommenderExamples} from './src/learn/data/recommender-examples.js';console.log(JSON.stringify(recommenderExamples));"], text=True, encoding="utf-8")
examples = json.loads(raw)
namespaces = {}
for key, example in examples.items():
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(example["code"], f"displayed-{key}.py", "exec"), namespace)
    assert stdout.getvalue().rstrip() == example["expected"], key
    namespaces[key] = namespace

# Execute the substantial changed tasks against the actual displayed helpers.
gradient = namespaces["gradient"]
gradient.update(mean=3., rating=2., penalty=.2, rate=.1)
state = np.array([.2, .4, -.3, .5, .05, .1])
updated = state-.1*gradient["gradient"](state)
close(updated, [.2347, .3275, -.3198, .4384, -.08, -.031])
close(gradient["objective"](state), .8873)
close(gradient["objective"](updated), .5048380542293619)
pair = namespaces["bpr"]
pair_state = np.array([1., .5, .5, 1., 1.5, 0.])
pair_new = pair_state-.1*pair["pair_gradient"](pair_state, .1)
close(pair_new[:2]@(pair_new[2:4]-pair_new[4:]), -.2166189241395764)
ranking = namespaces["ranking"]["metrics"]
before = ranking([4, 1, 2, 0, 3], [1, 1, 0, 0, 0], 2)
after = ranking([1, 4, 2, 0, 3], [1, 1, 0, 0, 0], 2)
close([before["average_precision"], after["average_precision"]], [.25, .5])
close([before["ndcg"], after["ndcg"]], [.38685280723454163, .6131471927654584])
changed_block = np.linalg.solve(np.array([[5., 2.], [2., 8.]]), [2., 6.])
close(changed_block, [1/9, 13/18])
assert (1-F(1, 4))*F(4, 5)+F(1, 4)*F(2, 5) == F(7, 10)

sources = ["src/learn/data/topics/recommender-systems-collaborative-filtering-matrix-factorization.jsx", "src/learn/data/recommender-models.js", "src/learn/data/recommender-examples.js", "src/learn/components/lesson-labs/RecommenderLabs.jsx", "src/learn/components/lesson-labs/RecommenderFigures.jsx", "src/learn/components/lesson-labs/recommender-labs.css", "src/learn/data/curriculum/blueprints/recommender-systems-collaborative-filtering-matrix-factorization.js"]
record = dict(timestamp=datetime.now(timezone.utc).isoformat(), assertions=total, programs=len(examples), cases={key: len(value) for key, value in cases.items()}, sourceHashes={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources}, changedPractice=dict(gradient=updated.tolist(), reversedPairGap=float(pair_new[:2]@(pair_new[2:4]-pair_new[4:])), rankingBefore=before, rankingAfter=after, implicitFactors=changed_block.tolist()))
(ROOT/"scratch/recommender-native/verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(f"Passed {total} independent numerical assertions and all {len(examples)} actual programs.")
