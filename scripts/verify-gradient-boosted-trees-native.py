"""Independent objective enumeration, rational arithmetic and actual-helper checks."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction as F
from itertools import combinations, product
import hashlib
import io
import json
from pathlib import Path
import runpy
import math
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "scratch/gradient-boosted-trees-verification"
data = json.loads((FOLDER / "oracle-input.json").read_text())
counts = {}
maximum_error = 0.0


def close(actual, expected, label, tolerance=2e-10):
    global maximum_error
    error = abs(float(actual) - float(expected)) / max(1, abs(float(expected)))
    assert math.isfinite(error) and error < tolerance, (label, actual, expected, error)
    maximum_error = max(maximum_error, error)
    counts[label] = counts.get(label, 0) + 1


def exact_leaf(values):
    total = sum(map(F, values))
    center = total / len(values)
    return center, sum((F(value) - center) ** 2 for value in values)


def inspect_tree(node, x, y, limit):
    indices = node["indices"]
    center, cost = exact_leaf([y[index] for index in indices])
    close(node["value"], center, "tree leaf-center")
    if node["depth"] == limit or cost == 0:
        assert node["leaf"]
        return
    # Enumerate legal *partitions*, evaluate objective directly with exact Fractions.
    partitions = {tuple(index for index in indices if x[index] <= threshold) for threshold in set(x[index] for index in indices)}
    options = []
    for left in partitions:
        right = tuple(index for index in indices if index not in left)
        if not left or not right:
            continue
        options.append((exact_leaf([y[i] for i in left])[1] + exact_leaf([y[i] for i in right])[1], left))
    optimal = min([cost] + [option[0] for option in options])
    if node["leaf"]:
        close(cost, optimal, "unsplit optimality")
    else:
        left, right = node["left"], node["right"]
        chosen = exact_leaf([y[i] for i in left["indices"]])[1] + exact_leaf([y[i] for i in right["indices"]])[1]
        close(chosen, optimal, "all-partition objective")
        inspect_tree(left, x, y, limit)
        inspect_tree(right, x, y, limit)


for case in data["trees"]:
    inspect_tree(case["result"], case["x"], case["y"], case["depth"])
for fitted in data["boosts"]:
    for before, after in zip(fitted["stages"], fitted["stages"][1:]):
        leaf_nodes = []
        def collect(node):
            if node["leaf"]:
                leaf_nodes.append(node)
            else:
                collect(node["left"])
                collect(node["right"])
        collect(after["tree"])
        projection_norm = sum(len(leaf["indices"]) * leaf["value"] ** 2 for leaf in leaf_nodes)
        predicted_sse = len(fitted["y"]) * before["mse"] - fitted["rate"] * (2-fitted["rate"]) * projection_norm
        close(len(fitted["y"]) * after["mse"], predicted_sse, "projection identity")
        for index, prediction in enumerate(after["prediction"]):
            close(prediction, after["before"][index] + fitted["rate"] * after["correction"][index], "additive prediction")

for case in data["newton"]:
    stats = case["statistics"]
    # Recover regularization from the actual stated fixture construction.
    # Curvature + weight are checked below using each enumerated parameter tuple.
    for name in ["parent", "left", "right"]:
        leaf = stats[name]
        close(leaf["objective"], -leaf["improvement"], "leaf objective dual form")
    close(stats["netGain"], -stats["left"]["objective"]-stats["right"]["objective"]+stats["parent"]["objective"]-stats["gamma"], "direct objective difference")
index = 0
for kind, split, lam, alpha, gamma in product(["square", "logistic", "confident"], [1,2,3,4], [0,.25,1,5], [0,.25,2,5], [0,.5,3]):
    case = data["newton"][index]
    index += 1
    for name in ["parent", "left", "right"]:
        leaf = case["statistics"][name]
        objective = lambda value: leaf["G"]*value + (leaf["H"]+lam)*value**2/2 + alpha*abs(value)
        if abs(leaf["G"]) <= alpha:
            optimum = 0.0
        else:
            result = minimize_scalar(objective, bounds=(-100,100), method="bounded", options={"xatol":1e-12})
            optimum = result.x
        close(objective(leaf["weight"]), objective(optimum), "independent scalar minimum", 3e-10)
    if kind != "square":
        score = np.asarray(case["before"])
        target = np.asarray(case["y"])
        loss = lambda value: np.logaddexp(0, value)-target*value
        step = 1e-4
        gradient = (loss(score+step)-loss(score-step))/(2*step)
        hessian = (loss(score+step)-2*loss(score)+loss(score-step))/step**2
        for actual, expected in zip(case["gradients"], gradient):
            close(actual, expected, "numeric logistic gradient", 2e-9)
        for actual, expected in zip(case["hessians"], hessian):
            close(actual, expected, "numeric logistic curvature", 2e-7)

for case in data["histogram"]:
    target = list(map(F, case["y"]))
    parent = exact_leaf(target)[1]/2
    gains = []
    for candidate in case["candidates"]:
        left = [target[index] for index in candidate["leftIndices"]]
        right = [target[index] for index in candidate["rightIndices"]]
        gain = parent - (exact_leaf(left)[1]+exact_leaf(right)[1])/2
        close(candidate["netGain"], gain, "histogram direct loss")
        gains.append(gain)
    close(case["best"]["netGain"], max(gains), "histogram exhaustive optimum")
for case in data["sampling"]:
    weight = F(len(case["remaining"]), len(case["selected"]))
    expected = []
    for subset in combinations(case["remaining"], len(case["selected"])):
        expected.append(sum(case["gradients"][index] for index in case["retained"])+weight*sum(case["gradients"][index] for index in subset))
    close(case["average"], sum(expected)/len(expected), "finite sampling mean")
    close(case["averageSquare"], sum(value**2 for value in expected)/len(expected), "finite sampling second moment")
for case in data["categories"]:
    for prefix in case["prefixes"]:
        row = prefix["index"]
        earlier = case["order"][:case["order"].index(row)]
        values = [case["rows"][i]["target"] for i in earlier if case["rows"][i]["category"] == case["rows"][row]["category"]]
        exact = (sum(values)+F(case["smoothing"])*F(1,2))/(len(values)+F(case["smoothing"]))
        close(prefix["value"], exact, "ordered prefix enumeration")

# Independently price splits through NumPy least-squares projection matrices.
def ls_tree(x, y, depth):
    if depth == 0 or len(y) < 2:
        return float(np.mean(y))
    candidates = []
    for cut in (np.unique(x)[:-1]+np.unique(x)[1:])/2:
        left = x <= cut
        design = np.column_stack([left, ~left]).astype(float)
        weights, *_ = np.linalg.lstsq(design, y, rcond=None)
        candidates.append((float(np.sum((y-design@weights)**2)), cut, left))
    baseline = float(np.sum((y-y.mean())**2))
    if not candidates or min(item[0] for item in candidates) >= baseline:
        return float(y.mean())
    _, cut, left = min(candidates, key=lambda item: (item[0], item[1]))
    return cut, ls_tree(x[left], y[left], depth-1), ls_tree(x[~left], y[~left], depth-1)


def predict(tree, x):
    if isinstance(tree, float):
        return tree
    return predict(tree[1] if x <= tree[0] else tree[2], x)


for case in data["validation"]:
    fitted = case["model"]
    x, y = np.array(fitted["x"]), np.array(fitted["y"])
    vx, vy = np.array(case["validationX"]), np.array(case["validationY"])
    train, valid = np.full(len(y), y.mean()), np.full(len(vy), y.mean())
    independent_trees = []
    for point in case["curves"]:
        close(point["train"], np.mean((y-train)**2), "LS training curve", 2e-8)
        close(point["validation"], np.mean((vy-valid)**2), "LS validation curve", 2e-8)
        for plotted in case["plottedSegments"]:
            if plotted["round"] != point["round"]:
                continue
            for segment in plotted["segments"]:
                # Compare independent fitted trees inside the open pieces. Two
                # algebraically equivalent float midpoint formulas can differ
                # by one ulp exactly at a discontinuity; audit those endpoints
                # separately using the saved model's declared boundaries.
                for fraction in [.25, .75]:
                    feature = segment["left"] + fraction*(segment["right"]-segment["left"])
                    reference = y.mean()+fitted["rate"]*sum(predict(tree,feature) for tree in independent_trees)
                    close(segment["value"], reference, "LS exact plotted step", 2e-8)
                def leaf_cells(node, low=-math.inf, high=math.inf):
                    if node["leaf"]:
                        return [(low, high, node["value"])]
                    return leaf_cells(node["left"], low, min(high, node["threshold"])) + leaf_cells(node["right"], max(low, node["threshold"]), high)
                feature = segment["right"]
                reference = fitted["base"]+fitted["rate"]*sum(value for saved in fitted["trees"][:point["round"]] for low,high,value in leaf_cells(saved) if low < feature <= high)
                close(segment["value"], reference, "saved-boundary interval-membership endpoint", 2e-8)
        tree = ls_tree(x, y-train, fitted["depth"])
        independent_trees.append(tree)
        train += fitted["rate"]*np.array([predict(tree,value) for value in x])
        valid += fitted["rate"]*np.array([predict(tree,value) for value in vx])

# Execute the actual published-source helpers, including old edge cases.
program_records = json.loads((FOLDER / "program-results.json").read_text(encoding="utf-8"))
for record in program_records["programs"]:
    captured = io.StringIO()
    with redirect_stdout(captured):
        namespace = runpy.run_path(str(ROOT/record["program"]))
    assert captured.getvalue().strip() == record["stdout"], record["key"]
    counts["complete program stdout"] = counts.get("complete program stdout",0)+1
    if record["key"] == "residual-corrections":
        import math
        for low in [float.fromhex('0x0.0000000000001p-1022'), 1., math.nextafter(1., 2.), -1., math.nextafter(-1., -2.), 999.]:
            high = math.nextafter(low, math.inf)
            fitted = namespace["fit_stump"]([low, high], [-2., 3.])
            assert low <= fitted["cut"] < high
            assert [namespace["predict"](fitted, x) for x in [low, high]] == [-2., 3.]
        counts["native adjacent representable features"] = 6
        fitted = namespace["fit_stump"]([1, 2, 3, 4], [-2, -2, 2, 2])
        predictions = [2+F(1,4)*namespace["predict"](fitted, x) for x in [1,2,3,4]]
        assert predictions == [1.5,1.5,2.5,2.5]
        close(sum((target-predicted)**2 for target,predicted in zip([0,0,4,4],predictions))/4, F(9,4), "changed quarter-rate practice")
    if record["key"] == "quantile-planning":
        candidates = [F(index, 10) for index in range(-10, 91)]
        losses = [sum(namespace["pinball"](value, candidate, F(3,5)) for value in [0,1,1,2,8]) for candidate in candidates]
        minimizers = [candidate for candidate, loss in zip(candidates, losses) if loss == min(losses)]
        assert minimizers == [F(index,10) for index in range(10,21)]
        counts["changed quantile decisions"] = len(candidates)
    if record["key"] == "recursive-regression":
        tree = namespace["RegressionTree"](max_depth=3).fit(np.ones((4,2)),np.array([-1.,0.,2.,7.]))
        assert np.allclose(tree.predict(np.ones((3,2))),2.)
        boosted = namespace["GradientBoostingScratch"](n_estimators=2)
        with redirect_stdout(io.StringIO()):
            boosted.fit(np.array([[0.],[1.],[2.]]),np.array([0.,1.,3.]))
            boosted.fit(np.array([[0.],[1.],[2.]]),np.array([0.,1.,3.]))
        assert len(boosted.trees) == 2
        counts["native unsplittable/refit"] = 2

for case in data["adjacent"]:
    assert case["low"] <= case["threshold"] < case["high"]
counts["JS adjacent representable features"] = len(data["adjacent"])

record = {"verifiedAt": datetime.now(timezone.utc).isoformat(), "modelSha256": data["modelSha256"],
          "examplesSha256": hashlib.sha256((ROOT/"src/learn/data/gradient-boosted-trees-examples.js").read_bytes()).hexdigest(),
          "counts": counts, "rejectedInputs": data["invalidInputsRejected"], "maximumScaledError":maximum_error}
(FOLDER/"native-results.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
