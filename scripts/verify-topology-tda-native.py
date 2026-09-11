"""Execute the actual published programs and compare changed inputs with checked fixtures."""
import contextlib
import hashlib
import io
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp

mp.mp.dps = 85
root = Path.cwd()
examples = json.loads(subprocess.run(
    ["node", "--input-type=module", "-e", "import {topologyTdaExamples as e} from './src/learn/data/topology-tda-examples.js'; console.log(JSON.stringify(e));"],
    check=True, capture_output=True, text=True, encoding="utf-8").stdout)
fixtures = json.loads(Path("scratch/topology-tda-review/model-fixtures.json").read_text())
archive = json.loads(Path("docs/teaching/evidence/topology-tda-original-content.json").read_text(encoding="utf-8"))
original = next(block["code"] for block in archive["blocks"] if block["language"] == "python")
assert examples["original"]["code"] == original
assert examples["original"]["expected"] == next(block["code"] for block in archive["blocks"] if block["language"] == "output")
namespaces, programs = {}, []
for name, example in examples.items():
    namespace, stream = {}, io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(example["code"], name + ".py", "exec"), namespace)
    assert stream.getvalue().rstrip("\n") == example["expected"], name
    namespaces[name] = namespace
    programs.append({"name": name, "codeSha256": hashlib.sha256(example["code"].encode()).hexdigest(), "stdout": stream.getvalue()})

assertions = 0
def check(condition, context):
    global assertions
    assert condition, context
    assertions += 1

def close(actual, expected, context):
    check(math.isclose(actual, expected, rel_tol=3e-10, abs_tol=1e-323), (context, actual, expected))

# Enumerate the entire row space independently of Gaussian elimination.
for mask in range(512):
    rows = [(mask >> (3*row)) & 7 for row in range(3)]
    span = {0}
    for row in rows:
        span |= {value ^ row for value in span}
    matrix = [[(row >> column) & 1 for column in range(3)] for row in rows]
    check(namespaces["boundaries"]["rank_f2"](matrix) == len(span).bit_length()-1, ("exact row span", mask))

pairs = list(itertools.combinations(range(5), 2))
for case in fixtures["graphCases"]:
    edges = {pair for index, pair in enumerate(pairs) if case["mask"] >> index & 1}
    facets = list(edges) + [(vertex,) for vertex in range(5)]
    facets += [triangle for triangle in itertools.combinations(range(5), 3) if all(edge in edges for edge in itertools.combinations(triangle, 2))]
    actual = namespaces["boundaries"]["betti_numbers"](namespaces["boundaries"]["closure"](facets))
    check(list(actual) == case["betti"][:3], ("native graph homology", case["mask"]))

def normalized_bars(bars):
    # A sub-ULP difference in two distance implementations can split an exact
    # tied event. Compare resolvable positive bars after 11-decimal rounding.
    return sorted((k, round(b,11), round(d,11) if math.isfinite(d) else math.inf)
                  for k,b,d in bars if k < 2 and (math.isinf(d) or round(d,11) > round(b,11)))

for case in fixtures["pointCases"]:
    expected = [(bar["dimension"], bar["birth"], math.inf if bar["death"] is None else bar["death"]) for bar in case["intervals"]]
    observed = namespaces["persistence"]["rips_diagram"](case["points"])
    check(normalized_bars(observed) == normalized_bars(expected), ("native Rips intervals", case["name"], case["scale"]))

for case in fixtures["pixels"]:
    values = [0 if case["mask"] >> pixel & 1 else 1 for pixel in range(9)]
    counts, betti = namespaces["pixels"]["pixel_betti"](values, 0)
    check(list(betti) == case["result"]["betti"], ("native cubical counts", case["mask"]))

for case in fixtures["matching"]:
    value, assignment = namespaces["matching"]["diagram_distance"](case["first"], case["second"], float(case["power"]))
    close(value, case["result"]["value"], "native matching")

for case in fixtures["normal"]:
    lower, upper = mp.mpf(case["lower"]), mp.mpf(case["upper"])
    width, midpoint = upper-lower, (upper+lower)/2
    nearest = min(max(mp.mpf(0), lower), upper)
    expected = width * mp.exp(-nearest**2/2)/mp.sqrt(2*mp.pi) * mp.quad(lambda t: mp.exp(-((lower+width*t)**2-nearest**2)/2), [0,1])
    close(namespaces["features"]["normal_interval"](case["lower"], case["upper"]), float(expected), "native Gaussian quadrature")

for case in fixtures["imageCases"]:
    if "xEdges" in case:
        continue  # Native teaching program declares its fixed 4x4 unit grid.
    actual = namespaces["features"]["pixel_integrals"](case["diagram"], case["bandwidth"])
    for (_, contributions), expected_pixel in zip(actual, case["result"]["pixels"]):
        for observed, expected in zip(contributions, expected_pixel["contributions"]):
            close(observed, expected, "native integrated pixel")

record = {"passed": True, "verifiedAt": datetime.now(timezone.utc).isoformat(), "python": sys.version,
          "programs": programs, "originalPreservedExactly": True, "changedInputAssertions": assertions,
          "finiteRipsComparisonRounding": 11, "limits": "Actual published stdlib programs; finite case checks. The model fixtures have a separately executed RREF/triangulation/assignment/high-precision oracle. This does not certify arbitrary complexes or statistical inference."}
Path("docs/teaching/evidence/topology-tda-native-verification.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({key:value for key,value in record.items() if key != "programs"}))
