"""Complementary finite review, using series resistances and closed-form modes.

This does not rerun or relabel the author's broad browser/native suite.
"""

import contextlib
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/graph-fundamentals-independent-review"
OUT.mkdir(parents=True, exist_ok=True)


def node(script):
    return json.loads(subprocess.check_output(
        ["node", "--input-type=module", "-e", script], cwd=ROOT, encoding="utf-8"
    ))


examples = node("import {graphFundamentalsExamples as x} from './src/learn/data/graph-fundamentals-examples.js'; console.log(JSON.stringify(x));")
functions = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name + ".py", "exec"), namespace)
    functions[name] = namespace

# Opposite anchored vertices split a cycle into two independent series paths.
# The oracle uses their resistance ratios, not graph elimination or adjacency.
cases = []
for a, b, c, d in itertools.product([1, 2, 4], repeat=4):
    edges = [[0, 1, a], [1, 2, b], [2, 3, c], [3, 0, d]]
    expected = [F(8), F(8) - F(12 * b, a + b), F(-4), F(8) - F(12 * c, c + d)]
    native = functions["interpolation"]["harmonic_values"](4, edges, {0: 8, 2: -4})
    assert native == expected
    cases.append({"edges": edges, "expected": [float(x) for x in expected]})

(OUT / "cycle-inputs.json").write_text(json.dumps(cases), encoding="utf-8")
actual = node("""
import fs from 'node:fs';
import {graphHarmonicInterpolation as harmonic, graphEnergy, graphNormalizations as normalize} from './src/learn/data/graph-fundamentals-models.js';
const cases=JSON.parse(fs.readFileSync('scratch/graph-fundamentals-independent-review/cycle-inputs.json','utf8'));
const cycle=cases.map(({edges})=>({
  harmonic:harmonic(4,edges,{0:8,2:-4}),
  loops:harmonic(4,[...edges,[1,1,3],[3,3,2]],{0:8,2:-4}),
  partial:harmonic(5,edges,{0:8,2:-4}),
  energy:graphEnergy(4,edges,[8,1,-4,3]),
  reversed:graphEnergy(4,edges.map(([u,v,w])=>[v,u,w]),[8,1,-4,3]),
  normal:normalize(5,[...edges,[1,1,3],[3,3,2]])
}));
const boundaries=[Number.MIN_VALUE,1e-320,1e-310,1e-308].map(weight=>{
  try {const x=normalize(2,[[0,1,weight]]);return {weight,accepted:true,finite:[x.inverse,x.randomWalk,x.transition,x.symmetric].flat(2).every(Number.isFinite),transition:x.transition};}
  catch(error){return {weight,accepted:false,name:error.name,message:error.message};}
});
console.log(JSON.stringify({cycle,boundaries}));
""")

max_error = 0.0
for case, result in zip(cases, actual["cycle"], strict=True):
    expected = np.array(case["expected"])
    np.testing.assert_allclose(result["harmonic"]["values"], expected, rtol=0, atol=8e-14)
    np.testing.assert_allclose(result["loops"]["values"], expected, rtol=0, atol=8e-14)
    assert result["partial"]["values"][-1] is None
    assert result["partial"]["unanchored"] == [[4]]
    max_error = max(max_error, float(np.max(np.abs(np.array(result["harmonic"]["values"]) - expected))))
    energy = result["energy"]
    drops = np.array([8, 1, -4, 3])
    expected_energy = sum(w * (drops[u] - drops[v]) ** 2 for u, v, w in case["edges"])
    assert energy["edgeEnergy"] == expected_energy
    assert energy["action"] == result["reversed"]["action"]
    np.testing.assert_array_equal(np.array(energy["incidence"]), -np.array(result["reversed"]["incidence"]))
    normal = result["normal"]
    p = np.array(normal["transition"])
    s = np.array(normal["symmetric"])
    r = np.array(normal["randomWalk"])
    degree = np.array(normal["degrees"])
    root_degree = np.sqrt(degree)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=5e-16)
    np.testing.assert_allclose(degree @ p, degree, atol=2e-14)
    np.testing.assert_allclose(s @ root_degree, 0, atol=2e-15)
    np.testing.assert_allclose(s @ (root_degree * [1, 3, -2, 4, 9]), root_degree * (r @ [1, 3, -2, 4, 9]), atol=1e-14)
    assert p[4, 4] == 1 and s[4, 4] == 0
    assert normal["naiveIdentity"][4][4] == 1

# The two independent nonconstant eigenvectors of the three-node path give an
# exact changed-input closed-form exchange solution; no repeated loop oracle.
initial = [F(2), F(-1), F(4)]
mean = sum(initial) / 3
v1, v3 = [1, 0, -1], [1, -2, 1]
c1 = sum(x * y for x, y in zip(initial, v1)) / 2
c3 = sum(x * y for x, y in zip(initial, v3)) / 6
exchange = [[F(3, 4), F(1, 4), 0], [F(1, 4), F(1, 2), F(1, 4)], [0, F(1, 4), F(3, 4)]]
for step in range(13):
    expected = [mean + c1 * F(3, 4) ** step * a + c3 * F(1, 4) ** step * b for a, b in zip(v1, v3)]
    assert functions["averaging"]["iterate"](exchange, initial, step) == expected

# Combinatorial triples, independent of trace(A^3), check actual native helper.
pairs = list(itertools.combinations(range(4), 2))
for mask in range(64):
    edges = [pair for bit, pair in enumerate(pairs) if mask & (1 << bit)]
    edge_set = set(edges)
    triangles = [triple for triple in itertools.combinations(range(4), 3) if all(pair in edge_set for pair in itertools.combinations(triple, 2))]
    degrees, local, clustering, count = functions["features"]["triangle_features"](5, edges)
    assert count == len(triangles)
    assert local == [sum(vertex in triple for triple in triangles) for vertex in range(5)]
    assert clustering[-1] == degrees[-1] == 0

boundary_native = []
for weight in [5e-324, 1e-320, 1e-310, 1e-308]:
    try:
        output = functions["normalization"]["normalized_operators"]([[0, weight], [weight, 0]])
        finite = bool(np.isfinite(np.array(output[-1])).all())
        boundary_native.append({"weight": weight, "accepted": True, "finite": finite})
        assert weight == 1e-308 and finite
    except ValueError as error:
        boundary_native.append({"weight": weight, "accepted": False, "message": str(error)})
        assert weight < 1e-308
for result in actual["boundaries"]:
    assert result["accepted"] == (result["weight"] == 1e-308)
    if result["accepted"]:
        assert result["finite"]
        np.testing.assert_allclose(result["transition"], [[0, 1], [1, 0]], atol=2e-15)
    else:
        assert result["name"] == "RangeError"

freeze = json.loads((ROOT / "docs/teaching/evidence/graph-fundamentals-author-review.json").read_text(encoding="utf-8"))
hashes = {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in freeze["productionHashes"]}
record = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "allPassed": True,
    "counts": {"changedCycleCircuits": 81, "loopPartialAndCoordinateCases": 81, "exactClosedModeIterations": 13, "simpleFourVertexGraphsWithIsolate": 64, "jsBoundaryCases": 4, "nativeBoundaryCases": 4},
    "maximumHarmonicError": max_error,
    "jsBoundaries": actual["boundaries"], "nativeBoundaries": boundary_native,
    "productionHashes": hashes,
    "authorFreezeHashMatches": hashes == freeze["productionHashes"],
    "limits": "Scoped complementary oracles and full source read; the author's broad browser/native suites remain separately attributed. Not an arbitrary floating-point guarantee.",
}
(OUT / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
