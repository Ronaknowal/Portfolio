"""Independent finite/exact oracles and changed runs of actual lesson helpers."""
import contextlib
from datetime import datetime, timezone
from fractions import Fraction as Q
import io
import itertools
import json
import math
from pathlib import Path

root = Path("scratch/category-theory-verification")
fixtures = json.loads((root / "model-fixtures.json").read_text(encoding="utf-8"))
counts = {}

for key, rows in fixtures.items():
    counts[key] = len(rows)
    for r in rows:
        if key == "compositions":
            f, g, h = r["maps"]
            expected = [h[g[f[i]]] for i in range(3)]
            assert r["left"] == r["right"] == expected
            assert r["path"] == [r["input"], f[r["input"]], g[f[r["input"]]], expected[r["input"]]]
        elif key == "naturality":
            transform = lambda xs: list(reversed(xs)) if r["operation"] == "reverse" else sorted(xs)
            first = [r["mapping"][x] for x in transform(r["values"])]
            second = transform([r["mapping"][x] for x in r["values"]])
            assert r["topThenRight"] == first
            assert r["leftThenBottom"] == second
            assert r["commutesHere"] == (first == second)
        elif key == "products":
            candidates = [(a, b) for a in range(2) for b in range(2)]
            if r["mode"] == "missing":
                candidates.remove((0, 0))
            elif r["mode"] == "duplicate":
                candidates.append((0, 0))
            assert len(r["mediators"]) == candidates.count((r["row"], r["column"]))
            assert r["universal"] == (r["mode"] == "complete")
        elif key == "probability":
            p = Q(r["percent"], 100)
            atoms = [(0, 1-p), (1, p)]
            joint = [[sum(a*b for x, a in atoms for y, b in atoms if (x, y) == (i, j))
                      for j in range(2)] for i in range(2)]
            copy = [[sum(a for x, a in atoms if (x, x) == (i, j)) for j in range(2)] for i in range(2)]
            for name, expected in [("independent", joint), ("copied", copy)]:
                for row, other in zip(r[name], expected):
                    for value, exact in zip(row, other):
                        assert math.isclose(value, float(exact), abs_tol=1e-15)
        elif key == "tangents":
            x, v, w = Q(r["tenth"], 10), Q(r["tangent"]), Q(r["cotangent"])
            # Polynomial coefficients are composed independently of the browser stages.
            polynomial = [Q(1), Q(2), Q(1)] if r["choice"] == "shiftedSquare" else (
                [Q(-1), Q(0), Q(0), Q(2)] if r["choice"] == "cubicAffine" else [Q(0), Q(0), Q(0), Q(0), Q(1)])
            value = sum(c*x**i for i, c in enumerate(polynomial))
            derivative = sum(i*polynomial[i]*x**(i-1) for i in range(1, len(polynomial)))
            for field, expected in [("secondValue", value), ("directDerivative", derivative),
                                    ("composedDerivative", derivative), ("outputTangent", derivative*v),
                                    ("inputCotangent", derivative*w), ("outputPairing", derivative*v*w)]:
                assert math.isclose(r[field], float(expected), rel_tol=3e-14, abs_tol=2e-12), (field, r)
        elif key == "adjunctions":
            s = {i for i in range(4) if r["sourceMask"] >> i & 1}
            t = {i for i in range(3) if r["targetMask"] >> i & 1}
            relation = {(i, y) for i, y in enumerate(r["mapping"])}
            image = {y for x, y in relation if x in s}
            preimage = {x for x, y in relation if y in t}
            assert set(r["image"]) == image and set(r["preimage"]) == preimage
            assert r["imageContained"] == (image <= t) == (s <= preimage) == r["sourceContained"]

text = Path("src/learn/data/category-theory-examples.js").read_text(encoding="utf-8")
examples = json.loads(text[text.index("{"):].rstrip().removesuffix(";"))
namespaces = {}
for name, example in examples.items():
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example["code"], "<displayed-" + name + ">", "exec"), namespace)
    assert output.getvalue().rstrip() == example["expected"], name
    namespaces[name] = namespace
original = json.loads(Path("docs/teaching/evidence/category-theory-original-content.json").read_text(encoding="utf-8"))
assert examples["original"]["code"].rstrip() == original["blocks"][0]["code"].rstrip()

# Test the actual schema helper on every tiny direct-site assignment.
schema = namespaces["schema"]
for direct in itertools.product(range(2), repeat=3):
    assert schema["audit"]((0, 0, 1), (0, 1), direct) == [i for i, site in enumerate(direct) if site != (0, 0, 1)[i]]

# Actual effect helpers: success, failure, and successful None remain distinct.
option = namespaces["option"]
Some, pure, bind, kleisli = (option[key] for key in ("Some", "pure", "bind", "kleisli"))
functions = [pure, lambda x: None, lambda x: Some(x+1), lambda x: None if x == 0 else Some(-x)]
option_cases = 0
for f, g, h in itertools.product(functions, repeat=3):
    for x in range(-3, 4):
        assert kleisli(h, kleisli(g, f))(x) == kleisli(kleisli(h, g), f)(x)
        option_cases += 1
assert bind(Some(None), pure) == Some(None)

# Independent quotient dimension oracle: enumerate F2 rank from integer XOR rows.
homology = namespaces["homology"]
def rank(columns):
    basis = {}
    for column in columns:
        while column:
            pivot = column.bit_length()-1
            if pivot not in basis:
                basis[pivot] = column
                break
            column ^= basis[pivot]
    return len(basis)

homology_cases = 0
vertices = range(4)
all_edges = list(itertools.combinations(vertices, 2))
all_faces = list(itertools.combinations(vertices, 3))
for edge_mask in range(1 << len(all_edges)):
    edges = [e for i, e in enumerate(all_edges) if edge_mask >> i & 1]
    allowed_faces = [f for f in all_faces if all(e in edges for e in itertools.combinations(f, 2))]
    for face_mask in range(1 << len(allowed_faces)):
        faces = [f for i, f in enumerate(allowed_faces) if face_mask >> i & 1]
        columns1 = [sum(1 << v for v in edge) for edge in edges]
        columns2 = [sum(1 << edges.index(e) for e in itertools.combinations(face, 2)) for face in faces]
        beta = len(edges)-rank(columns1)-rank(columns2)
        assert len(homology["homology_classes"](edges, faces)[2]) == 2**beta
        homology_cases += 1

# Every rational binary channel on this grid, with actual native composition.
compose = namespaces["probability"]["compose"]
channels = [[[Q(a, 4), 1-Q(a, 4)], [Q(b, 4), 1-Q(b, 4)]] for a in range(5) for b in range(5)]
channel_cases = 0
for first in channels:
    for second in channels:
        actual = compose(first, second)
        for x in range(2):
            for z in range(2):
                assert actual[x][z] == sum(first[x][y]*second[y][z] for y in range(2))
            assert sum(actual[x]) == 1
        channel_cases += 1

record = {
    "passed": True, "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "programs": len(examples), "modelCases": counts,
    "actualHelperChangedCases": {"schema": 8, "option": option_cases, "homology": homology_cases, "channels": channel_cases},
    "limits": "Finite exact oracles and displayed-program execution support these implementation cases; the lesson's general laws require its separate mathematical proofs.",
}
(root / "native-results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
