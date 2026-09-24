"""Actual native programs plus independent exact oracles for lesson models."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from io import StringIO
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "scratch/computational-geometry-verification"
FOLDER.mkdir(parents=True, exist_ok=True)
examples = json.loads(subprocess.check_output([
    "node", "--input-type=module", "-e",
    "import {computationalGeometryExamples as e} from './src/learn/data/computational-geometry-examples.js'; console.log(JSON.stringify(e));"
], cwd=ROOT, text=True, encoding="utf-8"))
namespaces = {}
for name, example in examples.items():
    filename = FOLDER / f"{name}.py"
    filename.write_text(example["code"], encoding="utf-8")
    actual = subprocess.check_output([sys.executable, "-X", "utf8", "-I", str(filename)], text=True, encoding="utf-8")
    assert actual.strip() == example["expected"], (name, actual, example["expected"])
    namespace = {}
    with redirect_stdout(StringIO()):
        exec(compile(example["code"], str(filename), "exec"), namespace)
    namespaces[name] = namespace

grid = list(itertools.product(range(3), repeat=2))
randomizer = random.Random(739)
orientation_cases = list(itertools.product(grid, repeat=3))
orientation_cases += [tuple(tuple(randomizer.randint(-10**6, 10**6) for _ in range(2))
                            for _ in range(3)) for _ in range(100)]
segment_cases = list(itertools.product(grid, repeat=4))
hull_cases = [[grid[index] for index in range(9) if mask & (1 << index)] for mask in range(512)]
hull_cases += [[tuple(randomizer.randrange(9) for _ in range(2)) for _ in range(randomizer.randrange(21))]
               for _ in range(100)]
polygons = [
    [(1, 1), (7, 1), (7, 6), (5, 6), (5, 3), (3, 3), (3, 6), (1, 6)],
    [(1, 1), (7, 1), (7, 6), (1, 6)],
    [(1, 1), (7, 1), (4, 6)],
    [(0, 0), (6, 0), (6, 4), (4, 4), (4, 2), (2, 2), (2, 4), (0, 4)],
]
polygon_cases = [(shape, query) for polygon in polygons for shape in [polygon, polygon[::-1]]
                 for query in itertools.product(range(9), repeat=2)]
payload = {"orientation": orientation_cases, "segments": segment_cases, "hulls": hull_cases,
           "polygons": polygon_cases}
fixture_file = FOLDER / "fixtures.json"
fixture_file.write_text(json.dumps(payload), encoding="utf-8")
javascript = r"""
import fs from 'node:fs';
import * as geometry from './src/learn/data/computational-geometry-models.js';
const fixture = JSON.parse(fs.readFileSync('scratch/computational-geometry-verification/fixtures.json','utf8'));
const result = {
  orientation: fixture.orientation.map(points => geometry.orientation(...points)),
  segments: fixture.segments.map(points => geometry.segmentState(...points).kind),
  hulls: fixture.hulls.map(points => [false,true].map(policy => {
    const trace=geometry.hullTrace(points,policy);
    if (JSON.stringify(trace.frames.at(-1).stack) !== JSON.stringify(trace.hull)) throw Error('final trace mismatch');
    return trace.hull;
  })),
  polygons: fixture.polygons.map(([polygon,query]) => geometry.polygonState(polygon,query)),
  precision: [20,26,27,30].map(value=>geometry.precisionState(value)),
  lostInput: geometry.precisionState(27,'input'),
};
const invalid = [()=>geometry.orientation([.5,0],[0,0],[1,1]),
  ()=>geometry.orientation([1000001,0],[0,0],[1,1]),
  ()=>geometry.parseGridPoints('1.5,2'),()=>geometry.parseGridPoints('9,2'),
  ()=>geometry.parseGridPoints('1,2\n'.repeat(21)),()=>geometry.hullTrace(Array(51).fill([0,0])),
  ()=>geometry.polygonState([[0,0],[1,1]],[0,0]),()=>geometry.precisionState(19),
  ()=>geometry.precisionState(27,'other')];
result.invalid = invalid.map(action=>{try {action(); return false;}catch(error){return error instanceof RangeError;}});
fs.writeFileSync('scratch/computational-geometry-verification/model-results.json',JSON.stringify(result));
"""
subprocess.check_call(["node", "--input-type=module", "-e", javascript], cwd=ROOT)
models = json.loads((FOLDER / "model-results.json").read_text(encoding="utf-8"))


def determinant(a, b, c):
    # Independent three-point shoelace expansion.
    return (a[0]*b[1] + b[0]*c[1] + c[0]*a[1]
            - a[1]*b[0] - b[1]*c[0] - c[1]*a[0])


def between(a, b, p):
    return determinant(a, b, p) == 0 and sum((p[i]-a[i])*(p[i]-b[i]) for i in range(2)) <= 0


def exact_segment(a, b, c, d):
    """Parametric line solve, with projected intervals for the parallel branch."""
    r = (b[0]-a[0], b[1]-a[1])
    s = (d[0]-c[0], d[1]-c[1])
    cross = lambda left, right: left[0]*right[1] - left[1]*right[0]
    if a == b:
        return "touch" if between(c, d, a) else "disjoint"
    if c == d:
        return "touch" if between(a, b, c) else "disjoint"
    denominator = cross(r, s)
    offset = (c[0]-a[0], c[1]-a[1])
    if denominator:
        t, u = Fraction(cross(offset, s), denominator), Fraction(cross(offset, r), denominator)
        if not (0 <= t <= 1 and 0 <= u <= 1):
            return "disjoint"
        return "proper crossing" if 0 < t < 1 and 0 < u < 1 else "touch"
    if cross(offset, r):
        return "disjoint"
    axis = 0 if r[0] else 1
    interval = sorted((Fraction(c[axis]-a[axis], r[axis]), Fraction(d[axis]-a[axis], r[axis])))
    lower, upper = max(Fraction(0), interval[0]), min(Fraction(1), interval[1])
    return "overlap" if lower < upper else "touch" if lower == upper else "disjoint"


def supporting_oracle(records, boundary):
    points = sorted(set(records))
    if len(points) < 3:
        return set(points)
    supported = set()
    for a, b in itertools.combinations(points, 2):
        signs = [determinant(a, b, p) for p in points]
        if min(signs) >= 0 or max(signs) <= 0:
            supported.update(p for p in points if determinant(a, b, p) == 0)
    if boundary:
        return supported
    return {p for p in supported if not any(between(a, b, p)
            for a, b in itertools.combinations([q for q in points if q != p], 2))}


def winding_oracle(polygon, query):
    # Count signed vertical crossings of the upward ray, with rational x parameters;
    # different ray/direction and arithmetic from the lesson's horizontal determinant rule.
    winding = 0
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if between(a, b, query):
            return "boundary"
        if min(a[0], b[0]) <= query[0] < max(a[0], b[0]):
            t = Fraction(query[0]-a[0], b[0]-a[0])
            height = a[1] + t*(b[1]-a[1])
            if height > query[1]:
                winding += 1 if b[0] > a[0] else -1
    return "inside" if winding else "outside"


for arguments, actual in zip(orientation_cases, models["orientation"]):
    expected = determinant(*arguments)
    assert actual == expected
    assert namespaces["turnAndArea"]["orient"](*arguments) == expected

for arguments, actual in zip(segment_cases, models["segments"]):
    expected = exact_segment(*arguments)
    assert actual == expected, (arguments, actual, expected)
    assert namespaces["segments"]["classify_segments"](*arguments)[0] == expected

for records, outputs in zip(hull_cases, models["hulls"]):
    for boundary, actual in zip([False, True], outputs):
        actual = list(map(tuple, actual))
        expected = supporting_oracle(records, boundary)
        assert len(actual) == len(set(actual)) and set(actual) == expected, (records, boundary, actual, expected)
        native = namespaces["monotoneHull"]["convex_hull"](records, boundary)
        assert native == actual
        if actual:
            assert actual[0] == min(expected)
        if len(actual) >= 3:
            for a, b in zip(actual, actual[1:] + actual[:1]):
                assert all(determinant(a, b, p) >= 0 for p in records)

for (polygon, query), actual in zip(polygon_cases, models["polygons"]):
    expected = winding_oracle(polygon, query)
    assert actual["classification"] == expected, (polygon, query, expected)
    assert namespaces["polygonQuery"]["locate"](polygon, query) == expected
    # A triangulated signed fan is a separate expansion from consecutive shoelace terms.
    fan = sum(determinant(polygon[0], polygon[index], polygon[index+1]) for index in range(1, len(polygon)-1))
    assert actual["doubledArea"] == fan

for state in models["precision"]:
    intended = [tuple(map(int, value)) for value in state["points"]]
    rounded = [tuple(map(float, value)) for value in intended]
    assert str(determinant(*intended)) == state["exactDeterminant"] == "-1"
    assert namespaces["precision"]["orient"](*rounded) == state["floatDeterminant"]
assert models["lostInput"]["exactDeterminant"] == "1"
assert models["lostInput"]["floatDeterminant"] == 0 and not models["lostInput"]["allInputsPreserved"]
assert all(models["invalid"])

line_tests = []
for _ in range(200):
    records = [randomizer.choice(grid) for _ in range(randomizer.randrange(12))]
    distinct = sorted(set(records))
    if len(distinct) < 2:
        expected = len(records)
    else:
        expected = max(sum(determinant(a, b, p) == 0 for p in records)
                       for a, b in itertools.combinations(distinct, 2))
    assert namespaces["lineGroups"]["most_collinear"](records) == expected
    line_tests.append(records)

# Independently exercise every closed-form/acceptance group from the page.
assert determinant((-2, 1), (4, 4), (1, 5)) == 15
assert exact_segment((0, 0), (4, 0), (2, 0), (6, 0)) == "overlap"
assert exact_segment((0, 0), (4, 0), (4, 0), (4, 3)) == "touch"
assert exact_segment((0, 0), (4, 0), (5, 0), (5, 0)) == "disjoint"
assert namespaces["rationalIntersection"]["proper_intersection"]((0, 0), (3, 3), (0, 2), (3, 0)) == (Fraction(6, 5), Fraction(6, 5))
records = [(0,0),(2,0),(4,0),(4,3),(2,3),(0,3),(2,1),(0,0)]
corners = namespaces["monotoneHull"]["convex_hull"](records)
assert corners == [(0,0),(4,0),(4,3),(0,3)]
assert namespaces["monotoneHull"]["convex_hull"](records, True) == [(0,0),(2,0),(4,0),(4,3),(2,3),(0,3)]
assert namespaces["polygonQuery"]["signed_area_twice"](polygons[-1]) == 40
assert [winding_oracle(polygons[-1], p) for p in [(3,3),(1,3),(3,2),(6,4)]] == ["outside","inside","boundary","boundary"]
assert [exact_segment(a,b,(-1,1),(5,1)) for a,b in zip(corners,corners[1:]+corners[:1])] == ["disjoint","proper crossing","disjoint","proper crossing"]
assert [max(p[0]*w[0]+p[1]*w[1] for p in corners) for w in [(1,1),(-2,1)]] == [7,3]

sources = [ROOT / "src/learn/data/computational-geometry-models.js", ROOT / "src/learn/data/computational-geometry-examples.js"]
result = {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.version.split()[0], "exact_stdout_programs": len(examples),
    "orientation_cases": len(orientation_cases), "segment_fraction_oracle_cases": len(segment_cases),
    "hull_supporting_oracle_cases_both_policies": 2*len(hull_cases),
    "polygon_vertical_winding_and_signed_fan_cases": len(polygon_cases),
    "precision_cases": 5, "invalid_model_cases": len(models["invalid"]),
    "native_line_group_bruteforce_cases": len(line_tests), "independent_practice_groups": 6,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(FOLDER / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
