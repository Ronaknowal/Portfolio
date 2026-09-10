"""Author complete local programs and capture their actual Python output."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/computational-geometry-native"
DIRECTORY.mkdir(parents=True, exist_ok=True)
ORIENTATION = '''def orient(a, b, c):
    """Doubled signed area; this lesson's contract is integer coordinate pairs."""
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))

def on_segment(a, b, query):
    return (orient(a, b, query) == 0
            and min(a[0], b[0]) <= query[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= query[1] <= max(a[1], b[1]))

'''
SEGMENT = ORIENTATION + '''def classify_segments(a, b, c, d):
    shared = sorted({p for p in (a, b, c, d)
                     if on_segment(a, b, p) and on_segment(c, d, p)})
    if len(shared) >= 2:
        return "overlap", (shared[0], shared[-1])
    if shared:
        return "touch", shared[0]
    sides = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if sides[0] * sides[1] < 0 and sides[2] * sides[3] < 0:
        return "proper crossing", None
    return "disjoint", None

'''
HULL = ORIENTATION + '''def convex_hull(points, include_boundary=False):
    points = sorted(set(map(tuple, points)))
    if len(points) <= 1:
        return points
    if all(orient(points[0], points[-1], p) == 0 for p in points):
        return points if include_boundary else [points[0], points[-1]]

    def chain(sequence):
        result = []
        for candidate in sequence:
            while len(result) >= 2:
                turn = orient(result[-2], result[-1], candidate)
                keep_turn = turn >= 0 if include_boundary else turn > 0
                if keep_turn:
                    break
                result.pop()
            result.append(candidate)
        return result

    lower = chain(points)
    upper = chain(reversed(points))
    return lower[:-1] + upper[:-1]

'''
POLYGON = ORIENTATION + '''def signed_area_twice(vertices):
    return sum(a[0]*b[1] - a[1]*b[0]
               for a, b in zip(vertices, vertices[1:] + vertices[:1]))

def locate(vertices, query):
    """Simple polygon, at least three vertices; boundary is a separate result."""
    crossings = 0
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if on_segment(a, b, query):
            return "boundary"
        if (a[1] > query[1]) != (b[1] > query[1]):
            determinant = orient(a, b, query)
            to_right = determinant > 0 if b[1] > a[1] else determinant < 0
            crossings += to_right
    return "inside" if crossings % 2 else "outside"

'''
PROGRAMS = {
    "turnAndArea": ("One determinant, a turn and an area", ORIENTATION + '''from fractions import Fraction

a, b = (1, 1), (7, 3)
for c in [(4, 5), (4, 2), (4, 1), a]:
    determinant = orient(a, b, c)
    label = "left" if determinant > 0 else "right" if determinant < 0 else "collinear"
    print(c, "det:", determinant, label, "area:", Fraction(abs(determinant), 2))
print("reverse baseline:", orient(b, a, (4, 5)))
'''),
    "segments": ("Classify closed segments, including point segments", SEGMENT + '''fixtures = [
    ((1, 1), (7, 5), (1, 5), (7, 1)),
    ((1, 1), (4, 3), (4, 3), (7, 1)),
    ((1, 3), (6, 3), (3, 3), (7, 3)),
    ((1, 1), (6, 5), (1, 2), (5, 6)),
    ((4, 3), (4, 3), (1, 3), (7, 3)),
    ((2, 2), (2, 2), (3, 3), (3, 3)),
]
for fixture in fixtures:
    print(classify_segments(*fixture))
'''),
    "rationalIntersection": ("Construct a crossing point without rounding it", SEGMENT + '''from fractions import Fraction

def cross(u, v):
    return u[0]*v[1] - u[1]*v[0]

def proper_intersection(a, b, c, d):
    if classify_segments(a, b, c, d)[0] != "proper crossing":
        raise ValueError("This construction requires a proper crossing.")
    direction = b[0]-a[0], b[1]-a[1]
    other = d[0]-c[0], d[1]-c[1]
    offset = c[0]-a[0], c[1]-a[1]
    parameter = Fraction(cross(offset, other), cross(direction, other))
    return tuple(Fraction(a[i]) + parameter*direction[i] for i in range(2))

intersection = proper_intersection((0, 0), (3, 3), (0, 2), (3, 0))
print("exact crossing:", tuple(map(str, intersection)))
print("on both segments:", on_segment((0, 0), (3, 3), intersection)
      and on_segment((0, 2), (3, 0), intersection))
'''),
    "precision": ("Separate product rounding, input rounding and intended decimal data", ORIENTATION + '''from fractions import Fraction

n = 2**27
points = [(0, 0), (n+1, n), (n, n-1)]
print("integer det:", orient(*points))
print("float det:", orient(*(tuple(map(float, p)) for p in points)))
n = 2**53
points = [(n, 0), (n+1, 0), (n, 1)]
rounded = [tuple(map(float, p)) for p in points]
print("input distinction lost:", rounded[0] == rounded[1])
print("int det, rounded-input det:", orient(*points), orient(*rounded))
print("intended decimal:", Fraction("0.1"))
print("stored binary float:", Fraction(0.1))
epsilon = 1e-9
for scale in [1, 10**6]:
    determinant = orient((0., 0.), (scale, 0.), (scale, scale*1e-10))
    print("scale:", scale, "epsilon says collinear:", abs(determinant) <= epsilon)
'''),
    "monotoneHull": ("Construct corners or every boundary record", HULL + '''points = [(1, 1), (3, 1), (7, 1), (7, 6), (4, 6),
          (1, 6), (3, 3), (5, 4), (1, 1)]
print("corners:", convex_hull(points))
print("boundary:", convex_hull(points, True))
line = [(1, 2), (3, 3), (5, 4), (7, 5)]
print("line corners:", convex_hull(line))
print("line boundary:", convex_hull(line, True))
print("empty and repeated singleton:", convex_hull([]), convex_hull([(2, 2)]*3))
'''),
    "polygonQuery": ("Keep the courtyard opening outside", POLYGON + '''from fractions import Fraction

polygon = [(1, 1), (7, 1), (7, 6), (5, 6),
           (5, 3), (3, 3), (3, 6), (1, 6)]
print("signed area:", Fraction(signed_area_twice(polygon), 2))
print("reversed area:", Fraction(signed_area_twice(list(reversed(polygon))), 2))
for query in [(4, 5), (2, 5), (4, 2), (3, 4), (4, 3), (7, 6), (8, 5)]:
    print(query, locate(polygon, query))
'''),
    "supportAndBounds": ("Use a convex envelope as a directional summary", HULL + '''points = [(1, 1), (3, 1), (7, 1), (7, 6), (4, 6),
          (1, 6), (3, 3), (5, 4)]
hull = convex_hull(points)
for direction in [(1, 0), (0, 1), (2, -1), (-1, 2)]:
    score = lambda p: p[0]*direction[0] + p[1]*direction[1]
    full_maximum = max(map(score, points))
    hull_maximum = max(map(score, hull))
    print(direction, "maximum:", hull_maximum, "agrees:", full_maximum == hull_maximum)

def rectangle_overlap(first, second, positive_area=True):
    widths = [min(first[i+2], second[i+2]) - max(first[i], second[i])
              for i in range(2)]
    return all(width > 0 if positive_area else width >= 0 for width in widths)

print("touch: positive area / closed contact:",
      rectangle_overlap((0, 0, 2, 2), (2, 0, 4, 2)),
      rectangle_overlap((0, 0, 2, 2), (2, 0, 4, 2), False))
'''),
    "lineGroups": ("Normalize line directions instead of hashing rounded slopes", '''from collections import Counter
from math import gcd

def most_collinear(records):
    multiplicities = Counter(map(tuple, records))
    best = 0
    for anchor, own_count in multiplicities.items():
        groups = Counter()
        for other, count in multiplicities.items():
            if other == anchor:
                continue
            dx, dy = other[0]-anchor[0], other[1]-anchor[1]
            divisor = gcd(abs(dx), abs(dy))
            dx, dy = dx//divisor, dy//divisor
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            groups[dx, dy] += count
        best = max(best, own_count + max(groups.values(), default=0))
    return best

for records in [[], [(2, 2)], [(2, 2)]*3,
                [(0, 0), (1, 1), (-2, -2), (3, 3), (0, 2)],
                [(1, 0), (1, 2), (1, 3), (2, 4)],
                [(0, 0), (0, 0), (1, 1), (2, 2), (1, 0)]]:
    print(most_collinear(records))
'''),
}

examples = {}
for key, (title, code) in PROGRAMS.items():
    filename = DIRECTORY / f"{key}.py"
    filename.write_text(code, encoding="utf-8")
    output = subprocess.check_output([sys.executable, "-X", "utf8", "-I", str(filename)], text=True, encoding="utf-8")
    examples[key] = {"title": title, "code": code, "expected": output.strip()}
destination = ROOT / "src/learn/data/computational-geometry-examples.js"
destination.write_text("// Complete Python 3.12 programs; outputs captured by the owned preparation script.\nexport const computationalGeometryExamples = " + json.dumps(examples, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
(DIRECTORY / "captured.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
print(json.dumps({key: value["expected"] for key, value in examples.items()}, indent=2))
