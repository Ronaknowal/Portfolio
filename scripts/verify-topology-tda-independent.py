from pathlib import Path
import contextlib, datetime, hashlib, io, itertools, json, math, subprocess
import numpy as np
import mpmath as mp
from scipy.optimize import linear_sum_assignment

root = Path.cwd()
mp.mp.dps = 85
js = r"""import * as m from './src/learn/data/topology-tda-models.js';
import { topologyTdaExamples as examples } from './src/learn/data/topology-tda-examples.js';
let seed = 42040;
const random = () => {
  seed = Math.imul(seed, 1664525) + 1013904223 >>> 0;
  return seed / 2 ** 32;
};
const choose = (n, k) => {
  const out = [];
  const visit = (a, v) => {
    if (v.length === k) {
      out.push(v);
      return;
    }
    for (let i = a; i < n; i++) visit(i + 1, [...v, i]);
  };
  visit(0, []);
  return out;
};
const filtrations = [];
for (let trial = 0; trial < 18; trial++) {
  const n = 4 + trial % 3,
    cells = [],
    births = new Map();
  for (let k = 1; k <= 4; k++) for (const v of choose(n, k)) {
    const minimum = k === 1 ? 0 : Math.max(...m.simplexFaces(v).map(f => births.get(f.join('-'))));
    const birth = minimum + Math.floor(random() * 3);
    births.set(v.join('-'), birth);
    cells.push({
      vertices: v,
      birth
    });
  }
  const result = m.persistentHomology(cells, {
    captureTrace: true
  });
  const permutation = Array.from({
    length: n
  }, (_, i) => n - 1 - i);
  const permuted = m.persistentHomology(cells.map(c => ({
    vertices: c.vertices.map(i => permutation[i]),
    birth: c.birth
  })));
  filtrations.push({
    result,
    permuted
  });
}
const perturbations = [];
for (let trial = 0; trial < 18; trial++) {
  const points = Array.from({
    length: 5 + trial % 2
  }, () => [random() * 2 - 1, random() * 2 - 1]);
  const changed = points.map(([x, y]) => {
    const angle = random() * 2 * Math.PI,
      radius = .015 * random();
    return [x + radius * Math.cos(angle), y + radius * Math.sin(angle)];
  });
  const first = m.persistentHomology(m.ripsFiltration(points)),
    second = m.persistentHomology(m.ripsFiltration(changed));
  perturbations.push({
    points,
    changed,
    first: first.intervals,
    second: second.intervals
  });
}
const mapper = [];
for (const intervalCount of [3, 4, 5]) for (const overlap of [.25, .4, .65]) for (const clusterDistance of [.45, .6, 1.8]) mapper.push(m.mapperGraph({
  intervalCount,
  overlap,
  clusterDistance
}));
const intervals = [[0, 1e-20], [-1e-20, 1e-20], [1, 1 + 1e-12], [8, 8 + 1e-8], [-3, -3 + 1e-10], [0, .1], [1.7, 1.8], [20, 20.001], [-35, -34.999]];
console.log(JSON.stringify({
  examples,
  filtrations,
  perturbations,
  mapper,
  normal: intervals.map(([a, b]) => ({
    a,
    b,
    value: m.normalInterval(a, b)
  })),
  tinyPixel: m.persistenceImage([[0, 1]], {
    bandwidth: .5,
    xEdges: [0, 1e-20],
    yEdges: [0, 1]
  }),
  skeleton: m.bettiAt(m.ripsFiltration([[0, 0], [1, 0], [0, 1], [1, 1]], {
    maximumDimension: 2
  })),
  solid: m.bettiAt(m.ripsFiltration([[0, 0], [1, 0], [0, 1], [1, 1]], {
    maximumDimension: 3
  }))
}));"""
payload = json.loads(
    subprocess.run(
        ["node", "--input-type=module", "-e", js],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
)
counts = {}
namespaces = {}
for key, e in payload["examples"].items():
    ns = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(e["code"], key + ".py", "exec"), ns)
    assert output.getvalue().rstrip("\n") == e["expected"]
    namespaces[key] = ns
counts["actualPrograms"] = len(namespaces)
archive = json.loads(
    (root / "docs/teaching/evidence/topology-tda-original-content.json").read_text(
        encoding="utf-8"
    )
)
assert (
    hashlib.sha256(archive["source"].encode("utf-8")).hexdigest() == archive["sha256"]
)
assert payload["examples"]["original"]["code"] == archive["blocks"][0]["code"]
assert payload["examples"]["original"]["expected"] == archive["blocks"][1]["code"]
counts["preservedOriginalCodeAndOutput"] = True


def rank(vectors):
    basis = {}
    for v in vectors:
        while v:
            pivot = (v & -v).bit_length() - 1
            if pivot in basis:
                v ^= basis[pivot]
            else:
                basis[pivot] = v
                break
    return len(basis)


def mask(indices):
    return sum(1 << i for i in indices)


def bd(vector, columns):
    result = 0
    for i, col in enumerate(columns):
        if vector >> i & 1:
            result ^= mask(col)
    return result


traces = representatives = 0
for case in payload["filtrations"]:
    r = case["result"]
    cells = r["ordered"]
    columns = r["boundaries"]
    n = len(cells)
    for tr in r["trace"]:
        previous = None
        for stage in tr["stages"]:
            combination = mask(stage["combination"])
            boundary = mask(stage["boundary"])
            assert bd(combination, columns) == boundary
            assert all(
                i <= tr["column"]
                and cells[i]["dimension"] == cells[tr["column"]]["dimension"]
                for i in stage["combination"]
            )
            if stage["addedColumn"] is not None:
                assert stage["addedColumn"] < tr["column"]
                assert boundary.bit_length() < (previous or 0).bit_length()
            previous = boundary
            traces += 1
    for bar in r["intervals"]:
        k = bar["dimension"]
        v = mask(bar["representative"])
        i = bar["creator"]
        j = bar["destroyer"]
        assert v and not bd(v, columns)
        assert all(
            idx <= i and cells[idx]["dimension"] == k for idx in bar["representative"]
        )
        # At its creator prefix this representative is independent of all available boundaries.
        earlier = [
            mask(columns[t]) for t in range(i + 1) if cells[t]["dimension"] == k + 1
        ]
        assert rank(earlier + [v]) == rank(earlier) + 1
        if j is not None:
            before = [
                mask(columns[t]) for t in range(j) if cells[t]["dimension"] == k + 1
            ]
            after = before + [mask(columns[j])]
            assert rank(before + [v]) == rank(before) + 1
            assert rank(after + [v]) == rank(after)
        representatives += 1
    # Active representatives form a quotient basis at every actual scale (not every tie step).
    for time in sorted({c["birth"] for c in cells}):
        for k in range(4):
            boundaries = [
                mask(columns[t])
                for t, c in enumerate(cells)
                if c["birth"] <= time and c["dimension"] == k + 1
            ]
            active = [
                mask(b["representative"])
                for b in r["intervals"]
                if b["dimension"] == k
                and b["birth"] <= time
                and (b["death"] is None or time < b["death"])
            ]
            assert rank(boundaries + active) == rank(boundaries) + len(active)

    def bars(rs):
        return sorted(
            (b["dimension"], b["birth"], math.inf if b["death"] is None else b["death"])
            for b in rs["intervals"]
            if b["death"] is None or b["death"] > b["birth"]
        )

    assert bars(r) == bars(case["permuted"])
counts.update(
    {
        "changedFiltrations": len(payload["filtrations"]),
        "actualTraceStages": traces,
        "birthDeathRepresentativeCertificates": representatives,
    }
)
assert payload["skeleton"]["betti"] == [1, 0, 1, 0]
assert payload["solid"]["betti"] == [1, 0, 0, 0]


def bottleneck(a, b):
    n, m = len(a), len(b)
    if n + m == 0:
        return 0.0
    c = np.full((n + m, n + m), np.inf)
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            c[i, j] = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
        c[i, m + i] = (p[1] - p[0]) / 2
    for j, q in enumerate(b):
        c[n + j, j] = (q[1] - q[0]) / 2
    c[n:, m:] = 0
    candidates = sorted(set(c[np.isfinite(c)].tolist()))
    for value in candidates:
        rows, cols = linear_sum_assignment((c > value).astype(int))
        if np.all(c[rows, cols] <= value):
            return value
    raise AssertionError("no diagonal matching")


for case in payload["perturbations"]:
    delta = max(math.dist(p, q) for p, q in zip(case["points"], case["changed"]))
    for k in [0, 1]:
        a = [
            (b["birth"], b["death"])
            for b in case["first"]
            if b["dimension"] == k
            and b["death"] is not None
            and b["death"] > b["birth"]
        ]
        b = [
            (b["birth"], b["death"])
            for b in case["second"]
            if b["dimension"] == k
            and b["death"] is not None
            and b["death"] > b["birth"]
        ]
        assert bottleneck(a, b) <= 2 * delta + 2e-14
counts["PairedCloudStabilityBipartiteChecks"] = 36

higher = 0
for state in payload["mapper"]:
    nodes = [set(n["members"]) for n in state["nodes"]]
    edges = {(e["first"], e["second"]): e["members"] for e in state["edges"]}
    for i, j in itertools.combinations(range(len(nodes)), 2):
        intersection = nodes[i] & nodes[j]
        assert bool(intersection) == ((i, j) in edges)
        if intersection:
            assert sorted(intersection) == edges[i, j]
    # Full finite nerve has one simplex for every nonempty common intersection.
    triangles = [
        (i, j, k)
        for i, j, k in itertools.combinations(range(len(nodes)), 3)
        if nodes[i] & nodes[j] & nodes[k]
    ]
    if triangles:
        higher += 1
        vertices = list(range(len(nodes)))
        edge_list = list(edges)
        edge_id = {e: i for i, e in enumerate(edge_list)}
        triangle_boundaries = [
            sum(1 << edge_id[e] for e in itertools.combinations(t, 2))
            for t in triangles
        ]
        assert rank(triangle_boundaries) > 0
    for point, memberlist in enumerate(state["membership"]):
        assert memberlist == [i for i, s in enumerate(nodes) if point in s]
counts["ChangedMapperNerves"] = len(payload["mapper"])
counts["HigherNerveFillingCases"] = higher

normal_results = []
for interval in payload["normal"]:
    # Preserve actual binary64 endpoint values; do not mistake decimal input rounding for integration error.
    a, b = mp.mpf(interval["a"]), mp.mpf(interval["b"])
    expected = mp.quad(lambda t: mp.exp(-t * t / 2) / mp.sqrt(2 * mp.pi), [a, b])
    actual = interval["value"]
    relative = float(abs(mp.mpf(actual) - expected) / expected) if expected else 0
    normal_results.append(
        {
            **interval,
            "expected": str(expected),
            "relativeError": relative,
            "passed": relative < 3e-10,
        }
    )
px = mp.quad(
    lambda t: mp.exp(-t * t / 2) / mp.sqrt(2 * mp.pi), [0, mp.mpf(1e-20) / mp.mpf(".5")]
)
py = mp.quad(lambda t: mp.exp(-t * t / 2) / mp.sqrt(2 * mp.pi), [-2, 0])
pixel_expected = px * py
pixel_actual = payload["tinyPixel"]["capturedWeight"]
pixel_relative = float(abs(mp.mpf(pixel_actual) - pixel_expected) / pixel_expected)
fn = namespaces["features"]["normal_interval"]
native_tiny = fn(0, 1e-20)
native_normal_results = []
for interval in normal_results:
    actual = fn(interval["a"], interval["b"])
    expected = mp.mpf(interval["expected"])
    relative = float(abs(mp.mpf(actual) - expected) / expected)
    native_normal_results.append(
        {
            "a": interval["a"],
            "b": interval["b"],
            "actual": actual,
            "expected": str(expected),
            "relativeError": relative,
            "passed": relative < 3e-10,
        }
    )
passed = (
    all(c["passed"] for c in normal_results + native_normal_results)
    and pixel_relative < 3e-10
)
production = [
    "src/learn/data/topics/topology-topological-data-analysis-tda.jsx",
    "src/learn/data/topology-tda-models.js",
    "src/learn/data/topology-tda-examples.js",
    "src/learn/components/lesson-labs/TopologyTdaLabs.jsx",
    "src/learn/components/lesson-labs/topology-tda-labs.css",
    "src/learn/data/curriculum/blueprints/topology-topological-data-analysis-tda.js",
]
r = {
    "checkedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "passed" if passed else "normal interval arithmetic finding remains",
    "counts": counts,
    "normalIntegrals": normal_results,
    "nativeNormalIntegrals": native_normal_results,
    "tinyPixel": {
        "actual": pixel_actual,
        "expected": str(pixel_expected),
        "relativeError": pixel_relative,
    },
    "actualNativeTinyInterval": native_tiny,
    "production": [
        {"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()}
        for p in production
    ],
    "limits": "Complementary finite-F2, quotient-representative, bipartite matching, finite-nerve and high-precision integration checks; not a second exhaustive browser run or a proof of general persistence theorems by simulation.",
}
p = root / "scratch/topology-tda-independent-review"
p.mkdir(parents=True, exist_ok=True)
(p / "results.json").write_text(json.dumps(r, indent=2) + "\n")
if not passed and not (p / "initial-arithmetic-finding.json").exists():
    (p / "initial-arithmetic-finding.json").write_text(json.dumps(r, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in r.items() if k not in ["production", "normalIntegrals"]},
        indent=2,
    )
)
