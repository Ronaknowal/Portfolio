"""Independent row-space, triangulation, assignment and quadrature oracles.

Run scripts/export-topology-tda-model-fixtures.mjs first. This script does not
translate the lesson's left-to-right persistence-column implementation.
"""
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp

mp.mp.dps = 65
fixtures = json.loads(Path("scratch/topology-tda-review/model-fixtures.json").read_text())
assertions = 0


def check(condition, context):
    global assertions
    assert condition, context
    assertions += 1


def close(actual, expected, context, relative=2e-11, absolute=2e-14):
    check(abs(actual - expected) <= absolute + relative * abs(expected), (context, actual, expected))


def rref(rows, columns):
    rows = list(rows)
    pivots = []
    rank = 0
    for column in range(columns):
        eligible = next((row for row in range(rank, len(rows)) if rows[row] >> column & 1), None)
        if eligible is None:
            continue
        rows[rank], rows[eligible] = rows[eligible], rows[rank]
        for row in range(len(rows)):
            if row != rank and rows[row] >> column & 1:
                rows[row] ^= rows[rank]
        pivots.append(column)
        rank += 1
    return rows[:rank], pivots


def matrix_rows(columns, row_count):
    return [sum((int(bool(column >> row & 1)) << index) for index, column in enumerate(columns)) for row in range(row_count)]


def boundaries(complex_cells, dimension):
    columns = sorted(cell for cell in complex_cells if len(cell) == dimension + 1)
    rows = sorted(cell for cell in complex_cells if len(cell) == dimension)
    indices = {cell: index for index, cell in enumerate(rows)}
    vectors = []
    for cell in columns:
        vector = 0
        if dimension:
            for excluded in range(len(cell)):
                face = cell[:excluded] + cell[excluded + 1 :]
                vector ^= 1 << indices[face]
        vectors.append(vector)
    return rows, columns, vectors


def betti(complex_cells):
    counts = [sum(len(cell) == dimension + 1 for cell in complex_cells) for dimension in range(4)]
    ranks = [0]
    for dimension in range(1, 4):
        rows, columns, vectors = boundaries(complex_cells, dimension)
        ranks.append(len(rref(matrix_rows(vectors, len(rows)), len(columns))[1]))
    ranks.append(0)
    return [counts[k] - ranks[k] - ranks[k + 1] for k in range(4)]


def image_rank(source, target, dimension):
    lower, source_cells, columns = boundaries(source, dimension)
    reduced_rows, pivots = rref(matrix_rows(columns, len(lower)), len(source_cells))
    free_columns = [column for column in range(len(source_cells)) if column not in pivots]
    cycles = []
    for free in free_columns:
        vector = 1 << free
        for row, pivot in zip(reduced_rows, pivots):
            if row >> free & 1:
                vector |= 1 << pivot
        cycles.append(vector)
    target_cells, higher, target_boundaries = boundaries(target, dimension + 1)
    target_index = {cell: index for index, cell in enumerate(target_cells)}
    embedded = [sum(1 << target_index[cell] for index, cell in enumerate(source_cells) if vector >> index & 1) for vector in cycles]
    boundary_rank = len(rref(matrix_rows(target_boundaries, len(target_cells)), len(higher))[1])
    combined = target_boundaries + embedded
    combined_rank = len(rref(matrix_rows(combined, len(target_cells)), len(combined))[1])
    return combined_rank - boundary_rank


pairs = list(itertools.combinations(range(5), 2))
for case in fixtures["graphCases"]:
    edges = {pair for index, pair in enumerate(pairs) if case["mask"] >> index & 1}
    cells = {(vertex,) for vertex in range(5)} | edges
    cells |= {triangle for triangle in itertools.combinations(range(5), 3) if all(edge in edges for edge in itertools.combinations(triangle, 2))}
    expected = betti(cells)
    check(case["betti"] == expected, ("graph ranks", case["mask"]))
    check([sum(bar["dimension"] == dimension and bar["death"] is None for bar in case["bars"]) for dimension in range(4)] == expected, ("graph essential intervals", case["mask"]))

image_rank_cases = 0
for case in fixtures["pointCases"]:
    births = {tuple(simplex["vertices"]): simplex["birth"] for simplex in case["ordered"]}
    for snapshot in case["snapshots"]:
        threshold = snapshot["threshold"]
        active = {cell for cell, birth in births.items() if birth <= threshold}
        expected = betti(active)
        check(snapshot["betti"] == expected, ("point ranks", case["name"], threshold))
        observed = [sum(bar["dimension"] == dimension and bar["birth"] <= threshold and (bar["death"] is None or bar["death"] > threshold) for bar in case["intervals"]) for dimension in range(4)]
        check(observed == expected, ("barcode snapshot", case["name"], threshold))
    times = [row["threshold"] for row in case["snapshots"]]
    for first, second in itertools.combinations_with_replacement(times, 2):
        source = {cell for cell, birth in births.items() if birth <= first}
        target = {cell for cell, birth in births.items() if birth <= second}
        for dimension in (0, 1):
            expected = image_rank(source, target, dimension)
            observed = sum(bar["dimension"] == dimension and bar["birth"] <= first and (bar["death"] is None or bar["death"] > second) for bar in case["intervals"])
            check(observed == expected, ("persistent image rank", case["name"], first, second, dimension))
            image_rank_cases += 1
    if case["name"] in ("square", "rectangle"):
        positive_h1 = [bar for bar in case["intervals"] if bar["dimension"] == 1 and bar["death"] is not None and bar["death"] > bar["birth"]]
        check(len(positive_h1) == 1, "rectangle has one positive H1 bar")
        scale = case["scale"]
        width, height = (2, 2) if case["name"] == "square" else (3, 1)
        close(positive_h1[0]["birth"], max(width, height) * scale, "rectangle birth")
        close(positive_h1[0]["death"], math.hypot(width, height) * scale, "rectangle death")

for case in fixtures["pixels"]:
    cells = set()
    for pixel in range(9):
        if not case["mask"] >> pixel & 1:
            continue
        x, y = pixel % 3, pixel // 3
        corners = [4 * y + x, 4 * y + x + 1, 4 * (y + 1) + x + 1, 4 * (y + 1) + x]
        for triangle in (corners[:3], [corners[0], corners[2], corners[3]]):
            for size in range(1, 4):
                cells.update(tuple(sorted(face)) for face in itertools.combinations(triangle, size))
    check(case["result"]["betti"] == betti(cells)[:2], ("pixel triangulation", case["mask"]))

worst_normal_relative = 0
worst_normal_range_relative = 0
subnormal_intervals = 0
rounded_to_zero = 0
for case in fixtures["normal"]:
    lower, upper = mp.mpf(case["lower"]), mp.mpf(case["upper"])
    width = upper - lower
    if width * (1 + abs(lower + width / 2)) <= mp.mpf("0.01"):
        midpoint = lower + width / 2
        expected = width * mp.exp(-midpoint**2/2) / mp.sqrt(2*mp.pi) * mp.quad(lambda t: mp.exp(-((lower + width*t)**2-midpoint**2)/2), [0, 1])
    elif lower >= 0:
        expected = (mp.erfc(lower / mp.sqrt(2)) - mp.erfc(upper / mp.sqrt(2))) / 2
    elif upper <= 0:
        expected = (mp.erfc(-upper / mp.sqrt(2)) - mp.erfc(-lower / mp.sqrt(2))) / 2
    else:
        expected = (mp.erf(upper / mp.sqrt(2)) - mp.erf(lower / mp.sqrt(2))) / 2
    relative = float(abs(mp.mpf(case["value"]) - expected) / expected)
    worst_normal_relative = max(worst_normal_relative, relative)
    if expected >= mp.mpf(2.2250738585072014e-308):
        worst_normal_range_relative = max(worst_normal_range_relative, relative)
    else:
        subnormal_intervals += 1
        rounded_to_zero += int(case["value"] == 0)
    close(case["value"], float(expected), "normal interval", relative=2e-10, absolute=1e-323)

for case in fixtures["imageCases"]:
    sigma = mp.mpf(case["bandwidth"])
    for pixel in case["result"]["pixels"]:
        expected_values = []
        for birth, death in case["diagram"]:
            persistence = mp.mpf(death) - mp.mpf(birth)
            weight = min(persistence, 1)
            # Independently integrate the separable normalized density.
            integrals = []
            for center, index, edges in ((mp.mpf(birth), pixel["column"], case.get("xEdges", [0,1,2,3,4])), (persistence, pixel["row"], case.get("yEdges", [0,1,2,3,4]))):
                start, end = mp.mpf(edges[index]), mp.mpf(edges[index+1])
                width = end - start
                nearest = min(max(center, start), end)
                minimum_squared = ((nearest-center)/sigma)**2
                scale = width * mp.exp(-minimum_squared/2)/(sigma*mp.sqrt(2*mp.pi))
                integrals.append(scale * mp.quad(lambda t: mp.exp(-(((start + width*t-center)/sigma)**2-minimum_squared)/2), [0, 1]))
            expected_values.append(float(weight * integrals[0] * integrals[1]))
        for observed, expected in zip(pixel["contributions"], expected_values):
            close(observed, expected, "Gaussian pixel quadrature", relative=3e-10, absolute=1e-323)
        close(pixel["value"], sum(expected_values), "pixel total")
    close(case["result"]["capturedWeight"] + case["result"]["omittedWeight"], case["result"]["totalWeight"], "image mass accounting")

for case in fixtures["matching"]:
    first, second = case["first"], case["second"]
    size = len(first) + len(second)
    costs = []
    for row in range(size):
        values = []
        for column in range(size):
            if row < len(first) and column < len(second):
                cost = max(abs(first[row][0] - second[column][0]), abs(first[row][1] - second[column][1]))
            elif row < len(first):
                cost = (first[row][1] - first[row][0]) / 2
            elif column < len(second):
                cost = (second[column][1] - second[column][0]) / 2
            else:
                cost = 0
            values.append(cost)
        costs.append(values)
    power = math.inf if case["power"] == "Infinity" else float(case["power"])
    candidates = []
    for permutation in itertools.permutations(range(size)):
        values = [costs[row][column] for row, column in enumerate(permutation)]
        candidates.append(max(values, default=0) if power == math.inf else sum(value**power for value in values)**(1/power))
    close(case["result"]["value"], min(candidates), "augmented assignment oracle")

for case in fixtures["mapper"]:
    result = case["result"]
    expected_nodes = []
    for index, interval in enumerate(result["cover"]):
        members = [point for point, coordinates in enumerate(result["points"]) if interval["lower"] <= coordinates[0] <= interval["upper"]]
        reach = [[math.dist(result["points"][left], result["points"][right]) <= case["clusterDistance"] for right in members] for left in members]
        for middle in range(len(members)):
            for left in range(len(members)):
                for right in range(len(members)):
                    reach[left][right] |= reach[left][middle] and reach[middle][right]
        components = sorted({tuple(members[right] for right in range(len(members)) if reach[left][right]) for left in range(len(members))})
        expected_nodes.extend((index, component) for component in components)
    check([(node["interval"], tuple(node["members"])) for node in result["nodes"]] == expected_nodes, "Mapper transitive closure")
    expected_edges = {(first, second): sorted(set(expected_nodes[first][1]) & set(expected_nodes[second][1])) for first, second in itertools.combinations(range(len(expected_nodes)), 2)}
    expected_edges = {key: members for key, members in expected_edges.items() if members}
    check({(edge["first"], edge["second"]): edge["members"] for edge in result["edges"]} == expected_edges, "Mapper nerve intersections")
    check(all(result["membership"]), "cover includes every observation")

result = {"passed": True, "checkedAt": datetime.now(timezone.utc).isoformat(), "assertions": assertions,
          "persistentImageRankComparisons": image_rank_cases, "graphCases": len(fixtures["graphCases"]),
          "pixelTriangulations": len(fixtures["pixels"]), "worstNormalIntervalRelativeError": worst_normal_relative,
          "worstNormalRangeRelativeError": worst_normal_range_relative,
          "subnormalIntervals": subnormal_intervals, "underflowedIntervals": rounded_to_zero,
          "fixtureCreatedAt": fixtures["createdAt"], "limits": "Finite bounded cases; no general theorem, empirical topology inference or browser review is established by this check."}
Path("scratch/topology-tda-review/model-oracle-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
